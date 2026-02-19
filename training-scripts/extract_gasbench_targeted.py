#!/usr/bin/env python3
"""Extract frames from gasbench-cached video datasets on vast.ai for targeted training.
Focuses on pe-video (real) and semisynthetic-video to fix model failures.
Also extracts from other cached real video datasets for diversity.
"""
import os
import io
import json
import subprocess
import random
from pathlib import Path
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

CACHE_DIR = '/.cache/gasbench/datasets'
OUTPUT_DIR = '/root/train_data'
FRAMES_PER_VIDEO = 8
FRAME_SIZE = 224

# Datasets to extract and their labels
# real=0, synthetic=1, semisynthetic=1
TARGETS = {
    # Critical - model fails on these
    'pe-video': 0,           # real - model at 17% accuracy
    'semisynthetic-video': 1, # semisynthetic - model at 28%
    # Additional real video datasets for diversity
    'deepaction-pexels': 0,   # real - already good but adds diversity
    # Additional synthetic that were weaker
    'ByteDance_Synthetic_Videos': 1,
    'klingai-videos': 1,
    'moviegen-bench': 1,
    'aura-video': 1,
}


def extract_frames_ffmpeg(video_path, num_frames=FRAMES_PER_VIDEO, size=FRAME_SIZE):
    frames = []
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', str(video_path)],
            capture_output=True, text=True, timeout=10
        )
        info = json.loads(result.stdout)
        duration = float(info['format'].get('duration', 5))

        for i in range(num_frames):
            t = duration * (i + 0.5) / num_frames
            result = subprocess.run(
                ['ffmpeg', '-ss', str(t), '-i', str(video_path),
                 '-vframes', '1', '-f', 'image2pipe', '-vcodec', 'mjpeg', '-'],
                capture_output=True, timeout=15
            )
            if result.returncode == 0 and len(result.stdout) > 100:
                img = Image.open(io.BytesIO(result.stdout)).convert('RGB')
                img = img.resize((size, size), Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format='JPEG', quality=95)
                frames.append(buf.getvalue())
    except Exception:
        pass
    return frames


def process_dataset(name, label):
    dataset_dir = Path(CACHE_DIR) / name
    samples_dir = dataset_dir / 'samples'
    if not samples_dir.exists():
        print(f'  {name}: no samples dir, skipping')
        return []

    video_files = sorted(samples_dir.glob('*.mp4'))
    if not video_files:
        print(f'  {name}: no mp4 files')
        return []

    print(f'  {name}: processing {len(video_files)} videos (label={label})')
    samples = []
    for i, vf in enumerate(video_files):
        frames = extract_frames_ffmpeg(vf)
        for fb in frames:
            samples.append((fb, label))
        if (i + 1) % 20 == 0:
            print(f'    {i+1}/{len(video_files)} ({len(samples)} frames)')

    print(f'  {name}: {len(samples)} frames extracted')
    return samples


def save_parquet(samples, output_path):
    if not samples:
        return
    images = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    image_dicts = [{'bytes': img_bytes} for img_bytes in images]
    table = pa.table({'image': image_dicts, 'label': labels})
    pq.write_table(table, output_path)
    sz = os.path.getsize(output_path) / 1e6
    print(f'Saved {len(samples)} frames to {output_path} ({sz:.1f} MB)')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Group by what we're fixing
    pe_video_samples = []
    semi_samples = []
    synth_boost_samples = []

    for name, label in TARGETS.items():
        samples = process_dataset(name, label)
        if name == 'pe-video':
            pe_video_samples.extend(samples)
        elif name == 'semisynthetic-video':
            semi_samples.extend(samples)
        elif label == 0:
            pe_video_samples.extend(samples)  # Add to real pool
        else:
            synth_boost_samples.extend(samples)

    # Save targeted parquets
    if pe_video_samples:
        save_parquet(pe_video_samples, f'{OUTPUT_DIR}/video_pe_real.parquet')
    if semi_samples:
        save_parquet(semi_samples, f'{OUTPUT_DIR}/video_bitmind_semi.parquet')
    if synth_boost_samples:
        save_parquet(synth_boost_samples, f'{OUTPUT_DIR}/video_synth_boost.parquet')

    print(f'\nDone! PE-video real: {len(pe_video_samples)}, Semi: {len(semi_samples)}, Synth boost: {len(synth_boost_samples)}')


if __name__ == '__main__':
    main()
