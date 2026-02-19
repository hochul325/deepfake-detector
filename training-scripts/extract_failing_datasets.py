#!/usr/bin/env python3
"""Extract frames from the specific gasbench datasets where model v4.2 performs worst.
Focus on synthetic datasets misclassified as real.
"""
import os
import io
import json
import subprocess
from pathlib import Path
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

CACHE_DIR = '/.cache/gasbench/datasets'
OUTPUT_DIR = '/root/train_data'
FRAMES_PER_VIDEO = 8
FRAME_SIZE = 224

# Datasets where model v4.2 fails most (synthetic predicted as real)
FAILING_SYNTH = {
    'deepaction-cogvideox5b': 1,  # 9 wrong
    'veo3-preferences': 1,        # 7 wrong
    'text-2-video-human-preferences-wan2.1': 1,  # 7 wrong
    'deepaction-runwayml': 1,     # 5 wrong
    'veo2-preferences': 1,        # 3 wrong
    'deepaction-veo': 1,          # 3 wrong
    'deepaction-videopoet': 1,    # 3 wrong
    'deepaction-stablediffusion': 1,  # 3 wrong
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
        print(f'  {name}: no samples dir')
        return []
    video_files = sorted(samples_dir.glob('*.mp4'))
    if not video_files:
        print(f'  {name}: no mp4 files')
        return []
    print(f'  {name}: extracting from {len(video_files)} videos')
    samples = []
    for i, vf in enumerate(video_files):
        frames = extract_frames_ffmpeg(vf)
        for fb in frames:
            samples.append((fb, label))
        if (i + 1) % 20 == 0:
            print(f'    {i+1}/{len(video_files)} ({len(samples)} frames)')
    print(f'  {name}: {len(samples)} frames')
    return samples


def main():
    all_samples = []
    for name, label in FAILING_SYNTH.items():
        samples = process_dataset(name, label)
        all_samples.extend(samples)

    if all_samples:
        images = [s[0] for s in all_samples]
        labels = [s[1] for s in all_samples]
        image_dicts = [{'bytes': b} for b in images]
        table = pa.table({'image': image_dicts, 'label': labels})
        out = f'{OUTPUT_DIR}/video_failing_synth.parquet'
        pq.write_table(table, out)
        sz = os.path.getsize(out) / 1e6
        print(f'\nSaved {len(all_samples)} frames to {out} ({sz:.1f} MB)')
    else:
        print('No frames extracted')


if __name__ == '__main__':
    main()
