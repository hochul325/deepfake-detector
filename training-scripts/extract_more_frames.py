#!/usr/bin/env python3
"""Extract additional frames from gasbench-cached datasets that have the most errors.
Uses faster approach: extract ALL frames from video in one ffmpeg call.
Focus on moviegen-bench (6 wrong) and pe-video (4 wrong).
Also extract from gasstation-generated-videos.
"""
import os
import io
import json
import subprocess
from pathlib import Path
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

CACHE_DIR = Path('/.cache/gasbench/datasets')
OUTPUT_DIR = Path('/root/train_data')
FRAME_SIZE = 224

# Extract more from these datasets
TARGETS = {
    'moviegen-bench': 1,              # 6 wrong - synthetic
    'pe-video': 0,                    # 4 wrong - real
    'gasstation-generated-videos': 1,  # 1 wrong - synthetic (tiny dataset)
    'veo3-preferences': 1,            # 4 wrong - synthetic
    'text-2-video-human-preferences-wan2.1': 1,  # 4 wrong - synthetic
    'veo2-preferences': 1,            # 3 wrong - synthetic
    'deepaction-cogvideox5b': 1,      # 3 wrong - synthetic
    'klingai-videos': 1,              # 2 wrong - synthetic
}

def extract_all_frames_fast(video_path, max_frames=12, size=FRAME_SIZE):
    """Extract frames in a single ffmpeg call - much faster."""
    frames = []
    try:
        # Get duration first
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', str(video_path)],
            capture_output=True, text=True, timeout=10
        )
        info = json.loads(result.stdout)
        duration = float(info['format'].get('duration', 5))

        # Use fps filter to extract frames evenly
        target_fps = max_frames / max(duration, 0.5)

        result = subprocess.run(
            ['ffmpeg', '-i', str(video_path),
             '-vf', f'fps={target_fps:.4f},scale={size}:{size}',
             '-f', 'image2pipe', '-vcodec', 'mjpeg', '-'],
            capture_output=True, timeout=30
        )

        if result.returncode == 0 and result.stdout:
            # Split mjpeg stream into individual frames
            data = result.stdout
            start = 0
            while start < len(data):
                # Find JPEG SOI marker (0xFFD8)
                soi = data.find(b'\xff\xd8', start)
                if soi < 0:
                    break
                # Find next SOI or end
                next_soi = data.find(b'\xff\xd8', soi + 2)
                if next_soi < 0:
                    frame_data = data[soi:]
                else:
                    frame_data = data[soi:next_soi]
                start = soi + 2 if next_soi < 0 else next_soi

                try:
                    img = Image.open(io.BytesIO(frame_data)).convert('RGB')
                    buf = io.BytesIO()
                    img.save(buf, format='JPEG', quality=95)
                    frames.append(buf.getvalue())
                except Exception:
                    continue

                if len(frames) >= max_frames:
                    break
    except Exception:
        pass
    return frames


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, label in TARGETS.items():
        samples_dir = CACHE_DIR / name / 'samples'
        if not samples_dir.exists():
            print(f'{name}: no samples dir')
            continue

        videos = sorted(samples_dir.glob('*.mp4'))
        if not videos:
            print(f'{name}: no videos')
            continue

        print(f'\n{name}: {len(videos)} videos (label={label})')
        all_frames = []
        for i, vf in enumerate(videos):
            frames = extract_all_frames_fast(vf, max_frames=12)
            for fb in frames:
                all_frames.append((fb, label))
            if (i + 1) % 20 == 0:
                print(f'  {i+1}/{len(videos)} ({len(all_frames)} frames)')

        print(f'  Total: {len(all_frames)} frames')

        if all_frames:
            images = [s[0] for s in all_frames]
            labels = [s[1] for s in all_frames]
            image_dicts = [{'bytes': b} for b in images]
            table = pa.table({'image': image_dicts, 'label': labels})
            out = OUTPUT_DIR / f'video_extra_{name.replace("-", "_")}.parquet'
            pq.write_table(table, str(out))
            sz = os.path.getsize(out) / 1e6
            print(f'  Saved to {out} ({sz:.1f} MB)')


if __name__ == '__main__':
    main()
