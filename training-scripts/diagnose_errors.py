#!/usr/bin/env python3
"""Diagnose model errors on gasbench-cached datasets.
Identifies specific wrong predictions with confidence levels.
"""
import os
import io
import json
import subprocess
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np

sys.path.insert(0, '/root/video_detector_v4')
from model import load_model

CACHE_DIR = Path('/.cache/gasbench/datasets')
MODEL_DIR = '/root/video_detector_v4'
FRAME_SIZE = 224
NUM_FRAMES = 16  # match gasbench input

def extract_frames(video_path, num_frames=NUM_FRAMES, size=FRAME_SIZE):
    """Extract frames from video using ffmpeg."""
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
                frames.append(img)
    except Exception as e:
        pass
    return frames

def frames_to_tensor(frames):
    """Convert frames to model input tensor [1, T, 3, H, W] in 0-255."""
    tfm = transforms.Compose([
        transforms.Resize((FRAME_SIZE, FRAME_SIZE)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255.0),
    ])
    tensors = [tfm(f) for f in frames]
    if not tensors:
        return None
    # Pad or truncate to NUM_FRAMES
    while len(tensors) < NUM_FRAMES:
        tensors.append(tensors[-1])
    tensors = tensors[:NUM_FRAMES]
    return torch.stack(tensors).unsqueeze(0)  # [1, T, 3, H, W]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(os.path.join(MODEL_DIR, 'model.safetensors'))
    model = model.to(device)
    model.eval()

    print(f"Model temperature: {model.temperature.item():.3f}")

    # Get video datasets
    datasets = {}
    for d in sorted(CACHE_DIR.iterdir()):
        info_file = d / 'dataset_info.json'
        if not info_file.exists():
            continue
        with open(info_file) as f:
            info = json.load(f)
        if info.get('modality') != 'video':
            continue
        datasets[d.name] = info

    print(f"\nFound {len(datasets)} video datasets")

    total_wrong = 0
    total_right = 0
    wrong_details = []

    for name, info in sorted(datasets.items()):
        media_type = info.get('media_type', 'unknown')
        true_label = 1 if media_type in ('synthetic', 'semisynthetic') else 0

        samples_dir = CACHE_DIR / name / 'samples'
        videos = sorted(samples_dir.glob('*.mp4'))
        if not videos:
            continue

        correct = 0
        wrong = 0
        wrong_list = []

        for vf in videos:
            frames = extract_frames(vf)
            if not frames:
                continue
            tensor = frames_to_tensor(frames)
            if tensor is None:
                continue

            with torch.no_grad():
                logits = model(tensor.to(device))
                probs = F.softmax(logits, dim=1)
                pred = logits.argmax(1).item()
                conf_real = probs[0, 0].item()
                conf_synth = probs[0, 1].item()

            if pred == true_label:
                correct += 1
            else:
                wrong += 1
                wrong_list.append({
                    'file': vf.name,
                    'true': 'real' if true_label == 0 else 'synthetic',
                    'pred': 'real' if pred == 0 else 'synthetic',
                    'conf_real': conf_real,
                    'conf_synth': conf_synth,
                })

        total_right += correct
        total_wrong += wrong
        acc = correct / max(correct + wrong, 1) * 100
        status = f"{'OK' if wrong == 0 else 'WRONG'}"
        print(f"  {name} ({media_type}): {acc:.0f}% ({correct}/{correct+wrong}) {status}")

        for w in wrong_list:
            conf = max(w['conf_real'], w['conf_synth'])
            print(f"    WRONG: {w['file']} true={w['true']} pred={w['pred']} "
                  f"P(real)={w['conf_real']:.4f} P(synth)={w['conf_synth']:.4f}")
            wrong_details.append({**w, 'dataset': name, 'media_type': media_type})

    print(f"\n{'='*60}")
    print(f"SUMMARY: {total_right}/{total_right+total_wrong} correct ({100*total_right/(total_right+total_wrong):.1f}%)")
    print(f"Wrong: {total_wrong}")

    # Analyze confidence on wrong predictions
    if wrong_details:
        print(f"\nWRONG PREDICTION ANALYSIS:")
        high_conf_wrong = sum(1 for w in wrong_details if max(w['conf_real'], w['conf_synth']) > 0.9)
        med_conf_wrong = sum(1 for w in wrong_details if 0.7 < max(w['conf_real'], w['conf_synth']) <= 0.9)
        low_conf_wrong = sum(1 for w in wrong_details if max(w['conf_real'], w['conf_synth']) <= 0.7)
        print(f"  High confidence (>0.9): {high_conf_wrong}")
        print(f"  Medium confidence (0.7-0.9): {med_conf_wrong}")
        print(f"  Low confidence (<0.7): {low_conf_wrong}")

        synth_as_real = sum(1 for w in wrong_details if w['true'] == 'synthetic' and w['pred'] == 'real')
        real_as_synth = sum(1 for w in wrong_details if w['true'] == 'real' and w['pred'] == 'synthetic')
        print(f"  Synthetic→Real: {synth_as_real}")
        print(f"  Real→Synthetic: {real_as_synth}")

if __name__ == '__main__':
    main()
