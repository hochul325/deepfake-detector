#!/usr/bin/env python3
"""Extract audio training data from gasbench cache."""
import os, io, json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torchaudio
import torch

CACHE_DIR = '/.cache/gasbench/datasets'
OUTPUT_DIR = '/root/audio_train_data'
TARGET_SR = 16000
TARGET_SAMPLES = 96000

os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_audio_file(filepath):
    """Load an audio file and convert to 16kHz, 6s float32."""
    try:
        data, sr = sf.read(filepath)
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)
        if sr != TARGET_SR:
            waveform = torch.from_numpy(data).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
            waveform = resampler(waveform)
            data = waveform.squeeze(0).numpy()
        max_val = np.abs(data).max()
        if max_val < 1e-6:
            return None
        data = data / max_val
        if len(data) < TARGET_SAMPLES // 4:
            return None
        if len(data) < TARGET_SAMPLES:
            data = np.pad(data, (0, TARGET_SAMPLES - len(data)))
        elif len(data) > TARGET_SAMPLES:
            start = (len(data) - TARGET_SAMPLES) // 2
            data = data[start:start + TARGET_SAMPLES]
        return data.astype(np.float32).tobytes()
    except Exception as e:
        return None

AUDIO_EXTS = ('.wav', '.mp3', '.flac', '.ogg', '.opus', '.m4a', '.webm')

# Process each cached audio dataset
for ds_name in sorted(os.listdir(CACHE_DIR)):
    ds_dir = os.path.join(CACHE_DIR, ds_name, 'samples')
    if not os.path.isdir(ds_dir):
        continue

    # Check if it has audio files
    audio_files = []
    for f in sorted(os.listdir(ds_dir)):
        if f.lower().endswith(AUDIO_EXTS):
            audio_files.append(os.path.join(ds_dir, f))

    if not audio_files:
        continue

    # Determine label from dataset_info.json
    info_path = os.path.join(CACHE_DIR, ds_name, 'dataset_info.json')
    label = 0  # default real
    if os.path.exists(info_path):
        try:
            with open(info_path) as f:
                info = json.load(f)
            media_type = info.get('media_type', 'real')
            if media_type in ('synthetic', 'semisynthetic'):
                label = 1
        except Exception:
            pass

    safe_name = ds_name.replace('-', '_')
    prefix = 'synth' if label == 1 else 'real'
    outpath = os.path.join(OUTPUT_DIR, prefix + '_cache_' + safe_name + '.parquet')
    if os.path.exists(outpath):
        print("SKIP " + ds_name + " (exists)")
        continue

    print("Processing " + ds_name + " (" + str(len(audio_files)) + " files, label=" + str(label) + ")")
    audio_bytes_list = []
    for af in audio_files:
        result = process_audio_file(af)
        if result:
            audio_bytes_list.append(result)

    if audio_bytes_list:
        table = pa.table({"audio_bytes": audio_bytes_list, "label": [label] * len(audio_bytes_list)})
        pq.write_table(table, outpath, compression="snappy")
        size_mb = os.path.getsize(outpath) / (1024**2)
        print("  -> " + str(len(audio_bytes_list)) + " samples, " + str(round(size_mb, 1)) + " MB")
    else:
        print("  No valid audio extracted")

print("\nAll audio training files:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / (1024**2)
    print("  " + f + " (" + str(round(size, 1)) + " MB)")
