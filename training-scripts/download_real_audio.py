#!/usr/bin/env python3
"""Download real audio data - limited to 1000 samples each for disk efficiency."""
import os, io, gc
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torchaudio
import torch
from huggingface_hub import HfApi, hf_hub_download

OUTPUT_DIR = '/root/audio_train_data'
TARGET_SR = 16000
TARGET_SAMPLES = 96000

def decode_and_process(audio_bytes):
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes))
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
    except Exception:
        return None

def download_from_hf(hf_path, config=None, max_samples=1000):
    api = HfApi()
    try:
        files = api.list_repo_files(hf_path, repo_type="dataset")
    except Exception as e:
        print("  Cannot list: " + str(e))
        return []
    parquet_files = sorted([f for f in files if f.endswith('.parquet')])
    if config:
        filtered = [f for f in parquet_files if config in f]
        if filtered:
            parquet_files = filtered
    parquet_files = parquet_files[:3]
    results = []
    for pf in parquet_files:
        if len(results) >= max_samples:
            break
        try:
            local = hf_hub_download(hf_path, pf, repo_type="dataset")
            table = pq.read_table(local)
            df = table.to_pandas()
            audio_col = None
            for col in ['audio', 'speech', 'wav']:
                if col in df.columns:
                    audio_col = col
                    break
            if not audio_col:
                print("  No audio col, cols=" + str(list(df.columns)))
                os.remove(local)
                continue
            for idx in range(len(df)):
                if len(results) >= max_samples:
                    break
                try:
                    val = df[audio_col].iloc[idx]
                    ab = val.get('bytes') if isinstance(val, dict) else (val if isinstance(val, bytes) else None)
                    if not ab:
                        continue
                    p = decode_and_process(ab)
                    if p:
                        results.append(p)
                except Exception:
                    continue
            print("  " + pf + " -> " + str(len(results)))
            os.remove(local)
        except Exception as e:
            print("  Error: " + str(e))
    return results

def save(data_list, label, outpath):
    if not data_list:
        return
    table = pa.table({"audio_bytes": data_list, "label": [label] * len(data_list)})
    pq.write_table(table, outpath, compression="snappy")
    size_mb = os.path.getsize(outpath) / (1024**2)
    print("  -> " + str(len(data_list)) + " samples, " + str(round(size_mb, 1)) + " MB")

tasks = [
    {"name": "parler-tts/mls_eng_10k", "output": "real_mls_eng_1k.parquet", "samples": 1000},
    {"name": "ymoslem/MediaSpeech", "config": "ar", "output": "real_mediaspeech_1k.parquet", "samples": 1000},
    {"name": "qmeeus/slurp", "output": "real_slurp_1k.parquet", "samples": 1000},
]

for task in tasks:
    outpath = os.path.join(OUTPUT_DIR, task["output"])
    if os.path.exists(outpath):
        print("SKIP " + task["name"])
        continue
    print("Downloading: " + task["name"])
    data = download_from_hf(task["name"], config=task.get("config"), max_samples=task["samples"])
    save(data, 0, outpath)
    gc.collect()
    import shutil
    cache_dir = "/workspace/.hf_home/hub"
    if os.path.isdir(cache_dir):
        for d in os.listdir(cache_dir):
            if d.startswith("datasets--"):
                shutil.rmtree(os.path.join(cache_dir, d), ignore_errors=True)

print("\nAll files:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / (1024**2)
    print("  " + f + " (" + str(round(size, 1)) + " MB)")
