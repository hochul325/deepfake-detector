#!/usr/bin/env python3
"""Download audio training data - focused on synthetic sources. Compact storage."""
import os, io, gc
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torchaudio
import torch
from huggingface_hub import HfApi, hf_hub_download

OUTPUT_DIR = '/root/audio_train_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_SR = 16000
TARGET_SAMPLES = 96000

def decode_and_process(audio_bytes):
    """Decode audio bytes, resample, normalize, pad/crop."""
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


def download_from_hf(hf_path, config=None, max_samples=2000):
    """Download audio from HF parquet files."""
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

    parquet_files = parquet_files[:5]  # limit files to download
    results = []

    for pf in parquet_files:
        if len(results) >= max_samples:
            break
        try:
            local = hf_hub_download(hf_path, pf, repo_type="dataset")
            table = pq.read_table(local)
            df = table.to_pandas()

            audio_col = None
            for col in ['audio', 'speech', 'wav', 'waveform', 'output_audio']:
                if col in df.columns:
                    audio_col = col
                    break
            if not audio_col:
                print("  No audio col in " + pf + ", cols=" + str(list(df.columns)))
                continue

            for idx in range(len(df)):
                if len(results) >= max_samples:
                    break
                try:
                    val = df[audio_col].iloc[idx]
                    audio_bytes = val.get('bytes') if isinstance(val, dict) else (val if isinstance(val, bytes) else None)
                    if not audio_bytes:
                        continue
                    processed = decode_and_process(audio_bytes)
                    if processed:
                        results.append(processed)
                except Exception:
                    continue

            print("  " + pf + " -> " + str(len(results)) + " total")
            os.remove(local)
        except Exception as e:
            print("  Error on " + pf + ": " + str(e))
    return results


def save(data_list, label, outpath):
    if not data_list:
        print("  WARNING: No data to save")
        return
    table = pa.table({"audio_bytes": data_list, "label": [label] * len(data_list)})
    pq.write_table(table, outpath, compression="snappy")
    size_mb = os.path.getsize(outpath) / (1024**2)
    print("  -> " + str(len(data_list)) + " samples, " + str(round(size_mb, 1)) + " MB")


# Focus on synthetic datasets we don't have yet
tasks = [
    # ElevenLabs (key synthetic source in gasbench)
    {"name": "skypro1111/elevenlabs_dataset", "output": "synth_elevenlabs_1.parquet", "label": 1, "samples": 1300},
    {"name": "velocity-engg/eleven_labs_dataset", "output": "synth_elevenlabs_2.parquet", "label": 1, "samples": 1000},
    {"name": "NeoBoy/elevenlabsSpeechTest", "output": "synth_elevenlabs_3.parquet", "label": 1, "samples": 800},
    {"name": "velocity-engg/eleven_labs_datase_latin", "output": "synth_elevenlabs_latin.parquet", "label": 1, "samples": 1000},
    # Arabic deepfake
    {"name": "DeepFake-Audio-Rangers/Arabic_Audio_Deepfake", "output": "synth_arabic_deepfake.parquet", "label": 1, "samples": 2000},
]

for task in tasks:
    outpath = os.path.join(OUTPUT_DIR, task["output"])
    if os.path.exists(outpath):
        print("SKIP " + task["name"])
        continue
    print("Downloading: " + task["name"])
    data = download_from_hf(task["name"], max_samples=task["samples"])
    save(data, task["label"], outpath)
    gc.collect()
    # Clean HF cache after each
    import shutil
    cache_dir = os.path.expanduser("/workspace/.hf_home/hub")
    for d in os.listdir(cache_dir):
        if d.startswith("datasets--"):
            shutil.rmtree(os.path.join(cache_dir, d), ignore_errors=True)

print("\nDone! All files:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / (1024**2)
    print("  " + f + " (" + str(round(size, 1)) + " MB)")
print("Disk: ", end="")
os.system("df -h / | tail -1")
