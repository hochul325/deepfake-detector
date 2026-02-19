#!/usr/bin/env python3
"""Download audio training data on vast.ai - using soundfile for decoding."""
import os, io, gc, struct
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torchaudio
import torch

OUTPUT_DIR = '/root/audio_train_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_SR = 16000
TARGET_SAMPLES = 96000

# Disable torchcodec, force soundfile
os.environ['HF_AUDIO_DECODER'] = 'soundfile'

def decode_audio_bytes(audio_bytes, target_sr=TARGET_SR):
    """Decode audio bytes using soundfile."""
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes))
        if data.ndim > 1:
            data = data.mean(axis=1)  # mono
        data = data.astype(np.float32)
        # Resample
        if sr != target_sr:
            waveform = torch.from_numpy(data).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)
            data = waveform.squeeze(0).numpy()
        return data
    except Exception:
        return None

def process_to_fixed_length(arr, target_samples=TARGET_SAMPLES):
    """Normalize and pad/crop to fixed length."""
    if arr is None or len(arr) < target_samples // 4:
        return None
    max_val = np.abs(arr).max()
    if max_val < 1e-6:
        return None
    arr = arr / max_val
    if len(arr) < target_samples:
        arr = np.pad(arr, (0, target_samples - len(arr)))
    elif len(arr) > target_samples:
        start = (len(arr) - target_samples) // 2
        arr = arr[start:start + target_samples]
    return arr.astype(np.float32).tobytes()

def download_parquet_audio(hf_path, config=None, max_samples=2000, split="train"):
    """Download audio from HF parquet files directly."""
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()

    # List parquet files
    try:
        files = api.list_repo_files(hf_path, repo_type="dataset")
    except Exception as e:
        print("  Cannot list files: " + str(e))
        return []

    parquet_files = [f for f in files if f.endswith('.parquet')]
    if config:
        parquet_files = [f for f in parquet_files if config in f or 'default' in f]
    if not parquet_files:
        # Try with split
        parquet_files = [f for f in files if f.endswith('.parquet') and split in f]
    if not parquet_files:
        parquet_files = [f for f in files if f.endswith('.parquet')]

    # Sort and limit
    parquet_files = sorted(parquet_files)[:10]  # max 10 parquet files

    audio_arrays = []
    for pf in parquet_files:
        if len(audio_arrays) >= max_samples:
            break
        try:
            local_path = hf_hub_download(hf_path, pf, repo_type="dataset")
            table = pq.read_table(local_path)
            df = table.to_pandas()

            # Find audio column
            audio_col = None
            for col in ['audio', 'speech', 'wav', 'waveform', 'output_audio']:
                if col in df.columns:
                    audio_col = col
                    break
            if audio_col is None:
                continue

            for idx in range(len(df)):
                if len(audio_arrays) >= max_samples:
                    break
                try:
                    val = df[audio_col].iloc[idx]
                    audio_bytes = None
                    if isinstance(val, dict):
                        audio_bytes = val.get('bytes')
                    elif isinstance(val, bytes):
                        audio_bytes = val

                    if audio_bytes is None:
                        continue

                    arr = decode_audio_bytes(audio_bytes)
                    if arr is not None:
                        result = process_to_fixed_length(arr)
                        if result is not None:
                            audio_arrays.append(result)
                except Exception:
                    continue

            if len(audio_arrays) % 500 < 10:
                print("  ..." + str(len(audio_arrays)) + "/" + str(max_samples))
        except Exception as e:
            print("  Parquet error: " + str(e))
            continue
        # Clean up
        try:
            os.remove(local_path)
        except Exception:
            pass

    return audio_arrays

def save_parquet(audio_bytes_list, label, outpath):
    if not audio_bytes_list:
        return
    table = pa.table({
        "audio_bytes": audio_bytes_list,
        "label": [label] * len(audio_bytes_list)
    })
    pq.write_table(table, outpath, compression="snappy")
    size_mb = os.path.getsize(outpath) / (1024**2)
    print("  Saved: " + str(len(audio_bytes_list)) + " samples, " + str(round(size_mb, 1)) + " MB")


datasets_to_download = [
    # Real speech
    {"name": "mozilla-foundation/common_voice_17_0", "config": "en", "output": "real_common_voice_en.parquet",
     "label": 0, "samples": 3000},
    {"name": "parler-tts/mls_eng_10k", "output": "real_mls_eng.parquet",
     "label": 0, "samples": 2000},
    {"name": "ymoslem/MediaSpeech", "config": "ar", "output": "real_mediaspeech.parquet",
     "label": 0, "samples": 2000},
    {"name": "qmeeus/slurp", "output": "real_slurp.parquet",
     "label": 0, "samples": 2000},
    {"name": "simon3000/genshin-voice", "output": "real_genshin.parquet",
     "label": 0, "samples": 2000},

    # Synthetic - ElevenLabs
    {"name": "skypro1111/elevenlabs_dataset", "output": "synth_elevenlabs_1.parquet",
     "label": 1, "samples": 1300},
    {"name": "velocity-engg/eleven_labs_dataset", "output": "synth_elevenlabs_2.parquet",
     "label": 1, "samples": 1000},
    {"name": "NeoBoy/elevenlabsSpeechTest", "output": "synth_elevenlabs_3.parquet",
     "label": 1, "samples": 800},
    {"name": "velocity-engg/eleven_labs_datase_latin", "output": "synth_elevenlabs_latin.parquet",
     "label": 1, "samples": 1000},

    # Synthetic - other
    {"name": "DeepFake-Audio-Rangers/Arabic_Audio_Deepfake", "output": "synth_arabic_deepfake.parquet",
     "label": 1, "samples": 2000},
    {"name": "Thorsten-Voice/TV-44kHz-Full", "output": "synth_thorsten.parquet",
     "label": 1, "samples": 2000},
]

for ds in datasets_to_download:
    outpath = os.path.join(OUTPUT_DIR, ds["output"])
    if os.path.exists(outpath):
        print("SKIP " + ds["name"] + " (exists)")
        continue

    print("Downloading: " + ds["name"])
    audio_data = download_parquet_audio(
        ds["name"],
        config=ds.get("config"),
        max_samples=ds["samples"]
    )
    save_parquet(audio_data, ds["label"], outpath)
    gc.collect()

print("\nDone! Files:")
if os.path.isdir(OUTPUT_DIR):
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / (1024**2)
        print("  " + f + " (" + str(round(size, 1)) + " MB)")
