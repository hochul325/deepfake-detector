#!/usr/bin/env python3
"""Download audio training data directly on vast.ai."""
import os, io, gc
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
import soundfile as sf
import torchaudio

OUTPUT_DIR = '/root/audio_train_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_SR = 16000
TARGET_SAMPLES = 96000  # 6 seconds

def process_audio(audio_data):
    """Convert HF audio dict to float32 array at 16kHz, 6 seconds."""
    if isinstance(audio_data, dict):
        arr = audio_data.get('array')
        sr = audio_data.get('sampling_rate', TARGET_SR)
        if arr is None:
            return None
        arr = np.array(arr, dtype=np.float32)
    else:
        return None

    # Resample if needed
    if sr != TARGET_SR:
        import torch
        waveform = torch.from_numpy(arr).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
        waveform = resampler(waveform)
        arr = waveform.squeeze(0).numpy()

    # Normalize
    max_val = np.abs(arr).max()
    if max_val > 1e-6:
        arr = arr / max_val
    else:
        return None  # Silent audio

    # Pad or center-crop
    if len(arr) < TARGET_SAMPLES // 4:
        return None  # Too short
    if len(arr) < TARGET_SAMPLES:
        arr = np.pad(arr, (0, TARGET_SAMPLES - len(arr)))
    elif len(arr) > TARGET_SAMPLES:
        start = (len(arr) - TARGET_SAMPLES) // 2
        arr = arr[start:start + TARGET_SAMPLES]

    return arr.astype(np.float32).tobytes()


datasets_to_download = [
    # Real speech - diverse sources matching gasbench
    {"name": "mozilla-foundation/common_voice_17_0", "config": "en", "output": "real_common_voice_en.parquet",
     "label": 0, "samples": 3000, "audio_col": "audio"},
    {"name": "parler-tts/mls_eng_10k", "output": "real_mls_eng.parquet",
     "label": 0, "samples": 2000, "audio_col": "audio"},
    {"name": "ymoslem/MediaSpeech", "config": "ar", "output": "real_mediaspeech_ar.parquet",
     "label": 0, "samples": 1000, "audio_col": "audio"},
    {"name": "ymoslem/MediaSpeech", "config": "fr", "output": "real_mediaspeech_fr.parquet",
     "label": 0, "samples": 1000, "audio_col": "audio"},
    {"name": "qmeeus/slurp", "output": "real_slurp.parquet",
     "label": 0, "samples": 2000, "audio_col": "audio"},
    {"name": "simon3000/genshin-voice", "output": "real_genshin.parquet",
     "label": 0, "samples": 2000, "audio_col": "audio"},

    # Synthetic - ElevenLabs (prominent in gasbench)
    {"name": "skypro1111/elevenlabs_dataset", "output": "synth_elevenlabs_1.parquet",
     "label": 1, "samples": 1300, "audio_col": "audio"},
    {"name": "velocity-engg/eleven_labs_dataset", "output": "synth_elevenlabs_2.parquet",
     "label": 1, "samples": 1000, "audio_col": "audio"},
    {"name": "NeoBoy/elevenlabsSpeechTest", "output": "synth_elevenlabs_3.parquet",
     "label": 1, "samples": 800, "audio_col": "audio"},
    {"name": "velocity-engg/eleven_labs_datase_latin", "output": "synth_elevenlabs_latin.parquet",
     "label": 1, "samples": 1000, "audio_col": "audio"},

    # Synthetic - Arabic deepfakes
    {"name": "DeepFake-Audio-Rangers/Arabic_Audio_Deepfake", "output": "synth_arabic_deepfake.parquet",
     "label": 1, "samples": 2000, "audio_col": "audio"},

    # Synthetic - Various TTS
    {"name": "Thorsten-Voice/TV-44kHz-Full", "output": "synth_thorsten.parquet",
     "label": 1, "samples": 2000, "audio_col": "audio"},
]

for ds in datasets_to_download:
    outpath = os.path.join(OUTPUT_DIR, ds["output"])
    if os.path.exists(outpath):
        print("SKIP " + ds["name"] + " (exists)")
        continue

    print("Downloading: " + ds["name"] + (" [" + ds.get("config", "") + "]" if ds.get("config") else ""))
    audio_bytes_list = []
    try:
        kwargs = {"split": "train", "streaming": True}
        if "config" in ds:
            dataset = load_dataset(ds["name"], ds["config"], **kwargs)
        else:
            dataset = load_dataset(ds["name"], **kwargs)

        count = 0
        errors = 0
        for row in dataset:
            try:
                audio_col = ds.get("audio_col", "audio")
                audio = row.get(audio_col)
                if audio is None:
                    for col in ["audio", "speech", "wav", "waveform"]:
                        if col in row and row[col] is not None:
                            audio = row[col]
                            break
                if audio is None:
                    errors += 1
                    if errors > 100:
                        break
                    continue

                audio_bytes = process_audio(audio)
                if audio_bytes is None:
                    errors += 1
                    continue

                audio_bytes_list.append(audio_bytes)
                count += 1
                if count >= ds["samples"]:
                    break
                if count % 500 == 0:
                    print("  ..." + str(count) + "/" + str(ds["samples"]))
            except Exception as e:
                errors += 1
                if errors > 200:
                    print("  Too many errors (" + str(errors) + "), last: " + str(e))
                    break
                continue

        if audio_bytes_list:
            table = pa.table({
                "audio_bytes": audio_bytes_list,
                "label": [ds["label"]] * len(audio_bytes_list)
            })
            pq.write_table(table, outpath, compression="snappy")
            size_mb = os.path.getsize(outpath) / (1024**2)
            print("  Saved: " + str(count) + " samples, " + str(round(size_mb, 1)) + " MB")
        else:
            print("  WARNING: No audio loaded from " + ds["name"])
    except Exception as e:
        print("  ERROR: " + str(e))
    gc.collect()

print("\nDone! Files:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / (1024**2)
    print("  " + f + " (" + str(round(size, 1)) + " MB)")
