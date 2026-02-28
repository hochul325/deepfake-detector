"""Deepfake Detector Web App — FastAPI backend serving image/audio/video detection."""

import asyncio
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

APP_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Model loading via importlib (avoids name collisions between model.py files)
# ---------------------------------------------------------------------------

def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_all_models(device: torch.device):
    models = {}

    img_mod = _load_module("image_model", APP_DIR / "image_detector" / "model.py")
    models["image"] = img_mod.load_model(
        str(APP_DIR / "image_detector" / "model.safetensors")
    ).to(device)

    aud_mod = _load_module("audio_model", APP_DIR / "audio_detector" / "model.py")
    models["audio"] = aud_mod.load_model(
        str(APP_DIR / "audio_detector" / "model.safetensors")
    ).to(device)

    vid_mod = _load_module("video_model", APP_DIR / "video_detector" / "model.py")
    models["video"] = vid_mod.load_model(
        str(APP_DIR / "video_detector" / "model.safetensors")
    ).to(device)

    return models


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}

AUDIO_SR = 16_000
AUDIO_SAMPLES = 96_000
VIDEO_MAX_FRAMES = 16
IMG_SIZE = 224


def detect_modality(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in VIDEO_EXTS:
        return "video"
    raise ValueError(f"Unsupported file extension: {ext}")


def preprocess_image(file_path: str, device: torch.device) -> torch.Tensor:
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError("Could not read image file")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # HWC uint8 → CHW float [0, 255]
    tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
    return tensor.to(device)


def preprocess_audio(file_path: str, device: torch.device) -> torch.Tensor:
    waveform, sr = torchaudio.load(file_path)
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # Resample to 16kHz
    if sr != AUDIO_SR:
        waveform = torchaudio.functional.resample(waveform, sr, AUDIO_SR)
    # Center-crop or pad to AUDIO_SAMPLES
    length = waveform.shape[1]
    if length > AUDIO_SAMPLES:
        start = (length - AUDIO_SAMPLES) // 2
        waveform = waveform[:, start : start + AUDIO_SAMPLES]
    elif length < AUDIO_SAMPLES:
        pad_total = AUDIO_SAMPLES - length
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        waveform = torch.nn.functional.pad(waveform, (pad_left, pad_right))
    return waveform.to(device)


def preprocess_video(file_path: str, device: torch.device) -> torch.Tensor:
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise ValueError("Video has no frames")

    # Select up to VIDEO_MAX_FRAMES evenly spaced
    n_frames = min(total_frames, VIDEO_MAX_FRAMES)
    indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frames.append(torch.from_numpy(frame).permute(2, 0, 1).float())
    cap.release()

    if not frames:
        raise ValueError("Could not extract any frames from video")

    # Stack as (1, T, 3, 224, 224)
    tensor = torch.stack(frames).unsqueeze(0)
    return tensor.to(device)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Deepfake Detector")

device: torch.device = None
models: dict = {}
gpu_lock: asyncio.Lock = None


@app.on_event("startup")
async def startup():
    global device, models, gpu_lock
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading models on {device} ...")
    models.update(load_all_models(device))
    gpu_lock = asyncio.Lock()
    print("All models loaded.")


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = APP_DIR / "templates" / "index.html"
    return html_path.read_text()


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "device": str(device),
        "models_loaded": list(models.keys()),
    }


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    try:
        modality = detect_modality(file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Save upload to temp file
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        if modality == "image":
            tensor = preprocess_image(tmp_path, device)
        elif modality == "audio":
            tensor = preprocess_audio(tmp_path, device)
        else:
            tensor = preprocess_video(tmp_path, device)

        model = models[modality]
        async with gpu_lock:
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)[0]

        real_prob = probs[0].item()
        fake_prob = probs[1].item()
        verdict = "REAL" if real_prob > fake_prob else "FAKE"
        confidence = max(real_prob, fake_prob)

        return {
            "verdict": verdict,
            "confidence": round(confidence * 100, 2),
            "modality": modality,
            "probabilities": {
                "real": round(real_prob * 100, 2),
                "fake": round(fake_prob * 100, 2),
            },
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    finally:
        os.unlink(tmp_path)
