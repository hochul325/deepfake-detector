"""Deepfake Detector Web App — FastAPI backend serving image/audio/video detection."""

import asyncio
import importlib.util
import os
import sys
import tempfile
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

APP_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = APP_DIR / "examples"

# ---------------------------------------------------------------------------
# Rate limiting — 10 free detections per IP per day
# ---------------------------------------------------------------------------
DAILY_LIMIT = 10
_rate_counts: dict[str, dict] = defaultdict(lambda: {"date": None, "count": 0})


def check_rate_limit(ip: str) -> tuple[bool, int]:
    """Returns (allowed, remaining)."""
    today = date.today().isoformat()
    entry = _rate_counts[ip]
    if entry["date"] != today:
        entry["date"] = today
        entry["count"] = 0
    if entry["count"] >= DAILY_LIMIT:
        return False, 0
    entry["count"] += 1
    return True, DAILY_LIMIT - entry["count"]


# ---------------------------------------------------------------------------
# Model loading via importlib (avoids name collisions between model.py files)
# ---------------------------------------------------------------------------
MODEL_NAMES = {
    "image": "CLIP ViT-B/16",
    "audio": "Wav2Vec2-Base",
    "video": "CLIP ViT-B/16 (temporal)",
}


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
    tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
    return tensor.to(device)


def preprocess_audio(file_path: str, device: torch.device) -> torch.Tensor:
    waveform, sr = torchaudio.load(file_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != AUDIO_SR:
        waveform = torchaudio.functional.resample(waveform, sr, AUDIO_SR)
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
    tensor = torch.stack(frames).unsqueeze(0)
    return tensor.to(device)


def _run_inference(file_path: str, modality: str) -> dict:
    """Run inference on a file. Returns result dict with timing."""
    file_size = os.path.getsize(file_path)
    t0 = time.perf_counter()

    if modality == "image":
        tensor = preprocess_image(file_path, device)
    elif modality == "audio":
        tensor = preprocess_audio(file_path, device)
    else:
        tensor = preprocess_video(file_path, device)

    with torch.no_grad():
        logits = models[modality](tensor)
        probs = torch.softmax(logits, dim=1)[0]

    elapsed = time.perf_counter() - t0
    real_prob = probs[0].item()
    fake_prob = probs[1].item()
    is_real = real_prob > fake_prob
    confidence = max(real_prob, fake_prob)

    return {
        "verdict": "REAL" if is_real else "FAKE",
        "confidence": round(confidence * 100, 2),
        "modality": modality,
        "probabilities": {
            "real": round(real_prob * 100, 2),
            "fake": round(fake_prob * 100, 2),
        },
        "details": {
            "model": MODEL_NAMES[modality],
            "processing_time_ms": round(elapsed * 1000),
            "file_size_bytes": file_size,
        },
    }


# ---------------------------------------------------------------------------
# Example files
# ---------------------------------------------------------------------------

EXAMPLES = [
    {"id": "real_face", "file": "real_face.jpg", "label": "Real Photo", "label_ko": "실제 사진", "modality": "image", "expected": "real"},
    {"id": "fake_face", "file": "fake_face.jpg", "label": "AI Face", "label_ko": "AI 얼굴", "modality": "image", "expected": "fake"},
    {"id": "real_speech", "file": "real_speech.wav", "label": "Real Speech", "label_ko": "실제 음성", "modality": "audio", "expected": "real"},
    {"id": "fake_speech", "file": "fake_speech.mp3", "label": "TTS Speech", "label_ko": "TTS 음성", "modality": "audio", "expected": "fake"},
    {"id": "real_video", "file": "real_video.mp4", "label": "Real Video", "label_ko": "실제 영상", "modality": "video", "expected": "real"},
    {"id": "fake_video", "file": "fake_video.mp4", "label": "Deepfake", "label_ko": "딥페이크", "modality": "video", "expected": "fake"},
]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Deepfake Detector API",
    description="Detect AI-generated images, audio, and video using deep learning.",
    version="1.0.0",
)

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
async def detect(request: Request, file: UploadFile = File(...)):
    # Rate limit
    client_ip = request.headers.get("x-forwarded-for", request.client.host).split(",")[0].strip()
    allowed, remaining = check_rate_limit(client_ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Daily limit reached. You can perform 10 free detections per day.",
        )

    try:
        modality = detect_modality(file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        async with gpu_lock:
            result = _run_inference(tmp_path, modality)
        result["remaining_today"] = remaining
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    finally:
        os.unlink(tmp_path)


@app.get("/api/examples")
async def list_examples():
    """List available example files for one-click testing."""
    available = []
    for ex in EXAMPLES:
        path = EXAMPLES_DIR / ex["file"]
        if path.exists():
            available.append({
                "id": ex["id"],
                "label": ex["label"],
                "label_ko": ex["label_ko"],
                "modality": ex["modality"],
                "expected": ex["expected"],
                "file_size": path.stat().st_size,
            })
    return {"examples": available}


@app.post("/api/detect/example/{example_id}")
async def detect_example(example_id: str, request: Request):
    """Run detection on a pre-loaded example file."""
    client_ip = request.headers.get("x-forwarded-for", request.client.host).split(",")[0].strip()
    allowed, remaining = check_rate_limit(client_ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Daily limit reached. You can perform 10 free detections per day.",
        )

    ex = next((e for e in EXAMPLES if e["id"] == example_id), None)
    if not ex:
        raise HTTPException(status_code=404, detail="Example not found")
    path = EXAMPLES_DIR / ex["file"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="Example file missing")

    try:
        async with gpu_lock:
            result = _run_inference(str(path), ex["modality"])
        result["remaining_today"] = remaining
        result["example_id"] = example_id
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")


@app.get("/api/examples/{example_id}/thumb")
async def example_thumbnail(example_id: str):
    """Serve example file thumbnail / file."""
    ex = next((e for e in EXAMPLES if e["id"] == example_id), None)
    if not ex:
        raise HTTPException(status_code=404, detail="Example not found")
    path = EXAMPLES_DIR / ex["file"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="File missing")
    return FileResponse(path)
