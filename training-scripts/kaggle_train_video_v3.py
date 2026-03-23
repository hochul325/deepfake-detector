#!/usr/bin/env python3
"""Video deepfake detector v3 - Kaggle-ready training notebook.

Key improvements over v2 (which failed entrance exam at 77%):
1. First 16 CONSECUTIVE frames (matches gasbench/decord evaluation pipeline)
2. REAL_CLASS_WEIGHT=2.0 (addresses false positives on real videos)
3. Aspect-preserving resize + center crop (matches gasbench preprocessing)
4. More real datasets needed in training data

Usage on Kaggle:
1. Upload this script as a Kaggle notebook
2. Upload your existing model weights as a Kaggle Dataset
3. Upload gasbench video datasets as Kaggle Datasets (or download in notebook)
4. Enable GPU accelerator (T4 x2 or P100)
5. Run all cells

The script will:
- Download gasbench datasets if not cached
- Extract first 16 consecutive frames per video
- Fine-tune from existing weights (or from pretrained CLIP ViT)
- Calibrate temperature for optimal sn34 score
- Package model in GAS format (model.py + model_config.yaml + model.safetensors)
- Output a zip ready to push via push_standalone.py
"""
import os, sys, io, json, random, logging, signal, zipfile
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import timm
from safetensors.torch import save_file, load_file

# === CONFIGURATION ===
# Adjust these paths for your Kaggle environment
KAGGLE_INPUT = '/kaggle/input'  # Kaggle dataset mount point
KAGGLE_WORKING = '/kaggle/working'  # Kaggle output directory

# If running locally or on Colab, override these:
# KAGGLE_INPUT = '/content/drive/MyDrive/deepfake-data'
# KAGGLE_WORKING = '/content/output'

# Where to find gasbench datasets
# Option 1: Uploaded as Kaggle dataset
GASBENCH_CACHE = os.environ.get('GASBENCH_CACHE',
    os.path.join(KAGGLE_INPUT, 'gasbench-video-datasets'))
# Option 2: Download via gasbench CLI (uncomment below)
# GASBENCH_CACHE = os.path.join(KAGGLE_WORKING, 'gasbench_cache')

# Where to find existing model weights (uploaded as Kaggle dataset)
PRETRAINED_WEIGHTS = os.environ.get('PRETRAINED_WEIGHTS',
    os.path.join(KAGGLE_INPUT, 'video-detector-weights', 'model.safetensors'))

OUTPUT_DIR = os.path.join(KAGGLE_WORKING, 'video_detector_v3')
FRAME_DIR = os.path.join(KAGGLE_WORKING, 'video_frames_cache')

# === KEY v3 CHANGES ===
FRAMES_PER_VIDEO = 16       # Was 8 in v2 — must match gasbench evaluation
REAL_CLASS_WEIGHT = 2.0     # Was 1.1 in v2 — reduces false positives on real videos
CONSECUTIVE_FRAMES = True   # Was False (evenly-spaced) in v2 — matches gasbench/decord

SEED = 42
EPOCHS = 35
BATCH_SIZE = 16  # Smaller for Kaggle T4 (16GB VRAM)
LR = 3e-6
VAL_SPLIT = 0.20

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)


# ============================================================
# Frame Extraction — CRITICAL: must match gasbench pipeline
# ============================================================

def extract_frames_consecutive(video_path, num_frames=FRAMES_PER_VIDEO):
    """Extract first N consecutive frames — matches gasbench/decord pipeline.

    This is the KEY fix from v2→v3. Gasbench uses decord to read the first N
    consecutive frames, NOT evenly-spaced frames. Training must match evaluation.
    """
    frames = []
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return frames

        for i in range(min(num_frames, total)):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Aspect-preserving resize to 256px on short side, then center crop
            h, w = frame.shape[:2]
            if h < w:
                new_h, new_w = 256, int(w * 256 / h)
            else:
                new_h, new_w = int(h * 256 / w), 256
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            # Center crop to 256x256
            y = (new_h - 256) // 2
            x = (new_w - 256) // 2
            frame = frame[y:y+256, x:x+256]
            frames.append(frame)
        cap.release()
    except Exception as e:
        logger.warning(f"Failed to extract from {video_path}: {e}")
    return frames


def extract_all_frames_to_disk(videos, output_dir):
    """Extract frames from all videos and save to disk as JPEG."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / 'manifest.json'

    # Check cache
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        if len(manifest) > 0 and os.path.exists(manifest[0]['path']):
            logger.info(f"Found cached frames: {len(manifest)} frames in {output_dir}")
            return [(m['path'], m['label']) for m in manifest]

    manifest = []
    frame_idx = 0
    for i, (vpath, label, ds_name) in enumerate(videos):
        frames = extract_frames_consecutive(vpath, FRAMES_PER_VIDEO)
        for j, frame in enumerate(frames):
            fname = f"frame_{frame_idx:06d}.jpg"
            fpath = str(output_dir / fname)
            img = Image.fromarray(frame)
            img.save(fpath, quality=95)
            manifest.append({'path': fpath, 'label': label})
            frame_idx += 1

        if (i + 1) % 100 == 0:
            logger.info(f"  Extracted {i+1}/{len(videos)} videos ({frame_idx} frames)")
            sys.stdout.flush()

    logger.info(f"  Total: {frame_idx} frames from {len(videos)} videos")

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f)

    return [(m['path'], m['label']) for m in manifest]


# ============================================================
# Dataset Discovery
# ============================================================

def load_gasbench_video_datasets():
    """Load all video datasets from gasbench cache."""
    cache = Path(GASBENCH_CACHE)
    if not cache.exists():
        logger.error(f"Gasbench cache not found at {cache}")
        logger.error("Upload video datasets as a Kaggle Dataset or set GASBENCH_CACHE")
        sys.exit(1)

    all_videos = []

    for ds_dir in sorted(cache.iterdir()):
        if not ds_dir.is_dir():
            continue

        info_path = ds_dir / 'dataset_info.json'
        week_dirs = [d for d in ds_dir.iterdir() if d.is_dir() and d.name.startswith('20')]

        if week_dirs:
            for week_dir in week_dirs:
                info_path = week_dir / 'dataset_info.json'
                if not info_path.exists():
                    continue
                with open(info_path) as f:
                    info = json.load(f)
                if info.get('modality') != 'video':
                    continue
                media_type = info.get('media_type', '')
                label = 0 if media_type == 'real' else 1
                ds_name = f"{ds_dir.name}/{week_dir.name}"
                samples_dir = week_dir / 'samples'
                if samples_dir.exists():
                    count = 0
                    for vid_file in sorted(samples_dir.iterdir()):
                        if vid_file.suffix.lower() in ('.mp4', '.avi', '.webm', '.mov'):
                            all_videos.append((str(vid_file), label, ds_name))
                            count += 1
                    label_str = 'real' if label == 0 else 'synth'
                    logger.info(f"  {ds_name}: {count} {label_str} videos")
            continue

        if not info_path.exists():
            continue

        with open(info_path) as f:
            info = json.load(f)

        if info.get('modality') != 'video':
            continue

        media_type = info.get('media_type', '')
        label = 0 if media_type == 'real' else 1
        ds_name = info.get('name', ds_dir.name)

        samples_dir = ds_dir / 'samples'
        if not samples_dir.exists():
            continue

        count = 0
        for vid_file in sorted(samples_dir.iterdir()):
            if vid_file.suffix.lower() in ('.mp4', '.avi', '.webm', '.mov'):
                all_videos.append((str(vid_file), label, ds_name))
                count += 1

        label_str = 'real' if label == 0 else 'synth'
        logger.info(f"  {ds_name}: {count} {label_str} videos")

    return all_videos


# ============================================================
# Model Architecture (identical to GAS evaluation model.py)
# ============================================================

class VideoDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2, backbone='vit_base_patch16_clip_224.openai'):
        super().__init__()
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.encoder = timm.create_model(backbone, pretrained=True, num_classes=0)
        feature_dim = self.encoder.num_features
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        self.register_buffer('temperature', torch.ones(1))

    def extract_features(self, x):
        x = x / 255.0
        x = (x - self.mean) / self.std
        return self.encoder(x)

    def forward(self, x):
        if x.dim() == 4:
            features = self.extract_features(x)
            return self.classifier(features) / self.temperature
        B, T = x.shape[:2]
        frames = x.reshape(B * T, *x.shape[2:])
        features = self.extract_features(frames)
        features = features.view(B, T, -1)
        pooled = features.mean(dim=1)
        return self.classifier(pooled) / self.temperature


# ============================================================
# Augmentations
# ============================================================

class JPEGCompress:
    def __init__(self, quality_range=(30, 95)):
        self.quality_range = quality_range
    def __call__(self, img):
        quality = random.randint(*self.quality_range)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        return Image.open(buf).convert('RGB')


class RandomNoise:
    def __init__(self, std_range=(0.005, 0.03)):
        self.std_range = std_range
    def __call__(self, img):
        arr = np.array(img, dtype=np.float32) / 255.0
        std = random.uniform(*self.std_range)
        noise = np.random.randn(*arr.shape).astype(np.float32) * std
        arr = np.clip(arr + noise, 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))


class RandomDownscale:
    def __init__(self, scale_range=(0.4, 0.8)):
        self.scale_range = scale_range
    def __call__(self, img):
        w, h = img.size
        scale = random.uniform(*self.scale_range)
        new_w, new_h = max(16, int(w * scale)), max(16, int(h * scale))
        img = img.resize((new_w, new_h), Image.BILINEAR)
        img = img.resize((w, h), Image.BILINEAR)
        return img


def get_train_transform(size=224):
    return transforms.Compose([
        transforms.Resize((size + 16, size + 16)),
        transforms.RandomCrop((size, size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.05),
        transforms.RandomApply([JPEGCompress(quality_range=(30, 95))], p=0.3),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.15),
        transforms.RandomApply([RandomNoise(std_range=(0.005, 0.03))], p=0.15),
        transforms.RandomApply([RandomDownscale(scale_range=(0.4, 0.8))], p=0.15),
        transforms.RandomGrayscale(p=0.03),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.10, scale=(0.02, 0.10)),
        transforms.Lambda(lambda x: x * 255.0),
    ])


def get_val_transform(size=224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255.0),
    ])


# ============================================================
# Dataset
# ============================================================

class FrameDataset(Dataset):
    def __init__(self, frame_list, transform=None):
        self.frame_list = frame_list
        self.transform = transform

    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self, idx):
        path, label = self.frame_list[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (256, 256), (128, 128, 128))
        if self.transform:
            img = self.transform(img)
        return img, label


# ============================================================
# Training
# ============================================================

def train_epoch(model, loader, optimizer, scheduler, device, epoch, mixup_alpha=0.2):
    model.train()
    # v3 KEY CHANGE: REAL_CLASS_WEIGHT=2.0 (class 0 = real, class 1 = synth)
    weight = torch.tensor([REAL_CLASS_WEIGHT, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.08, weight=weight)
    total_loss, correct, total = 0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        if mixup_alpha > 0 and random.random() < 0.3:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            lam = max(lam, 1 - lam)
            idx = torch.randperm(images.size(0), device=device)
            mixed_images = lam * images + (1 - lam) * images[idx]
            logits = model(mixed_images)
            loss = lam * criterion(logits, labels) + (1 - lam) * criterion(logits, labels[idx])
        else:
            logits = model(images)
            loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 50 == 0:
            logger.info(f"Ep{epoch} [{batch_idx+1}/{len(loader)}] Loss:{total_loss/(batch_idx+1):.4f} Acc:{100*correct/total:.1f}%")
            sys.stdout.flush()

    return total_loss / max(len(loader), 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    for images, labels in loader:
        logits = model(images.to(device))
        probs = F.softmax(logits, dim=1)
        all_labels.extend(labels.numpy())
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    labels = np.array(all_labels)
    preds = np.array(all_preds)
    probs = np.array(all_probs)

    acc = (preds == labels).mean()
    tp = ((preds == 1) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()

    denom = np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    mcc = float(tp*tn - fp*fn) / denom if denom > 0 else 0
    brier = float(np.mean((probs[:, 1] - labels) ** 2))

    mcc_norm = max(0, ((mcc + 1) / 2)) ** 1.2
    brier_norm = max(0, (0.25 - brier) / 0.25) ** 1.8
    sn34 = float(np.sqrt(max(1e-12, mcc_norm * brier_norm)))

    logger.info(f"  Acc:{acc:.4f} MCC:{mcc:.4f} Brier:{brier:.6f} sn34:{sn34:.4f} (TP={tp} TN={tn} FP={fp} FN={fn})")
    return {'accuracy': acc, 'mcc': mcc, 'brier': brier, 'sn34_score': sn34}


def calibrate_temperature(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        old_temp = model.temperature.clone()
        model.temperature.fill_(1.0)
        for images, labels in loader:
            logits = model(images.to(device))
            all_logits.append(logits.cpu())
            all_labels.append(labels)
        model.temperature.copy_(old_temp)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    best_sn34, best_temp = 0, 1.0
    for temp in np.arange(0.1, 8.0, 0.02):
        probs = F.softmax(logits / temp, dim=1)
        synth_probs = probs[:, 1]
        brier = ((synth_probs - labels.float()) ** 2).mean().item()
        preds = (synth_probs > 0.5).long()
        tp = ((preds == 1) & (labels == 1)).sum().item()
        tn = ((preds == 0) & (labels == 0)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()
        denom = np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
        mcc = float(tp*tn - fp*fn) / denom if denom > 0 else 0
        mcc_norm = max(0, ((mcc + 1) / 2)) ** 1.2
        brier_norm = max(0, (0.25 - brier) / 0.25) ** 1.8
        sn34 = float(np.sqrt(max(1e-12, mcc_norm * brier_norm)))
        if sn34 > best_sn34:
            best_sn34, best_temp = sn34, temp

    # Fine search
    for temp in np.arange(max(0.01, best_temp - 0.15), best_temp + 0.15, 0.001):
        probs = F.softmax(logits / temp, dim=1)
        synth_probs = probs[:, 1]
        brier = ((synth_probs - labels.float()) ** 2).mean().item()
        preds = (synth_probs > 0.5).long()
        tp = ((preds == 1) & (labels == 1)).sum().item()
        tn = ((preds == 0) & (labels == 0)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()
        denom = np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
        mcc = float(tp*tn - fp*fn) / denom if denom > 0 else 0
        mcc_norm = max(0, ((mcc + 1) / 2)) ** 1.2
        brier_norm = max(0, (0.25 - brier) / 0.25) ** 1.8
        sn34 = float(np.sqrt(max(1e-12, mcc_norm * brier_norm)))
        if sn34 > best_sn34:
            best_sn34, best_temp = sn34, temp

    logger.info(f"  Calibrated: temp={best_temp:.3f} sn34={best_sn34:.4f}")
    model.temperature.fill_(best_temp)
    return best_temp


# ============================================================
# Packaging — creates GAS-ready zip
# ============================================================

def package_model(model, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), str(output_dir / 'model.safetensors'))

    (output_dir / 'model_config.yaml').write_text("""name: "clip-vit-video-deepfake-detector-gas"
version: "12.0.0"
modality: "video"

preprocessing:
  resize: [224, 224]
  max_frames: 16

model:
  num_classes: 2
  weights_file: "model.safetensors"
""")

    (output_dir / 'model.py').write_text('''import torch
import torch.nn as nn
import timm
from safetensors.torch import load_file


class VideoDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.encoder = timm.create_model('vit_base_patch16_clip_224.openai', pretrained=False, num_classes=0)
        feature_dim = self.encoder.num_features
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        self.register_buffer('temperature', torch.ones(1))

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)
        B, T = x.shape[:2]
        frames = x.reshape(B * T, *x.shape[2:])
        frames = frames.float() / 255.0
        frames = (frames - self.mean) / self.std
        features = self.encoder(frames)
        features = features.view(B, T, -1)
        pooled = features.mean(dim=1)
        return self.classifier(pooled) / self.temperature


def load_model(weights_path, num_classes=2):
    model = VideoDeepfakeDetector(num_classes=num_classes)
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    model.train(False)
    return model
''')

    # Create zip ready for push
    zip_path = output_dir.parent / 'video_detector_v3.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_dir / 'model.safetensors', 'model.safetensors')
        zf.write(output_dir / 'model_config.yaml', 'model_config.yaml')
        zf.write(output_dir / 'model.py', 'model.py')

    logger.info(f"Model packaged in {output_dir}")
    logger.info(f"Push-ready zip: {zip_path}")
    logger.info(f"To push: python push_standalone.py --video-model {zip_path} --wallet-name miner")


# ============================================================
# Main
# ============================================================

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # === Load video paths ===
    logger.info("Discovering video datasets...")
    all_videos = load_gasbench_video_datasets()

    n_real = sum(1 for _, l, _ in all_videos if l == 0)
    n_synth = sum(1 for _, l, _ in all_videos if l == 1)
    logger.info(f"\nTotal videos: {len(all_videos)} ({n_real} real, {n_synth} synth)")

    if len(all_videos) == 0:
        logger.error("No video datasets found! Check GASBENCH_CACHE path.")
        logger.error(f"Looking in: {GASBENCH_CACHE}")
        sys.exit(1)

    # Oversample small datasets to median
    ds_counts = Counter(ds for _, _, ds in all_videos)
    median_count = max(sorted(ds_counts.values())[len(ds_counts) // 2], 50)
    logger.info(f"Median dataset count: {median_count}")

    balanced_videos = []
    for ds_name, count in ds_counts.items():
        ds_items = [(p, l, d) for p, l, d in all_videos if d == ds_name]
        if count < median_count:
            repeats = max(1, median_count // count)
            ds_items = ds_items * repeats
            logger.info(f"  Oversampled {ds_name}: {count} -> {len(ds_items)}")
        balanced_videos.extend(ds_items)

    all_videos = balanced_videos
    n_real = sum(1 for _, l, _ in all_videos if l == 0)
    n_synth = sum(1 for _, l, _ in all_videos if l == 1)
    logger.info(f"After oversampling: {len(all_videos)} ({n_real} real, {n_synth} synth)")

    # Split
    random.shuffle(all_videos)
    val_size = int(len(all_videos) * VAL_SPLIT)
    val_videos = all_videos[:val_size]
    train_videos = all_videos[val_size:]

    logger.info(f"Train: {len(train_videos)} videos, Val: {len(val_videos)} videos")

    # Extract frames
    logger.info("Extracting train frames (first 16 consecutive)...")
    train_frames = extract_all_frames_to_disk(train_videos, os.path.join(FRAME_DIR, 'train'))
    logger.info("Extracting val frames...")
    val_frames = extract_all_frames_to_disk(val_videos, os.path.join(FRAME_DIR, 'val'))

    logger.info(f"Train frames: {len(train_frames)}, Val frames: {len(val_frames)}")

    train_ds = FrameDataset(train_frames, get_train_transform())
    val_ds = FrameDataset(val_frames, get_val_transform())

    # Weighted sampler
    train_labels = [l for _, l in train_frames]
    class_counts = [train_labels.count(0), train_labels.count(1)]
    logger.info(f"Train class counts: real={class_counts[0]}, synth={class_counts[1]}")

    if min(class_counts) > 0:
        weights = [1.0 / class_counts[l] for l in train_labels]
        sampler = WeightedRandomSampler(weights, len(train_labels), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                                  num_workers=2, pin_memory=True, drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=2, pin_memory=True, drop_last=True)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)

    # === Create model ===
    logger.info("Creating model...")
    model = VideoDeepfakeDetector().to(device)

    # Load pretrained weights if available
    if os.path.exists(PRETRAINED_WEIGHTS):
        logger.info(f"Loading pretrained weights from {PRETRAINED_WEIGHTS}...")
        state = load_file(PRETRAINED_WEIGHTS)
        state = {k: v for k, v in state.items() if k != 'temperature'}
        model.load_state_dict(state, strict=False)
        logger.info("Pretrained weights loaded")
    else:
        logger.info("No pretrained weights found, training from CLIP ViT scratch")

    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,} / {total_params:,}")

    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and 'classifier' not in n]
    classifier_params = [p for n, p in model.named_parameters()
                         if p.requires_grad and 'classifier' in n]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': LR},
        {'params': classifier_params, 'lr': LR * 10},
    ], weight_decay=0.02)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = len(train_loader) * 2

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_sn34, best_epoch = 0, 0
    patience, patience_counter = 10, 0

    for epoch in range(1, EPOCHS + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch}/{EPOCHS}")
        logger.info(f"{'='*50}")

        loss, acc = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        logger.info(f"Train Loss:{loss:.4f} Acc:{acc:.4f}")

        metrics = evaluate(model, val_loader, device)

        if metrics['sn34_score'] > best_sn34:
            best_sn34 = metrics['sn34_score']
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(KAGGLE_WORKING, 'best_video_v3.pt'))
            logger.info(f"  *** New best sn34: {best_sn34:.4f} ***")
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch > 5:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    logger.info(f"\nBest epoch: {best_epoch} (sn34: {best_sn34:.4f})")
    model.load_state_dict(torch.load(os.path.join(KAGGLE_WORKING, 'best_video_v3.pt'), weights_only=True))

    logger.info("Calibrating temperature...")
    calibrate_temperature(model, val_loader, device)

    logger.info("\nFinal evaluation (after calibration):")
    final = evaluate(model, val_loader, device)

    package_model(model, OUTPUT_DIR)

    logger.info(f"\n{'='*50}")
    logger.info(f"DONE! Final sn34_score: {final['sn34_score']:.4f}")
    logger.info(f"Model: {OUTPUT_DIR}")
    logger.info(f"Zip: {KAGGLE_WORKING}/video_detector_v3.zip")
    logger.info(f"{'='*50}")
    logger.info(f"\nNext steps:")
    logger.info(f"1. Download video_detector_v3.zip from Kaggle output")
    logger.info(f"2. On your Ubuntu machine:")
    logger.info(f"   cd ~/bitmind-subnet")
    logger.info(f"   source .venv/bin/activate")
    logger.info(f"   python push_standalone.py --video-model /path/to/video_detector_v3.zip --wallet-name miner")


if __name__ == '__main__':
    main()
