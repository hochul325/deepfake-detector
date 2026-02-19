#!/usr/bin/env python3
"""Video deepfake detector training script v3.
Fixes RGB/BGR bug, improved augmentations, better calibration.

Input format from gasbench: float32 [B, T, 3, H, W] RGB 0-255
Output format: [B, 2] logits (index 0=real, 1=synthetic)
Score: sn34 = sqrt(mcc_norm^1.2 * brier_norm^1.8)
"""

import os
import sys
import io
import time
import random
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image, ImageFilter
import timm
from safetensors.torch import save_file
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# Model Architecture
# ============================================================

class VideoDeepfakeDetector(nn.Module):
    """CLIP ViT-B/16 based deepfake detector.

    Input: float32 [B, T, 3, H, W] RGB 0-255 (or [B, 3, H, W] for single images)
    Output: [B, 2] logits (real, synthetic)

    NOTE: Input is RGB (matches gasbench pipeline which sends RGB).
    No BGR->RGB flip needed.
    """
    def __init__(self, num_classes=2, backbone='vit_base_patch16_clip_224.openai'):
        super().__init__()
        # ImageNet normalization (for RGB input)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.encoder = timm.create_model(backbone, pretrained=True, num_classes=0)
        feature_dim = self.encoder.num_features  # 768 for ViT-B

        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        self.register_buffer('temperature', torch.ones(1))
        self._freeze_backbone()

    def _freeze_backbone(self):
        """Freeze backbone except last 3 transformer blocks and norms."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        for name, param in self.encoder.named_parameters():
            if any(f'blocks.{i}.' in name for i in [9, 10, 11]):
                param.requires_grad = True
            elif 'norm' in name.lower():
                param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    def extract_features(self, x):
        """Extract features from RGB 0-255 float32 images.
        x: [B, 3, H, W] float32 RGB 0-255
        """
        # Normalize: RGB 0-255 -> 0-1 -> ImageNet normalized
        x = x / 255.0
        x = (x - self.mean) / self.std
        return self.encoder(x)

    def forward(self, x):
        """Forward pass.
        x: [B, 3, H, W] or [B, T, 3, H, W] float32 RGB 0-255
        """
        if x.dim() == 4:
            # Single image: [B, 3, H, W]
            features = self.extract_features(x)
            return self.classifier(features) / self.temperature
        # Video: [B, T, 3, H, W]
        B, T = x.shape[:2]
        frames = x.reshape(B * T, *x.shape[2:])
        features = self.extract_features(frames)
        features = features.view(B, T, -1)
        pooled = features.mean(dim=1)  # temporal mean pooling
        return self.classifier(pooled) / self.temperature


# ============================================================
# Data Loading
# ============================================================

def load_images_from_parquet(parquet_path, label, max_samples=5000):
    """Load images from parquet. Returns list of (PIL.Image RGB, label)."""
    samples = []
    try:
        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        # Find image column
        img_col = None
        for col in ['image', 'img', 'pixel_values', 'data']:
            if col in df.columns:
                img_col = col
                break
        if img_col is None:
            for col in df.columns:
                if df[col].dtype == object and len(df) > 0:
                    val = df[col].iloc[0]
                    if isinstance(val, dict) and 'bytes' in val:
                        img_col = col
                        break
                    elif isinstance(val, bytes):
                        img_col = col
                        break

        if img_col is None:
            logger.warning(f"No image column in {parquet_path}. Cols: {list(df.columns)}")
            return samples

        count = 0
        for idx in range(min(len(df), max_samples)):
            try:
                val = df[img_col].iloc[idx]
                if isinstance(val, dict):
                    img_bytes = val.get('bytes')
                    if img_bytes:
                        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                        samples.append((img, label))
                        count += 1
                elif isinstance(val, bytes):
                    img = Image.open(io.BytesIO(val)).convert('RGB')
                    samples.append((img, label))
                    count += 1
                elif isinstance(val, Image.Image):
                    samples.append((val.convert('RGB'), label))
                    count += 1
            except Exception:
                continue
            if count >= max_samples:
                break

        logger.info(f"  Loaded {count} from {Path(parquet_path).name} (label={'real' if label==0 else 'synth'})")
    except Exception as e:
        logger.warning(f"Error loading {parquet_path}: {e}")
    return samples


class DeepfakeDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


class JPEGCompress:
    """Random JPEG compression augmentation."""
    def __init__(self, quality_range=(30, 95)):
        self.quality_range = quality_range

    def __call__(self, img):
        quality = random.randint(*self.quality_range)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        return Image.open(buf).convert('RGB')


class RandomNoise:
    """Add random Gaussian noise."""
    def __init__(self, std_range=(0.01, 0.05)):
        self.std_range = std_range

    def __call__(self, img):
        arr = np.array(img, dtype=np.float32) / 255.0
        std = random.uniform(*self.std_range)
        noise = np.random.randn(*arr.shape).astype(np.float32) * std
        arr = np.clip(arr + noise, 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))


class RandomDownscale:
    """Random downscale then upscale to simulate low quality."""
    def __init__(self, scale_range=(0.3, 0.8)):
        self.scale_range = scale_range

    def __call__(self, img):
        w, h = img.size
        scale = random.uniform(*self.scale_range)
        new_w, new_h = max(16, int(w * scale)), max(16, int(h * scale))
        img = img.resize((new_w, new_h), Image.BILINEAR)
        img = img.resize((w, h), Image.BILINEAR)
        return img


def get_train_transform(size=224):
    """Training augmentations that simulate real-world degradations.
    Output: float32 tensor [3, H, W] RGB 0-255 (matching gasbench format).
    """
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15, hue=0.05),
        transforms.RandomApply([JPEGCompress(quality_range=(30, 95))], p=0.3),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.2),
        transforms.RandomApply([RandomNoise(std_range=(0.01, 0.04))], p=0.15),
        transforms.RandomApply([RandomDownscale(scale_range=(0.3, 0.8))], p=0.15),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),  # [0,1] RGB
        transforms.Lambda(lambda x: x * 255.0),  # scale to 0-255 RGB (matches gasbench)
    ])


def get_val_transform(size=224):
    """Validation transform: simple resize + to tensor.
    Output: float32 tensor [3, H, W] RGB 0-255.
    """
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255.0),
    ])


# ============================================================
# Training
# ============================================================

def train_epoch(model, loader, optimizer, scheduler, device, epoch, mixup_alpha=0.2):
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    total_loss, correct, total = 0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        # Mixup augmentation
        if mixup_alpha > 0 and random.random() < 0.3:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            lam = max(lam, 1 - lam)  # keep lam >= 0.5
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

    # sn34 scoring
    mcc_norm = max(0, ((mcc + 1) / 2)) ** 1.2
    brier_norm = max(0, (0.25 - brier) / 0.25) ** 1.8
    sn34 = float(np.sqrt(max(1e-12, mcc_norm * brier_norm)))

    logger.info(f"  Acc:{acc:.4f} MCC:{mcc:.4f} Brier:{brier:.6f} sn34:{sn34:.4f} (TP={tp} TN={tn} FP={fp} FN={fn})")
    return {'accuracy': acc, 'mcc': mcc, 'brier': brier, 'sn34_score': sn34}


def calibrate_temperature(model, loader, device):
    """Find optimal temperature for Brier score via grid search."""
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

    # Coarse search
    best_brier, best_temp = float('inf'), 1.0
    for temp in np.arange(0.2, 8.0, 0.02):
        probs = F.softmax(logits / temp, dim=1)
        brier = ((probs[:, 1] - labels.float()) ** 2).mean().item()
        if brier < best_brier:
            best_brier, best_temp = brier, temp

    # Fine search around best
    for temp in np.arange(max(0.05, best_temp - 0.15), best_temp + 0.15, 0.002):
        probs = F.softmax(logits / temp, dim=1)
        brier = ((probs[:, 1] - labels.float()) ** 2).mean().item()
        if brier < best_brier:
            best_brier, best_temp = brier, temp

    logger.info(f"  Calibrated temperature: {best_temp:.3f} (Brier: {best_brier:.6f})")
    model.temperature.fill_(best_temp)
    return best_temp


def package_model(model, output_dir):
    """Package model for gasbench/GAS Station submission."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_file(model.state_dict(), str(output_dir / 'model.safetensors'))

    (output_dir / 'model_config.yaml').write_text("""name: "clip-vit-video-deepfake-detector-v3"
version: "3.0.0"
modality: "video"

preprocessing:
  resize: [224, 224]
  max_frames: 16

model:
  num_classes: 2
  weights_file: "model.safetensors"
""")

    # Model.py for submission - must match architecture exactly
    # Input from gasbench: float32 [B, T, 3, H, W] RGB 0-255
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
        """Input: float32 [B, T, 3, H, W] or [B, 3, H, W] RGB 0-255"""
        if x.dim() == 4:
            x = x.unsqueeze(1)
        B, T = x.shape[:2]
        frames = x.reshape(B * T, *x.shape[2:])
        # RGB 0-255 -> 0-1 -> ImageNet normalized
        frames = frames / 255.0
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

    logger.info(f"Model packaged in {output_dir}")
    for f in output_dir.iterdir():
        logger.info(f"  {f.name}: {f.stat().st_size / 1024:.1f} KB")


# ============================================================
# Main Training Loop
# ============================================================

def main():
    DATA_DIR = '/root/train_data'
    OUTPUT_DIR = '/root/video_detector_v3'
    SEED = 42
    EPOCHS = 15
    BATCH_SIZE = 32
    LR = 2e-5
    VAL_SPLIT = 0.15
    MAX_PER_FILE = 5000

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load data
    logger.info("Loading data from parquet files...")
    all_samples = []

    real_keywords = ['ffhq', 'celeb', 'coco', 'ms-', 'lsun', 'imagenet']
    synth_keywords = ['genimage', 'journey', 'midjourney', 'ideogram', 'dalle', 'sdxl', 'stable']

    data_dir = Path(DATA_DIR)
    for f in sorted(data_dir.glob('*.parquet')):
        name = f.stem.lower()
        if any(k in name for k in real_keywords):
            samples = load_images_from_parquet(str(f), label=0, max_samples=MAX_PER_FILE)
            all_samples.extend(samples)
        elif any(k in name for k in synth_keywords):
            samples = load_images_from_parquet(str(f), label=1, max_samples=MAX_PER_FILE)
            all_samples.extend(samples)
        else:
            logger.warning(f"Unknown dataset type for {f.name}, skipping")

    if len(all_samples) < 100:
        logger.error(f"Only {len(all_samples)} samples! Need more data.")
        sys.exit(1)

    n_real = sum(1 for _, l in all_samples if l == 0)
    n_synth = sum(1 for _, l in all_samples if l == 1)
    logger.info(f"Total: {len(all_samples)} samples ({n_real} real, {n_synth} synthetic)")

    # Balance classes by undersampling majority
    if n_real > 0 and n_synth > 0:
        min_class = min(n_real, n_synth)
        max_per_class = int(min_class * 1.2)  # allow slight imbalance

        real_samples = [(img, l) for img, l in all_samples if l == 0]
        synth_samples = [(img, l) for img, l in all_samples if l == 1]

        random.shuffle(real_samples)
        random.shuffle(synth_samples)

        all_samples = real_samples[:max_per_class] + synth_samples[:max_per_class]
        n_real = sum(1 for _, l in all_samples if l == 0)
        n_synth = sum(1 for _, l in all_samples if l == 1)
        logger.info(f"After balancing: {len(all_samples)} ({n_real} real, {n_synth} synthetic)")

    # Split train/val
    random.shuffle(all_samples)
    val_size = int(len(all_samples) * VAL_SPLIT)
    val_samples = all_samples[:val_size]
    train_samples = all_samples[val_size:]
    logger.info(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Create datasets and loaders
    train_ds = DeepfakeDataset(train_samples, get_train_transform())
    val_ds = DeepfakeDataset(val_samples, get_val_transform())

    # Weighted sampler for balanced batches
    train_labels = [l for _, l in train_samples]
    class_counts = [train_labels.count(0), train_labels.count(1)]
    if min(class_counts) > 0:
        weights = [1.0 / class_counts[l] for l in train_labels]
        sampler = WeightedRandomSampler(weights, len(train_labels), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                                  num_workers=4, pin_memory=True, drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=4, pin_memory=True, drop_last=True)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Create model
    logger.info("Creating model...")
    model = VideoDeepfakeDetector().to(device)

    # Optimizer with different LR for backbone vs classifier
    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and 'classifier' not in n]
    classifier_params = [p for n, p in model.named_parameters()
                         if p.requires_grad and 'classifier' in n]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': LR},
        {'params': classifier_params, 'lr': LR * 5},  # higher LR for classifier
    ], weight_decay=0.01)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = len(train_loader) * 2  # 2 epoch warmup

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    best_sn34, best_epoch = 0, 0
    patience, patience_counter = 5, 0

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
            torch.save(model.state_dict(), '/tmp/best_model_v3.pt')
            logger.info(f"  *** New best sn34: {best_sn34:.4f} ***")
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch > 5:
                logger.info(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # Load best model
    logger.info(f"\nBest epoch: {best_epoch} (sn34: {best_sn34:.4f})")
    model.load_state_dict(torch.load('/tmp/best_model_v3.pt', weights_only=True))

    # Temperature calibration
    logger.info("Calibrating temperature...")
    calibrate_temperature(model, val_loader, device)

    # Final evaluation
    logger.info("\nFinal evaluation (after calibration):")
    final = evaluate(model, val_loader, device)

    # Package model
    package_model(model, OUTPUT_DIR)

    # Also create zip
    import subprocess
    subprocess.run(['zip', '-r', '-j', f'{OUTPUT_DIR}.zip', OUTPUT_DIR], check=True)

    logger.info(f"\nDone! Model: {OUTPUT_DIR}")
    logger.info(f"Final sn34_score: {final['sn34_score']:.4f}")
    logger.info(f"MCC: {final['mcc']:.4f}, Brier: {final['brier']:.6f}")
    logger.info(f"Target: sn34 > 0.95 to pass entrance exam")

    if final['sn34_score'] >= 0.95:
        logger.info("PASSED! Score meets entrance exam threshold.")
    else:
        logger.info(f"Need {0.95 - final['sn34_score']:.4f} more to pass. Consider more data or longer training.")


if __name__ == '__main__':
    main()
