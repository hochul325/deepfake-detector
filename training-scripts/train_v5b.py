#!/usr/bin/env python3
"""Video deepfake detector training v5b.
Changes from v4:
- More training data from failing datasets (12 extra frames per video)
- Label smoothing 0.08 (was 0.05)
- Mixup 0.3 alpha, 40% chance
- 1.5x class balance cap (was 1.15x)
- TTA in model.py (horizontal flip feature averaging)
- 25 epochs with patience 8
- Slightly lower backbone LR (8e-6)
- Class weight on synthetic 1.15x
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
        self._freeze_backbone()

    def _freeze_backbone(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for name, param in self.encoder.named_parameters():
            if any(f'blocks.{i}.' in name for i in [6, 7, 8, 9, 10, 11]):
                param.requires_grad = True
            elif 'norm' in name.lower():
                param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

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


def load_video_frame_parquet(parquet_path, max_samples=10000):
    samples = []
    try:
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        if 'label' not in df.columns or 'image' not in df.columns:
            return samples
        count = 0
        for idx in range(min(len(df), max_samples)):
            try:
                val = df['image'].iloc[idx]
                label = int(df['label'].iloc[idx])
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
            except Exception:
                continue
            if count >= max_samples:
                break
        n_real = sum(1 for _, l in samples if l == 0)
        n_synth = sum(1 for _, l in samples if l == 1)
        logger.info(f"  Loaded {count} from {Path(parquet_path).name} ({n_real}r, {n_synth}s)")
    except Exception as e:
        logger.warning(f"Error loading {parquet_path}: {e}")
    return samples


def load_image_parquet(parquet_path, label, max_samples=2000):
    samples = []
    try:
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
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
            except Exception:
                continue
            if count >= max_samples:
                break
        logger.info(f"  Loaded {count} from {Path(parquet_path).name} ({'real' if label==0 else 'synth'})")
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
    def __init__(self, quality_range=(20, 95)):
        self.quality_range = quality_range
    def __call__(self, img):
        quality = random.randint(*self.quality_range)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        return Image.open(buf).convert('RGB')


class RandomNoise:
    def __init__(self, std_range=(0.005, 0.04)):
        self.std_range = std_range
    def __call__(self, img):
        arr = np.array(img, dtype=np.float32) / 255.0
        std = random.uniform(*self.std_range)
        noise = np.random.randn(*arr.shape).astype(np.float32) * std
        arr = np.clip(arr + noise, 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))


class RandomDownscale:
    def __init__(self, scale_range=(0.25, 0.75)):
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
        transforms.Resize((size + 32, size + 32)),
        transforms.RandomCrop((size, size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomApply([JPEGCompress(quality_range=(20, 95))], p=0.4),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.25),
        transforms.RandomApply([RandomNoise(std_range=(0.005, 0.04))], p=0.2),
        transforms.RandomApply([RandomDownscale(scale_range=(0.25, 0.75))], p=0.2),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255.0),
    ])


def get_val_transform(size=224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255.0),
    ])


def train_epoch(model, loader, optimizer, scheduler, device, epoch, mixup_alpha=0.3):
    model.train()
    # CE with label smoothing + class weight
    weight = torch.tensor([1.0, 1.15]).to(device)  # Weight synthetic slightly
    criterion = nn.CrossEntropyLoss(label_smoothing=0.08, weight=weight)

    total_loss, correct, total = 0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        if mixup_alpha > 0 and random.random() < 0.4:
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

    best_brier, best_temp = float('inf'), 1.0
    for temp in np.arange(0.1, 10.0, 0.02):
        probs = F.softmax(logits / temp, dim=1)
        brier = ((probs[:, 1] - labels.float()) ** 2).mean().item()
        if brier < best_brier:
            best_brier, best_temp = brier, temp

    for temp in np.arange(max(0.05, best_temp - 0.15), best_temp + 0.15, 0.001):
        probs = F.softmax(logits / temp, dim=1)
        brier = ((probs[:, 1] - labels.float()) ** 2).mean().item()
        if brier < best_brier:
            best_brier, best_temp = brier, temp

    logger.info(f"  Calibrated temperature: {best_temp:.3f} (Brier: {best_brier:.6f})")
    model.temperature.fill_(best_temp)
    return best_temp


def package_model(model, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), str(output_dir / 'model.safetensors'))

    (output_dir / 'model_config.yaml').write_text("""name: "clip-vit-video-deepfake-detector-v5"
version: "5.0.0"
modality: "video"

preprocessing:
  resize: [224, 224]
  max_frames: 16

model:
  num_classes: 2
  weights_file: "model.safetensors"
""")

    # Model with TTA (horizontal flip feature averaging)
    (output_dir / 'model.py').write_text('''import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def _encode(self, x):
        x = x / 255.0
        x = (x - self.mean) / self.std
        return self.encoder(x)

    def forward(self, x):
        """Input: float32 [B, T, 3, H, W] or [B, 3, H, W] RGB 0-255"""
        if x.dim() == 4:
            x = x.unsqueeze(1)
        B, T = x.shape[:2]
        frames = x.reshape(B * T, *x.shape[2:])

        # TTA: average features from original + horizontally flipped
        feat_orig = self._encode(frames)
        feat_flip = self._encode(frames.flip(-1))
        features = (feat_orig + feat_flip) / 2.0

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


def main():
    DATA_DIR = '/root/train_data'
    OUTPUT_DIR = '/root/video_detector_v4'
    SEED = 42
    EPOCHS = 25
    BATCH_SIZE = 32
    LR = 8e-6
    VAL_SPLIT = 0.15

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    logger.info("Loading data...")
    video_samples = []
    image_samples = []
    data_dir = Path(DATA_DIR)

    video_parquets = sorted(data_dir.glob('video_*.parquet'))
    logger.info(f"\nFound {len(video_parquets)} video frame parquets:")
    for f in video_parquets:
        samples = load_video_frame_parquet(str(f), max_samples=10000)
        video_samples.extend(samples)

    v_real = sum(1 for _, l in video_samples if l == 0)
    v_synth = sum(1 for _, l in video_samples if l == 1)
    logger.info(f"Video frames: {len(video_samples)} ({v_real} real, {v_synth} synth)")

    IMAGE_MAX_PER_FILE = 2000
    real_keywords = ['ffhq', 'celeb', 'coco', 'ms_coco']
    synth_keywords = ['genimage', 'journey', 'midjourney']

    logger.info(f"\nLoading image data (max {IMAGE_MAX_PER_FILE} per file):")
    for f in sorted(data_dir.glob('*.parquet')):
        if f.name.startswith('video_'):
            continue
        name = f.stem.lower()
        if any(k in name for k in real_keywords):
            samples = load_image_parquet(str(f), label=0, max_samples=IMAGE_MAX_PER_FILE)
            image_samples.extend(samples)
        elif any(k in name for k in synth_keywords):
            samples = load_image_parquet(str(f), label=1, max_samples=IMAGE_MAX_PER_FILE)
            image_samples.extend(samples)

    i_real = sum(1 for _, l in image_samples if l == 0)
    i_synth = sum(1 for _, l in image_samples if l == 1)
    logger.info(f"Image frames: {len(image_samples)} ({i_real} real, {i_synth} synth)")

    all_samples = list(image_samples)
    if video_samples:
        oversample_factor = max(1, len(image_samples) // max(len(video_samples), 1))
        oversample_factor = min(oversample_factor, 10)
        oversampled_video = video_samples * oversample_factor
        while len(oversampled_video) < len(image_samples):
            oversampled_video.extend(random.sample(video_samples,
                min(len(video_samples), len(image_samples) - len(oversampled_video))))
        all_samples.extend(oversampled_video)
        logger.info(f"Video oversampled {oversample_factor}x: {len(oversampled_video)} samples")

    n_real = sum(1 for _, l in all_samples if l == 0)
    n_synth = sum(1 for _, l in all_samples if l == 1)
    logger.info(f"\nTotal: {len(all_samples)} ({n_real} real, {n_synth} synth)")

    # Soft balance: 1.5x cap
    if n_real > 0 and n_synth > 0:
        min_class = min(n_real, n_synth)
        max_per_class = int(min_class * 1.5)
        real_all = [(img, l) for img, l in all_samples if l == 0]
        synth_all = [(img, l) for img, l in all_samples if l == 1]
        random.shuffle(real_all)
        random.shuffle(synth_all)
        all_samples = real_all[:max_per_class] + synth_all[:max_per_class]
        n_real = sum(1 for _, l in all_samples if l == 0)
        n_synth = sum(1 for _, l in all_samples if l == 1)
        logger.info(f"After soft balancing: {len(all_samples)} ({n_real} real, {n_synth} synth)")

    random.shuffle(all_samples)
    val_size = int(len(all_samples) * VAL_SPLIT)
    val_samples = all_samples[:val_size]
    train_samples = all_samples[val_size:]
    logger.info(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    train_ds = DeepfakeDataset(train_samples, get_train_transform())
    val_ds = DeepfakeDataset(val_samples, get_val_transform())

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

    logger.info("Creating model...")
    model = VideoDeepfakeDetector().to(device)

    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and 'classifier' not in n]
    classifier_params = [p for n, p in model.named_parameters()
                         if p.requires_grad and 'classifier' in n]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': LR},
        {'params': classifier_params, 'lr': LR * 10},
    ], weight_decay=0.01)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = len(train_loader) * 2

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_sn34, best_epoch = 0, 0
    patience, patience_counter = 8, 0

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
            torch.save(model.state_dict(), '/tmp/best_model_v5b.pt')
            logger.info(f"  *** New best sn34: {best_sn34:.4f} ***")
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch > 5:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    logger.info(f"\nBest epoch: {best_epoch} (sn34: {best_sn34:.4f})")
    model.load_state_dict(torch.load('/tmp/best_model_v5b.pt', weights_only=True))

    logger.info("Calibrating temperature...")
    calibrate_temperature(model, val_loader, device)

    logger.info("\nFinal evaluation (after calibration):")
    final = evaluate(model, val_loader, device)

    package_model(model, OUTPUT_DIR)

    logger.info(f"\nDone! Model: {OUTPUT_DIR}")
    logger.info(f"Final sn34_score: {final['sn34_score']:.4f}")
    logger.info(f"MCC: {final['mcc']:.4f}, Brier: {final['brier']:.6f}")


if __name__ == '__main__':
    main()
