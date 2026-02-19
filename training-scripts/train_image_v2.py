#!/usr/bin/env python3
"""Image deepfake detector training v2.
Key change from v1: expanded data with modern generators (FLUX, SDXL, DALL-E 3, Midjourney).
Old generators (BigGAN, GLIDE, ADM) capped at 2K to reduce dominance.
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
from PIL import Image
import timm
from safetensors.torch import save_file
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


class ImageDeepfakeDetector(nn.Module):
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

    def forward(self, x):
        x = x / 255.0
        x = (x - self.mean) / self.std
        features = self.encoder(x)
        return self.classifier(features) / self.temperature


def load_parquet_with_label(parquet_path, label, max_samples=5000):
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
            logger.warning(f"  No image column found in {Path(parquet_path).name}")
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


def load_parquet_with_label_col(parquet_path, max_samples=10000):
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
        transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.08),
        transforms.RandomApply([JPEGCompress(quality_range=(20, 95))], p=0.4),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.25),
        transforms.RandomApply([RandomNoise(std_range=(0.005, 0.04))], p=0.2),
        transforms.RandomApply([RandomDownscale(scale_range=(0.25, 0.75))], p=0.2),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),
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
    weight = torch.tensor([1.0, 1.15]).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.10, weight=weight)
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

    (output_dir / 'model_config.yaml').write_text("""name: "clip-vit-image-deepfake-detector-v2"
version: "2.0.0"
modality: "image"

preprocessing:
  resize: [224, 224]

model:
  num_classes: 2
  weights_file: "model.safetensors"
""")

    (output_dir / 'model.py').write_text('''import torch
import torch.nn as nn
import timm
from safetensors.torch import load_file


class ImageDeepfakeDetector(nn.Module):
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
        x = x / 255.0
        x = (x - self.mean) / self.std
        features = self.encoder(x)
        return self.classifier(features) / self.temperature


def load_model(weights_path, num_classes=2):
    model = ImageDeepfakeDetector(num_classes=num_classes)
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    model.train(False)
    return model
''')

    logger.info(f"Model packaged in {output_dir}")


def main():
    DATA_DIR = '/root/train_data'
    OUTPUT_DIR = '/root/image_detector_v2'
    SEED = 42
    EPOCHS = 30
    BATCH_SIZE = 64
    LR = 8e-6
    VAL_SPLIT = 0.15

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    logger.info("Loading data...")
    all_samples = []
    data_dir = Path(DATA_DIR)

    real_keywords = ['ffhq', 'celeb', 'coco', 'ms_coco', 'lfw', 'afhq', 'open_image',
                     'flickr', 'fashion', 'caltech', 'food', 'wikiart']
    synth_keywords = ['genimage', 'journey', 'midjourney', 'dalle', 'flux',
                      'diffusion', 'sdxl', 'stable', 'ideogram', 'grok',
                      'leonardo', 'sfhq', 'aura', 'kling', 'biggan', 'glide', 'adm']
    # Old generators to cap more aggressively
    old_gen_keywords = ['biggan', 'glide', 'adm']

    for f in sorted(data_dir.glob('*.parquet')):
        if f.name.startswith('video_'):
            continue
        try:
            table = pq.read_table(str(f), columns=['label'])
            has_label = 'label' in table.column_names
        except Exception:
            has_label = False

        if has_label:
            samples = load_parquet_with_label_col(str(f), max_samples=10000)
            all_samples.extend(samples)
            continue

        name = f.stem.lower()
        if any(k in name for k in real_keywords):
            samples = load_parquet_with_label(str(f), label=0, max_samples=5000)
            all_samples.extend(samples)
        elif any(k in name for k in synth_keywords):
            # Cap old generators at 2K, others at 5K
            cap = 2000 if any(k in name for k in old_gen_keywords) else 5000
            samples = load_parquet_with_label(str(f), label=1, max_samples=cap)
            all_samples.extend(samples)
        else:
            logger.info(f"  Skipping {f.name} (unknown type)")

    # Also load from image_train_data if it exists
    img_data_dir = Path('/root/image_train_data')
    if img_data_dir.exists():
        for f in sorted(img_data_dir.glob('*.parquet')):
            try:
                table = pq.read_table(str(f), columns=['label'])
                has_label = 'label' in table.column_names
            except Exception:
                has_label = False

            if has_label:
                samples = load_parquet_with_label_col(str(f), max_samples=10000)
                all_samples.extend(samples)
            else:
                name = f.stem.lower()
                if any(k in name for k in real_keywords):
                    samples = load_parquet_with_label(str(f), label=0, max_samples=5000)
                    all_samples.extend(samples)
                elif any(k in name for k in synth_keywords):
                    cap = 2000 if any(k in name for k in old_gen_keywords) else 5000
                    samples = load_parquet_with_label(str(f), label=1, max_samples=cap)
                    all_samples.extend(samples)

    n_real = sum(1 for _, l in all_samples if l == 0)
    n_synth = sum(1 for _, l in all_samples if l == 1)
    logger.info(f"\nTotal loaded: {len(all_samples)} ({n_real} real, {n_synth} synth)")

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
    model = ImageDeepfakeDetector().to(device)

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
            torch.save(model.state_dict(), '/tmp/best_image_model_v2.pt')
            logger.info(f"  *** New best sn34: {best_sn34:.4f} ***")
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch > 5:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    logger.info(f"\nBest epoch: {best_epoch} (sn34: {best_sn34:.4f})")
    model.load_state_dict(torch.load('/tmp/best_image_model_v2.pt', weights_only=True))

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
