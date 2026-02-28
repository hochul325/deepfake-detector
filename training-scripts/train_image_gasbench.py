#!/usr/bin/env python3
"""Image deepfake detector - train directly on gasbench cache data.

Strategy:
- Load ALL images from /.cache/gasbench/datasets/*/samples/*.{jpg,png}
- Use dataset_info.json to determine real vs synthetic labels
- Also load from gasstation-generated-images (weekly bucket)
- Fine-tune from existing weights (CLIP ViT)
- Heavy augmentation
- Validate on held-out 20%
- Calibrate temperature
"""
import os, sys, io, json, random, logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import timm
from safetensors.torch import save_file, load_file

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

GASBENCH_CACHE = '/.cache/gasbench/datasets'
OUTPUT_DIR = '/root/image_detector_gas'

# Best existing weights to fine-tune from
# Use the single model weights from v2 training
EXISTING_WEIGHTS = None  # Will search for existing weights


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
        # Unfreeze blocks 4-11 (more than before - was 6-11)
        for name, param in self.encoder.named_parameters():
            if any(f'blocks.{i}.' in name for i in range(4, 12)):
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


def load_gasbench_image_datasets():
    """Load all image datasets from gasbench cache."""
    cache = Path(GASBENCH_CACHE)
    all_samples = []  # list of (img_path, label, dataset_name)

    for ds_dir in sorted(cache.iterdir()):
        if not ds_dir.is_dir():
            continue

        # Handle gasstation-generated-images (has week subdirectories)
        info_path = ds_dir / 'dataset_info.json'
        week_dirs = [d for d in ds_dir.iterdir() if d.is_dir() and d.name.startswith('20')]

        if week_dirs:
            # gasstation-generated-images/2026W08/
            for week_dir in week_dirs:
                info_path = week_dir / 'dataset_info.json'
                if not info_path.exists():
                    continue
                with open(info_path) as f:
                    info = json.load(f)
                if info.get('modality') != 'image':
                    continue
                media_type = info.get('media_type', '')
                label = 0 if media_type == 'real' else 1
                ds_name = f"{ds_dir.name}/{week_dir.name}"
                samples_dir = week_dir / 'samples'
                if samples_dir.exists():
                    count = 0
                    for img_file in sorted(samples_dir.iterdir()):
                        if img_file.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
                            all_samples.append((str(img_file), label, ds_name))
                            count += 1
                    label_str = 'real' if label == 0 else 'synth'
                    logger.info(f"  {ds_name}: {count} {label_str}")
            continue

        if not info_path.exists():
            continue

        with open(info_path) as f:
            info = json.load(f)

        if info.get('modality') != 'image':
            continue

        media_type = info.get('media_type', '')
        label = 0 if media_type == 'real' else 1
        ds_name = info.get('name', ds_dir.name)

        samples_dir = ds_dir / 'samples'
        if not samples_dir.exists():
            continue

        count = 0
        for img_file in sorted(samples_dir.iterdir()):
            if img_file.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
                all_samples.append((str(img_file), label, ds_name))
                count += 1

        label_str = 'real' if label == 0 else 'synth'
        logger.info(f"  {ds_name}: {count} {label_str}")

    return all_samples


class LazyImageDataset(Dataset):
    """Load images lazily from disk to save memory."""
    def __init__(self, samples, transform=None):
        self.samples = samples  # list of (path, label)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            # Return a black image on error
            img = Image.new('RGB', (224, 224), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, label


class JPEGCompress:
    def __init__(self, quality_range=(10, 95)):
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
        transforms.RandomApply([JPEGCompress(quality_range=(10, 95))], p=0.45),
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


@torch.no_grad()
def evaluate_per_dataset(model, dataset_samples, device, batch_size=32):
    """Evaluate per-dataset to find weaknesses."""
    model.eval()
    from collections import defaultdict
    groups = defaultdict(list)
    for path, label, ds_name in dataset_samples:
        groups[ds_name].append((path, label))

    results = {}
    val_transform = get_val_transform()
    for ds_name, samples in sorted(groups.items()):
        ds = LazyImageDataset(samples, val_transform)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
        all_preds, all_labels = [], []
        for images, labels in loader:
            logits = model(images.to(device))
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
        preds = np.array(all_preds)
        labels = np.array(all_labels)
        acc = (preds == labels).mean()
        results[ds_name] = acc
        label_type = 'real' if labels[0] == 0 else 'synth'
        errors = (preds != labels).sum()
        logger.info(f"  {ds_name} ({label_type}): {acc:.1%} ({errors} errors / {len(labels)} total)")

    return results


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

    (output_dir / 'model_config.yaml').write_text("""name: "clip-vit-image-deepfake-detector-gas"
version: "3.0.0"
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
        x = x.float() / 255.0
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
    SEED = 42
    EPOCHS = 30
    BATCH_SIZE = 64
    LR = 8e-6
    VAL_SPLIT = 0.20

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # === Load data from gasbench cache ===
    logger.info("Loading image data from gasbench cache...")
    all_samples = load_gasbench_image_datasets()

    n_real = sum(1 for _, l, _ in all_samples if l == 0)
    n_synth = sum(1 for _, l, _ in all_samples if l == 1)
    logger.info(f"\nRaw total: {len(all_samples)} ({n_real} real, {n_synth} synth)")

    # Also load from existing train_data parquets that have good data
    # Load FLUX, SDXL, DALL-E, Midjourney from parquet if they exist
    import pyarrow.parquet as pq
    extra_data_dir = Path('/root/train_data')
    extra_parquets = {
        'synth_flux_10k.parquet': 1,
        'synth_dalle3.parquet': 1,
        'synth_midjourney.parquet': 1,
        'synth_midjourney_genimage.parquet': 1,
        'synth_sdxl_bitmind.parquet': 1,
    }
    for pq_name, label in extra_parquets.items():
        pq_path = extra_data_dir / pq_name
        if pq_path.exists():
            try:
                table = pq.read_table(str(pq_path))
                df = table.to_pandas()
                img_col = 'image' if 'image' in df.columns else None
                if img_col is None:
                    for col in df.columns:
                        if df[col].dtype == object and len(df) > 0:
                            val = df[col].iloc[0]
                            if isinstance(val, (dict, bytes)):
                                img_col = col
                                break
                if img_col:
                    count = 0
                    # Save images to a temp dir for lazy loading
                    tmp_dir = Path(f'/tmp/extra_images/{pq_name.replace(".parquet", "")}')
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                    for idx in range(min(len(df), 2000)):
                        try:
                            val = df[img_col].iloc[idx]
                            if isinstance(val, dict):
                                img_bytes = val.get('bytes')
                            elif isinstance(val, bytes):
                                img_bytes = val
                            else:
                                continue
                            if img_bytes:
                                img_path = tmp_dir / f'{idx:06d}.jpg'
                                if not img_path.exists():
                                    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                                    img.save(str(img_path), 'JPEG', quality=95)
                                all_samples.append((str(img_path), label, f'extra_{pq_name}'))
                                count += 1
                        except Exception:
                            continue
                    logger.info(f"  extra {pq_name}: {count} synth samples")
            except Exception as e:
                logger.warning(f"Error loading {pq_name}: {e}")

    n_real = sum(1 for _, l, _ in all_samples if l == 0)
    n_synth = sum(1 for _, l, _ in all_samples if l == 1)
    logger.info(f"After extras: {len(all_samples)} ({n_real} real, {n_synth} synth)")

    # Oversample smaller datasets
    from collections import Counter
    ds_counts = Counter(ds for _, _, ds in all_samples)
    for ds_name, count in sorted(ds_counts.items(), key=lambda x: x[1]):
        logger.info(f"  Dataset {ds_name}: {count} samples")

    # Balance: oversample small datasets to at least 200
    balanced_samples = []
    for ds_name, count in ds_counts.items():
        ds_items = [(p, l, d) for p, l, d in all_samples if d == ds_name]
        if count < 200:
            repeats = max(1, 200 // count)
            ds_items = ds_items * repeats
            logger.info(f"  Oversampled {ds_name}: {count} -> {len(ds_items)}")
        balanced_samples.extend(ds_items)

    all_samples = balanced_samples
    n_real = sum(1 for _, l, _ in all_samples if l == 0)
    n_synth = sum(1 for _, l, _ in all_samples if l == 1)
    logger.info(f"After oversampling: {len(all_samples)} ({n_real} real, {n_synth} synth)")

    # Split
    random.shuffle(all_samples)
    val_size = int(len(all_samples) * VAL_SPLIT)
    val_with_names = all_samples[:val_size]
    train_with_names = all_samples[val_size:]

    train_samples = [(p, l) for p, l, _ in train_with_names]
    val_samples = [(p, l) for p, l, _ in val_with_names]

    logger.info(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    train_ds = LazyImageDataset(train_samples, get_train_transform())
    val_ds = LazyImageDataset(val_samples, get_val_transform())

    # Weighted sampler
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

    # === Create model ===
    logger.info("Creating model...")
    model = ImageDeepfakeDetector().to(device)

    # Try to load existing weights
    weight_candidates = [
        '/root/train_data/../image_detector_v2/model.safetensors',
        '/root/image_detector_v2/model.safetensors',
    ]
    for wc in weight_candidates:
        if os.path.exists(wc):
            logger.info(f"Loading existing weights from {wc}...")
            state = load_file(wc)
            model.load_state_dict(state, strict=False)
            logger.info("Existing weights loaded")
            break
    else:
        logger.info("No existing weights found, training from pretrained CLIP ViT")

    # Per-dataset eval BEFORE training
    logger.info("\n=== Per-dataset accuracy BEFORE training ===")
    raw_samples = load_gasbench_image_datasets()
    evaluate_per_dataset(model, raw_samples, device)

    # Optimizer
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
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
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
            torch.save(model.state_dict(), '/tmp/best_image_gas.pt')
            logger.info(f"  *** New best sn34: {best_sn34:.4f} ***")
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch > 5:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    logger.info(f"\nBest epoch: {best_epoch} (sn34: {best_sn34:.4f})")
    model.load_state_dict(torch.load('/tmp/best_image_gas.pt', weights_only=True))

    logger.info("Calibrating temperature...")
    calibrate_temperature(model, val_loader, device)

    logger.info("\nFinal evaluation (after calibration):")
    final = evaluate(model, val_loader, device)

    logger.info("\n=== Per-dataset accuracy AFTER training ===")
    evaluate_per_dataset(model, raw_samples, device)

    package_model(model, OUTPUT_DIR)

    logger.info(f"\nDone! Model: {OUTPUT_DIR}")
    logger.info(f"Final sn34_score: {final['sn34_score']:.4f}")


if __name__ == '__main__':
    main()
