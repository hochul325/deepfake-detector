#!/usr/bin/env python3
"""
Video Deepfake Detector Training Script v2
Target: sn34_score > 0.95 on gasbench video benchmark

Architecture: CLIP ViT-B/16 per-frame encoder + temporal mean pooling + classifier
- Strong pre-trained features from CLIP (generalizes across generators)
- Per-frame processing captures frame-level artifacts
- Mean pooling across frames gives robust video-level predictions
- Temperature scaling for calibration (critical for Brier score)

Training: Fine-tune on diverse real/synthetic image data from HuggingFace
(image data generalizes to per-frame video analysis)
"""

import os
import sys
import time
import random
import logging
import argparse
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# Model Architecture
# ============================================================

class VideoDeepfakeDetector(nn.Module):
    """
    Per-frame CLIP ViT-B/16 encoder with temporal mean pooling.

    Input:  [B, T, 3, H, W] float32, values 0-255, BGR format
    Output: [B, 2] logits for [real, synthetic]
    """

    def __init__(self, num_classes: int = 2, backbone: str = 'vit_base_patch16_clip_224.openai'):
        super().__init__()

        # ImageNet normalization constants (after BGR->RGB and /255)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Load CLIP ViT-B/16 as feature extractor (no classification head)
        self.encoder = timm.create_model(backbone, pretrained=True, num_classes=0)
        feature_dim = self.encoder.num_features  # 768 for ViT-B

        # Classification head with dropout for regularization
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

        # Temperature for post-training calibration (set after training)
        self.register_buffer('temperature', torch.ones(1))

        self._freeze_backbone()

    def _freeze_backbone(self):
        """Freeze most of encoder, fine-tune last 3 blocks + norm."""
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False

        # Unfreeze last 3 transformer blocks + final norm
        for name, param in self.encoder.named_parameters():
            if any(f'blocks.{i}.' in name for i in [9, 10, 11]):
                param.requires_grad = True
            elif 'norm' in name.lower():
                param.requires_grad = True

        # Always train classifier
        for param in self.classifier.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    def extract_features(self, x):
        """Extract features from a batch of images [B, 3, H, W] float32 0-255 BGR."""
        # BGR -> RGB
        x = x.flip(1)
        # Normalize to [0, 1]
        x = x / 255.0
        # ImageNet normalization
        x = (x - self.mean) / self.std
        # Extract features
        return self.encoder(x)

    def forward(self, x):
        """
        Forward pass for video input.

        Args:
            x: [B, T, 3, H, W] float32 values 0-255 BGR format
               OR [B, 3, H, W] for single image input
        """
        if x.dim() == 4:
            # Single image: [B, 3, H, W]
            features = self.extract_features(x)
            logits = self.classifier(features)
            return logits / self.temperature

        # Video: [B, T, 3, H, W]
        B, T = x.shape[:2]

        # Process all frames at once
        frames = x.reshape(B * T, *x.shape[2:])  # [B*T, 3, H, W]
        features = self.extract_features(frames)   # [B*T, D]
        features = features.view(B, T, -1)         # [B, T, D]

        # Temporal aggregation: mean pooling
        pooled = features.mean(dim=1)  # [B, D]

        # Classify
        logits = self.classifier(pooled)
        return logits / self.temperature


# ============================================================
# Dataset
# ============================================================

class DeepfakeImageDataset(Dataset):
    """Dataset that loads images from HuggingFace datasets for training."""

    def __init__(self, samples, transform=None):
        """
        Args:
            samples: list of (image_path_or_pil, label) tuples
                     label: 0=real, 1=synthetic
            transform: torchvision transforms
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_data, label = self.samples[idx]

        if isinstance(img_data, str):
            img = Image.open(img_data).convert('RGB')
        elif isinstance(img_data, Image.Image):
            img = img_data.convert('RGB')
        else:
            img = img_data

        if self.transform:
            img = self.transform(img)

        return img, label


def load_hf_dataset_samples(dataset_name, split='train', max_samples=5000,
                            image_col='image', label_col='label',
                            real_label=0, synthetic_label=1):
    """Load samples from a HuggingFace dataset using streaming."""
    from datasets import load_dataset

    logger.info(f"Loading {dataset_name} (max {max_samples} samples)...")

    samples = []
    try:
        ds = load_dataset(dataset_name, split=split, streaming=True, trust_remote_code=True)

        for i, item in enumerate(ds):
            if i >= max_samples:
                break

            if image_col in item and item[image_col] is not None:
                img = item[image_col]
                if isinstance(img, Image.Image):
                    # Determine label
                    if label_col in item:
                        raw_label = item[label_col]
                        if isinstance(raw_label, str):
                            label = 1 if raw_label.lower() in ('synthetic', 'fake', 'ai', 'generated', '1') else 0
                        else:
                            label = int(raw_label)
                    else:
                        label = synthetic_label

                    samples.append((img.copy(), label))

            if (i + 1) % 1000 == 0:
                logger.info(f"  Loaded {i+1} samples from {dataset_name}")

    except Exception as e:
        logger.warning(f"Error loading {dataset_name}: {e}")

    n_real = sum(1 for _, l in samples if l == 0)
    n_synth = sum(1 for _, l in samples if l == 1)
    logger.info(f"  {dataset_name}: {len(samples)} samples ({n_real} real, {n_synth} synthetic)")
    return samples


def prepare_training_data(data_dir='/tmp/train_data', max_per_dataset=3000):
    """Prepare training data from multiple HuggingFace datasets."""

    all_samples = []

    # === REAL IMAGE DATASETS ===
    real_datasets = [
        ('bitmind/celeb-a-hq', 'train', 'image', None, 3000),
        ('bitmind/ffhq-256', 'train', 'image', None, 3000),
        ('bitmind/MS-COCO-unique-256', 'train', 'image', None, 3000),
    ]

    for ds_name, split, img_col, lbl_col, max_n in real_datasets:
        try:
            samples = load_hf_dataset_samples(ds_name, split=split, max_samples=max_n,
                                               image_col=img_col)
            # Force label=0 (real) for these datasets
            samples = [(img, 0) for img, _ in samples]
            all_samples.extend(samples)
        except Exception as e:
            logger.warning(f"Skipping {ds_name}: {e}")

    # === SYNTHETIC IMAGE DATASETS ===
    synthetic_datasets = [
        ('bitmind/ideogram-27k', 'train', 'image', None, 3000),
        ('bitmind/JourneyDB', 'train', 'image', None, 3000),
        ('bitmind/GenImage_MidJourney', 'train', 'image', None, 3000),
    ]

    for ds_name, split, img_col, lbl_col, max_n in synthetic_datasets:
        try:
            samples = load_hf_dataset_samples(ds_name, split=split, max_samples=max_n,
                                               image_col=img_col)
            # Force label=1 (synthetic) for these datasets
            samples = [(img, 1) for img, _ in samples]
            all_samples.extend(samples)
        except Exception as e:
            logger.warning(f"Skipping {ds_name}: {e}")

    # === MIXED DATASETS (with labels) ===
    mixed_datasets = [
        ('bitmind/FakeClue', 'flux_dev', 'image', 'label', 2000),
        ('bitmind/FakeClue', 'sd_xl', 'image', 'label', 2000),
        ('bitmind/FakeClue', 'dalle3', 'image', 'label', 2000),
        ('bitmind/FakeClue', 'midjourney_v6', 'image', 'label', 2000),
    ]

    for ds_name, config, img_col, lbl_col, max_n in mixed_datasets:
        try:
            from datasets import load_dataset
            ds = load_dataset(ds_name, config, split='train', streaming=True, trust_remote_code=True)
            samples = []
            for i, item in enumerate(ds):
                if i >= max_n:
                    break
                if img_col in item and item[img_col] is not None:
                    img = item[img_col]
                    if isinstance(img, Image.Image):
                        label = int(item.get(lbl_col, 1))
                        samples.append((img.copy(), label))

            n_real = sum(1 for _, l in samples if l == 0)
            n_synth = sum(1 for _, l in samples if l == 1)
            logger.info(f"  {ds_name}/{config}: {len(samples)} ({n_real} real, {n_synth} synth)")
            all_samples.extend(samples)
        except Exception as e:
            logger.warning(f"Skipping {ds_name}/{config}: {e}")

    random.shuffle(all_samples)

    n_real = sum(1 for _, l in all_samples if l == 0)
    n_synth = sum(1 for _, l in all_samples if l == 1)
    logger.info(f"\nTotal: {len(all_samples)} samples ({n_real} real, {n_synth} synthetic)")

    return all_samples


# ============================================================
# Training
# ============================================================

def get_transforms(is_train=True, size=224):
    """Get data transforms. Train includes augmentation."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.2),
            transforms.ToTensor(),  # [0, 1] RGB
            # Convert to BGR 0-255 format (matching gasbench input)
            transforms.Lambda(lambda x: x.flip(0) * 255.0),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.flip(0) * 255.0),
        ])


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """Train for one epoch with label smoothing."""
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)  # [B, 3, H, W] input for single images
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 50 == 0:
            acc = 100 * correct / total
            logger.info(f"Epoch {epoch} [{batch_idx+1}/{len(dataloader)}] "
                       f"Loss: {total_loss/(batch_idx+1):.4f} Acc: {acc:.1f}%")

    return total_loss / len(dataloader), correct / total


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate model, return metrics including MCC and Brier score."""
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    for images, labels in dataloader:
        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_labels.extend(labels.numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    labels = np.array(all_labels)
    preds = np.array(all_preds)
    probs = np.array(all_probs)

    # Accuracy
    accuracy = (preds == labels).mean()

    # MCC (Matthews Correlation Coefficient)
    tp = ((preds == 1) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()

    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = float(tp * tn - fp * fn) / denom if denom > 0 else 0.0

    # Brier score (for binary: mean squared error of probability estimates)
    # prob_synthetic is the probability of class 1 (synthetic)
    brier = np.mean((probs[:, 1] - labels) ** 2)

    # sn34_score
    alpha, beta = 1.2, 1.8
    mcc_norm = ((mcc + 1) / 2) ** alpha
    brier_norm = ((0.25 - brier) / 0.25) ** beta
    sn34_score = np.sqrt(mcc_norm * brier_norm) if mcc_norm > 0 and brier_norm > 0 else 0

    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  MCC: {mcc:.4f}")
    logger.info(f"  Brier: {brier:.6f}")
    logger.info(f"  sn34_score: {sn34_score:.4f}")
    logger.info(f"  TP={tp} TN={tn} FP={fp} FN={fn}")

    return {
        'accuracy': accuracy, 'mcc': mcc, 'brier': brier,
        'sn34_score': sn34_score, 'probs': probs, 'labels': labels
    }


def calibrate_temperature(model, val_loader, device, lr=0.01, max_iter=200):
    """
    Optimize temperature parameter to minimize Brier score on validation set.
    This is critical for sn34_score since Brier has exponent 1.8.
    """
    logger.info("Calibrating temperature...")
    model.eval()

    # Collect all logits and labels
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            # Temporarily set temperature to 1 to get raw logits
            old_temp = model.temperature.clone()
            model.temperature.fill_(1.0)
            logits = model(images)
            model.temperature.copy_(old_temp)

            all_logits.append(logits.cpu())
            all_labels.append(labels)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    # Grid search for best temperature
    best_brier = float('inf')
    best_temp = 1.0

    for temp in np.arange(0.3, 5.0, 0.05):
        scaled_logits = logits / temp
        probs = F.softmax(scaled_logits, dim=1)
        brier = ((probs[:, 1] - labels.float()) ** 2).mean().item()

        if brier < best_brier:
            best_brier = brier
            best_temp = temp

    # Fine-grained search around best
    for temp in np.arange(max(0.1, best_temp - 0.2), best_temp + 0.2, 0.01):
        scaled_logits = logits / temp
        probs = F.softmax(scaled_logits, dim=1)
        brier = ((probs[:, 1] - labels.float()) ** 2).mean().item()

        if brier < best_brier:
            best_brier = brier
            best_temp = temp

    logger.info(f"  Best temperature: {best_temp:.3f} (Brier: {best_brier:.6f})")
    model.temperature.fill_(best_temp)

    return best_temp


# ============================================================
# Packaging
# ============================================================

def package_model(model, output_dir, temperature):
    """Package model for gasbench submission."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save weights
    state_dict = model.state_dict()
    weights_path = output_dir / 'model.safetensors'
    save_file(state_dict, str(weights_path))
    logger.info(f"Saved weights: {weights_path} ({weights_path.stat().st_size / 1e6:.1f} MB)")

    # Write model_config.yaml
    config_content = """name: "clip-vit-video-deepfake-detector-v2"
version: "2.0.0"
modality: "video"

preprocessing:
  resize: [224, 224]
  max_frames: 16

model:
  num_classes: 2
  weights_file: "model.safetensors"
"""
    (output_dir / 'model_config.yaml').write_text(config_content)

    # Write model.py
    model_py = '''import torch
import torch.nn as nn
import timm
from safetensors.torch import load_file


class VideoDeepfakeDetector(nn.Module):
    """
    Per-frame CLIP ViT-B/16 encoder with temporal mean pooling.

    Input:  [B, T, 3, H, W] float32, values 0-255, BGR format
    Output: [B, 2] logits for [real, synthetic]
    """

    def __init__(self, num_classes=2):
        super().__init__()

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.encoder = timm.create_model(
            'vit_base_patch16_clip_224.openai', pretrained=False, num_classes=0
        )
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
        frames = frames.flip(1)
        frames = frames / 255.0
        frames = (frames - self.mean) / self.std

        features = self.encoder(frames)
        features = features.view(B, T, -1)
        pooled = features.mean(dim=1)

        logits = self.classifier(pooled)
        return logits / self.temperature


def load_model(weights_path, num_classes=2):
    """Required entry point for gasbench."""
    model = VideoDeepfakeDetector(num_classes=num_classes)
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    model.train(False)
    return model
'''
    (output_dir / 'model.py').write_text(model_py)

    logger.info(f"Model packaged in {output_dir}")
    return output_dir


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Train Video Deepfake Detector v2')
    parser.add_argument('--output-dir', default='/root/video_detector_v2', help='Output directory')
    parser.add_argument('--epochs', type=int, default=8, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max-per-dataset', type=int, default=3000, help='Max samples per dataset')
    parser.add_argument('--val-split', type=float, default=0.15, help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # === Load Training Data ===
    logger.info("=" * 60)
    logger.info("LOADING TRAINING DATA")
    logger.info("=" * 60)

    all_samples = prepare_training_data(max_per_dataset=args.max_per_dataset)

    if len(all_samples) < 100:
        logger.error("Not enough training data!")
        sys.exit(1)

    # Split train/val
    random.shuffle(all_samples)
    val_size = int(len(all_samples) * args.val_split)
    val_samples = all_samples[:val_size]
    train_samples = all_samples[val_size:]

    logger.info(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Create datasets
    train_dataset = DeepfakeImageDataset(train_samples, transform=get_transforms(is_train=True))
    val_dataset = DeepfakeImageDataset(val_samples, transform=get_transforms(is_train=False))

    # Balanced sampling
    train_labels = [l for _, l in train_samples]
    class_counts = [train_labels.count(0), train_labels.count(1)]
    weights = [1.0 / class_counts[l] for l in train_labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_labels), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # === Create Model ===
    logger.info("=" * 60)
    logger.info("CREATING MODEL")
    logger.info("=" * 60)

    model = VideoDeepfakeDetector(num_classes=2)
    model = model.to(device)

    # Optimizer: only train unfrozen params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-7)

    # === Training Loop ===
    logger.info("=" * 60)
    logger.info("TRAINING")
    logger.info("=" * 60)

    best_sn34 = 0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n--- Epoch {epoch}/{args.epochs} ---")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        logger.info("Evaluating on validation set...")
        metrics = evaluate(model, val_loader, device)

        if metrics['sn34_score'] > best_sn34:
            best_sn34 = metrics['sn34_score']
            best_epoch = epoch
            # Save best model
            torch.save(model.state_dict(), '/tmp/best_model.pt')
            logger.info(f"  *** New best sn34_score: {best_sn34:.4f} (epoch {epoch}) ***")

    # Load best model
    logger.info(f"\nLoading best model from epoch {best_epoch} (sn34: {best_sn34:.4f})")
    model.load_state_dict(torch.load('/tmp/best_model.pt', weights_only=True))

    # === Temperature Calibration ===
    logger.info("=" * 60)
    logger.info("TEMPERATURE CALIBRATION")
    logger.info("=" * 60)

    best_temp = calibrate_temperature(model, val_loader, device)

    # Final evaluation with calibrated temperature
    logger.info("\nFinal evaluation with calibrated temperature:")
    final_metrics = evaluate(model, val_loader, device)

    # === Package Model ===
    logger.info("=" * 60)
    logger.info("PACKAGING MODEL")
    logger.info("=" * 60)

    output_dir = package_model(model, args.output_dir, best_temp)

    # Create zip
    import subprocess
    zip_path = f"{args.output_dir}.zip"
    subprocess.run(['zip', '-r', '-j', zip_path, str(output_dir)], check=True)
    logger.info(f"\nModel zip: {zip_path}")

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"  Best sn34_score (val): {final_metrics['sn34_score']:.4f}")
    logger.info(f"  MCC: {final_metrics['mcc']:.4f}")
    logger.info(f"  Brier: {final_metrics['brier']:.6f}")
    logger.info(f"  Temperature: {best_temp:.3f}")
    logger.info(f"  Output: {zip_path}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
