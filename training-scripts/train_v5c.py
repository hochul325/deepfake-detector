#!/usr/bin/env python3
"""v5c: Different seed (1234), ALL 12 blocks unfrozen, lower LR (5e-6)."""
import os, sys, io, time, random, logging
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
            if 'blocks.' in name:
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
            return self.classifier(self.extract_features(x)) / self.temperature
        B, T = x.shape[:2]
        frames = x.reshape(B * T, *x.shape[2:])
        features = self.extract_features(frames).view(B, T, -1).mean(dim=1)
        return self.classifier(features) / self.temperature

def load_video_frame_parquet(path, max_s=10000):
    samples = []
    try:
        df = pq.read_table(path).to_pandas()
        if 'label' not in df.columns or 'image' not in df.columns:
            return samples
        for idx in range(min(len(df), max_s)):
            try:
                val = df['image'].iloc[idx]
                label = int(df['label'].iloc[idx])
                if isinstance(val, dict) and val.get('bytes'):
                    samples.append((Image.open(io.BytesIO(val['bytes'])).convert('RGB'), label))
                elif isinstance(val, bytes):
                    samples.append((Image.open(io.BytesIO(val)).convert('RGB'), label))
            except Exception:
                continue
            if len(samples) >= max_s:
                break
        nr = sum(1 for _, l in samples if l == 0)
        logger.info(f"  {Path(path).name}: {len(samples)} ({nr}r, {len(samples)-nr}s)")
    except Exception as e:
        logger.warning(f"Error: {path}: {e}")
    return samples

def load_image_parquet(path, label, max_s=2000):
    samples = []
    try:
        df = pq.read_table(path).to_pandas()
        img_col = None
        for col in ['image', 'img']:
            if col in df.columns:
                img_col = col
                break
        if not img_col:
            for col in df.columns:
                if df[col].dtype == object and len(df) > 0:
                    val = df[col].iloc[0]
                    if isinstance(val, dict) and 'bytes' in val:
                        img_col = col
                        break
                    elif isinstance(val, bytes):
                        img_col = col
                        break
        if not img_col:
            return samples
        for idx in range(min(len(df), max_s)):
            try:
                val = df[img_col].iloc[idx]
                if isinstance(val, dict) and val.get('bytes'):
                    samples.append((Image.open(io.BytesIO(val['bytes'])).convert('RGB'), label))
                elif isinstance(val, bytes):
                    samples.append((Image.open(io.BytesIO(val)).convert('RGB'), label))
            except Exception:
                continue
            if len(samples) >= max_s:
                break
        logger.info(f"  {Path(path).name}: {len(samples)} ({'real' if label==0 else 'synth'})")
    except Exception:
        pass
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
    def __init__(self, qr=(20, 95)):
        self.qr = qr
    def __call__(self, img):
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=random.randint(*self.qr))
        buf.seek(0)
        return Image.open(buf).convert('RGB')

class RandomNoise:
    def __init__(self, sr=(0.005, 0.04)):
        self.sr = sr
    def __call__(self, img):
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.clip(arr + np.random.randn(*arr.shape).astype(np.float32) * random.uniform(*self.sr), 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))

class RandomDownscale:
    def __init__(self, sr=(0.25, 0.75)):
        self.sr = sr
    def __call__(self, img):
        w, h = img.size
        s = random.uniform(*self.sr)
        return img.resize((max(16, int(w*s)), max(16, int(h*s))), Image.BILINEAR).resize((w, h), Image.BILINEAR)

def get_train_transform(size=224):
    return transforms.Compose([
        transforms.Resize((size+32, size+32)),
        transforms.RandomCrop((size, size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomApply([JPEGCompress()], p=0.4),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.25),
        transforms.RandomApply([RandomNoise()], p=0.2),
        transforms.RandomApply([RandomDownscale()], p=0.2),
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
    labels, preds, probs = np.array(all_labels), np.array(all_preds), np.array(all_probs)
    acc = (preds == labels).mean()
    tp = ((preds==1)&(labels==1)).sum()
    tn = ((preds==0)&(labels==0)).sum()
    fp = ((preds==1)&(labels==0)).sum()
    fn = ((preds==0)&(labels==1)).sum()
    d = np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    mcc = float(tp*tn - fp*fn) / d if d > 0 else 0
    brier = float(np.mean((probs[:, 1] - labels) ** 2))
    mcc_norm = max(0, ((mcc+1)/2)) ** 1.2
    brier_norm = max(0, (0.25-brier)/0.25) ** 1.8
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
            all_logits.append(model(images.to(device)).cpu())
            all_labels.append(labels)
        model.temperature.copy_(old_temp)
    logits, labels = torch.cat(all_logits), torch.cat(all_labels)
    best_brier, best_temp = float('inf'), 1.0
    for temp in np.arange(0.1, 10.0, 0.02):
        brier = ((F.softmax(logits / temp, dim=1)[:, 1] - labels.float()) ** 2).mean().item()
        if brier < best_brier:
            best_brier, best_temp = brier, temp
    for temp in np.arange(max(0.05, best_temp-0.15), best_temp+0.15, 0.001):
        brier = ((F.softmax(logits / temp, dim=1)[:, 1] - labels.float()) ** 2).mean().item()
        if brier < best_brier:
            best_brier, best_temp = brier, temp
    logger.info(f"  Calibrated temperature: {best_temp:.3f} (Brier: {best_brier:.6f})")
    model.temperature.fill_(best_temp)
    return best_temp

def package_model(model, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), str(output_dir / 'model.safetensors'))
    (output_dir / 'model_config.yaml').write_text('name: "clip-vit-video-deepfake-detector-v5c"\nversion: "5.2.0"\nmodality: "video"\npreprocessing:\n  resize: [224, 224]\n  max_frames: 16\nmodel:\n  num_classes: 2\n  weights_file: "model.safetensors"\n')
    (output_dir / 'model.py').write_text('import torch\nimport torch.nn as nn\nimport timm\nfrom safetensors.torch import load_file\n\nclass VideoDeepfakeDetector(nn.Module):\n    def __init__(self, num_classes=2):\n        super().__init__()\n        self.register_buffer(\'mean\', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))\n        self.register_buffer(\'std\', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))\n        self.encoder = timm.create_model(\'vit_base_patch16_clip_224.openai\', pretrained=False, num_classes=0)\n        feature_dim = self.encoder.num_features\n        self.classifier = nn.Sequential(\n            nn.LayerNorm(feature_dim),\n            nn.Dropout(0.3),\n            nn.Linear(feature_dim, 256),\n            nn.GELU(),\n            nn.Dropout(0.2),\n            nn.Linear(256, num_classes)\n        )\n        self.register_buffer(\'temperature\', torch.ones(1))\n\n    def forward(self, x):\n        if x.dim() == 4:\n            x = x.unsqueeze(1)\n        B, T = x.shape[:2]\n        frames = x.reshape(B * T, *x.shape[2:])\n        frames = frames / 255.0\n        frames = (frames - self.mean) / self.std\n        features = self.encoder(frames)\n        features = features.view(B, T, -1)\n        pooled = features.mean(dim=1)\n        return self.classifier(pooled) / self.temperature\n\ndef load_model(weights_path, num_classes=2):\n    model = VideoDeepfakeDetector(num_classes=num_classes)\n    state_dict = load_file(weights_path)\n    model.load_state_dict(state_dict)\n    model.train(False)\n    return model\n')
    logger.info(f"Model packaged in {output_dir}")

def main():
    DATA_DIR = '/root/train_data'
    OUTPUT_DIR = '/root/video_detector_v4'
    SEED = 1234
    EPOCHS = 25
    BATCH_SIZE = 32
    LR = 5e-6
    VAL_SPLIT = 0.15

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    logger.info("Loading data...")
    video_samples, image_samples = [], []
    data_dir = Path(DATA_DIR)
    for f in sorted(data_dir.glob('video_*.parquet')):
        video_samples.extend(load_video_frame_parquet(str(f), 10000))
    logger.info(f"Video: {len(video_samples)}")

    for f in sorted(data_dir.glob('*.parquet')):
        if f.name.startswith('video_'):
            continue
        name = f.stem.lower()
        if any(k in name for k in ['ffhq', 'celeb', 'coco', 'ms_coco']):
            image_samples.extend(load_image_parquet(str(f), 0, 2000))
        elif any(k in name for k in ['genimage', 'journey', 'midjourney']):
            image_samples.extend(load_image_parquet(str(f), 1, 2000))
    logger.info(f"Image: {len(image_samples)}")

    all_samples = list(image_samples)
    if video_samples:
        of = max(1, min(10, len(image_samples) // max(len(video_samples), 1)))
        ov = video_samples * of
        while len(ov) < len(image_samples):
            ov.extend(random.sample(video_samples, min(len(video_samples), len(image_samples)-len(ov))))
        all_samples.extend(ov)

    nr = sum(1 for _, l in all_samples if l == 0)
    ns = len(all_samples) - nr
    logger.info(f"Total: {len(all_samples)} ({nr}r, {ns}s)")

    if nr > 0 and ns > 0:
        mc = min(nr, ns)
        mpc = int(mc * 1.5)
        ra = [(i, l) for i, l in all_samples if l == 0]
        sa = [(i, l) for i, l in all_samples if l == 1]
        random.shuffle(ra)
        random.shuffle(sa)
        all_samples = ra[:mpc] + sa[:mpc]
    logger.info(f"Balanced: {len(all_samples)}")

    random.shuffle(all_samples)
    vs = int(len(all_samples) * VAL_SPLIT)
    val_samples, train_samples = all_samples[:vs], all_samples[vs:]
    logger.info(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    train_ds = DeepfakeDataset(train_samples, get_train_transform())
    val_ds = DeepfakeDataset(val_samples, get_val_transform())
    tl = [l for _, l in train_samples]
    cc = [tl.count(0), tl.count(1)]
    w = [1.0/cc[l] for l in tl]
    sampler = WeightedRandomSampler(w, len(tl), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = VideoDeepfakeDetector().to(device)
    weight = torch.tensor([1.0, 1.15]).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.08, weight=weight)

    bp = [p for n, p in model.named_parameters() if p.requires_grad and 'classifier' not in n]
    cp = [p for n, p in model.named_parameters() if p.requires_grad and 'classifier' in n]
    optimizer = torch.optim.AdamW([{'params': bp, 'lr': LR}, {'params': cp, 'lr': LR * 20}], weight_decay=0.01)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = len(train_loader) * 2
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 0.5 * (1.0 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_sn34, best_epoch, patience_counter = 0, 0, 0
    for epoch in range(1, EPOCHS + 1):
        logger.info(f"\nEpoch {epoch}/{EPOCHS}")
        model.train()
        total_loss, correct, total = 0, 0, 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            if random.random() < 0.4:
                lam = max(np.random.beta(0.3, 0.3), 1 - np.random.beta(0.3, 0.3))
                idx = torch.randperm(images.size(0), device=device)
                images = lam * images + (1-lam) * images[idx]
                logits = model(images)
                loss = lam * criterion(logits, labels) + (1-lam) * criterion(logits, labels[idx])
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
            if (batch_idx+1) % 100 == 0:
                logger.info(f"  [{batch_idx+1}/{len(train_loader)}] Loss:{total_loss/(batch_idx+1):.4f} Acc:{100*correct/total:.1f}%")
        logger.info(f"Train Loss:{total_loss/len(train_loader):.4f} Acc:{correct/total:.4f}")
        metrics = evaluate(model, val_loader, device)
        if metrics['sn34_score'] > best_sn34:
            best_sn34 = metrics['sn34_score']
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), '/tmp/best_model_v5c.pt')
            logger.info(f"  *** New best sn34: {best_sn34:.4f} ***")
        else:
            patience_counter += 1
            if patience_counter >= 8 and epoch > 5:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    logger.info(f"\nBest epoch: {best_epoch} (sn34: {best_sn34:.4f})")
    model.load_state_dict(torch.load('/tmp/best_model_v5c.pt', weights_only=True))
    calibrate_temperature(model, val_loader, device)
    logger.info("\nFinal evaluation:")
    final = evaluate(model, val_loader, device)
    package_model(model, OUTPUT_DIR)
    logger.info(f"Done! sn34={final['sn34_score']:.4f} MCC={final['mcc']:.4f} Brier={final['brier']:.6f}")

if __name__ == '__main__':
    main()
