#!/usr/bin/env python3
"""Audio deepfake detector training v1.
Uses Wav2Vec2 backbone fine-tuned for real vs synthetic audio classification.
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
from safetensors.torch import save_file
import pyarrow.parquet as pq
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

TARGET_SR = 16000
TARGET_SAMPLES = 96000  # 6 seconds at 16kHz


class AudioDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Load pretrained wav2vec2-base
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            'facebook/wav2vec2-base', num_labels=num_classes, ignore_mismatched_sizes=True
        )
        # GASBench expects [real, synthetic] = [0, 1]
        # wav2vec2 classifier output order matches this after training

    def forward(self, x):
        """x: [B, 96000] float32 in [-1, 1]"""
        outputs = self.model(input_values=x)
        return outputs.logits  # [B, 2]


class AudioDeepfakeDetectorFineTune(nn.Module):
    """Version that loads from pretrained HF deepfake model and fine-tunes."""
    def __init__(self, num_classes=2, pretrained_name=None):
        super().__init__()
        if pretrained_name:
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                pretrained_name, num_labels=num_classes, ignore_mismatched_sizes=True
            )
        else:
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                'facebook/wav2vec2-base', num_labels=num_classes, ignore_mismatched_sizes=True
            )
        self.swap_labels = False  # Set True if model outputs [fake, real] instead of [real, fake]
        self.register_buffer('temperature', torch.ones(1))

    def forward(self, x):
        outputs = self.model(input_values=x)
        logits = outputs.logits
        if self.swap_labels:
            logits = logits[:, [1, 0]]
        return logits / self.temperature


def load_audio_parquet(parquet_path, label=None, max_samples=5000):
    """Load audio samples from parquet file."""
    samples = []
    try:
        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        # Find audio column
        audio_col = None
        for col in ['audio_bytes', 'audio', 'speech', 'wav', 'waveform']:
            if col in df.columns:
                audio_col = col
                break
        if audio_col is None:
            logger.warning(f"  No audio column in {Path(parquet_path).name}, cols: {list(df.columns)}")
            return samples

        # Find label column
        has_label = 'label' in df.columns
        if label is None and not has_label:
            logger.warning(f"  No label column and no label provided for {Path(parquet_path).name}")
            return samples

        count = 0
        for idx in range(min(len(df), max_samples)):
            try:
                val = df[audio_col].iloc[idx]
                sample_label = int(df['label'].iloc[idx]) if has_label else label

                if isinstance(val, bytes):
                    arr = np.frombuffer(val, dtype=np.float32)
                elif isinstance(val, dict):
                    arr_data = val.get('array') or val.get('bytes')
                    if isinstance(arr_data, bytes):
                        arr = np.frombuffer(arr_data, dtype=np.float32)
                    elif isinstance(arr_data, (list, np.ndarray)):
                        arr = np.array(arr_data, dtype=np.float32)
                    else:
                        continue
                elif isinstance(val, (list, np.ndarray)):
                    arr = np.array(val, dtype=np.float32)
                else:
                    continue

                # Ensure correct length
                if len(arr) < TARGET_SAMPLES // 4:  # Too short, skip
                    continue
                if len(arr) < TARGET_SAMPLES:
                    arr = np.pad(arr, (0, TARGET_SAMPLES - len(arr)))
                elif len(arr) > TARGET_SAMPLES:
                    start = (len(arr) - TARGET_SAMPLES) // 2
                    arr = arr[start:start + TARGET_SAMPLES]

                # Normalize
                max_val = np.abs(arr).max()
                if max_val > 0:
                    arr = arr / max_val

                samples.append((arr.copy(), sample_label))
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


class AudioDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        arr, label = self.samples[idx]
        arr = arr.copy()

        if self.augment:
            # Random gain
            if random.random() < 0.5:
                gain = random.uniform(0.5, 1.5)
                arr = arr * gain
                arr = np.clip(arr, -1.0, 1.0)

            # Random noise
            if random.random() < 0.3:
                noise_std = random.uniform(0.001, 0.02)
                arr = arr + np.random.randn(*arr.shape).astype(np.float32) * noise_std
                arr = np.clip(arr, -1.0, 1.0)

            # Random time shift
            if random.random() < 0.3:
                shift = random.randint(-4800, 4800)  # up to 0.3 seconds
                arr = np.roll(arr, shift)

            # Random speed perturbation (simple via resampling)
            if random.random() < 0.2:
                speed = random.uniform(0.9, 1.1)
                new_len = int(len(arr) / speed)
                indices = np.linspace(0, len(arr) - 1, new_len)
                arr = np.interp(indices, np.arange(len(arr)), arr).astype(np.float32)
                if len(arr) < TARGET_SAMPLES:
                    arr = np.pad(arr, (0, TARGET_SAMPLES - len(arr)))
                else:
                    arr = arr[:TARGET_SAMPLES]

            # Random chunk removal (silence insertion)
            if random.random() < 0.15:
                chunk_len = random.randint(1600, 8000)
                start = random.randint(0, max(0, len(arr) - chunk_len))
                arr[start:start + chunk_len] = 0.0

        return torch.from_numpy(arr), label


def train_epoch(model, loader, optimizer, scheduler, device, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    total_loss, correct, total = 0, 0, 0

    for batch_idx, (audio, labels) in enumerate(loader):
        audio, labels = audio.to(device), labels.to(device)
        logits = model(audio)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 20 == 0:
            logger.info(f"Ep{epoch} [{batch_idx+1}/{len(loader)}] Loss:{total_loss/(batch_idx+1):.4f} Acc:{100*correct/total:.1f}%")

    return total_loss / max(len(loader), 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    for audio, labels in loader:
        logits = model(audio.to(device))
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
        for audio, labels in loader:
            logits = model(audio.to(device))
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


def package_model(model, output_dir, swap_labels=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the inner wav2vec2 model state dict with wrapper prefix
    state_dict = {}
    for k, v in model.model.state_dict().items():
        state_dict[f"model.{k}"] = v
    # Also save temperature and swap_labels flag
    state_dict['temperature'] = model.temperature
    save_file(state_dict, str(output_dir / 'model.safetensors'))

    swap_str = "True" if swap_labels else "False"

    (output_dir / 'model_config.yaml').write_text("""name: "wav2vec2-audio-deepfake-detector-v1"
version: "1.0.0"
modality: "audio"

preprocessing:
  sample_rate: 16000
  duration_seconds: 6.0

model:
  num_classes: 2
  weights_file: "model.safetensors"
""")

    # Save config.json from the HF model
    model.model.config.save_pretrained(str(output_dir))

    (output_dir / 'model.py').write_text('''import os
import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Config


class AudioDeepfakeDetector(nn.Module):
    """Wav2Vec2-based audio deepfake detector for GASBench SN34."""

    def __init__(self, num_classes=2):
        super().__init__()
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        if os.path.exists(config_path):
            config = Wav2Vec2Config.from_pretrained(os.path.dirname(__file__))
            config.num_labels = num_classes
            self.model = Wav2Vec2ForSequenceClassification(config)
        else:
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                "facebook/wav2vec2-base", num_labels=num_classes, ignore_mismatched_sizes=True
            )
        self.swap_labels = ''' + swap_str + '''
        self.register_buffer("temperature", torch.ones(1))

    def forward(self, x):
        outputs = self.model(input_values=x)
        logits = outputs.logits
        if self.swap_labels:
            logits = logits[:, [1, 0]]
        return logits / self.temperature


def load_model(weights_path, num_classes=2):
    model = AudioDeepfakeDetector(num_classes=num_classes)
    state_dict = load_file(weights_path)

    # Extract temperature
    if "temperature" in state_dict:
        temp_val = state_dict.pop("temperature")
        model.temperature.copy_(temp_val)

    # Load wav2vec2 weights (stored with "model." prefix)
    inner_state = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            inner_state[k[6:]] = v
        else:
            inner_state[k] = v
    model.model.load_state_dict(inner_state)
    model.train(False)
    return model
''')

    logger.info(f"Model packaged in {output_dir}")


def main():
    DATA_DIR = '/root/audio_train_data'
    OUTPUT_DIR = '/root/audio_detector_v1'
    SEED = 42
    EPOCHS = 20
    BATCH_SIZE = 16
    LR = 1e-5
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

    if not data_dir.exists():
        logger.error(f"Data directory {DATA_DIR} does not exist!")
        sys.exit(1)

    for f in sorted(data_dir.glob('*.parquet')):
        samples = load_audio_parquet(str(f))
        all_samples.extend(samples)

    # Also check /root/train_data for any audio parquets
    alt_dir = Path('/root/train_data')
    if alt_dir.exists():
        for f in sorted(alt_dir.glob('audio_*.parquet')):
            samples = load_audio_parquet(str(f))
            all_samples.extend(samples)

    n_real = sum(1 for _, l in all_samples if l == 0)
    n_synth = sum(1 for _, l in all_samples if l == 1)
    logger.info(f"\nTotal loaded: {len(all_samples)} ({n_real} real, {n_synth} synth)")

    if len(all_samples) < 100:
        logger.error("Not enough data!")
        sys.exit(1)

    # Soft balance
    if n_real > 0 and n_synth > 0:
        min_class = min(n_real, n_synth)
        max_per_class = int(min_class * 1.5)
        real_all = [(a, l) for a, l in all_samples if l == 0]
        synth_all = [(a, l) for a, l in all_samples if l == 1]
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

    train_ds = AudioDataset(train_samples, augment=True)
    val_ds = AudioDataset(val_samples, augment=False)

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
    model = AudioDeepfakeDetectorFineTune(num_classes=2, pretrained_name='facebook/wav2vec2-base').to(device)

    # Freeze feature extractor, train transformer + classifier
    for name, param in model.model.named_parameters():
        if 'feature_extractor' in name or 'feature_projection' in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01
    )

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
            torch.save(model.state_dict(), '/tmp/best_audio_model_v1.pt')
            logger.info(f"  *** New best sn34: {best_sn34:.4f} ***")
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch > 3:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    logger.info(f"\nBest epoch: {best_epoch} (sn34: {best_sn34:.4f})")
    model.load_state_dict(torch.load('/tmp/best_audio_model_v1.pt', weights_only=True))

    logger.info("Calibrating temperature...")
    calibrate_temperature(model, val_loader, device)

    logger.info("\nFinal evaluation (after calibration):")
    final = evaluate(model, val_loader, device)

    package_model(model, OUTPUT_DIR, swap_labels=model.swap_labels)

    logger.info(f"\nDone! Model: {OUTPUT_DIR}")
    logger.info(f"Final sn34_score: {final['sn34_score']:.4f}")
    logger.info(f"MCC: {final['mcc']:.4f}, Brier: {final['brier']:.6f}")


if __name__ == '__main__':
    main()
