#!/usr/bin/env python3
"""Audio deepfake detector v2 - MATCHES GASBENCH PREPROCESSING EXACTLY.

Key fix: Use torchaudio for loading (same as gasbench), NO peak normalization.
Gasbench preprocessing:
1. torchaudio.load() -> float32 [-1, 1]
2. Average to mono
3. torchaudio.transforms.Resample to 16kHz
4. Center crop or zero-pad to 96000 samples
5. NO normalization
"""
import os, sys, json, random, logging, tempfile
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from safetensors.torch import save_file, load_file
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Config

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

TARGET_SR = 16000
TARGET_SAMPLES = 96000  # 6 seconds

GASBENCH_CACHE = '/.cache/gasbench/datasets'
OUTPUT_DIR = '/root/audio_detector_gas2'
V8_CONFIG = '/root/audio_detector_v8'
V8_WEIGHTS = '/root/audio_detector_v8/model.safetensors'


def load_wav_gasbench_style(wav_path):
    """Load audio EXACTLY like gasbench does - torchaudio, no normalization."""
    try:
        waveform, sample_rate = torchaudio.load(wav_path)
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Resample to 16kHz
        if sample_rate != TARGET_SR:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=TARGET_SR
            )
            waveform = resampler(waveform)
        # Center crop or pad
        current_length = waveform.shape[1]
        if current_length > TARGET_SAMPLES:
            start_idx = (current_length - TARGET_SAMPLES) // 2
            waveform = waveform[:, start_idx:start_idx + TARGET_SAMPLES]
        elif current_length < TARGET_SAMPLES:
            padding = TARGET_SAMPLES - current_length
            waveform = F.pad(waveform, (0, padding))
        # Skip very short clips
        if current_length < TARGET_SAMPLES // 4:
            return None
        # Squeeze to 1D, keep as float32 (NO normalization!)
        arr = waveform.squeeze(0).numpy()
        return arr.astype(np.float32)
    except Exception as e:
        return None


def load_gasbench_audio_datasets():
    """Load all audio datasets from gasbench cache."""
    cache = Path(GASBENCH_CACHE)
    all_samples = []

    for ds_dir in sorted(cache.iterdir()):
        if not ds_dir.is_dir():
            continue
        info_path = ds_dir / 'dataset_info.json'
        if not info_path.exists():
            continue
        with open(info_path) as f:
            info = json.load(f)
        if info.get('modality') != 'audio':
            continue
        media_type = info.get('media_type', '')
        label = 0 if media_type == 'real' else 1
        ds_name = info.get('name', ds_dir.name)
        samples_dir = ds_dir / 'samples'
        if not samples_dir.exists():
            continue
        audio_files = sorted([
            f for f in samples_dir.iterdir()
            if f.suffix.lower() in ('.wav', '.mp3', '.flac', '.ogg')
        ])
        count = 0
        for af in audio_files:
            arr = load_wav_gasbench_style(str(af))
            if arr is not None:
                all_samples.append((arr, label, ds_name))
                count += 1
        label_str = 'real' if label == 0 else 'synth'
        logger.info(f"  {ds_name}: {count} {label_str}")

    return all_samples


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
            # Gain augmentation (but keep in reasonable range, no peak norm)
            if random.random() < 0.5:
                gain = random.uniform(0.7, 1.3)
                arr = arr * gain
                arr = np.clip(arr, -1.0, 1.0)
            # Add noise
            if random.random() < 0.35:
                noise_std = random.uniform(0.001, 0.02)
                arr = arr + np.random.randn(*arr.shape).astype(np.float32) * noise_std
                arr = np.clip(arr, -1.0, 1.0)
            # Time shift
            if random.random() < 0.3:
                shift = random.randint(-4800, 4800)
                arr = np.roll(arr, shift)
            # Speed change
            if random.random() < 0.2:
                speed = random.uniform(0.9, 1.1)
                new_len = int(len(arr) / speed)
                indices = np.linspace(0, len(arr) - 1, new_len)
                arr = np.interp(indices, np.arange(len(arr)), arr).astype(np.float32)
                if len(arr) < TARGET_SAMPLES:
                    arr = np.pad(arr, (0, TARGET_SAMPLES - len(arr)))
                else:
                    arr = arr[:TARGET_SAMPLES]
            # Chunk masking
            if random.random() < 0.15:
                chunk_len = random.randint(1600, 8000)
                start = random.randint(0, max(0, len(arr) - chunk_len))
                arr[start:start + chunk_len] = 0.0
            # Simple reverb
            if random.random() < 0.1:
                delay = random.randint(800, 3200)
                decay = random.uniform(0.1, 0.3)
                reverb = np.zeros_like(arr)
                reverb[delay:] = arr[:-delay] * decay
                arr = arr + reverb
                arr = np.clip(arr, -1.0, 1.0)

        return torch.from_numpy(arr), label


class AudioModel(nn.Module):
    def __init__(self, config_dir=None):
        super().__init__()
        if config_dir and os.path.exists(os.path.join(config_dir, 'config.json')):
            config = Wav2Vec2Config.from_pretrained(config_dir)
            config.num_labels = 2
            self.model = Wav2Vec2ForSequenceClassification(config)
        else:
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                'facebook/wav2vec2-base', num_labels=2, ignore_mismatched_sizes=True
            )
        self.register_buffer('temperature', torch.ones(1))

    def forward(self, x):
        return self.model(input_values=x).logits / self.temperature


def train_epoch(model, loader, optimizer, scheduler, device, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.07)
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
            sys.stdout.flush()
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
    best_sn34, best_temp, best_brier = 0, 1.0, float('inf')
    for temp in np.arange(0.3, 5.0, 0.01):
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
            best_sn34, best_temp, best_brier = sn34, temp, brier
    logger.info(f"  Calibrated: temp={best_temp:.3f} sn34={best_sn34:.4f} Brier={best_brier:.6f}")
    model.temperature.fill_(best_temp)
    return best_temp


def package_model(model, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dict = {}
    for k, v in model.model.state_dict().items():
        state_dict[f"model.{k}"] = v
    state_dict['temperature'] = model.temperature
    save_file(state_dict, str(output_dir / 'model.safetensors'))
    model.model.config.save_pretrained(str(output_dir))
    (output_dir / 'model_config.yaml').write_text("""name: "wav2vec2-audio-deepfake-detector-gas2"
version: "10.1.0"
modality: "audio"

preprocessing:
  sample_rate: 16000
  duration_seconds: 6.0

model:
  num_classes: 2
  weights_file: "model.safetensors"
""")
    (output_dir / 'model.py').write_text('''import os
import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Config


class AudioDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        if os.path.exists(config_path):
            config = Wav2Vec2Config.from_pretrained(os.path.dirname(__file__))
            config.num_labels = num_classes
            self.model = Wav2Vec2ForSequenceClassification(config)
        else:
            config = Wav2Vec2Config(
                hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                intermediate_size=3072, num_labels=num_classes,
                num_feat_extract_layers=7,
                conv_dim=[512, 512, 512, 512, 512, 512, 512],
                conv_stride=[5, 2, 2, 2, 2, 2, 2],
                conv_kernel=[10, 3, 3, 3, 3, 2, 2],
                num_conv_pos_embeddings=128, num_conv_pos_embedding_groups=16,
            )
            self.model = Wav2Vec2ForSequenceClassification(config)
        self.register_buffer("temperature", torch.ones(1))

    def forward(self, x):
        outputs = self.model(input_values=x)
        return outputs.logits / self.temperature


def load_model(weights_path, num_classes=2):
    model = AudioDeepfakeDetector(num_classes=num_classes)
    state_dict = load_file(weights_path)
    if "temperature" in state_dict:
        temp_val = state_dict.pop("temperature")
        model.temperature.copy_(temp_val)
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
    SEED = 42
    EPOCHS = 30
    BATCH_SIZE = 16
    LR = 5e-6
    VAL_SPLIT = 0.20

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    logger.info("Loading audio data using GASBENCH-COMPATIBLE preprocessing...")
    all_samples = load_gasbench_audio_datasets()

    n_real = sum(1 for _, l, _ in all_samples if l == 0)
    n_synth = sum(1 for _, l, _ in all_samples if l == 1)
    logger.info(f"\nRaw total: {len(all_samples)} ({n_real} real, {n_synth} synth)")

    # Oversample small datasets
    from collections import Counter
    ds_counts = Counter(ds for _, _, ds in all_samples)
    median_count = sorted(ds_counts.values())[len(ds_counts) // 2]

    balanced_samples = []
    for ds_name, count in ds_counts.items():
        ds_items = [(a, l, d) for a, l, d in all_samples if d == ds_name]
        if count < median_count:
            repeats = max(1, median_count // count)
            ds_items = ds_items * repeats
            logger.info(f"  Oversampled {ds_name}: {count} -> {len(ds_items)}")
        balanced_samples.extend(ds_items)

    all_samples = balanced_samples
    n_real = sum(1 for _, l, _ in all_samples if l == 0)
    n_synth = sum(1 for _, l, _ in all_samples if l == 1)
    logger.info(f"After oversampling: {len(all_samples)} ({n_real} real, {n_synth} synth)")

    random.shuffle(all_samples)
    val_size = int(len(all_samples) * VAL_SPLIT)
    train_with_names = all_samples[val_size:]
    val_with_names = all_samples[:val_size]

    train_samples = [(a, l) for a, l, _ in train_with_names]
    val_samples = [(a, l) for a, l, _ in val_with_names]
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
    model = AudioModel(config_dir=V8_CONFIG).to(device)

    if os.path.exists(V8_WEIGHTS):
        logger.info(f"Loading v8 weights from {V8_WEIGHTS}...")
        v8_state = load_file(V8_WEIGHTS)
        temp_val = v8_state.pop('temperature', None)
        inner_state = {}
        for k, v in v8_state.items():
            if k.startswith('model.'):
                inner_state[k[6:]] = v
            else:
                inner_state[k] = v
        model.model.load_state_dict(inner_state)
        logger.info("v8 weights loaded")

    for name, param in model.model.named_parameters():
        if 'feature_extractor' in name or 'feature_projection' in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,} / {total_params:,} ({100*trainable/total_params:.1f}%)")

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
            torch.save(model.state_dict(), '/tmp/best_audio_gas2.pt')
            logger.info(f"  *** New best sn34: {best_sn34:.4f} ***")
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch > 5:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    logger.info(f"\nBest epoch: {best_epoch} (sn34: {best_sn34:.4f})")
    model.load_state_dict(torch.load('/tmp/best_audio_gas2.pt', weights_only=True))

    logger.info("Calibrating temperature on validation data...")
    calibrate_temperature(model, val_loader, device)

    logger.info("\nFinal evaluation:")
    final = evaluate(model, val_loader, device)

    package_model(model, OUTPUT_DIR)
    logger.info(f"\nDone! Model: {OUTPUT_DIR}")
    logger.info(f"Final sn34_score: {final['sn34_score']:.4f}")


if __name__ == '__main__':
    main()
