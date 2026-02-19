"""
Fine-tune a ViT image detector for BitMind Subnet 34.
Uses diverse datasets: BitMind's own data + public deepfake datasets.
Outputs a model ready for gascli push.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
from transformers import ViTForImageClassification, ViTConfig, ViTImageProcessor
from safetensors.torch import save_file
from datasets import load_dataset
from PIL import Image
import yaml
import os
import shutil
import zipfile
import time


# === Config ===
OUTPUT_DIR = "/root/trained_image_detector"
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
IMAGE_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# === Dataset wrapper ===
class DeepfakeDataset(Dataset):
    def __init__(self, hf_dataset, transform, label_map=None):
        self.dataset = hf_dataset
        self.transform = transform
        self.label_map = label_map  # maps dataset labels to {0: real, 1: fake}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Handle different dataset formats
        if "image" in item:
            img = item["image"]
        elif "img" in item:
            img = item["img"]
        else:
            raise KeyError(f"No image key found in {item.keys()}")

        if not isinstance(img, Image.Image):
            img = Image.open(img)
        img = img.convert("RGB")
        img = self.transform(img)

        # Get label
        if "label" in item:
            label = item["label"]
        elif "is_ai" in item:
            label = item["is_ai"]
        elif "fake" in item:
            label = item["fake"]
        else:
            raise KeyError(f"No label key found in {item.keys()}")

        if self.label_map:
            label = self.label_map.get(label, label)

        return img, int(label)


# === Model ===
class ImageDetector(nn.Module):
    def __init__(self, num_classes=2, pretrained_name="google/vit-base-patch16-224-in21k"):
        super().__init__()
        self.vit = ViTForImageClassification.from_pretrained(
            pretrained_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x):
        # Input: [B, 3, 224, 224] float32, already normalized
        return self.vit(pixel_values=x).logits


def load_datasets():
    """Load multiple diverse datasets for training."""
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_datasets = []
    val_datasets = []

    # Dataset 1: BitMind's FakeClue dataset
    print("Loading BitMind FakeClue dataset...")
    try:
        ds = load_dataset("bitmind/FakeClue", split="train", trust_remote_code=True)
        # Check structure
        print(f"  FakeClue columns: {ds.column_names}")
        print(f"  FakeClue size: {len(ds)}")
        if len(ds) > 0:
            split = ds.train_test_split(test_size=0.1, seed=42)
            train_datasets.append(DeepfakeDataset(split["train"], transform))
            val_datasets.append(DeepfakeDataset(split["test"], val_transform))
            print(f"  Added FakeClue: {len(split['train'])} train, {len(split['test'])} val")
    except Exception as e:
        print(f"  Skipping FakeClue: {e}")

    # Dataset 2: AI vs Real images
    print("Loading AI vs Real dataset...")
    try:
        ds = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets", split="train", trust_remote_code=True)
        print(f"  AI vs Real columns: {ds.column_names}")
        print(f"  AI vs Real size: {len(ds)}")
        if len(ds) > 0:
            split = ds.train_test_split(test_size=0.1, seed=42)
            train_datasets.append(DeepfakeDataset(split["train"], transform))
            val_datasets.append(DeepfakeDataset(split["test"], val_transform))
            print(f"  Added AI vs Real: {len(split['train'])} train, {len(split['test'])} val")
    except Exception as e:
        print(f"  Skipping AI vs Real: {e}")

    # Dataset 3: Deepfake face classification
    print("Loading Deepfake Face dataset...")
    try:
        ds = load_dataset("pujanpaudel/deepfake_face_classification", split="train", trust_remote_code=True)
        print(f"  Deepfake Face columns: {ds.column_names}")
        print(f"  Deepfake Face size: {len(ds)}")
        if len(ds) > 0:
            split = ds.train_test_split(test_size=0.1, seed=42)
            train_datasets.append(DeepfakeDataset(split["train"], transform))
            val_datasets.append(DeepfakeDataset(split["test"], val_transform))
            print(f"  Added Deepfake Face: {len(split['train'])} train, {len(split['test'])} val")
    except Exception as e:
        print(f"  Skipping Deepfake Face: {e}")

    # Dataset 4: OpenFake
    print("Loading OpenFake dataset...")
    try:
        ds = load_dataset("ComplexDataLab/OpenFake", split="train", trust_remote_code=True)
        print(f"  OpenFake columns: {ds.column_names}")
        print(f"  OpenFake size: {len(ds)}")
        if len(ds) > 0:
            # Limit to 50k samples if very large
            if len(ds) > 50000:
                ds = ds.shuffle(seed=42).select(range(50000))
            split = ds.train_test_split(test_size=0.1, seed=42)
            train_datasets.append(DeepfakeDataset(split["train"], transform))
            val_datasets.append(DeepfakeDataset(split["test"], val_transform))
            print(f"  Added OpenFake: {len(split['train'])} train, {len(split['test'])} val")
    except Exception as e:
        print(f"  Skipping OpenFake: {e}")

    if not train_datasets:
        raise RuntimeError("No datasets loaded! Check network connectivity.")

    print(f"\nTotal: {len(train_datasets)} datasets loaded")
    train_combined = ConcatDataset(train_datasets)
    val_combined = ConcatDataset(val_datasets)
    print(f"Combined: {len(train_combined)} train, {len(val_combined)} val samples")

    return train_combined, val_combined


def train():
    print(f"Device: {DEVICE}")
    print(f"Loading datasets...")

    train_dataset, val_dataset = load_datasets()

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    print(f"\nInitializing model...")
    model = ImageDetector(num_classes=2).to(DEVICE)

    # Freeze most layers, only fine-tune last few + classifier
    for name, param in model.vit.named_parameters():
        if "classifier" in name or "layernorm" in name.lower():
            param.requires_grad = True
        elif "encoder.layer.11" in name or "encoder.layer.10" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        start = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | Acc: {100.*train_correct/train_total:.1f}%")

        train_acc = 100. * train_correct / train_total
        elapsed = time.time() - start

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        scheduler.step()

        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} ({elapsed:.0f}s)")
        print(f"  Train: Loss={train_loss/len(train_loader):.4f}, Acc={train_acc:.1f}%")
        print(f"  Val:   Loss={val_loss/len(val_loader):.4f}, Acc={val_acc:.1f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  New best! Saving model (val_acc={val_acc:.1f}%)")
            save_best(model)

    print(f"\nTraining complete! Best val accuracy: {best_val_acc:.1f}%")
    package_model()


def save_best(model):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    state_dict = model.vit.state_dict()
    save_file(state_dict, os.path.join(OUTPUT_DIR, "model.safetensors"))
    model.vit.config.save_pretrained(OUTPUT_DIR)


def package_model():
    """Package the model into a zip ready for gascli push."""
    print("\nPackaging model for BitMind submission...")

    # Write model_config.yaml
    config = {
        "name": "vit-deepfake-detector-finetuned",
        "version": "2.0.0",
        "modality": "image",
        "preprocessing": {
            "resize": [224, 224],
            "normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        "model": {
            "num_classes": 2,
            "weights_file": "model.safetensors",
        },
    }

    with open(os.path.join(OUTPUT_DIR, "model_config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Write model.py for BitMind's sandbox
    model_py = '''import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import ViTForImageClassification, ViTConfig


class ImageDetector(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        config = ViTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            image_size=224,
            patch_size=16,
            num_labels=num_classes,
            num_channels=3,
        )
        self.vit = ViTForImageClassification(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        return self.vit(pixel_values=x).logits


def load_model(weights_path: str, num_classes: int = 2) -> nn.Module:
    model = ImageDetector(num_classes=num_classes)
    state_dict = load_file(weights_path)
    model.vit.load_state_dict(state_dict)
    model.train(False)
    return model
'''

    with open(os.path.join(OUTPUT_DIR, "model.py"), "w") as f:
        f.write(model_py)

    # Create zip
    zip_path = "/root/trained_image_detector.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in ["model_config.yaml", "model.py", "model.safetensors", "config.json"]:
            fpath = os.path.join(OUTPUT_DIR, fname)
            if os.path.exists(fpath):
                zf.write(fpath, fname)

    size_mb = os.path.getsize(zip_path) / 1024 / 1024
    print(f"Model packaged: {zip_path} ({size_mb:.1f} MB)")
    print(f"\nTo push: gascli d push --image-model {zip_path} --wallet-name miner --wallet-hotkey default")


if __name__ == "__main__":
    train()
