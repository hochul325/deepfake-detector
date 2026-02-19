"""
Build a video deepfake detector for BitMind Subnet 34.
Downloads a pre-trained VideoMAE model fine-tuned for deepfake detection,
wraps it in GASBench-compatible format, and packages as a zip.
"""

import torch
import os
import shutil
import zipfile
import yaml
from safetensors.torch import save_file
from transformers import VideoMAEForVideoClassification


OUTPUT_DIR = "/root/video_detector"
ZIP_PATH = "/root/video_detector.zip"


def build():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Try the pre-fine-tuned deepfake model first, fall back to base
    print("Downloading VideoMAE model...")
    try:
        hf_model = VideoMAEForVideoClassification.from_pretrained(
            "Ammar2k/videomae-base-finetuned-deepfake-subset",
            num_labels=2,
            ignore_mismatched_sizes=True,
        )
        print("  Loaded Ammar2k/videomae-base-finetuned-deepfake-subset")
    except Exception as e:
        print(f"  Failed to load fine-tuned model: {e}")
        print("  Falling back to MCG-NJU/videomae-base...")
        hf_model = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base",
            num_labels=2,
            ignore_mismatched_sizes=True,
        )
        print("  Loaded MCG-NJU/videomae-base")

    # Save weights
    print("Saving model weights...")
    state_dict = {}
    for k, v in hf_model.state_dict().items():
        state_dict[f"model.{k}"] = v
    save_file(state_dict, os.path.join(OUTPUT_DIR, "model.safetensors"))

    # Save config.json
    hf_model.config.num_labels = 2
    hf_model.config.save_pretrained(OUTPUT_DIR)

    # Write model_config.yaml
    config = {
        "name": "videomae-deepfake-detector",
        "version": "1.0.0",
        "modality": "video",
        "preprocessing": {
            "resize": [224, 224],
            "max_frames": 16,
        },
        "model": {
            "num_classes": 2,
            "weights_file": "model.safetensors",
        },
    }
    with open(os.path.join(OUTPUT_DIR, "model_config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Write model.py
    model_py = '''import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import VideoMAEForVideoClassification, VideoMAEConfig


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class VideoDeepfakeDetector(nn.Module):
    """VideoMAE-based deepfake detector for GASBench SN34."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        config = VideoMAEConfig(
            image_size=224,
            patch_size=16,
            num_channels=3,
            num_frames=16,
            tubelet_size=2,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            num_labels=num_classes,
        )
        self.model = VideoMAEForVideoClassification(config)

        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, 3, H, W] uint8 BGR [0, 255] from GASBench
        Returns:
            logits: [B, num_classes] raw logits
        """
        # BGR -> RGB
        x = x.flip(2)

        # uint8 -> float, normalize
        x = x.float() / 255.0
        x = (x - self.mean) / self.std

        # [B, T, C, H, W] -> [B, C, T, H, W] for VideoMAE
        x = x.permute(0, 2, 1, 3, 4)

        outputs = self.model(pixel_values=x)
        return outputs.logits


def load_model(weights_path: str, num_classes: int = 2) -> nn.Module:
    """Required entry point for GASBench."""
    model = VideoDeepfakeDetector(num_classes=num_classes)
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    model.train(False)
    return model
'''
    with open(os.path.join(OUTPUT_DIR, "model.py"), "w") as f:
        f.write(model_py)

    # Create zip
    print("Packaging zip...")
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in ["model_config.yaml", "model.py", "model.safetensors", "config.json"]:
            fpath = os.path.join(OUTPUT_DIR, fname)
            if os.path.exists(fpath):
                zf.write(fpath, fname)

    size_mb = os.path.getsize(ZIP_PATH) / 1024 / 1024
    print(f"\nVideo detector packaged: {ZIP_PATH} ({size_mb:.1f} MB)")
    print(f"Push with: gascli d push --video-model {ZIP_PATH} --wallet-name miner --wallet-hotkey default")


if __name__ == "__main__":
    build()
