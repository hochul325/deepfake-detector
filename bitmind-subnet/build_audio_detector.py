"""
Build an audio deepfake detector for BitMind Subnet 34.
Downloads a pre-trained Wav2Vec2 model fine-tuned for deepfake detection,
wraps it in GASBench-compatible format, and packages as a zip.
"""

import torch
import os
import shutil
import zipfile
import yaml
from safetensors.torch import save_file
from transformers import Wav2Vec2ForSequenceClassification


OUTPUT_DIR = "/root/audio_detector"
ZIP_PATH = "/root/audio_detector.zip"


def build():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Downloading Wav2Vec2 deepfake detection model...")
    try:
        hf_model = Wav2Vec2ForSequenceClassification.from_pretrained(
            "MelodyMachine/Deepfake-audio-detection-V2",
            num_labels=2,
            ignore_mismatched_sizes=True,
        )
        print("  Loaded MelodyMachine/Deepfake-audio-detection-V2")
    except Exception as e:
        print(f"  Failed: {e}")
        print("  Trying Gustking/wav2vec2-large-xlsr-deepfake-audio-classification...")
        hf_model = Wav2Vec2ForSequenceClassification.from_pretrained(
            "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification",
            num_labels=2,
            ignore_mismatched_sizes=True,
        )
        print("  Loaded Gustking model")

    # Check label mapping - we need [real, synthetic] order
    if hasattr(hf_model.config, 'id2label'):
        print(f"  id2label: {hf_model.config.id2label}")

    # Save weights with wrapper prefix
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
        "name": "wav2vec2-deepfake-audio-detector",
        "version": "1.0.0",
        "modality": "audio",
        "preprocessing": {
            "sample_rate": 16000,
            "duration_seconds": 6.0,
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
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Config


class AudioDeepfakeDetector(nn.Module):
    """Wav2Vec2-based audio deepfake detector for GASBench SN34."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        config = Wav2Vec2Config(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            num_labels=num_classes,
            # wav2vec2-base defaults
            num_feat_extract_layers=7,
            conv_dim=[512, 512, 512, 512, 512, 512, 512],
            conv_stride=[5, 2, 2, 2, 2, 2, 2],
            conv_kernel=[10, 3, 3, 3, 3, 2, 2],
            num_conv_pos_embeddings=128,
            num_conv_pos_embedding_groups=16,
        )
        self.model = Wav2Vec2ForSequenceClassification(config)
        # label order: index 0 = fake, index 1 = real (from MelodyMachine model)
        # GASBench expects: index 0 = real, index 1 = synthetic
        self.swap_labels = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 96000] float32 in [-1, 1] (16kHz, 6 seconds)
        Returns:
            logits: [B, num_classes] raw logits [real, synthetic]
        """
        outputs = self.model(input_values=x)
        logits = outputs.logits  # [B, 2]

        if self.swap_labels:
            # Swap from [fake, real] to [real, synthetic]
            logits = logits[:, [1, 0]]

        return logits


def load_model(weights_path: str, num_classes: int = 2) -> nn.Module:
    """Required entry point for GASBench."""
    model = AudioDeepfakeDetector(num_classes=num_classes)
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
    print(f"\nAudio detector packaged: {ZIP_PATH} ({size_mb:.1f} MB)")
    print(f"Push with: gascli d push --audio-model {ZIP_PATH} --wallet-name miner --wallet-hotkey default")


if __name__ == "__main__":
    build()
