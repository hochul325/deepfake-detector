import torch
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
