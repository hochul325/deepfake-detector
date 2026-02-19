import os
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
        self.swap_labels = False
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
