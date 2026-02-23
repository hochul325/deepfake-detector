import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Config


class SingleAudioModel(nn.Module):
    def __init__(self, config_dir=None, num_classes=2):
        super().__init__()
        if config_dir:
            config_path = os.path.join(config_dir, "config.json")
        else:
            config_path = None

        if config_path and os.path.exists(config_path):
            config = Wav2Vec2Config.from_pretrained(config_dir)
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


class AudioDeepfakeDetector(nn.Module):
    """Weighted ensemble: model_a (v5) + model_b (v6) + model_c (v8)."""

    def __init__(self, num_classes=2):
        super().__init__()
        model_dir = os.path.dirname(__file__)
        self.model_a = SingleAudioModel(config_dir=model_dir, num_classes=num_classes)
        self.model_b = SingleAudioModel(config_dir=model_dir, num_classes=num_classes)
        self.model_c = SingleAudioModel(config_dir=model_dir, num_classes=num_classes)
        self.weight_a = 0.30  # v5
        self.weight_b = 0.30  # v6
        self.weight_c = 0.40  # v8

    def forward(self, x):
        logits_a = self.model_a(x)
        logits_b = self.model_b(x)
        logits_c = self.model_c(x)
        probs_a = F.softmax(logits_a, dim=1)
        probs_b = F.softmax(logits_b, dim=1)
        probs_c = F.softmax(logits_c, dim=1)
        avg_probs = self.weight_a * probs_a + self.weight_b * probs_b + self.weight_c * probs_c
        avg_logits = torch.log(avg_probs.clamp(min=1e-7))
        return avg_logits


def load_model(weights_path, num_classes=2):
    model_dir = os.path.dirname(weights_path)
    weights_b = os.path.join(model_dir, "model_b.safetensors")
    weights_c = os.path.join(model_dir, "model_c.safetensors")

    def _load_single(single_model, state_dict):
        if "temperature" in state_dict:
            temp_val = state_dict.pop("temperature")
            single_model.temperature.copy_(temp_val)
        inner_state = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                inner_state[k[6:]] = v
            else:
                inner_state[k] = v
        single_model.model.load_state_dict(inner_state)

    if os.path.exists(weights_b) and os.path.exists(weights_c):
        model = AudioDeepfakeDetector(num_classes=num_classes)
        _load_single(model.model_a, load_file(weights_path))
        _load_single(model.model_b, load_file(weights_b))
        _load_single(model.model_c, load_file(weights_c))
        model.train(False)
        return model
    elif os.path.exists(weights_b):
        model = AudioDeepfakeDetector(num_classes=num_classes)
        _load_single(model.model_a, load_file(weights_path))
        _load_single(model.model_b, load_file(weights_b))
        model.weight_a = 0.50
        model.weight_b = 0.50
        model.weight_c = 0.00
        model.train(False)
        return model
    else:
        model = SingleAudioModel(config_dir=model_dir, num_classes=num_classes)
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
