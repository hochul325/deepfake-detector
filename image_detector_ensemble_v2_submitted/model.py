import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from safetensors.torch import load_file


class SingleModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.encoder = timm.create_model('vit_base_patch16_clip_224.openai', pretrained=False, num_classes=0)
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
        x = x / 255.0
        x = (x - self.mean) / self.std
        features = self.encoder(x)
        return self.classifier(features) / self.temperature


class ImageDeepfakeDetector(nn.Module):
    """Ensemble of three models - weighted probability averaging."""
    def __init__(self, num_classes=2):
        super().__init__()
        self.model_a = SingleModel(num_classes)  # v3
        self.model_b = SingleModel(num_classes)  # v3b
        self.model_c = SingleModel(num_classes)  # v4

    def forward(self, x):
        logits_a = self.model_a(x)
        logits_b = self.model_b(x)
        logits_c = self.model_c(x)
        # Average probabilities with weights: v3=0.2, v3b=0.2, v4=0.6
        probs_a = F.softmax(logits_a, dim=1)
        probs_b = F.softmax(logits_b, dim=1)
        probs_c = F.softmax(logits_c, dim=1)
        avg_probs = 0.2 * probs_a + 0.2 * probs_b + 0.6 * probs_c
        # Convert back to logits for gasbench (which applies softmax)
        avg_logits = torch.log(avg_probs.clamp(min=1e-7))
        return avg_logits


def load_model(weights_path, num_classes=2):
    model_dir = os.path.dirname(weights_path)
    weights_b = os.path.join(model_dir, 'model_b.safetensors')
    weights_c = os.path.join(model_dir, 'model_c.safetensors')

    if os.path.exists(weights_b) and os.path.exists(weights_c):
        model = ImageDeepfakeDetector(num_classes=num_classes)
        state_a = load_file(weights_path)
        state_b = load_file(weights_b)
        state_c = load_file(weights_c)
        model.model_a.load_state_dict(state_a)
        model.model_b.load_state_dict(state_b)
        model.model_c.load_state_dict(state_c)
        model.train(False)
        return model
    else:
        model = SingleModel(num_classes=num_classes)
        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict)
        model.train(False)
        return model
