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
        if x.dim() == 4:
            x = x.unsqueeze(1)
        B, T = x.shape[:2]
        frames = x.reshape(B * T, *x.shape[2:])
        frames = frames / 255.0
        frames = (frames - self.mean) / self.std
        features = self.encoder(frames)
        features = features.view(B, T, -1)
        pooled = features.mean(dim=1)
        return self.classifier(pooled) / self.temperature


class VideoDeepfakeDetector(nn.Module):
    """Ensemble of two models - averages probabilities then converts back to logits."""
    def __init__(self, num_classes=2):
        super().__init__()
        self.model_a = SingleModel(num_classes)
        self.model_b = SingleModel(num_classes)

    def forward(self, x):
        logits_a = self.model_a(x)
        logits_b = self.model_b(x)
        # Average probabilities (not logits) for better calibration
        probs_a = F.softmax(logits_a, dim=1)
        probs_b = F.softmax(logits_b, dim=1)
        avg_probs = (probs_a + probs_b) / 2.0
        # Convert back to logits for gasbench (which applies softmax)
        avg_logits = torch.log(avg_probs.clamp(min=1e-7))
        return avg_logits


def load_model(weights_path, num_classes=2):
    model_dir = os.path.dirname(weights_path)
    weights_b = os.path.join(model_dir, 'model_b.safetensors')

    if os.path.exists(weights_b):
        model = VideoDeepfakeDetector(num_classes=num_classes)
        state_a = load_file(weights_path)
        state_b = load_file(weights_b)
        model.model_a.load_state_dict(state_a)
        model.model_b.load_state_dict(state_b)
        model.train(False)
        return model
    else:
        model = SingleModel(num_classes=num_classes)
        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict)
        model.train(False)
        return model
