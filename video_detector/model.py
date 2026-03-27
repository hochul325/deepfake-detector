import torch
import torch.nn as nn
import timm
from safetensors.torch import load_file

class VideoDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.encoder = timm.create_model("vit_base_patch16_clip_224.openai", pretrained=False, num_classes=0)
        feature_dim = self.encoder.num_features
        self.temporal_attention = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Dropout(0.3),
            nn.Linear(feature_dim, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        self.register_buffer("temperature", torch.ones(1))

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)
        B, T = x.shape[:2]
        frames = x.reshape(B * T, *x.shape[2:])
        frames = frames.float() / 255.0
        frames = (frames - self.mean) / self.std
        features = self.encoder(frames)
        features = features.view(B, T, -1)
        # Attention-weighted temporal pooling
        attn = torch.softmax(self.temporal_attention(features), dim=1)
        pooled = (features * attn).sum(dim=1)
        return self.classifier(pooled) / self.temperature

def load_model(weights_path, num_classes=2):
    model = VideoDeepfakeDetector(num_classes=num_classes)
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    model.train(False)
    return model
