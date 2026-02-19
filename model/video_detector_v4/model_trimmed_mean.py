import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from safetensors.torch import load_file


class VideoDeepfakeDetector(nn.Module):
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

        # Trimmed mean: remove frames most distant from mean
        if T > 4:
            mean_feat = features.mean(dim=1, keepdim=True)
            dists = ((features - mean_feat) ** 2).sum(dim=-1)  # [B, T]
            k = max(1, T // 4)  # remove 25% most distant frames
            _, top_idx = dists.topk(k, dim=1, largest=True)
            mask = torch.ones(B, T, device=features.device, dtype=torch.bool)
            mask.scatter_(1, top_idx, False)
            # masked mean
            mask_f = mask.unsqueeze(-1).float()
            pooled = (features * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        else:
            pooled = features.mean(dim=1)

        return self.classifier(pooled) / self.temperature


def load_model(weights_path, num_classes=2):
    model = VideoDeepfakeDetector(num_classes=num_classes)
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    model.train(False)
    return model
