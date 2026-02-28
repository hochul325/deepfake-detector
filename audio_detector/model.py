from pathlib import Path
import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Config


class AudioDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        config_path = Path(__file__).parent / "config.json"
        if config_path.exists():
            config = Wav2Vec2Config.from_pretrained(str(Path(__file__).parent))
            config.num_labels = num_classes
            self.model = Wav2Vec2ForSequenceClassification(config)
        else:
            config = Wav2Vec2Config(
                hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                intermediate_size=3072, num_labels=num_classes,
                num_feat_extract_layers=7,
                conv_dim=[512, 512, 512, 512, 512, 512, 512],
                conv_stride=[5, 2, 2, 2, 2, 2, 2],
                conv_kernel=[10, 3, 3, 3, 3, 2, 2],
                num_conv_pos_embeddings=128, num_conv_pos_embedding_groups=16,
            )
            self.model = Wav2Vec2ForSequenceClassification(config)
        self.register_buffer("temperature", torch.ones(1))

    def forward(self, x):
        outputs = self.model(input_values=x)
        return outputs.logits / self.temperature


def load_model(weights_path, num_classes=2):
    model = AudioDeepfakeDetector(num_classes=num_classes)
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
