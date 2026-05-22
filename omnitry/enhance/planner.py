from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .taxonomy import class_index_map, normalize_class_name


class AffordancePlanner(nn.Module):
    def __init__(self, num_classes: int, class_dim: int = 32):
        super().__init__()
        self.class_embedding = nn.Embedding(num_classes, class_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(3 + class_dim, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.SiLU(),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 96, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.SiLU(),
            nn.ConvTranspose2d(96, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, image: torch.Tensor, class_ids: torch.Tensor) -> torch.Tensor:
        emb = self.class_embedding(class_ids)
        emb = emb[:, :, None, None].expand(-1, -1, image.shape[-2], image.shape[-1])
        x = torch.cat([image, emb], dim=1)
        logits = self.decoder(self.encoder(x))
        if logits.shape[-2:] != image.shape[-2:]:
            logits = F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return logits


def default_class_to_idx() -> Dict[str, int]:
    return class_index_map()


def class_to_id(name: str, mapping: Dict[str, int]) -> int:
    normalized = normalize_class_name(name)
    if normalized not in mapping:
        return 0
    return mapping[normalized]


def save_planner(path, model: AffordancePlanner, class_to_idx: Dict[str, int], meta: Dict):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_to_idx": class_to_idx,
            "meta": meta,
        },
        path,
    )


def load_planner(path, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location)
    class_to_idx = checkpoint["class_to_idx"]
    model = AffordancePlanner(num_classes=len(class_to_idx))
    model.load_state_dict(checkpoint["state_dict"])
    return model, class_to_idx, checkpoint.get("meta", {})
