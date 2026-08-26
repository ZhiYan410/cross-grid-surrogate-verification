from __future__ import annotations

import torch.nn as nn

from models.darcy_corrected_fno import CorrectedFNO2D
from models.resnet import ResNetED2D
from models.unet import UNet2D


ARCHITECTURES = ("corrected_fno", "unet", "resnet")


def build_model(architecture: str) -> nn.Module:
    """Build one audited Darcy architecture using the public model routes."""
    key = architecture.strip().lower()
    if key == "corrected_fno":
        return CorrectedFNO2D(in_channels=3, width=64, modes=12, depth=4)
    if key == "unet":
        return UNet2D(in_ch=1, out_ch=1, base_ch=32, depth=4)
    if key == "resnet":
        return ResNetED2D(in_ch=1, out_ch=1, base_ch=32, levels=2, blocks_per_level=2, bottleneck_blocks=4)
    raise ValueError(f"Unknown Darcy architecture {architecture!r}; expected {ARCHITECTURES}")
