from __future__ import annotations

"""Task-specific model wrappers for the full hyperelasticity study.

The wrappers adapt channel counts, rectangular-grid padding/cropping, and the
common physical left-clamp projection required by the final protocol.
"""

from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet import UNet2D
from models.resnet import ResNetED2D
from models.hyperelasticity_corrected_fno import CorrectedHyperelasticFNO
from data.hyperelasticity import NormalizerStats


ARCHITECTURES = ("corrected_fno", "unet", "resnet")


@dataclass(frozen=True)
class ModelDescription:
    architecture: str
    input_channels: int
    output_channels: int
    padding_multiple: int | None
    padding_mode: str | None
    real_parameter_count: int
    details: Dict[str, Any]


class _NormalizedClampMixin:
    """Apply physical u=(0,0) at x=0 in standardized output coordinates."""

    def _register_normalized_zero(self, normalizer: NormalizerStats) -> None:
        means = torch.tensor(normalizer.displacement_mean, dtype=torch.float32)
        stds = torch.tensor(normalizer.displacement_std, dtype=torch.float32)
        self.register_buffer(
            "normalized_physical_zero",
            (-means / stds).view(1, 2, 1, 1),
        )

    def _project_normalized_left_clamp(self, output: torch.Tensor) -> torch.Tensor:
        if output.ndim != 4 or output.shape[1] != 2:
            raise ValueError(f"Expected normalized output [B,2,H,W], got {tuple(output.shape)}")
        # Clone avoids an in-place write on an activation needed by autograd.
        projected = output.clone()
        projected[..., 0] = self.normalized_physical_zero.to(output.dtype)[..., 0]
        return projected


class HyperelasticUNet(_NormalizedClampMixin, nn.Module):
    """U-Net with high-index replicate padding and exact cropping."""

    padding_multiple = 16

    def __init__(
        self,
        normalizer: NormalizerStats,
        *,
        base_channels: int = 32,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.base_channels = int(base_channels)
        self.depth = int(depth)
        self.network = UNet2D(
            in_ch=1,
            out_ch=2,
            base_ch=self.base_channels,
            depth=self.depth,
        )
        self._register_normalized_zero(normalizer)

    def forward(self, normalized_traction: torch.Tensor) -> torch.Tensor:
        if normalized_traction.ndim != 4 or normalized_traction.shape[1] != 1:
            raise ValueError(f"Expected traction [B,1,H,W], got {tuple(normalized_traction.shape)}")
        ny, nx = normalized_traction.shape[-2:]
        pad_y = (-ny) % self.padding_multiple
        pad_x = (-nx) % self.padding_multiple
        padded = F.pad(
            normalized_traction,
            (0, pad_x, 0, pad_y),
            mode="replicate",
        )
        output = self.network(padded)[..., :ny, :nx]
        if output.shape[-2:] != (ny, nx):
            raise RuntimeError(f"U-Net crop failed: {tuple(output.shape)} vs {(ny, nx)}")
        return self._project_normalized_left_clamp(output)


class HyperelasticResNet(_NormalizedClampMixin, nn.Module):
    """Residual encoder-decoder with multiple-of-four padding and cropping."""

    padding_multiple = 4

    def __init__(
        self,
        normalizer: NormalizerStats,
        *,
        base_channels: int = 32,
        levels: int = 2,
        blocks_per_level: int = 2,
        bottleneck_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.base_channels = int(base_channels)
        self.levels = int(levels)
        self.blocks_per_level = int(blocks_per_level)
        self.bottleneck_blocks = int(bottleneck_blocks)
        self.network = ResNetED2D(
            in_ch=1,
            out_ch=2,
            base_ch=self.base_channels,
            levels=self.levels,
            blocks_per_level=self.blocks_per_level,
            bottleneck_blocks=self.bottleneck_blocks,
        )
        self._register_normalized_zero(normalizer)

    def forward(self, normalized_traction: torch.Tensor) -> torch.Tensor:
        if normalized_traction.ndim != 4 or normalized_traction.shape[1] != 1:
            raise ValueError(f"Expected traction [B,1,H,W], got {tuple(normalized_traction.shape)}")
        ny, nx = normalized_traction.shape[-2:]
        pad_y = (-ny) % self.padding_multiple
        pad_x = (-nx) % self.padding_multiple
        padded = F.pad(
            normalized_traction,
            (0, pad_x, 0, pad_y),
            mode="replicate",
        )
        output = self.network(padded)[..., :ny, :nx]
        if output.shape[-2:] != (ny, nx):
            raise RuntimeError(f"ResNet crop failed: {tuple(output.shape)} vs {(ny, nx)}")
        return self._project_normalized_left_clamp(output)


def build_model(
    architecture: str,
    normalizer: NormalizerStats,
) -> nn.Module:
    """Build exactly one approved task-specific architecture."""

    architecture = architecture.strip().lower()
    if architecture == "corrected_fno":
        return CorrectedHyperelasticFNO(
            normalizer,
            width=64,
            depth=4,
            modes=8,
        )
    if architecture == "unet":
        return HyperelasticUNet(normalizer, base_channels=32, depth=4)
    if architecture == "resnet":
        return HyperelasticResNet(
            normalizer,
            base_channels=32,
            levels=2,
            blocks_per_level=2,
            bottleneck_blocks=4,
        )
    raise ValueError(f"Unknown architecture {architecture!r}; expected {ARCHITECTURES}")


def describe_model(architecture: str, model: nn.Module) -> ModelDescription:
    """Return an immutable config description and real parameter count."""

    architecture = architecture.strip().lower()
    count = int(sum(parameter.numel() for parameter in model.parameters()))
    if architecture == "corrected_fno":
        details = {
            "spectral_layer": "standard two-block positive/negative first-axis",
            "width": 64,
            "depth": 4,
            "modes1": 8,
            "modes2": 8,
            "coordinate_channels": "physical x in [0,4], y in [0,1]",
            "spatial_padding": False,
        }
        return ModelDescription(architecture, 3, 2, None, None, count, details)
    if architecture == "unet":
        details = {"base_channels": 32, "depth": 4, "crop": "exact physical Ny,Nx"}
        return ModelDescription(architecture, 1, 2, 16, "high-index replicate", count, details)
    if architecture == "resnet":
        details = {
            "base_channels": 32,
            "levels": 2,
            "blocks_per_level": 2,
            "bottleneck_blocks": 4,
            "crop": "exact physical Ny,Nx",
        }
        return ModelDescription(architecture, 1, 2, 4, "high-index replicate", count, details)
    raise ValueError(architecture)
