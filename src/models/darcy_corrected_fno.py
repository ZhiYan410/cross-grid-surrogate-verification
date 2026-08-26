from __future__ import annotations

"""Corrected standard two-block FNO for DarcyFlow.

The layer follows the already verified two-block implementation: rfft2 keeps
the last-axis nonnegative frequencies, while the first axis receives two
independent blocks with independent real/imaginary parameters.
"""

from typing import Tuple

import torch
import torch.nn as nn


class StandardSpectralConv2d(nn.Module):
    """Apply independent learned weights to positive/negative first-axis blocks."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        if min(self.in_channels, self.out_channels, self.modes1, self.modes2) <= 0:
            raise ValueError("channels and modes must be positive")
        scale = 1.0 / float(self.in_channels * self.out_channels)
        shape = (self.in_channels, self.out_channels, self.modes1, self.modes2)
        self.weight_pos_real = nn.Parameter(scale * torch.randn(*shape))
        self.weight_pos_imag = nn.Parameter(scale * torch.randn(*shape))
        self.weight_neg_real = nn.Parameter(scale * torch.randn(*shape))
        self.weight_neg_imag = nn.Parameter(scale * torch.randn(*shape))

    @staticmethod
    def complex_multiply(x: torch.Tensor, wr: torch.Tensor, wi: torch.Tensor) -> torch.Tensor:
        weight = torch.complex(wr.to(dtype=x.real.dtype), wi.to(dtype=x.real.dtype))
        return torch.einsum("bixy,ioxy->boxy", x, weight)

    def retained_modes(self, height: int, width: int) -> Tuple[int, int]:
        m1 = min(self.modes1, height // 2)
        m2 = min(self.modes2, width // 2 + 1)
        if m1 <= 0 or m2 <= 0:
            raise ValueError(f"grid too small for spectral layer: {height}x{width}")
        return m1, m2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"expected [B,{self.in_channels},H,W], got {tuple(x.shape)}")
        b, _, h, w = x.shape
        m1, m2 = self.retained_modes(h, w)
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(
            b, self.out_channels, h, w // 2 + 1, dtype=x_ft.dtype, device=x.device
        )
        out_ft[:, :, :m1, :m2] = self.complex_multiply(
            x_ft[:, :, :m1, :m2],
            self.weight_pos_real[:, :, :m1, :m2],
            self.weight_pos_imag[:, :, :m1, :m2],
        )
        out_ft[:, :, -m1:, :m2] = self.complex_multiply(
            x_ft[:, :, -m1:, :m2],
            self.weight_neg_real[:, :, :m1, :m2],
            self.weight_neg_imag[:, :, :m1, :m2],
        )
        return torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")


class CorrectedFNO2D(nn.Module):
    """Original Darcy FNO topology with only the spectral layer corrected."""

    def __init__(self, in_channels: int = 3, width: int = 64, modes: int = 12, depth: int = 4) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.width = int(width)
        self.modes = int(modes)
        self.depth = int(depth)
        self.lift = nn.Conv2d(self.in_channels, self.width, kernel_size=1)
        self.spec_convs = nn.ModuleList(
            [StandardSpectralConv2d(self.width, self.width, self.modes, self.modes) for _ in range(self.depth)]
        )
        self.point_convs = nn.ModuleList(
            [nn.Conv2d(self.width, self.width, kernel_size=1) for _ in range(self.depth)]
        )
        self.act = nn.GELU()
        self.head = nn.Sequential(
            nn.Conv2d(self.width, self.width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.width, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        for spectral, local in zip(self.spec_convs, self.point_convs):
            x = self.act(spectral(x) + local(x))
        return self.head(x)


def build_fno_input(a: torch.Tensor) -> torch.Tensor:
    """Build the unchanged Darcy input [coefficient, x-coordinate, y-coordinate]."""
    if a.ndim != 4 or a.shape[1] != 1 or a.shape[-2] != a.shape[-1]:
        raise ValueError(f"expected coefficient [B,1,H,W], got {tuple(a.shape)}")
    b, _, h, w = a.shape
    xs = torch.linspace(0.0, 1.0, w, dtype=a.dtype, device=a.device).view(1, 1, 1, w)
    ys = torch.linspace(0.0, 1.0, h, dtype=a.dtype, device=a.device).view(1, 1, h, 1)
    return torch.cat([a, xs.expand(b, 1, h, w), ys.expand(b, 1, h, w)], dim=1)


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
