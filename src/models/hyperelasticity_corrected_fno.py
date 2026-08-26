from __future__ import annotations

"""Standard two-block 2D FNO for the final audited hyperelasticity protocol.

The implementation uses the audited topology and ``norm="ortho"`` FFT
convention, with independent learned positive- and negative-first-axis
spectral blocks.
"""

from typing import Tuple

import torch
import torch.nn as nn

from data.hyperelasticity import (
    DOMAIN_LENGTH,
    DOMAIN_WIDTH,
    NormalizerStats,
)


class StandardSpectralConv2d(nn.Module):
    """Two-block spectral convolution for BCHW tensors.

    ``rfft2`` compresses only the last transformed axis.  The first transformed
    axis still contains positive and negative frequencies, so independent
    learned weights are applied to ``0:modes1`` and ``-modes1:``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        if min(self.in_channels, self.out_channels, self.modes1, self.modes2) <= 0:
            raise ValueError("Channels and retained mode counts must be positive")

        scale = 1.0 / float(self.in_channels * self.out_channels)
        shape = (
            self.in_channels,
            self.out_channels,
            self.modes1,
            self.modes2,
        )
        # Real tensors keep parameter counts interpretable as real degrees of
        # freedom while representing two independent complex weight blocks.
        self.weight_pos_real = nn.Parameter(scale * torch.randn(*shape))
        self.weight_pos_imag = nn.Parameter(scale * torch.randn(*shape))
        self.weight_neg_real = nn.Parameter(scale * torch.randn(*shape))
        self.weight_neg_imag = nn.Parameter(scale * torch.randn(*shape))

    @staticmethod
    def complex_multiply(
        input_ft: torch.Tensor,
        weight_real: torch.Tensor,
        weight_imag: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the complex channel mixing used by the reference FNO layer."""

        weight = torch.complex(
            weight_real.to(dtype=input_ft.real.dtype),
            weight_imag.to(dtype=input_ft.real.dtype),
        )
        return torch.einsum("bixy,ioxy->boxy", input_ft, weight)

    def retained_modes(self, height: int, width: int) -> Tuple[int, int]:
        """Return non-overlapping runtime mode counts for a rectangular grid."""

        # At most half of the first FFT axis is assigned to each block.  For
        # Ny=16 and modes1=8, the two slices exactly partition that axis.
        m1 = min(self.modes1, height // 2)
        m2 = min(self.modes2, width // 2 + 1)
        if m1 <= 0 or m2 <= 0:
            raise ValueError(f"Grid is too small for spectral convolution: {height}x{width}")
        return m1, m2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected [B,{self.in_channels},H,W], got {tuple(x.shape)}"
            )
        batch, _, height, width = x.shape
        m1, m2 = self.retained_modes(height, width)
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(
            batch,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=x_ft.dtype,
            device=x.device,
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
        return torch.fft.irfft2(
            out_ft,
            s=(height, width),
            norm="ortho",
        )


class StandardFNO2D(nn.Module):
    """Audited FNO topology with a standard two-block spectral layer."""

    def __init__(
        self,
        in_channels: int = 3,
        width: int = 64,
        modes: int = 8,
        depth: int = 4,
        out_channels: int = 2,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.width = int(width)
        self.modes = int(modes)
        self.depth = int(depth)
        self.out_channels = int(out_channels)

        self.lift = nn.Conv2d(self.in_channels, self.width, kernel_size=1)
        self.spec_convs = nn.ModuleList(
            [
                StandardSpectralConv2d(
                    self.width,
                    self.width,
                    self.modes,
                    self.modes,
                )
                for _ in range(self.depth)
            ]
        )
        self.point_convs = nn.ModuleList(
            [
                nn.Conv2d(self.width, self.width, kernel_size=1)
                for _ in range(self.depth)
            ]
        )
        self.act = nn.GELU()
        self.head = nn.Sequential(
            nn.Conv2d(self.width, self.width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.width, self.out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        for spectral, local in zip(self.spec_convs, self.point_convs):
            x = self.act(spectral(x) + local(x))
        return self.head(x)


class CorrectedHyperelasticFNO(nn.Module):
    """Task wrapper preserving coordinates, normalization, and left clamp."""

    def __init__(
        self,
        normalizer: NormalizerStats,
        *,
        width: int = 64,
        depth: int = 4,
        modes: int = 8,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.depth = int(depth)
        self.modes = int(modes)
        self.fno = StandardFNO2D(
            in_channels=3,
            width=self.width,
            modes=self.modes,
            depth=self.depth,
            out_channels=2,
        )

        means = torch.tensor(normalizer.displacement_mean, dtype=torch.float32)
        stds = torch.tensor(normalizer.displacement_std, dtype=torch.float32)
        normalized_physical_zero = -means / stds
        self.register_buffer(
            "normalized_physical_zero",
            normalized_physical_zero.view(1, 2, 1, 1),
        )

    @staticmethod
    def coordinate_channels(reference: torch.Tensor) -> torch.Tensor:
        batch, _, ny, nx = reference.shape
        x = torch.linspace(
            0.0,
            DOMAIN_LENGTH,
            nx,
            dtype=reference.dtype,
            device=reference.device,
        ).view(1, 1, 1, nx)
        y = torch.linspace(
            0.0,
            DOMAIN_WIDTH,
            ny,
            dtype=reference.dtype,
            device=reference.device,
        ).view(1, 1, ny, 1)
        return torch.cat(
            [
                x.expand(batch, 1, ny, nx),
                y.expand(batch, 1, ny, nx),
            ],
            dim=1,
        )

    def forward(self, normalized_traction: torch.Tensor) -> torch.Tensor:
        if normalized_traction.ndim != 4 or normalized_traction.shape[1] != 1:
            raise ValueError(
                "Expected normalized traction [B,1,Ny,Nx], got "
                f"{tuple(normalized_traction.shape)}"
            )
        coordinates = self.coordinate_channels(normalized_traction)
        output = self.fno(torch.cat([normalized_traction, coordinates], dim=1))

        # The physical clamp u=(0,0) is represented in standardized output
        # coordinates, exactly as in the final audited hyperelasticity protocol.
        mask = torch.ones(
            (1, 1, 1, output.shape[-1]),
            dtype=output.dtype,
            device=output.device,
        )
        mask[..., 0] = 0.0
        return output * mask + self.normalized_physical_zero.to(output.dtype) * (
            1.0 - mask
        )


def real_parameter_count(module: nn.Module) -> int:
    """Count trainable real scalar degrees of freedom."""

    return int(sum(parameter.numel() for parameter in module.parameters()))
