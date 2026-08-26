from __future__ import annotations

import torch

from models.darcy_corrected_fno import StandardSpectralConv2d as DarcySpectral
from models.hyperelasticity_corrected_fno import StandardSpectralConv2d as HyperSpectral


def verify_two_block_layer(layer_type) -> None:
    layer = layer_type(2, 3, 4, 4)
    for name in ("weight_pos_real", "weight_pos_imag", "weight_neg_real", "weight_neg_imag"):
        if not hasattr(layer, name): raise AssertionError(f"missing independent complex parameter: {name}")
    x = torch.randn(2, 2, 16, 16, requires_grad=True)
    out = layer(x)
    out.square().mean().backward()
    if layer.weight_pos_real.grad is None or layer.weight_neg_real.grad is None: raise AssertionError("both retained blocks must receive gradients")


def run() -> None:
    verify_two_block_layer(DarcySpectral)
    verify_two_block_layer(HyperSpectral)
