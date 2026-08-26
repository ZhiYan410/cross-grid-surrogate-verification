from __future__ import annotations


























from typing import Sequence
import torch


def ensure_tensor(x: torch.Tensor, name: str = "x") -> torch.Tensor:





















    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, but got {type(x)}")
    return x


def assert_bchw(x: torch.Tensor, name: str = "x") -> None:















    x = ensure_tensor(x, name=name)
    if x.ndim != 4:
        raise ValueError(
            f"{name} must have shape [B, C, H, W] (4 dims), "
            f"but got shape {tuple(x.shape)} with ndim={x.ndim}"
        )


def assert_b1hw(x: torch.Tensor, name: str = "x") -> None:












    assert_bchw(x, name=name)
    if x.shape[1] != 1:
        raise ValueError(
            f"{name} must have shape [B, 1, H, W], "
            f"but got channel dimension C={x.shape[1]} in shape {tuple(x.shape)}"
        )


def to_b1hw(x: torch.Tensor, name: str = "x") -> torch.Tensor:






















    x = ensure_tensor(x, name=name)

    if x.ndim == 4:
        
        if x.shape[1] != 1:
            raise ValueError(
                f"{name} has 4 dims and is expected to be [B,1,H,W], "
                f"but got shape {tuple(x.shape)} with channel C={x.shape[1]}"
            )
        return x

    if x.ndim == 3:
        
        return x.unsqueeze(1)

    if x.ndim == 2:
        
        return x.unsqueeze(0).unsqueeze(0)

    raise ValueError(
        f"{name} cannot be converted to [B,1,H,W]. "
        f"Supported input shapes are [H,W], [B,H,W], [B,1,H,W], "
        f"but got shape {tuple(x.shape)} with ndim={x.ndim}"
    )


def get_spatial_shape(x: torch.Tensor, name: str = "x") -> tuple[int, int]:










    x = ensure_tensor(x, name=name)
    if x.ndim < 2:
        raise ValueError(
            f"{name} must have at least 2 dims to carry spatial shape, "
            f"but got shape {tuple(x.shape)}"
        )
    return int(x.shape[-2]), int(x.shape[-1])


def assert_same_spatial(*xs: torch.Tensor, names: Sequence[str] | None = None) -> None:


























    if len(xs) == 0:
        raise ValueError("assert_same_spatial received no tensors")

    if names is not None and len(names) != len(xs):
        raise ValueError(
            f"Length mismatch: got {len(xs)} tensors but {len(names)} names"
        )

    shapes = [get_spatial_shape(x, name=f"x{i}") for i, x in enumerate(xs)]
    ref_shape = shapes[0]

    for i, shp in enumerate(shapes[1:], start=1):
        if shp != ref_shape:
            if names is None:
                raise ValueError(
                    f"Spatial shapes are inconsistent: reference shape {ref_shape}, "
                    f"but tensor #{i} has shape {shp}"
                )
            else:
                raise ValueError(
                    f"Spatial shapes are inconsistent: "
                    f"{names[0]} has shape {ref_shape}, but {names[i]} has shape {shp}"
                )


def describe_shape(x: torch.Tensor, name: str = "x") -> str:














    x = ensure_tensor(x, name=name)
    return (
        f"{name}: shape={tuple(x.shape)}, "
        f"dtype={x.dtype}, device={x.device}"
    )
