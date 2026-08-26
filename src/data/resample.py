from __future__ import annotations






























from typing import Tuple

import torch
import torch.nn.functional as F

from data.shapes import ensure_tensor, assert_bchw, assert_same_spatial


def is_square_bchw(x: torch.Tensor, name: str = "x") -> None:











    assert_bchw(x, name=name)
    h, w = x.shape[-2], x.shape[-1]
    if h != w:
        raise ValueError(
            f"{name} must have square spatial shape [B, C, H, W] with H==W, "
            f"but got shape {tuple(x.shape)}"
        )


def get_current_res(x: torch.Tensor, name: str = "x") -> int:










    is_square_bchw(x, name=name)
    return int(x.shape[-1])


def _validate_target_res(target_res: int) -> int:








    if not isinstance(target_res, int):
        raise TypeError(f"target_res must be int, but got {type(target_res)}")
    if target_res <= 0:
        raise ValueError(f"target_res must be positive, but got {target_res}")
    return target_res


def resample_bchw(
    x: torch.Tensor,
    target_res: int,
    *,
    down_mode: str = "area",
    up_mode: str = "bilinear",
    align_corners: bool = False,
    name: str = "x",
) -> torch.Tensor:







































    x = ensure_tensor(x, name=name)
    current_res = get_current_res(x, name=name)
    target_res = _validate_target_res(target_res)

    if current_res == target_res:
        
        return x

    size = (target_res, target_res)

    if target_res < current_res:
        
        return F.interpolate(x, size=size, mode=down_mode)

    
    if up_mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        return F.interpolate(x, size=size, mode=up_mode, align_corners=align_corners)

    
    return F.interpolate(x, size=size, mode=up_mode)


def resample_pair(
    a: torch.Tensor,
    u: torch.Tensor,
    target_res: int,
    *,
    down_mode: str = "area",
    up_mode: str = "bilinear",
    align_corners: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:





































    a = ensure_tensor(a, name="a")
    u = ensure_tensor(u, name="u")

    assert_same_spatial(a, u, names=["a", "u"])

    a_res = resample_bchw(
        a,
        target_res=target_res,
        down_mode=down_mode,
        up_mode=up_mode,
        align_corners=align_corners,
        name="a",
    )
    u_res = resample_bchw(
        u,
        target_res=target_res,
        down_mode=down_mode,
        up_mode=up_mode,
        align_corners=align_corners,
        name="u",
    )

    assert_same_spatial(a_res, u_res, names=["a_res", "u_res"])
    return a_res, u_res
