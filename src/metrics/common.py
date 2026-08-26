from __future__ import annotations



























from typing import Any, Dict

import numpy as np
import torch

from data.shapes import ensure_tensor, assert_bchw


def check_same_shape(pred: torch.Tensor, target: torch.Tensor) -> None:












    pred = ensure_tensor(pred, name="pred")
    target = ensure_tensor(target, name="target")

    if tuple(pred.shape) != tuple(target.shape):
        raise ValueError(
            f"pred and target must have exactly the same shape, "
            f"but got pred={tuple(pred.shape)} and target={tuple(target.shape)}"
        )


def safe_flatten_per_sample(x: torch.Tensor, name: str = "x") -> torch.Tensor:











    x = ensure_tensor(x, name=name)
    assert_bchw(x, name=name)
    return x.reshape(x.shape[0], -1)


def sample_rel_l2(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:






























    pred = ensure_tensor(pred, name="pred")
    target = ensure_tensor(target, name="target")
    assert_bchw(pred, name="pred")
    assert_bchw(target, name="target")
    check_same_shape(pred, target)

    diff = safe_flatten_per_sample(pred - target, name="pred_minus_target")
    tgt = safe_flatten_per_sample(target, name="target")

    num = torch.linalg.norm(diff, ord=2, dim=1)
    den = torch.linalg.norm(tgt, ord=2, dim=1)

    values = num / (den + eps)
    return values


def sample_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:












    pred = ensure_tensor(pred, name="pred")
    target = ensure_tensor(target, name="target")
    assert_bchw(pred, name="pred")
    assert_bchw(target, name="target")
    check_same_shape(pred, target)

    diff2 = (pred - target) ** 2
    values = diff2.reshape(diff2.shape[0], -1).mean(dim=1)
    return values


def summarize_scalar_tensor(values: torch.Tensor | np.ndarray) -> Dict[str, Any]:





















    if isinstance(values, torch.Tensor):
        values_np = values.detach().cpu().numpy()
    elif isinstance(values, np.ndarray):
        values_np = values
    else:
        raise TypeError(f"values must be torch.Tensor or np.ndarray, but got {type(values)}")

    values_np = np.asarray(values_np).reshape(-1)

    if values_np.size == 0:
        raise ValueError("summarize_scalar_tensor received empty values")

    summary = {
        "mean": float(np.mean(values_np)),
        "std": float(np.std(values_np)),
        "p50": float(np.percentile(values_np, 50)),
        "p90": float(np.percentile(values_np, 90)),
        "n": int(values_np.size),
    }
    return summary


def make_metric_block(values: torch.Tensor | np.ndarray) -> Dict[str, Any]:









    return summarize_scalar_tensor(values)


def summarize_prediction_against_target(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, Dict[str, Any]]:


















    rel_vals = sample_rel_l2(pred, target)
    mse_vals = sample_mse(pred, target)

    return {
        "rel_l2": make_metric_block(rel_vals),
        "mse": make_metric_block(mse_vals),
    }
