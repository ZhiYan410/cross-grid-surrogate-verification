from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from metrics.hyperelasticity import compute_physical_metrics


def interpolate_prediction(prediction: np.ndarray, nx: int = 200, ny: int = 50) -> np.ndarray:
    """Frozen matched-grid route: bilinear interpolation with align_corners=True."""
    tensor = torch.from_numpy(np.asarray(prediction, dtype=np.float32))
    return F.interpolate(tensor, size=(ny, nx), mode="bilinear", align_corners=True).numpy()


def direct_vs_interpolated_metrics(native_prediction: np.ndarray, matched_prediction: np.ndarray, target: np.ndarray, traction: np.ndarray) -> dict:
    interpolated = interpolate_prediction(matched_prediction)
    return {
        "direct": compute_physical_metrics(native_prediction, target, traction),
        "interpolated": compute_physical_metrics(interpolated, target, traction),
        "direct_vs_interpolated": compute_physical_metrics(native_prediction, interpolated, traction),
    }


def directional_path_metrics(transfer_prediction: np.ndarray, reference_prediction: np.ndarray, traction: np.ndarray) -> dict:
    """Directional field/gradient/energy path diagnostics with source rules unchanged."""
    return compute_physical_metrics(transfer_prediction, reference_prediction, traction)
