from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


EPS = 1.0e-12
SPECTRAL_BIN_COUNT = 72
SPECTRAL_NEAR_NYQUIST_START = 0.85
SPECTRAL_NEAR_NYQUIST_END = 1.00
SPECTRAL_EPS_SCALE = 1.0e-12
SPECTRAL_EPS_FLOOR = 1.0e-30
SPECTRAL_PROTOCOL_SEED = "0001"
SPECTRAL_PROTOCOL_TEST_SAMPLES = 1000


def dxdy(height: int, width: int) -> Tuple[float, float]:
    return 1.0 / float(width), 1.0 / float(height)


def physical_metrics(pred: np.ndarray, target: np.ndarray, coefficient: np.ndarray) -> Dict[str, np.ndarray]:
    """Darcy relative L2, H1-seminorm, and coefficient-weighted energy metrics."""
    n, h, w = pred.shape
    dx, dy = dxdy(h, w)
    diff = pred.astype(np.float64) - target.astype(np.float64)
    truth = target.astype(np.float64)
    coeff = coefficient.astype(np.float64)
    l2 = np.linalg.norm(diff.reshape(n, -1), axis=1) / (np.linalg.norm(truth.reshape(n, -1), axis=1) + EPS)
    diff_gx = (diff[:, 1:-1, 2:] - diff[:, 1:-1, :-2]) / (2.0 * dx)
    true_gx = (truth[:, 1:-1, 2:] - truth[:, 1:-1, :-2]) / (2.0 * dx)
    diff_gy = (diff[:, 2:, 1:-1] - diff[:, :-2, 1:-1]) / (2.0 * dy)
    true_gy = (truth[:, 2:, 1:-1] - truth[:, :-2, 1:-1]) / (2.0 * dy)
    h1_num = np.sqrt(np.sum(diff_gx ** 2 + diff_gy ** 2, axis=(1, 2)))
    h1_den = np.sqrt(np.sum(true_gx ** 2 + true_gy ** 2, axis=(1, 2)))
    coeff_i = coeff[:, 1:-1, 1:-1]
    energy_num = np.sqrt(np.sum(coeff_i * (diff_gx ** 2 + diff_gy ** 2), axis=(1, 2)))
    energy_den = np.sqrt(np.sum(coeff_i * (true_gx ** 2 + true_gy ** 2), axis=(1, 2)))
    return {
        "rel_l2": l2,
        "rel_h1_seminorm": h1_num / (h1_den + EPS),
        "rel_energy_norm": energy_num / (energy_den + EPS),
    }


def operator_response(coefficient: np.ndarray, field: np.ndarray) -> np.ndarray:
    """G_a(u)=div(-a grad u), harmonic faces, interior output only."""
    h, w = field.shape[-2:]
    dx, dy = dxdy(h, w)
    a = coefficient.astype(np.float64)
    u = field.astype(np.float64)
    ax = 2.0 * a[:, :, :-1] * a[:, :, 1:] / np.maximum(a[:, :, :-1] + a[:, :, 1:], EPS)
    ay = 2.0 * a[:, :-1, :] * a[:, 1:, :] / np.maximum(a[:, :-1, :] + a[:, 1:, :], EPS)
    qx = -ax * (u[:, :, 1:] - u[:, :, :-1]) / dx
    qy = -ay * (u[:, 1:, :] - u[:, :-1, :]) / dy
    return (qx[:, 1:-1, 1:] - qx[:, 1:-1, :-1]) / dx + (qy[:, 1:, 1:-1] - qy[:, :-1, 1:-1]) / dy


def relative_path_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.linalg.norm((left - right).reshape(left.shape[0], -1), axis=1) / (np.linalg.norm(right.reshape(right.shape[0], -1), axis=1) + EPS)


def paired_path_metrics(coefficient: np.ndarray, transfer_prediction: np.ndarray, reference_prediction: np.ndarray) -> Dict[str, np.ndarray]:
    d_field = relative_path_distance(transfer_prediction, reference_prediction)
    d_op = relative_path_distance(operator_response(coefficient, transfer_prediction), operator_response(coefficient, reference_prediction))
    return {"D_field": d_field, "D_op": d_op, "amplification": d_op / (d_field + EPS)}


def bilinear_align_corners_false(fields: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    """NumPy equivalent of torch.interpolate(..., mode='bilinear', align_corners=False)."""
    n, h, w = fields.shape
    yy = (np.arange(target_height, dtype=np.float64) + 0.5) * h / target_height - 0.5
    xx = (np.arange(target_width, dtype=np.float64) + 0.5) * w / target_width - 0.5
    y0 = np.floor(yy).astype(np.int64); x0 = np.floor(xx).astype(np.int64)
    y1 = np.minimum(y0 + 1, h - 1); x1 = np.minimum(x0 + 1, w - 1)
    wy = yy - y0; wx = xx - x0
    y0 = np.clip(y0, 0, h - 1); x0 = np.clip(x0, 0, w - 1)
    wy = np.where(yy < 0, 0.0, np.where(yy > h - 1, 1.0, wy))
    wx = np.where(xx < 0, 0.0, np.where(xx > w - 1, 1.0, wx))
    top = fields[:, y0[:, None], x0[None, :]] * (1.0 - wx)[None, None, :] + fields[:, y0[:, None], x1[None, :]] * wx[None, None, :]
    bottom = fields[:, y1[:, None], x0[None, :]] * (1.0 - wx)[None, None, :] + fields[:, y1[:, None], x1[None, :]] * wx[None, None, :]
    return top * (1.0 - wy)[None, :, None] + bottom * wy[None, :, None]


def spectral_frequency_geometry(height: int, width: int) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, ...], np.ndarray]:
    """Return frozen full-FFT circular-Nyquist geometry for the 72-bin diagnostic."""
    if height <= 0 or width <= 0:
        raise ValueError("Spectral grid dimensions must be positive")
    fy = np.fft.fftfreq(height)
    fx = np.fft.fftfreq(width)
    frequency = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2) / 0.5
    retained = (frequency >= 0.0) & (frequency <= 1.0)
    masks = []
    for index in range(SPECTRAL_BIN_COUNT):
        left = index / SPECTRAL_BIN_COUNT
        right = (index + 1) / SPECTRAL_BIN_COUNT
        masks.append(
            retained
            & (frequency >= left)
            & ((frequency < right) if index < SPECTRAL_BIN_COUNT - 1 else (frequency <= right))
        )
    centers = (np.arange(SPECTRAL_BIN_COUNT, dtype=np.float64) + 0.5) / SPECTRAL_BIN_COUNT
    return frequency, retained, tuple(masks), centers


def spectral_denominator_safeguard(retained_target_spectral_energy: np.ndarray) -> np.ndarray:
    """Apply the final sample-dependent target-energy-relative safeguard."""
    energy = np.asarray(retained_target_spectral_energy, dtype=np.float64)
    return np.maximum(SPECTRAL_EPS_SCALE * energy, SPECTRAL_EPS_FLOOR)


def spectral_profile(pred: np.ndarray, target: np.ndarray, bins: int = SPECTRAL_BIN_COUNT) -> np.ndarray:
    """Final mean-removed complete-FFT target-normalized radial residual profile."""
    if bins != SPECTRAL_BIN_COUNT:
        raise ValueError(f"Final spectral protocol fixes the radial bin count at {SPECTRAL_BIN_COUNT}")
    prediction = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if prediction.ndim != 3 or prediction.shape != truth.shape:
        raise ValueError(f"Expected matching [N,H,W] arrays, got {prediction.shape} and {truth.shape}")
    if prediction.shape[0] == 0:
        raise ValueError("Spectral profile requires at least one sample")
    prediction = prediction - prediction.mean(axis=(1, 2), keepdims=True)
    truth = truth - truth.mean(axis=(1, 2), keepdims=True)
    residual = prediction - truth
    residual_fft = np.fft.fft2(residual, axes=(-2, -1), norm="ortho")
    target_fft = np.fft.fft2(truth, axes=(-2, -1), norm="ortho")
    _, retained, masks, _ = spectral_frequency_geometry(residual.shape[1], residual.shape[2])
    residual_energy = np.abs(residual_fft) ** 2
    target_energy = np.abs(target_fft) ** 2
    denominator = np.sum(target_energy[:, retained], axis=1)
    safe_denominator = denominator + spectral_denominator_safeguard(denominator)
    radial = np.empty((residual.shape[0], SPECTRAL_BIN_COUNT), dtype=np.float64)
    for index, mask in enumerate(masks):
        radial[:, index] = np.sum(residual_energy[:, mask], axis=1) / safe_denominator
    return radial.mean(axis=0)
