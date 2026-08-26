from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metrics.darcy import (
    SPECTRAL_BIN_COUNT,
    SPECTRAL_EPS_FLOOR,
    SPECTRAL_EPS_SCALE,
    SPECTRAL_NEAR_NYQUIST_END,
    SPECTRAL_NEAR_NYQUIST_START,
    SPECTRAL_PROTOCOL_SEED,
    SPECTRAL_PROTOCOL_TEST_SAMPLES,
    spectral_denominator_safeguard,
    spectral_frequency_geometry,
    spectral_profile,
)


def verified_production_reference(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Independent transcription of the verified frozen complete-FFT calculation."""
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    pred = pred - pred.mean(axis=(1, 2), keepdims=True)
    truth = truth - truth.mean(axis=(1, 2), keepdims=True)
    residual_energy = np.abs(np.fft.fft2(pred - truth, axes=(-2, -1), norm="ortho")) ** 2
    target_energy = np.abs(np.fft.fft2(truth, axes=(-2, -1), norm="ortho")) ** 2
    height, width = pred.shape[-2:]
    fy = np.fft.fftfreq(height)
    fx = np.fft.fftfreq(width)
    radius = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2) / 0.5
    retained = (radius >= 0.0) & (radius <= 1.0)
    denominator = np.sum(target_energy[:, retained], axis=1)
    epsilon = np.maximum(1.0e-12 * denominator, 1.0e-30)
    profiles = np.empty((pred.shape[0], 72), dtype=np.float64)
    for index in range(72):
        left, right = index / 72.0, (index + 1) / 72.0
        mask = retained & (radius >= left) & ((radius < right) if index < 71 else (radius <= right))
        profiles[:, index] = np.sum(residual_energy[:, mask], axis=1) / (denominator + epsilon)
    return profiles.mean(axis=0)


def synthetic_fields() -> tuple[np.ndarray, np.ndarray]:
    y, x = np.meshgrid(np.arange(256) / 256.0, np.arange(256) / 256.0, indexing="ij")
    target = np.stack((
        np.sin(2.0 * np.pi * x) + 0.25 * np.cos(4.0 * np.pi * y),
        0.6 * np.cos(2.0 * np.pi * x - 2.0 * np.pi * y) + 0.1 * np.sin(6.0 * np.pi * y),
        np.sin(4.0 * np.pi * x) * np.cos(2.0 * np.pi * y) + 0.2 * np.cos(2.0 * np.pi * x),
    ))
    prediction = target + np.stack((
        0.11 * np.cos(4.0 * np.pi * x) + 0.03 * np.sin(2.0 * np.pi * y),
        0.07 * np.sin(2.0 * np.pi * x + 4.0 * np.pi * y),
        0.09 * np.cos(6.0 * np.pi * x - 2.0 * np.pi * y),
    ))
    return prediction, target


class SpectralProtocolTests(unittest.TestCase):
    def test_final_protocol_constants(self) -> None:
        self.assertEqual(SPECTRAL_BIN_COUNT, 72)
        self.assertEqual(SPECTRAL_PROTOCOL_SEED, "0001")
        self.assertEqual(SPECTRAL_PROTOCOL_TEST_SAMPLES, 1000)
        self.assertEqual(SPECTRAL_NEAR_NYQUIST_START, 0.85)
        self.assertEqual(SPECTRAL_NEAR_NYQUIST_END, 1.00)

    def test_full_fft_circular_nyquist_geometry_and_near_band(self) -> None:
        frequency, retained, masks, centers = spectral_frequency_geometry(256, 256)
        self.assertEqual(frequency.shape, (256, 256))
        self.assertFalse(retained[128, 128])
        self.assertTrue(retained[128, 0])
        self.assertEqual(len(masks), 72)
        self.assertTrue(np.array_equal(np.sum(np.stack(masks), axis=0), retained.astype(np.int64)))
        near_indices = np.flatnonzero((centers >= 0.85) & (centers <= 1.00))
        self.assertTrue(np.array_equal(near_indices, np.arange(61, 72)))

    def test_sample_dependent_scale_relative_safeguard(self) -> None:
        energy = np.asarray((0.0, 2.0e-20, 7.0), dtype=np.float64)
        expected = np.asarray((1.0e-30, 1.0e-30, 7.0e-12), dtype=np.float64)
        self.assertTrue(np.array_equal(spectral_denominator_safeguard(energy), expected))
        self.assertEqual(SPECTRAL_EPS_SCALE, 1.0e-12)
        self.assertEqual(SPECTRAL_EPS_FLOOR, 1.0e-30)

    def test_public_function_matches_verified_frozen_complete_fft_reference(self) -> None:
        prediction, target = synthetic_fields()
        expected = verified_production_reference(prediction, target)
        actual = spectral_profile(prediction, target)
        self.assertTrue(np.array_equal(actual, expected))
        with self.assertRaises(ValueError):
            spectral_profile(prediction, target, bins=71)


if __name__ == "__main__":
    unittest.main()
