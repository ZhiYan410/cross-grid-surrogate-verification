from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class CorrectedFNOStaticTests(unittest.TestCase):
    def test_both_retained_spectral_blocks_are_present(self) -> None:
        required = (
            "weight_pos_real",
            "weight_pos_imag",
            "weight_neg_real",
            "weight_neg_imag",
            "out_ft[:, :, :m1, :m2]",
            "out_ft[:, :, -m1:, :m2]",
        )
        for relative in (
            "src/models/darcy_corrected_fno.py",
            "src/models/hyperelasticity_corrected_fno.py",
        ):
            content = (ROOT / relative).read_text(encoding="utf-8")
            for token in required:
                self.assertIn(token, content, f"{relative}: {token}")


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires declared PyTorch runtime")
class ModelForwardShapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch
        from data.hyperelasticity import NormalizerStats
        cls.torch = torch
        cls.normalizer = NormalizerStats(0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 400)

    def test_darcy_output_shapes_at_all_locked_resolutions(self) -> None:
        from models.darcy import build_model
        from models.darcy_corrected_fno import build_fno_input
        for architecture in ("corrected_fno", "unet", "resnet"):
            model = build_model(architecture).eval()
            for resolution in (32, 64, 128, 256):
                source = self.torch.zeros(1, 1, resolution, resolution)
                with self.torch.no_grad():
                    result = model(build_fno_input(source)) if architecture == "corrected_fno" else model(source)
                self.assertEqual(tuple(result.shape), (1, 1, resolution, resolution))

    def test_hyperelastic_shapes_and_left_clamp_at_all_locked_grids(self) -> None:
        from models.hyperelasticity import build_model
        for architecture in ("corrected_fno", "unet", "resnet"):
            model = build_model(architecture, self.normalizer).eval()
            for nx, ny in ((64, 16), (128, 32), (200, 50)):
                source = self.torch.zeros(1, 1, ny, nx)
                with self.torch.no_grad():
                    result = model(source)
                self.assertEqual(tuple(result.shape), (1, 2, ny, nx))
                self.assertTrue(self.torch.equal(result[..., 0], self.torch.zeros_like(result[..., 0])))


if __name__ == "__main__":
    unittest.main()
