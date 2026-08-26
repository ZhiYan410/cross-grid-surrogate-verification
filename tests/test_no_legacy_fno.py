from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LegacyFNOExclusionTests(unittest.TestCase):
    def test_no_legacy_file_import_or_route(self) -> None:
        source_root = ROOT / "src"
        for path in source_root.rglob("*.py"):
            content = path.read_text(encoding="utf-8")
            self.assertNotIn("p1_m02_fno2d", content, path)
            self.assertNotIn("class HyperelasticFNO(", content, path)
            self.assertNotIn("import HyperelasticFNO", content, path)

    def test_corrected_layers_keep_both_retained_blocks(self) -> None:
        for relative in ("src/models/darcy_corrected_fno.py", "src/models/hyperelasticity_corrected_fno.py"):
            content = (ROOT / relative).read_text(encoding="utf-8")
            for token in ("weight_pos_real", "weight_pos_imag", "weight_neg_real", "weight_neg_imag", "out_ft[:, :, :m1, :m2]", "out_ft[:, :, -m1:, :m2]"):
                self.assertIn(token, content, f"{relative}: {token}")


if __name__ == "__main__": unittest.main()
