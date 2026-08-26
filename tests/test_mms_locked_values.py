from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from verification.mms import run_convergence, verify_locked_values


class ManufacturedStencilTests(unittest.TestCase):
    def test_full_locked_convergence_study(self) -> None:
        rows, checks = run_convergence()
        result = verify_locked_values(rows, ROOT / "results/mms/fd_manufactured_convergence_locked_copy.csv")
        self.assertEqual(len(rows), 12)
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(checks["A"]["status"], "PASS", checks["A"])
        self.assertEqual(checks["B"]["status"], "PASS", checks["B"])
        self.assertEqual(checks["A"]["grid"], 64)
        self.assertEqual(checks["B"]["grid"], 64)


if __name__ == "__main__":
    unittest.main()
