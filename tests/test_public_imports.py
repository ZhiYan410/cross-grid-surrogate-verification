from __future__ import annotations

import importlib
import importlib.util
import pkgutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class PublicImportTests(unittest.TestCase):
    def test_no_private_project_dependency_text(self) -> None:
        forbidden = ("p1_01_core", "p1_02_train", "revision_R1_strong", "Paper1Final")
        for path in (ROOT / "src").rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, text, path)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires declared PyTorch runtime")
    def test_import_every_public_src_module(self) -> None:
        packages = ("data", "metrics", "models", "training", "verification")
        modules = []
        for package_name in packages:
            package = importlib.import_module(package_name)
            modules.append(package_name)
            modules.extend(info.name for info in pkgutil.walk_packages(package.__path__, prefix=package.__name__ + "."))
        for module_name in sorted(modules):
            with self.subTest(module=module_name):
                importlib.import_module(module_name)


if __name__ == "__main__":
    unittest.main()
