from __future__ import annotations

"""Verify SHA-256 integrity of public locked processed-result files."""

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest().upper()


def main() -> None:
    manifest = ROOT / "results" / "LOCKED_RESULTS_MANIFEST.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    failed = []
    for row in rows:
        path = ROOT / row["public_path"]
        if not path.exists() or digest(path) != row["public_sha256"]:
            failed.append(row["public_path"])
    if failed:
        raise SystemExit("Locked result verification failed: " + ", ".join(failed))
    print(f"PASS: {len(rows)} public locked processed-result files match their release hashes.")


if __name__ == "__main__":
    main()
