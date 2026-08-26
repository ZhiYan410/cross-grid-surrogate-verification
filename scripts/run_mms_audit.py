from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate the two-case Darcy manufactured-stencil convergence audit.")
    parser.add_argument("--locked-csv", type=Path, default=ROOT / "results/mms/fd_manufactured_convergence_locked_copy.csv")
    parser.add_argument("--write-csv", type=Path, help="Optional new output path; existing files are never replaced.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    from verification.mms import run_convergence, verify_locked_values
    rows, checks = run_convergence()
    locked = verify_locked_values(rows, args.locked_csv)
    if args.write_csv:
        if args.write_csv.exists():
            raise FileExistsError(args.write_csv)
        args.write_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.write_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    print({"rows": len(rows), "locked_values": locked, "case_a_crosscheck": checks["A"], "case_b_crosscheck": checks["B"], "sympy": checks["sympy"]})
    if locked["status"] != "PASS" or checks["A"]["status"] != "PASS" or checks["B"]["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
