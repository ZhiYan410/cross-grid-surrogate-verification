from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one final-protocol hyperelasticity architecture.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--architecture", choices=("corrected_fno", "unet", "resnet"), required=True)
    parser.add_argument("--train-nx", type=int, required=True)
    parser.add_argument("--train-ny", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    from training.hyperelasticity import train
    print(train(dataset_path=args.dataset, split_path=args.split, architecture=args.architecture,
                train_nx=args.train_nx, train_ny=args.train_ny, seed=args.seed,
                output_dir=args.output, epochs=args.epochs))


if __name__ == "__main__":
    main()
