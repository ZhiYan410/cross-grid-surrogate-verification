from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from data.darcy_dataset import PDEBenchDarcyFlowDataset, collate_a_u_meta
from data.darcy_splits import load_split_json
from data.resample import resample_pair
from metrics.common import sample_rel_l2
from models.darcy import build_model
from models.darcy_corrected_fno import build_fno_input


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def forward(model: nn.Module, architecture: str, coefficient: torch.Tensor) -> torch.Tensor:
    return model(build_fno_input(coefficient)) if architecture == "corrected_fno" else model(coefficient)


@torch.no_grad()
def validation_mean(model: nn.Module, architecture: str, loader: DataLoader, resolution: int, device: torch.device) -> float:
    model.eval(); values = []
    for coefficient, target, _ in loader:
        coefficient, target = coefficient.to(device), target.to(device)
        coefficient, target = resample_pair(coefficient, target, resolution, align_corners=False)
        values.append(sample_rel_l2(forward(model, architecture, coefficient), target).cpu())
    return float(torch.cat(values).mean())


def train(
    *, h5_path: Path, split_path: Path, architecture: str, train_resolution: int,
    seed: int, output_dir: Path, epochs: int = 50, batch_size: int = 16,
    learning_rate: float = 1.0e-3, val_resolution: int = 128,
) -> dict:
    """Final supervised Darcy protocol; output_dir must not exist before use."""
    if output_dir.exists(): raise FileExistsError(output_dir)
    if architecture not in ("corrected_fno", "unet", "resnet"): raise ValueError(architecture)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = load_split_json(split_path)
    dataset = PDEBenchDarcyFlowDataset(h5_path)
    train_loader = DataLoader(Subset(dataset, split["train"]), batch_size=batch_size, shuffle=True, collate_fn=collate_a_u_meta)
    val_loader = DataLoader(Subset(dataset, split["val"]), batch_size=batch_size, shuffle=False, collate_fn=collate_a_u_meta)
    model = build_model(architecture).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)
    loss_fn = nn.MSELoss(reduction="mean")
    output_dir.mkdir(parents=True, exist_ok=False)
    config = {"architecture": architecture, "train_resolution": train_resolution, "validation_resolution": val_resolution, "seed": seed, "epochs": epochs, "batch_size": batch_size, "optimizer": "Adam", "learning_rate": learning_rate, "training_loss": "MSE on paired resampled coefficient/solution fields", "checkpoint_criterion": "minimum validation mean relative L2 at validation resolution", "resampling": "area downsampling; bilinear upsampling; align_corners=False"}
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    best, best_epoch, rows = float("inf"), 0, []
    for epoch in range(1, epochs + 1):
        model.train(); total = 0.0; count = 0
        for coefficient, target, _ in train_loader:
            coefficient, target = coefficient.to(device), target.to(device)
            coefficient, target = resample_pair(coefficient, target, train_resolution, align_corners=False)
            loss = loss_fn(forward(model, architecture, coefficient), target)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total += float(loss) * coefficient.shape[0]; count += coefficient.shape[0]
        value = validation_mean(model, architecture, val_loader, val_resolution, device)
        selected = value < best
        if selected:
            best, best_epoch = value, epoch
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "best_val_rel_l2": value, "config": config}, output_dir / "ckpt_best.pt")
        rows.append({"epoch": epoch, "train_mse": total / max(count, 1), "validation_mean_relative_l2": value, "selected_as_best": int(selected)})
    import csv
    with (output_dir / "val_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return {"best_epoch": best_epoch, "best_validation_mean_relative_l2": best, "output_dir": str(output_dir)}
