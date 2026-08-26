from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from data.hyperelasticity import (
    EXPECTED_DATASET_SHA256, EXPECTED_SPLIT_SHA256, HyperelasticityArrays,
    HyperelasticityGridDataset, LockedSplit, NormalizerStats,
    decode_and_project_displacement, fit_train_only_normalizers,
    joint_relative_l2_torch, load_hyperelasticity_arrays, load_locked_split,
    set_global_seed,
)
from models.hyperelasticity import build_model


def materialize_dataset(arrays: HyperelasticityArrays, indices: Sequence[int], normalizer: NormalizerStats, *, nx: int, ny: int) -> TensorDataset:
    source = HyperelasticityGridDataset(arrays, indices, normalizer, nx=nx, ny=ny)
    traction_n, disp_n, traction_p, disp_p, identities = [], [], [], [], []
    for local_index in range(len(source)):
        traction, displacement, sample_index = source.physical_sample(local_index)
        traction_encoded = (traction - float(normalizer.traction_mean)) / float(normalizer.traction_std)
        mean = torch.as_tensor(normalizer.displacement_mean, dtype=displacement.dtype).view(2, 1, 1)
        std = torch.as_tensor(normalizer.displacement_std, dtype=displacement.dtype).view(2, 1, 1)
        traction_n.append(traction_encoded); disp_n.append((displacement - mean) / std)
        traction_p.append(traction); disp_p.append(displacement); identities.append(int(sample_index))
    return TensorDataset(torch.stack(traction_n), torch.stack(disp_n), torch.tensor(identities, dtype=torch.int64), torch.stack(traction_p), torch.stack(disp_p))


@torch.no_grad()
def validation_decoded_joint_l2(model: nn.Module, loader: DataLoader, normalizer: NormalizerStats, device: torch.device) -> tuple[float, float]:
    model.eval(); l2 = []; mse_total = 0.0; count = 0
    for traction, target_n, _, _, target_p in loader:
        traction, target_n, target_p = traction.to(device), target_n.to(device), target_p.to(device)
        prediction_n = model(traction)
        prediction_p = decode_and_project_displacement(prediction_n, normalizer)
        l2.append(joint_relative_l2_torch(prediction_p.to(torch.float64), target_p.to(torch.float64)).cpu())
        mse_total += float(torch.mean((prediction_n - target_n) ** 2)) * traction.shape[0]; count += traction.shape[0]
    return float(torch.cat(l2).mean()), mse_total / max(count, 1)


def train(*, dataset_path: Path, split_path: Path, architecture: str, train_nx: int, train_ny: int, seed: int, output_dir: Path, epochs: int = 50, batch_size: int = 16, learning_rate: float = 1.0e-3) -> Dict[str, object]:
    """Final hyperelasticity loss/checkpoint protocol with no energy term in training loss."""
    if output_dir.exists(): raise FileExistsError(output_dir)
    arrays = load_hyperelasticity_arrays(dataset_path)
    split = load_locked_split(split_path)
    if arrays.dataset_sha256 != EXPECTED_DATASET_SHA256 or split.digest != EXPECTED_SPLIT_SHA256: raise ValueError("dataset or locked split hash mismatch")
    set_global_seed(seed, deterministic=True)
    normalizer = fit_train_only_normalizers(arrays, split.train)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(architecture, normalizer).to(device)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(materialize_dataset(arrays, split.train, normalizer, nx=train_nx, ny=train_ny), batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(materialize_dataset(arrays, split.validation, normalizer, nx=train_nx, ny=train_ny), batch_size=batch_size, shuffle=False)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=0.0); mse = nn.MSELoss(reduction="mean")
    output_dir.mkdir(parents=True, exist_ok=False)
    config = {"architecture": architecture, "training_grid": {"nx": train_nx, "ny": train_ny}, "seed": seed, "epochs": epochs, "batch_size": batch_size, "optimizer": "Adam", "learning_rate": learning_rate, "training_loss": "equal-weight MSE over train-standardized ux and uy", "checkpoint_criterion": "minimum decoded validation mean joint-vector relative L2", "interpolation": "endpoint-preserving linear/bilinear; align_corners=True; native 200x50 direct", "left_clamp_projection": "physical u=(0,0) at x=0"}
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    best, best_epoch, rows = float("inf"), 0, []
    for epoch in range(1, epochs + 1):
        model.train(); total = 0.0; count = 0
        for traction, target, _, _, _ in train_loader:
            traction, target = traction.to(device), target.to(device)
            loss = mse(model(traction), target)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total += float(loss) * traction.shape[0]; count += traction.shape[0]
        value, val_mse = validation_decoded_joint_l2(model, val_loader, normalizer, device)
        selected = value < best
        if selected:
            best, best_epoch = value, epoch
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "best_val_joint_l2": value, "config": config}, output_dir / "ckpt_best.pt")
        rows.append({"epoch": epoch, "train_standardized_mse": total / max(count, 1), "validation_decoded_mean_joint_relative_l2": value, "validation_standardized_mse": val_mse, "selected_as_best": int(selected)})
    with (output_dir / "val_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return {"best_epoch": best_epoch, "best_validation_joint_relative_l2": best, "output_dir": str(output_dir)}
