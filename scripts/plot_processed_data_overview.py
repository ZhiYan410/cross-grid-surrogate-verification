from __future__ import annotations

"""Convenience visualizations of locked processed data.

These helpers are not reproductions of the published figures or their exact
panel composition. The locked processed data remain the authoritative public
figure-support records.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES = ("FNO", "U-Net", "ResNet")
BLUE, ORANGE, GREEN = "#4C78A8", "#E28E4B", "#59A14F"


def read_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def finite(value: str) -> float:
    return float(value) if value not in ("", None) else float("nan")


def matrix(rows, architecture: str, metric: str, train, evaluation):
    result = np.full((len(evaluation), len(train)), np.nan)
    for row in rows:
        if row["architecture"] == architecture:
            result[evaluation.index(int(float(row["eval_resolution"])))][train.index(int(float(row["train_resolution"])))]=finite(row[metric])
    return result


def fig2(output: Path) -> None:
    rows = read_rows(ROOT / "results/fig2/TableS1_fno_darcy_full_physical_matrix.csv") + read_rows(ROOT / "results/fig2/TableS2_unet_resnet_darcy_full_physical_matrices.csv")
    grids = [32, 64, 128, 256]
    panels = [matrix(rows, architecture, "rel_l2_seed_mean_mean", grids, grids) for architecture in ARCHITECTURES]
    all_values = np.concatenate([panel[np.isfinite(panel)] for panel in panels])
    norm = LogNorm(vmin=all_values.min(), vmax=all_values.max())
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.9), constrained_layout=True)
    for axis, architecture, values in zip(axes, ARCHITECTURES, panels):
        image = axis.imshow(values, origin="upper", cmap="magma", norm=norm, aspect="equal")
        axis.set_title(architecture); axis.set_xticks(range(4), grids); axis.set_yticks(range(4), grids)
        axis.set_xlabel("Training resolution"); axis.set_ylabel("Evaluation resolution")
    fig.colorbar(image, ax=axes, label="Darcy relative L2 error")
    fig.savefig(output, dpi=300); plt.close(fig)


def fig3(output: Path) -> None:
    with np.load(ROOT / "results/fig3/canonical_spatial_error_maps_eval256.npz", allow_pickle=False) as data:
        maps = np.asarray(data["maps"], dtype=np.float64)
        labels = [str(x) for x in data["architecture_labels"]]
        grids = [int(x) for x in data["train_resolutions"]]
    norm = Normalize(vmin=0.0, vmax=float(np.quantile(maps, 0.995)))
    fig, axes = plt.subplots(3, 4, figsize=(8.2, 5.9), constrained_layout=True)
    for row, architecture in enumerate(labels):
        for column, grid in enumerate(grids):
            image = axes[row, column].imshow(maps[row, column], origin="lower", cmap="magma", norm=norm)
            axes[row, column].set_axis_off()
            if row == 0: axes[row, column].set_title(f"Train {grid}")
        axes[row, 0].set_ylabel(architecture, rotation=90, labelpad=26)
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Mean absolute field error")
    fig.savefig(output, dpi=300); plt.close(fig)


def fig4(output: Path) -> None:
    rows = read_rows(ROOT / "results/fig4/Supplementary_Data_S1_full_spectral_profiles.csv")
    names = ("Corrected standard FNO", "U-Net", "ResNet")
    colors = {32: "#4C78A8", 64: "#72A3C8", 128: "#E28E4B", 256: "#59A14F"}
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.9), sharey=True, constrained_layout=True)
    for axis, name, title in zip(axes, names, ARCHITECTURES):
        axis.axvspan(2.0 / 3.0, 1.0, color="#AAB7C4", alpha=0.10, zorder=0)
        for training in (32, 64, 128, 256):
            curve = sorted((row for row in rows if row["architecture"] == name and row["record_type"] == "radial_profile_bin" and int(row["training_resolution"]) == training), key=lambda row: float(row["k_norm"]))
            axis.semilogy([float(row["k_norm"]) for row in curve], [float(row["target_normalized_residual_energy_mean"]) for row in curve], color=colors[training], linewidth=1.6, label=f"{training}->256")
        axis.set_xlim(0, 1); axis.set_title(title); axis.set_xlabel("Normalized frequency")
        axis.grid(axis="y", color="0.85", linewidth=0.4)
    axes[0].set_ylabel("Target-normalized residual energy")
    axes[0].legend(frameon=False, fontsize=7)
    fig.savefig(output, dpi=300); plt.close(fig)


def fig5(output: Path) -> None:
    rows = [row for row in read_rows(ROOT / "results/fig5/TableS4_darcy_path_consistency_amplification_and_fd_baseline.csv") if row.get("architecture") in ARCHITECTURES]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), constrained_layout=True)
    for axis, metric, label in zip(axes, ("D_field_seed_mean", "D_op_seed_mean"), ("D_field", "D_op")):
        for index, architecture in enumerate(ARCHITECTURES):
            values = sorted((row for row in rows if row["architecture"] == architecture), key=lambda row: str(row.get("path", "")))
            if len(values) != 2:
                raise ValueError(f"Expected two directional rows for {architecture}, found {len(values)}")
            x = np.asarray([index - 0.10, index + 0.10])
            y = np.asarray([finite(row[metric]) for row in values])
            axis.plot(x, y, color="#B3B3B3", linewidth=1.1, zorder=1)
            axis.scatter(x[0], y[0], s=42, color=BLUE, edgecolor="#2F5D83", linewidth=0.8, zorder=2, label="128->256" if index == 0 else None)
            axis.scatter(x[1], y[1], s=36, marker="s", color=ORANGE, edgecolor="#A65F2E", linewidth=0.8, zorder=2, label="256->128" if index == 0 else None)
        axis.set_yscale("log"); axis.set_xticks(range(3), ARCHITECTURES); axis.set_ylabel(label)
        axis.grid(axis="y", color="0.85", linewidth=0.4)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.savefig(output, dpi=300); plt.close(fig)


def _hyper_matrix(rows, architecture: str, metric: str):
    train = (64, 128); evaluation = (64, 128, 200)
    values = np.full((3, 2), np.nan)
    for row in rows:
        if row["architecture"] == architecture:
            values[evaluation.index(int(row["eval_nx"]))][train.index(int(row["train_nx"]))] = finite(row[metric])
    return values


def fig6(output: Path) -> None:
    rows = read_rows(ROOT / "results/fig6/TableS6_hyperelasticity_complete_grid_summary.csv")
    metrics = ("joint_rel_l2_seed_mean", "vector_h1_seminorm_rel_seed_mean")
    labels = ("Joint relative L2", "Vector H1-seminorm relative error")
    panels = [[_hyper_matrix(rows, architecture, metric) for architecture in ARCHITECTURES] for metric in metrics]
    norms = [LogNorm(vmin=np.concatenate([p[np.isfinite(p)] for p in group]).min(), vmax=np.concatenate([p[np.isfinite(p)] for p in group]).max()) for group in panels]
    fig, axes = plt.subplots(2, 3, figsize=(8.8, 5.2), constrained_layout=True)
    for row, (metric_panels, norm, label) in enumerate(zip(panels, norms, labels)):
        for column, (architecture, values) in enumerate(zip(ARCHITECTURES, metric_panels)):
            image = axes[row, column].imshow(values, origin="upper", cmap="magma", norm=norm, aspect="equal")
            axes[row, column].set_xticks((0, 1), ("64x16", "128x32")); axes[row, column].set_yticks((0, 1, 2), ("64x16", "128x32", "200x50"))
            if row == 0: axes[row, column].set_title(architecture)
            if column == 0: axes[row, column].set_ylabel("Evaluation grid")
            if row == 1: axes[row, column].set_xlabel("Training grid")
        fig.colorbar(image, ax=axes[row, :].tolist(), label=label)
    fig.savefig(output, dpi=300); plt.close(fig)


def fig7(output: Path) -> None:
    table7 = read_rows(ROOT / "results/fig7/TableS7_hyperelasticity_directional_and_direct_interpolated.csv")
    table3 = read_rows(ROOT / "results/fig7/Table3_hyperelasticity_native_and_admissibility.csv")
    direct = [row for row in table7 if row["record_type"] == "direct_vs_interpolated"]
    directional = [row for row in table7 if row["record_type"] == "directional_path"]
    labels = [f"{row['architecture']}\n{int(float(row['train_nx']))}x{int(float(row['train_ny']))}" for row in direct]
    fig, axes = plt.subplots(1, 3, figsize=(11.3, 3.2), constrained_layout=True)
    x = np.arange(len(direct))
    for offset, prefix, color, name in ((-0.17, "direct", BLUE, "native direct"), (0.17, "interpolated", ORANGE, "interpolated matched")):
        axes[0].errorbar(x + offset, [finite(row[f"{prefix}_joint_rel_l2_seed_mean"]) for row in direct], yerr=[finite(row[f"{prefix}_joint_rel_l2_seed_std"]) for row in direct], fmt="o", color=color, ecolor="0.3", capsize=2, label=name)
    axes[0].set_yscale("log"); axes[0].set_title("Native and interpolated routes"); axes[0].set_ylabel("Joint relative L2"); axes[0].set_xticks(x, labels, fontsize=7); axes[0].legend(frameon=False, fontsize=7)
    for index, row in enumerate(directional):
        axes[1].plot([index - 0.10, index + 0.10], [finite(row["field_path_distance_seed_mean"]), finite(row["gradient_path_distance_seed_mean"])], color="#B3B3B3", linewidth=1)
        axes[1].scatter(index - 0.10, finite(row["field_path_distance_seed_mean"]), color=BLUE, s=28)
        axes[1].scatter(index + 0.10, finite(row["gradient_path_distance_seed_mean"]), color=ORANGE, s=28)
    axes[1].set_yscale("log"); axes[1].set_title("Directional path diagnostics"); axes[1].set_ylabel("Path distance"); axes[1].set_xticks(range(len(directional)), [f"{r['architecture']}\n{r['path']}" for r in directional], fontsize=6, rotation=20, ha="right")
    x = np.arange(len(table3))
    axes[2].bar(x, [finite(row["energy_valid_sample_fraction_seed_mean"]) for row in table3], color=GREEN, width=0.65)
    axes[2].set_ylim(0, 1); axes[2].set_title("Conditional-energy validity"); axes[2].set_ylabel("Energy-valid sample fraction"); axes[2].set_xticks(x, [f"{r['architecture']}\n{r['train_nx']}x{r['train_ny']}" for r in table3], fontsize=7)
    for axis in axes: axis.grid(axis="y", color="0.86", linewidth=0.4)
    fig.savefig(output, dpi=300); plt.close(fig)


FUNCTIONS = {"fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5, "fig6": fig6, "fig7": fig7}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", choices=sorted(FUNCTIONS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    FUNCTIONS[args.figure](args.output)


if __name__ == "__main__":
    main()
