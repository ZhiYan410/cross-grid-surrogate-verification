from __future__ import annotations

"""Final hyperelasticity data, normalization, interpolation, and metric utilities.

The obsolete one-sided FNO wrapper was deliberately excluded from this public release.
"""

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


EXPECTED_DATASET_SHA256 = (
    "E7D7821BD7295243F46C708BE360004408FA343D87536E25126BAEBE5704F599"
)
EXPECTED_SPLIT_SHA256 = (
    "E9D44585D00448B74457B5D26FD7DB7F58326497648E8095ED5CB115C9116BE0"
)
EXPECTED_DATASET_FILENAME = "Hyperelasticity_n550_mooneyrivlin_200X50.npz"

NATIVE_NX = 200
NATIVE_NY = 50
DOMAIN_LENGTH = 4.0
DOMAIN_WIDTH = 1.0
EPSILON = 1.0e-12

MOONEY_C1 = 630.0
MOONEY_C2 = -1.2
MOONEY_C = 100.0
MOONEY_D = 2.0 * (MOONEY_C1 + 2.0 * MOONEY_C2)


def sha256_file(path: Path, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    """Return an uppercase SHA-256 digest without loading the file at once."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest().upper()


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    """Hash the exact canonical JSON representation used by the split asset."""

    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest().upper()


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch for a single-GPU deterministic run."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


@dataclass(frozen=True)
class LockedSplit:
    train: Tuple[int, ...]
    validation: Tuple[int, ...]
    test: Tuple[int, ...]
    digest: str
    algorithm: str
    split_seed: int

    def all_indices(self) -> Tuple[int, ...]:
        return self.train + self.validation + self.test


def load_locked_split(path: Path) -> LockedSplit:
    """Load and fully validate the approved 400/75/75 positional split."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if "canonical_payload" not in raw or "canonical_payload_sha256" not in raw:
        raise ValueError(f"Unexpected split schema: {path}")
    payload = raw["canonical_payload"]
    stored_digest = str(raw["canonical_payload_sha256"]).upper()
    recomputed = canonical_json_sha256(payload)
    if stored_digest != recomputed:
        raise ValueError(
            f"Split canonical checksum mismatch: stored={stored_digest}, "
            f"recomputed={recomputed}"
        )
    if recomputed != EXPECTED_SPLIT_SHA256:
        raise ValueError(
            f"Split does not match approved digest: {recomputed} "
            f"!= {EXPECTED_SPLIT_SHA256}"
        )
    if str(payload.get("dataset_sha256", "")).upper() != EXPECTED_DATASET_SHA256:
        raise ValueError("Split is not tied to the approved dataset digest")

    train = tuple(int(x) for x in payload["train_indices"])
    validation = tuple(int(x) for x in payload["validation_indices"])
    test = tuple(int(x) for x in payload["test_indices"])
    if (len(train), len(validation), len(test)) != (400, 75, 75):
        raise ValueError(
            f"Expected split counts (400,75,75), got "
            f"{(len(train), len(validation), len(test))}"
        )
    train_set, val_set, test_set = set(train), set(validation), set(test)
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError("Train, validation, and test indices are not disjoint")
    if train_set | val_set | test_set != set(range(550)):
        raise ValueError("Locked split does not cover positional indices 0..549")

    return LockedSplit(
        train=train,
        validation=validation,
        test=test,
        digest=recomputed,
        algorithm=str(payload["algorithm"]),
        split_seed=int(payload["split_seed"]),
    )


@dataclass(frozen=True)
class NormalizerStats:
    traction_mean: float
    traction_std: float
    displacement_mean_ux: float
    displacement_mean_uy: float
    displacement_std_ux: float
    displacement_std_uy: float
    fitted_sample_count: int

    @property
    def displacement_mean(self) -> np.ndarray:
        return np.asarray(
            [self.displacement_mean_ux, self.displacement_mean_uy],
            dtype=np.float64,
        )

    @property
    def displacement_std(self) -> np.ndarray:
        return np.asarray(
            [self.displacement_std_ux, self.displacement_std_uy],
            dtype=np.float64,
        )

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HyperelasticityArrays:
    traction: np.ndarray
    displacement: np.ndarray
    dataset_path: Path
    dataset_sha256: str


def load_hyperelasticity_arrays(dataset_path: Path) -> HyperelasticityArrays:
    """Hash and load the exact official NPZ data contract."""

    dataset_path = dataset_path.resolve()
    if dataset_path.name != EXPECTED_DATASET_FILENAME:
        raise ValueError(
            f"Unexpected dataset filename: {dataset_path.name}; "
            f"expected {EXPECTED_DATASET_FILENAME}"
        )
    digest = sha256_file(dataset_path)
    if digest != EXPECTED_DATASET_SHA256:
        raise ValueError(
            f"Dataset SHA-256 mismatch: {digest} != {EXPECTED_DATASET_SHA256}"
        )
    with np.load(dataset_path, allow_pickle=False) as archive:
        keys = set(archive.files)
        if keys != {"traction", "disp2D"}:
            raise ValueError(f"Unexpected dataset keys: {sorted(keys)}")
        traction = np.asarray(archive["traction"], dtype=np.float64)
        displacement = np.asarray(archive["disp2D"], dtype=np.float64)

    if traction.shape != (550, NATIVE_NY):
        raise ValueError(f"Unexpected traction shape: {traction.shape}")
    if displacement.shape != (550, NATIVE_NY, NATIVE_NX, 2):
        raise ValueError(f"Unexpected displacement shape: {displacement.shape}")
    if not np.isfinite(traction).all() or not np.isfinite(displacement).all():
        raise ValueError("Official dataset contains NaN or Inf")

    return HyperelasticityArrays(
        traction=traction,
        displacement=displacement,
        dataset_path=dataset_path,
        dataset_sha256=digest,
    )


def fit_train_only_normalizers(
    arrays: HyperelasticityArrays,
    train_indices: Sequence[int],
) -> NormalizerStats:
    """Fit scalar statistics only on approved training sample identities."""

    idx = np.asarray(train_indices, dtype=np.int64)
    traction = arrays.traction[idx]
    displacement = arrays.displacement[idx]
    traction_mean = float(np.mean(traction, dtype=np.float64))
    traction_std = float(np.std(traction, dtype=np.float64))
    disp_mean = np.mean(displacement, axis=(0, 1, 2), dtype=np.float64)
    disp_std = np.std(displacement, axis=(0, 1, 2), dtype=np.float64)
    if traction_std <= 0.0 or np.any(disp_std <= 0.0):
        raise ValueError("Training-only normalizer has a non-positive std")
    return NormalizerStats(
        traction_mean=traction_mean,
        traction_std=traction_std,
        displacement_mean_ux=float(disp_mean[0]),
        displacement_mean_uy=float(disp_mean[1]),
        displacement_std_ux=float(disp_std[0]),
        displacement_std_uy=float(disp_std[1]),
        fitted_sample_count=len(idx),
    )


def physical_coordinates(nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return endpoint-inclusive coordinates for the fixed beam domain."""

    x = np.linspace(0.0, DOMAIN_LENGTH, nx, dtype=np.float64)
    y = np.linspace(0.0, DOMAIN_WIDTH, ny, dtype=np.float64)
    return x, y


def resample_traction_1d(
    traction_native: torch.Tensor,
    target_ny: int,
) -> torch.Tensor:
    """Linearly interpolate the endpoint-inclusive 1D traction along y."""

    if traction_native.ndim != 1 or traction_native.shape[0] != NATIVE_NY:
        raise ValueError(
            f"Expected native traction [{NATIVE_NY}], got "
            f"{tuple(traction_native.shape)}"
        )
    if target_ny == NATIVE_NY:
        return traction_native.clone()
    values = traction_native.view(1, 1, NATIVE_NY)
    return F.interpolate(
        values,
        size=target_ny,
        mode="linear",
        align_corners=True,
    ).view(target_ny)


def resample_displacement_nodal(
    displacement_native_chw: torch.Tensor,
    target_nx: int,
    target_ny: int,
) -> torch.Tensor:
    """Bilinearly interpolate two endpoint-inclusive nodal components."""

    if tuple(displacement_native_chw.shape) != (2, NATIVE_NY, NATIVE_NX):
        raise ValueError(
            "Expected native displacement [2,50,200], got "
            f"{tuple(displacement_native_chw.shape)}"
        )
    if target_nx == NATIVE_NX and target_ny == NATIVE_NY:
        return displacement_native_chw.clone()
    values = displacement_native_chw.unsqueeze(0)
    return F.interpolate(
        values,
        size=(target_ny, target_nx),
        mode="bilinear",
        align_corners=True,
    ).squeeze(0)


class HyperelasticityGridDataset(Dataset):
    """Positional-index dataset at one requested rectangular sampling grid."""

    def __init__(
        self,
        arrays: HyperelasticityArrays,
        indices: Sequence[int],
        normalizer: NormalizerStats,
        *,
        nx: int,
        ny: int,
    ) -> None:
        self.arrays = arrays
        self.indices = tuple(int(i) for i in indices)
        self.normalizer = normalizer
        self.nx = int(nx)
        self.ny = int(ny)
        if self.nx < 2 or self.ny < 2:
            raise ValueError("Both rectangular grid dimensions must be >= 2")
        self.native_direct = self.nx == NATIVE_NX and self.ny == NATIVE_NY

    def __len__(self) -> int:
        return len(self.indices)

    def physical_sample(
        self,
        local_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Return unnormalized traction/displacement and positional identity."""

        sample_index = self.indices[int(local_index)]
        traction_native = torch.from_numpy(
            self.arrays.traction[sample_index].astype(np.float32, copy=False)
        )
        displacement_native = torch.from_numpy(
            np.moveaxis(
                self.arrays.displacement[sample_index].astype(
                    np.float32, copy=False
                ),
                -1,
                0,
            )
        )
        if self.native_direct:
            traction_y = traction_native.clone()
            displacement = displacement_native.clone()
        else:
            traction_y = resample_traction_1d(traction_native, self.ny)
            displacement = resample_displacement_nodal(
                displacement_native,
                self.nx,
                self.ny,
            )
        traction = traction_y.view(1, self.ny, 1).expand(1, self.ny, self.nx)
        return traction.contiguous(), displacement.contiguous(), sample_index

    def __getitem__(
        self,
        local_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        traction, displacement, sample_index = self.physical_sample(local_index)
        traction = (
            traction - float(self.normalizer.traction_mean)
        ) / float(self.normalizer.traction_std)
        means = torch.tensor(
            self.normalizer.displacement_mean,
            dtype=displacement.dtype,
        ).view(2, 1, 1)
        stds = torch.tensor(
            self.normalizer.displacement_std,
            dtype=displacement.dtype,
        ).view(2, 1, 1)
        displacement = (displacement - means) / stds
        return traction, displacement, sample_index


def decode_displacement(
    normalized: torch.Tensor,
    normalizer: NormalizerStats,
) -> torch.Tensor:
    """Decode a BCHW two-component displacement tensor."""

    if normalized.ndim != 4 or normalized.shape[1] != 2:
        raise ValueError(
            f"Expected normalized displacement [B,2,H,W], got "
            f"{tuple(normalized.shape)}"
        )
    means = torch.as_tensor(
        normalizer.displacement_mean,
        dtype=normalized.dtype,
        device=normalized.device,
    ).view(1, 2, 1, 1)
    stds = torch.as_tensor(
        normalizer.displacement_std,
        dtype=normalized.dtype,
        device=normalized.device,
    ).view(1, 2, 1, 1)
    return normalized * stds + means


def decode_and_project_displacement(
    normalized: torch.Tensor,
    normalizer: NormalizerStats,
) -> torch.Tensor:
    """Decode and enforce the homogeneous physical clamp exactly at x=0."""

    physical = decode_displacement(normalized, normalizer)
    physical = physical.clone()
    physical[:, :, :, 0] = 0.0
    return physical


def encode_displacement(
    physical: torch.Tensor,
    normalizer: NormalizerStats,
) -> torch.Tensor:
    """Encode a BCHW physical displacement tensor."""

    means = torch.as_tensor(
        normalizer.displacement_mean,
        dtype=physical.dtype,
        device=physical.device,
    ).view(1, 2, 1, 1)
    stds = torch.as_tensor(
        normalizer.displacement_std,
        dtype=physical.dtype,
        device=physical.device,
    ).view(1, 2, 1, 1)
    return (physical - means) / stds


def trapezoidal_node_weights(ny: int, nx: int) -> np.ndarray:
    """Tensor-product endpoint weights; constant dx*dy cancels in ratios."""

    wy = np.ones(ny, dtype=np.float64)
    wx = np.ones(nx, dtype=np.float64)
    wy[[0, -1]] = 0.5
    wx[[0, -1]] = 0.5
    return wy[:, None] * wx[None, :]


def joint_relative_l2_torch(
    prediction: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = EPSILON,
) -> torch.Tensor:
    """Per-sample physical joint-vector L2 with trapezoidal nodal weights."""

    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("Prediction/target BCHW shapes do not match")
    ny, nx = prediction.shape[-2:]
    wy = torch.ones(ny, dtype=prediction.dtype, device=prediction.device)
    wx = torch.ones(nx, dtype=prediction.dtype, device=prediction.device)
    wy[[0, -1]] = 0.5
    wx[[0, -1]] = 0.5
    weights = (wy[:, None] * wx[None, :]).view(1, 1, ny, nx)
    numerator = torch.sqrt(
        torch.sum(weights * (prediction - target) ** 2, dim=(1, 2, 3))
    )
    denominator = torch.sqrt(
        torch.sum(weights * target**2, dim=(1, 2, 3))
    )
    return numerator / (denominator + float(epsilon))


def _q1_grad_primary(
    field: np.ndarray,
    dx: float,
    dy: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Q1 cell-center gradients using explicit four-node expressions."""

    grad_x = (
        (field[:, :-1, 1:, :] - field[:, :-1, :-1, :])
        + (field[:, 1:, 1:, :] - field[:, 1:, :-1, :])
    ) / (2.0 * dx)
    grad_y = (
        (field[:, 1:, :-1, :] - field[:, :-1, :-1, :])
        + (field[:, 1:, 1:, :] - field[:, :-1, 1:, :])
    ) / (2.0 * dy)
    return grad_x, grad_y


def _q1_grad_independent(
    field: np.ndarray,
    dx: float,
    dy: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Independent Q1 implementation based on edge differences and averaging."""

    x_edges = np.diff(field, axis=2) / dx
    y_edges = np.diff(field, axis=1) / dy
    grad_x = 0.5 * (x_edges[:, :-1, :, :] + x_edges[:, 1:, :, :])
    grad_y = 0.5 * (y_edges[:, :, :-1, :] + y_edges[:, :, 1:, :])
    return grad_x, grad_y


def _kinematics_primary(
    field: np.ndarray,
    dx: float,
    dy: float,
) -> Dict[str, np.ndarray]:
    gx, gy = _q1_grad_primary(field, dx, dy)
    fxx = 1.0 + gx[..., 0]
    fxy = gy[..., 0]
    fyx = gx[..., 1]
    fyy = 1.0 + gy[..., 1]
    jacobian = fxx * fyy - fxy * fyx
    i1 = fxx**2 + fxy**2 + fyx**2 + fyy**2
    i2 = jacobian**2
    return {
        "grad_x": gx,
        "grad_y": gy,
        "fxx": fxx,
        "fxy": fxy,
        "fyx": fyx,
        "fyy": fyy,
        "jacobian": jacobian,
        "i1": i1,
        "i2": i2,
    }


def _kinematics_independent(
    field: np.ndarray,
    dx: float,
    dy: float,
) -> Dict[str, np.ndarray]:
    gx, gy = _q1_grad_independent(field, dx, dy)
    shape = gx.shape[:-1] + (2, 2)
    deformation = np.zeros(shape, dtype=np.float64)
    deformation[..., 0, 0] = 1.0 + gx[..., 0]
    deformation[..., 0, 1] = gy[..., 0]
    deformation[..., 1, 0] = gx[..., 1]
    deformation[..., 1, 1] = 1.0 + gy[..., 1]
    c_tensor = np.einsum("...ki,...kj->...ij", deformation, deformation)
    jacobian = np.linalg.det(deformation)
    i1 = np.trace(c_tensor, axis1=-2, axis2=-1)
    c_squared = np.einsum("...ik,...kj->...ij", c_tensor, c_tensor)
    trace_c_squared = np.trace(c_squared, axis1=-2, axis2=-1)
    i2 = 0.5 * (i1**2 - trace_c_squared)
    return {
        "grad_x": gx,
        "grad_y": gy,
        "deformation": deformation,
        "jacobian": jacobian,
        "i1": i1,
        "i2": i2,
    }


def _mooney_energy_from_kinematics(
    kinematics: Mapping[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate energy only for positive J; invalid cells remain NaN."""

    jacobian = np.asarray(kinematics["jacobian"], dtype=np.float64)
    energy = np.full(jacobian.shape, np.nan, dtype=np.float64)
    valid = jacobian > 0.0
    energy[valid] = (
        MOONEY_C * (jacobian[valid] - 1.0) ** 2
        - MOONEY_D * np.log(jacobian[valid])
        + MOONEY_C1 * (np.asarray(kinematics["i1"])[valid] - 2.0)
        + MOONEY_C2 * (np.asarray(kinematics["i2"])[valid] - 1.0)
    )
    return energy, valid


def _relative_l2_numpy(
    prediction: np.ndarray,
    target: np.ndarray,
    component: int | None = None,
) -> np.ndarray:
    """Per-sample weighted nodal relative L2."""

    n, ny, nx, channels = prediction.shape
    if target.shape != prediction.shape or channels != 2:
        raise ValueError("Expected matching arrays [N,Ny,Nx,2]")
    weights = trapezoidal_node_weights(ny, nx)
    if component is None:
        error2 = np.sum((prediction - target) ** 2, axis=-1)
        target2 = np.sum(target**2, axis=-1)
    else:
        error2 = (prediction[..., component] - target[..., component]) ** 2
        target2 = target[..., component] ** 2
    numerator = np.sqrt(np.sum(error2 * weights[None, ...], axis=(1, 2)))
    denominator = np.sqrt(np.sum(target2 * weights[None, ...], axis=(1, 2)))
    return numerator / (denominator + EPSILON)


def compute_physical_metrics(
    prediction_nchw: np.ndarray,
    target_nchw: np.ndarray,
    traction_nchw: np.ndarray,
    *,
    implementation: str = "primary",
) -> Dict[str, np.ndarray]:
    """Compute every requested metric in float64 for all samples."""

    prediction = np.moveaxis(
        np.asarray(prediction_nchw, dtype=np.float64), 1, -1
    )
    target = np.moveaxis(np.asarray(target_nchw, dtype=np.float64), 1, -1)
    traction = np.asarray(traction_nchw, dtype=np.float64)
    if prediction.shape != target.shape or prediction.shape[-1] != 2:
        raise ValueError("Expected prediction/target [N,2,Ny,Nx]")
    if traction.shape != (
        prediction.shape[0],
        1,
        prediction.shape[1],
        prediction.shape[2],
    ):
        raise ValueError("Expected traction [N,1,Ny,Nx]")

    _, ny, nx, _ = prediction.shape
    dx = DOMAIN_LENGTH / float(nx - 1)
    dy = DOMAIN_WIDTH / float(ny - 1)
    if implementation == "primary":
        kin_pred = _kinematics_primary(prediction, dx, dy)
        kin_target = _kinematics_primary(target, dx, dy)
    elif implementation == "independent":
        kin_pred = _kinematics_independent(prediction, dx, dy)
        kin_target = _kinematics_independent(target, dx, dy)
    else:
        raise ValueError(f"Unknown metric implementation: {implementation}")

    pred_energy, pred_valid = _mooney_energy_from_kinematics(kin_pred)
    target_energy, target_valid = _mooney_energy_from_kinematics(kin_target)

    grad_error2 = np.sum(
        (kin_pred["grad_x"] - kin_target["grad_x"]) ** 2
        + (kin_pred["grad_y"] - kin_target["grad_y"]) ** 2,
        axis=(1, 2, 3),
    )
    grad_target2 = np.sum(
        kin_target["grad_x"] ** 2 + kin_target["grad_y"] ** 2,
        axis=(1, 2, 3),
    )
    rel_h1 = np.sqrt(grad_error2) / (
        np.sqrt(grad_target2) + EPSILON
    )

    if implementation == "primary":
        f_target2 = np.sum(
            kin_target["fxx"] ** 2
            + kin_target["fxy"] ** 2
            + kin_target["fyx"] ** 2
            + kin_target["fyy"] ** 2,
            axis=(1, 2),
        )
    else:
        f_target2 = np.sum(
            kin_target["deformation"] ** 2,
            axis=(1, 2, 3, 4),
        )
    rel_deformation = np.sqrt(grad_error2) / (
        np.sqrt(f_target2) + EPSILON
    )

    energy_valid_by_sample = np.all(
        pred_valid & target_valid, axis=(1, 2)
    )
    rel_energy_density = np.full(prediction.shape[0], np.nan, dtype=np.float64)
    total_potential_rel = np.full(
        prediction.shape[0], np.nan, dtype=np.float64
    )
    total_potential_abs = np.full(
        prediction.shape[0], np.nan, dtype=np.float64
    )
    internal_pred = np.full(prediction.shape[0], np.nan, dtype=np.float64)
    internal_target = np.full(prediction.shape[0], np.nan, dtype=np.float64)
    potential_pred = np.full(prediction.shape[0], np.nan, dtype=np.float64)
    potential_target = np.full(prediction.shape[0], np.nan, dtype=np.float64)

    for sample in range(prediction.shape[0]):
        if not energy_valid_by_sample[sample]:
            continue
        p_energy = pred_energy[sample]
        t_energy = target_energy[sample]
        rel_energy_density[sample] = math.sqrt(
            float(np.sum((p_energy - t_energy) ** 2) * dx * dy)
        ) / (
            math.sqrt(float(np.sum(t_energy**2) * dx * dy)) + EPSILON
        )
        internal_pred[sample] = float(np.sum(p_energy) * dx * dy)
        internal_target[sample] = float(np.sum(t_energy) * dx * dy)
        traction_right = traction[sample, 0, :, -1]
        work_pred = float(
            np.trapz(
                traction_right * prediction[sample, :, -1, 1],
                dx=dy,
            )
        )
        work_target = float(
            np.trapz(
                traction_right * target[sample, :, -1, 1],
                dx=dy,
            )
        )
        potential_pred[sample] = internal_pred[sample] - work_pred
        potential_target[sample] = internal_target[sample] - work_target
        total_potential_abs[sample] = abs(
            potential_pred[sample] - potential_target[sample]
        )
        total_potential_rel[sample] = total_potential_abs[sample] / (
            abs(potential_target[sample]) + EPSILON
        )

    jacobian = np.asarray(kin_pred["jacobian"], dtype=np.float64)
    # prediction has [N,Ny,Nx,2]; x=0 is the second spatial index.
    clamp = prediction[:, :, 0, :]

    return {
        "joint_rel_l2": _relative_l2_numpy(prediction, target),
        "ux_rel_l2": _relative_l2_numpy(prediction, target, component=0),
        "uy_rel_l2": _relative_l2_numpy(prediction, target, component=1),
        "vector_h1_seminorm_rel": rel_h1,
        "deformation_gradient_rel": rel_deformation,
        "strain_energy_density_rel": rel_energy_density,
        "sampled_total_potential_rel": total_potential_rel,
        "sampled_total_potential_abs": total_potential_abs,
        "clamped_boundary_rms": np.sqrt(
            np.mean(np.sum(clamp**2, axis=-1), axis=1)
        ),
        "min_det_f": np.min(jacobian, axis=(1, 2)),
        "nonpositive_j_fraction": np.mean(jacobian <= 0.0, axis=(1, 2)),
        "energy_valid": energy_valid_by_sample.astype(np.int64),
        "internal_energy_prediction": internal_pred,
        "internal_energy_target": internal_target,
        "total_potential_prediction": potential_pred,
        "total_potential_target": potential_target,
    }


def compare_metric_implementations(
    primary: Mapping[str, np.ndarray],
    independent: Mapping[str, np.ndarray],
) -> Dict[str, float | bool]:
    """Compare independent H1 and energy calculations without hiding invalids."""

    result: Dict[str, float | bool] = {}
    for key in ("vector_h1_seminorm_rel", "strain_energy_density_rel"):
        left = np.asarray(primary[key], dtype=np.float64)
        right = np.asarray(independent[key], dtype=np.float64)
        finite_match = np.array_equal(np.isfinite(left), np.isfinite(right))
        finite = np.isfinite(left) & np.isfinite(right)
        if np.any(finite):
            abs_difference = np.abs(left[finite] - right[finite])
            scale = np.maximum(
                np.maximum(np.abs(left[finite]), np.abs(right[finite])),
                EPSILON,
            )
            max_abs = float(np.max(abs_difference))
            max_rel = float(np.max(abs_difference / scale))
        else:
            max_abs = float("nan")
            max_rel = float("nan")
        result[f"{key}_finite_mask_match"] = bool(finite_match)
        result[f"{key}_max_abs_difference"] = max_abs
        result[f"{key}_max_rel_difference"] = max_rel
    result["pass"] = bool(
        result["vector_h1_seminorm_rel_finite_mask_match"]
        and result["strain_energy_density_rel_finite_mask_match"]
        and float(result["vector_h1_seminorm_rel_max_abs_difference"]) <= 1e-9
        and float(result["vector_h1_seminorm_rel_max_rel_difference"]) <= 1e-8
        and float(result["strain_energy_density_rel_max_abs_difference"]) <= 1e-9
        and float(result["strain_energy_density_rel_max_rel_difference"]) <= 1e-8
    )
    return result


def run_metric_unit_tests(
    arrays: HyperelasticityArrays,
    sample_index: int,
) -> Dict[str, Any]:
    """Exercise expected metric invariances on one stored ground-truth field."""

    target = np.moveaxis(
        arrays.displacement[int(sample_index)].astype(np.float64),
        -1,
        0,
    )[None, ...]
    traction_y = arrays.traction[int(sample_index)].astype(np.float64)
    traction = np.repeat(
        traction_y[None, None, :, None],
        NATIVE_NX,
        axis=3,
    )
    x, y = physical_coordinates(NATIVE_NX, NATIVE_NY)
    yy, xx = np.meshgrid(y, x, indexing="ij")

    cases: Dict[str, np.ndarray] = {
        "exact": target.copy(),
        "constant_shift": target + np.asarray([0.01, -0.01])[None, :, None, None],
        "scaled_1p02": target * 1.02,
        "local_y_gaussian": target.copy(),
        "left_boundary_ux": target.copy(),
    }
    bump = 0.005 * np.exp(
        -((xx - 2.0) ** 2 / 0.16 + (yy - 0.5) ** 2 / 0.04)
    )
    cases["local_y_gaussian"][0, 1] += bump
    cases["left_boundary_ux"][0, 0, :, 0] += 1.0e-4

    computed = {
        name: compute_physical_metrics(value, target, traction)
        for name, value in cases.items()
    }
    exact = computed["exact"]
    constant = computed["constant_shift"]
    scaled = computed["scaled_1p02"]
    local = computed["local_y_gaussian"]
    left = computed["left_boundary_ux"]
    checks = {
        "exact_zero": bool(
            abs(float(exact["joint_rel_l2"][0])) < 1e-12
            and abs(float(exact["vector_h1_seminorm_rel"][0])) < 1e-12
            and abs(float(exact["strain_energy_density_rel"][0])) < 1e-12
        ),
        "translation_gradient_invariance": bool(
            float(constant["vector_h1_seminorm_rel"][0]) < 1e-12
            and float(constant["strain_energy_density_rel"][0]) < 1e-11
        ),
        "translation_l2_and_clamp_sensitive": bool(
            float(constant["joint_rel_l2"][0]) > 0.0
            and float(constant["clamped_boundary_rms"][0]) > 0.0
        ),
        "scale_detected": bool(
            float(scaled["joint_rel_l2"][0]) > 0.0
            and float(scaled["vector_h1_seminorm_rel"][0]) > 0.0
            and float(scaled["strain_energy_density_rel"][0]) > 0.0
        ),
        "local_perturbation_detected": bool(
            float(local["joint_rel_l2"][0]) > 0.0
            and float(local["vector_h1_seminorm_rel"][0]) > 0.0
            and float(local["strain_energy_density_rel"][0]) > 0.0
        ),
        "left_boundary_detected": bool(
            float(left["clamped_boundary_rms"][0])
            > float(exact["clamped_boundary_rms"][0])
        ),
    }
    independent = compute_physical_metrics(
        cases["scaled_1p02"],
        target,
        traction,
        implementation="independent",
    )
    independent_check = compare_metric_implementations(scaled, independent)
    return {
        "sample_index": int(sample_index),
        "checks": checks,
        "independent_check": independent_check,
        "pass": bool(all(checks.values()) and independent_check["pass"]),
    }


def endpoint_and_identity_qc(
    arrays: HyperelasticityArrays,
    split: LockedSplit,
    normalizer: NormalizerStats,
    *,
    nx: int = 128,
    ny: int = 32,
) -> Dict[str, Any]:
    """Check all approved identities and endpoint-preserving interpolation."""

    split_sets = [set(split.train), set(split.validation), set(split.test)]
    identity_pass = bool(
        not (split_sets[0] & split_sets[1])
        and not (split_sets[0] & split_sets[2])
        and not (split_sets[1] & split_sets[2])
        and set(split.all_indices()) == set(range(550))
    )

    full_indices = split.all_indices()
    coarse = HyperelasticityGridDataset(
        arrays,
        full_indices,
        normalizer,
        nx=nx,
        ny=ny,
    )
    native = HyperelasticityGridDataset(
        arrays,
        split.test,
        normalizer,
        nx=NATIVE_NX,
        ny=NATIVE_NY,
    )

    max_traction_endpoint_difference = 0.0
    max_displacement_corner_difference = 0.0
    max_resampled_clamp = 0.0
    finite_pass = True
    for local, sample_index in enumerate(full_indices):
        traction, displacement, returned_index = coarse.physical_sample(local)
        if returned_index != sample_index:
            identity_pass = False
        original_t = arrays.traction[sample_index]
        original_u = arrays.displacement[sample_index]
        max_traction_endpoint_difference = max(
            max_traction_endpoint_difference,
            abs(float(traction[0, 0, 0]) - float(original_t[0])),
            abs(float(traction[0, -1, 0]) - float(original_t[-1])),
        )
        corners = (
            (0, 0),
            (0, -1),
            (-1, 0),
            (-1, -1),
        )
        for cy, cx in corners:
            coarse_corner = displacement[:, cy, cx].numpy()
            native_corner = original_u[cy, cx, :]
            max_displacement_corner_difference = max(
                max_displacement_corner_difference,
                float(np.max(np.abs(coarse_corner - native_corner))),
            )
        max_resampled_clamp = max(
            max_resampled_clamp,
            float(torch.max(torch.abs(displacement[:, :, 0])).item()),
        )
        finite_pass = bool(
            finite_pass
            and torch.isfinite(traction).all()
            and torch.isfinite(displacement).all()
        )

    native_direct_max_difference = 0.0
    for local, sample_index in enumerate(split.test):
        traction, displacement, returned_index = native.physical_sample(local)
        if returned_index != sample_index:
            identity_pass = False
        # The export contract is float32. Compare against a direct float32
        # cast of the stored native arrays so this check detects interpolation
        # or reordering without treating the declared dtype conversion as an
        # interpolation error.
        direct_t = np.repeat(
            arrays.traction[sample_index].astype(np.float32)[:, None],
            NATIVE_NX,
            axis=1,
        )
        direct_u = np.moveaxis(
            arrays.displacement[sample_index].astype(np.float32),
            -1,
            0,
        )
        native_direct_max_difference = max(
            native_direct_max_difference,
            float(np.max(np.abs(traction.numpy()[0] - direct_t))),
            float(np.max(np.abs(displacement.numpy() - direct_u))),
        )

    x, y = physical_coordinates(nx, ny)
    coordinate_pass = bool(
        x[0] == 0.0
        and x[-1] == DOMAIN_LENGTH
        and y[0] == 0.0
        and y[-1] == DOMAIN_WIDTH
    )
    result = {
        "identity_disjoint_and_complete": identity_pass,
        "coordinate_endpoints_exact": coordinate_pass,
        "max_traction_endpoint_difference": max_traction_endpoint_difference,
        "max_displacement_corner_difference": max_displacement_corner_difference,
        "max_resampled_target_clamp_abs": max_resampled_clamp,
        "native_dataset_direct_flag": bool(native.native_direct),
        "native_direct_max_difference": native_direct_max_difference,
        "all_checked_fields_finite": finite_pass,
    }
    result["pass"] = bool(
        identity_pass
        and coordinate_pass
        and max_traction_endpoint_difference <= 1e-6
        and max_displacement_corner_difference <= 1e-6
        and max_resampled_clamp <= 1e-6
        and native.native_direct
        and native_direct_max_difference == 0.0
        and finite_pass
    )
    return result


def training_mean_field_at_grid(
    arrays: HyperelasticityArrays,
    train_indices: Sequence[int],
    *,
    nx: int,
    ny: int,
) -> np.ndarray:
    """Return the train-only spatial mean displacement as [2,Ny,Nx]."""

    mean_native = np.mean(
        arrays.displacement[np.asarray(train_indices, dtype=np.int64)],
        axis=0,
        dtype=np.float64,
    )
    mean_chw = torch.from_numpy(
        np.moveaxis(mean_native.astype(np.float32), -1, 0)
    )
    mean_grid = resample_displacement_nodal(mean_chw, nx, ny).numpy()
    mean_grid[:, :, 0] = 0.0
    return mean_grid.astype(np.float32)
