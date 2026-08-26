from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


BOOTSTRAP_METRICS = (
    "joint_rel_l2", "ux_rel_l2", "uy_rel_l2", "vector_h1_seminorm_rel",
    "deformation_gradient_rel", "strain_energy_density_rel",
    "sampled_total_potential_rel", "min_det_f", "nonpositive_j_fraction",
)


def f(value: Any) -> float:
    return float(value) if value not in (None, "") else float("nan")


def bootstrap_mean(values: np.ndarray, replicates: int, seed: int) -> Tuple[float, float, int]:
    finite = np.isfinite(values); values = values[finite]
    if not values.size: return float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(replicates, values.size), endpoint=False)
    means = values[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)), int(values.size)


def bootstrap_primary(per_sample: Sequence[Mapping[str, Any]], replicates: int = 10000) -> List[Dict[str, Any]]:
    """Production paired bootstrap with a deterministic base seed plus counter."""
    groups: Dict[Tuple[str, int, int, int, int, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in per_sample:
        key = (str(row["architecture"]), int(row["train_nx"]), int(row["train_ny"]), int(row["eval_nx"]), int(row["eval_ny"]), int(row["sample_index"]))
        groups[key].append(row)
    output: List[Dict[str, Any]] = []; counter = 0
    for configuration in sorted({key[:5] for key in groups}):
        matching = sorted((key, rows) for key, rows in groups.items() if key[:5] == configuration)
        if len(matching) != 75 or any(len(rows) != 3 for _, rows in matching): raise RuntimeError(f"Paired seed/sample structure failed for {configuration}")
        for metric in BOOTSTRAP_METRICS:
            values = np.asarray([np.mean([f(row[metric]) for row in rows]) if all(np.isfinite(f(row[metric])) for row in rows) else float("nan") for _, rows in matching], dtype=np.float64)
            actual_seed = 20260801 + counter
            lo, hi, count = bootstrap_mean(values, replicates, actual_seed); counter += 1
            architecture, train_nx, train_ny, eval_nx, eval_ny = configuration
            output.append({"architecture": architecture, "train_nx": train_nx, "train_ny": train_ny, "eval_nx": eval_nx, "eval_ny": eval_ny, "metric": metric, "paired_sample_seed_averaged_mean": float(np.nanmean(values)) if count else float("nan"), "bootstrap_ci95_low": lo, "bootstrap_ci95_high": hi, "conditioning_sample_count": count, "bootstrap_replicates": replicates, "bootstrap_seed_base": 20260801, "bootstrap_seed_actual": actual_seed, "method": "resample paired sample identities after averaging the three training runs"})
    return output
