from __future__ import annotations

"""Self-contained manufactured-solution verification for the Darcy stencil."""

import csv
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from metrics.darcy import operator_response


DEFAULT_GRIDS = (16, 32, 64, 128, 256, 512)
ASYMPTOTIC_MIN_N = 64
CROSSCHECK_TOL = 1.0e-12


def cell_centres(n: int) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Return a square cell-centred mesh on [0, 1] x [0, 1]."""
    if n < 4:
        raise ValueError("n must be at least 4")
    x = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    y = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return xx, yy, 1.0 / float(n), 1.0 / float(n)


def manufactured_field(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def coeff_constant(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    del y
    return np.ones_like(x, dtype=np.float64)


def exact_constant(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 2.0 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)


def coeff_variable(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 1.0 + 0.25 * np.cos(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)


def exact_variable_manual(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Evaluate -grad(a).grad(u)-a Laplacian(u) for the smooth Case B."""
    a = coeff_variable(x, y)
    du_dx = np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
    du_dy = np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)
    da_dx = -0.5 * np.pi * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
    da_dy = -0.5 * np.pi * np.cos(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)
    laplace_u = -2.0 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    return -(da_dx * du_dx + da_dy * du_dy) - a * laplace_u


def verify_variable_expression_with_sympy() -> Dict[str, Any]:
    """Independently verify the Case B analytical expression when SymPy is installed."""
    try:
        import sympy as sp
    except ImportError:
        return {"status": "NOT_RUN_NO_SYMPY", "symbolic_equal": None, "numeric_max_abs_difference": None}
    x, y = sp.symbols("x y", real=True)
    pi = sp.pi
    a = 1 + sp.Rational(1, 4) * sp.cos(2 * pi * x) * sp.cos(2 * pi * y)
    u = sp.sin(pi * x) * sp.sin(pi * y)
    derived = -(sp.diff(a, x) * sp.diff(u, x) + sp.diff(a, y) * sp.diff(u, y)) - a * (sp.diff(u, x, 2) + sp.diff(u, y, 2))
    manual = (sp.Rational(1, 2) * pi**2 * (sp.sin(2*pi*x)*sp.cos(2*pi*y)*sp.cos(pi*x)*sp.sin(pi*y) + sp.cos(2*pi*x)*sp.sin(2*pi*y)*sp.sin(pi*x)*sp.cos(pi*y)) + 2*pi**2*a*sp.sin(pi*x)*sp.sin(pi*y))
    symbolic_equal = bool(sp.trigsimp(sp.expand_trig(derived - manual)) == 0)
    derived_fn = sp.lambdify((x, y), derived, modules="numpy")
    gx = (np.arange(37, dtype=np.float64) + 0.5) / 37.0
    gy = (np.arange(41, dtype=np.float64) + 0.5) / 41.0
    xx, yy = np.meshgrid(gx, gy, indexing="xy")
    maximum = float(np.max(np.abs(np.asarray(derived_fn(xx, yy), dtype=np.float64) - exact_variable_manual(xx, yy))))
    return {"status": "PASS" if symbolic_equal and maximum <= 1.0e-12 else "FAIL", "symbolic_equal": symbolic_equal, "numeric_max_abs_difference": maximum, "sympy_version": sp.__version__}


def harmonic_scalar(left: float, right: float, eps: float = 1.0e-12) -> float:
    return 2.0 * left * right / max(left + right, eps)


def independent_scalar_operator(a: np.ndarray, u: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Independent scalar-loop implementation of the audited finite-difference stencil."""
    if a.shape != u.shape or a.ndim != 2:
        raise ValueError("a and u must be matching two-dimensional arrays")
    h, w = u.shape
    output = np.empty((h - 2, w - 2), dtype=np.float64)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            a_e = harmonic_scalar(float(a[i, j]), float(a[i, j + 1]))
            a_w = harmonic_scalar(float(a[i, j - 1]), float(a[i, j]))
            a_n = harmonic_scalar(float(a[i, j]), float(a[i + 1, j]))
            a_s = harmonic_scalar(float(a[i - 1, j]), float(a[i, j]))
            q_e = -a_e * (float(u[i, j + 1]) - float(u[i, j])) / dx
            q_w = -a_w * (float(u[i, j]) - float(u[i, j - 1])) / dx
            q_n = -a_n * (float(u[i + 1, j]) - float(u[i, j])) / dy
            q_s = -a_s * (float(u[i, j]) - float(u[i - 1, j])) / dy
            output[i - 1, j - 1] = (q_e - q_w) / dx + (q_n - q_s) / dy
    return output


def metric_values(discrete: np.ndarray, exact: np.ndarray, dx: float, dy: float) -> Dict[str, float]:
    error = discrete - exact
    absolute = np.abs(error)
    exact_l2 = math.sqrt(float(np.sum(exact**2) * dx * dy))
    absolute_l2 = math.sqrt(float(np.sum(error**2) * dx * dy))
    exact_linf = float(np.max(np.abs(exact)))
    return {"relative_l2_error": absolute_l2 / exact_l2, "absolute_l2_error": absolute_l2,
            "relative_linf_error": float(np.max(absolute)) / exact_linf,
            "maximum_pointwise_abs_error": float(np.max(absolute)), "mean_pointwise_abs_error": float(np.mean(absolute)),
            "exact_l2_norm": exact_l2, "exact_linf_norm": exact_linf}


def _add_orders(rows: List[Dict[str, Any]]) -> None:
    metrics = ("relative_l2_error", "absolute_l2_error", "relative_linf_error", "maximum_pointwise_abs_error", "mean_pointwise_abs_error")
    for case in ("A", "B"):
        prior = None
        for row in sorted((r for r in rows if r["case"] == case), key=lambda r: int(r["nx"])):
            for metric in metrics:
                key = f"observed_order_{metric}"
                row[key] = "" if prior is None else math.log(float(prior[metric]) / float(row[metric])) / math.log(float(row["nx"]) / float(prior["nx"]))
            prior = row


def _slope(rows: Iterable[Dict[str, Any]], metric: str) -> float:
    selected = [r for r in rows if int(r["nx"]) >= ASYMPTOTIC_MIN_N]
    h = np.asarray([float(r["dx"]) for r in selected], dtype=np.float64)
    error = np.asarray([float(r[metric]) for r in selected], dtype=np.float64)
    return float(np.polyfit(np.log(h), np.log(error), 1)[0])


def run_convergence(grids: Sequence[int] = DEFAULT_GRIDS, crosscheck_grid: int = 64) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Run both audited smooth cases and return all rows plus verification metadata."""
    grids = tuple(sorted(set(int(n) for n in grids)))
    if grids != DEFAULT_GRIDS:
        if len(grids) < 3 or any(fine != 2 * coarse for coarse, fine in zip(grids, grids[1:])):
            raise ValueError("grids must be at least three successive factor-two refinements")
    cases = (("A", "constant_coefficient", coeff_constant, exact_constant, "2*pi^2*sin(pi*x)*sin(pi*y)"),
             ("B", "smooth_variable_coefficient", coeff_variable, exact_variable_manual, "-grad(a).grad(u)-a*laplacian(u)"))
    rows: List[Dict[str, Any]] = []
    checks: Dict[str, Dict[str, Any]] = {}
    for key, label, coefficient, exact_response, exact_description in cases:
        for n in grids:
            x, y, dx, dy = cell_centres(n)
            a = np.asarray(coefficient(x, y), dtype=np.float64)
            u = np.asarray(manufactured_field(x, y), dtype=np.float64)
            exact = np.asarray(exact_response(x, y), dtype=np.float64)[1:-1, 1:-1]
            discrete = np.asarray(operator_response(a[None, ...], u[None, ...])[0], dtype=np.float64)
            row: Dict[str, Any] = {"case": key, "case_label": label, "nx": n, "ny": n, "dx": dx, "dy": dy,
                "coordinate_placement": "cell_centred_(i+0.5)/N", "output_h": discrete.shape[0], "output_w": discrete.shape[1],
                "expected_output_h": n - 2, "expected_output_w": n - 2, "n_interior_cells": int(discrete.size),
                "coefficient_min": float(a.min()), "coefficient_max": float(a.max()),
                "finite_check": bool(np.isfinite(a).all() and np.isfinite(u).all() and np.isfinite(exact).all() and np.isfinite(discrete).all()),
                "shape_check": bool(discrete.shape == (n - 2, n - 2) and exact.shape == (n - 2, n - 2)),
                "exact_response": exact_description, **metric_values(discrete, exact, dx, dy)}
            rows.append(row)
            if n == crosscheck_grid:
                independent = independent_scalar_operator(a, u, dx, dy)
                difference = independent - discrete
                maximum = float(np.max(np.abs(difference)))
                relative = float(np.linalg.norm(difference.ravel()) / np.linalg.norm(discrete.ravel()))
                checks[key] = {"grid": n, "max_abs_difference": maximum, "relative_l2_difference": relative,
                               "status": "PASS" if maximum <= CROSSCHECK_TOL and independent.shape == discrete.shape else "FAIL"}
    _add_orders(rows)
    for key in ("A", "B"):
        group = sorted((r for r in rows if r["case"] == key), key=lambda r: int(r["nx"]))
        l2 = _slope(group, "relative_l2_error")
        linf = _slope(group, "relative_linf_error")
        status = "PASS" if all(bool(r["finite_check"]) and bool(r["shape_check"]) for r in group) and np.all(np.diff([float(r["relative_l2_error"]) for r in group]) < 0) and np.all(np.diff([float(r["relative_linf_error"]) for r in group]) < 0) and 1.8 <= l2 <= 2.2 and 1.8 <= linf <= 2.2 else "FAIL"
        for row in group:
            row["monotonic_relative_l2_all_grids"] = True
            row["monotonic_relative_linf_all_grids"] = True
            row["asymptotic_fit_order_relative_l2"] = l2
            row["asymptotic_fit_order_relative_linf"] = linf
            row["case_status"] = status
            row["independent_crosscheck_grid"] = checks[key]["grid"]
            row["independent_crosscheck_max_abs_difference"] = checks[key]["max_abs_difference"]
            row["independent_crosscheck_relative_l2_difference"] = checks[key]["relative_l2_difference"]
            row["independent_crosscheck_status"] = checks[key]["status"]
    symbolic = verify_variable_expression_with_sympy()
    for row in rows:
        row["sympy_case_b_status"] = symbolic["status"]
        row["sympy_case_b_numeric_max_abs_difference"] = symbolic.get("numeric_max_abs_difference", "")
    checks["sympy"] = symbolic
    return rows, checks


def verify_locked_values(rows: Sequence[Dict[str, Any]], locked_csv: str | Path, rtol: float = 5.0e-9, atol: float = 5.0e-12) -> Dict[str, Any]:
    """Compare regenerated values with the immutable locked convergence CSV."""
    with Path(locked_csv).open(encoding="utf-8-sig", newline="") as handle:
        locked = list(csv.DictReader(handle))
    regenerated = {(str(row["case"]), int(row["nx"])): row for row in rows}
    if len(locked) != len(regenerated):
        return {"status": "FAIL", "reason": f"row count {len(regenerated)} != locked {len(locked)}"}
    numeric = ("relative_l2_error", "absolute_l2_error", "relative_linf_error", "maximum_pointwise_abs_error", "mean_pointwise_abs_error", "exact_l2_norm", "exact_linf_norm", "asymptotic_fit_order_relative_l2", "asymptotic_fit_order_relative_linf")
    for row in locked:
        key = (row["case"], int(row["nx"]))
        if key not in regenerated:
            return {"status": "FAIL", "reason": f"missing {key}"}
        actual = regenerated[key]
        for name in numeric:
            if not np.isclose(float(actual[name]), float(row[name]), rtol=rtol, atol=atol):
                return {"status": "FAIL", "reason": f"{key} {name}: {actual[name]} != {row[name]}"}
    return {"status": "PASS", "rows": len(locked), "rtol": rtol, "atol": atol}
