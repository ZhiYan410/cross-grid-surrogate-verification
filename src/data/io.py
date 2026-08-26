from __future__ import annotations

"""Small, dependency-free JSON and CSV helpers for the public release."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


def ensure_dir(path: str | Path) -> Path:
    """Create *path* and its parents when needed, then return it."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _to_jsonable(obj: Any) -> Any:
    """Convert common array, tensor, and path values to JSON-safe values."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(key): _to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(value) for value in obj]
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
    except ImportError:
        pass
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            tensor = obj.detach().cpu()
            return tensor.item() if tensor.ndim == 0 else tensor.tolist()
    except ImportError:
        pass
    return obj


def load_json(path: str | Path) -> Dict[str, Any]:
    """Load a JSON object and reject non-object top-level values."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"JSON file not found: {source}")
    with source.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object (dict), got {type(data)} from {source}")
    return data


def save_json(obj: Dict[str, Any], path: str | Path) -> Path:
    """Write a JSON object with stable key ordering."""
    if not isinstance(obj, dict):
        raise TypeError(f"save_json expects a dict, got {type(obj)}")
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(obj), handle, indent=2, sort_keys=True, ensure_ascii=False)
    return target


def load_config_json(config_path: str | Path) -> Dict[str, Any]:
    """Load a configuration JSON object."""
    return load_json(config_path)


def save_result_json(result: Dict[str, Any], path: str | Path) -> Path:
    """Write a result JSON object."""
    return save_json(result, path)


def save_dict_rows_to_csv(rows: Sequence[Dict[str, Any]], path: str | Path) -> Path:
    """Write a non-empty sequence of homogeneous dictionary rows to CSV."""
    items = list(rows)
    if not items:
        raise ValueError("save_dict_rows_to_csv received no rows")
    if not isinstance(items[0], dict):
        raise TypeError("Each CSV row must be a dict")
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(items[0]))
        writer.writeheader()
        for row in items:
            if not isinstance(row, dict):
                raise TypeError("Each CSV row must be a dict")
            writer.writerow(row)
    return target


def load_csv_as_dict_rows(path: str | Path) -> List[Dict[str, str]]:
    """Load a CSV file without implicit type coercion."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"CSV file not found: {source}")
    with source.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def append_dict_row_to_csv(row: Dict[str, Any], path: str | Path) -> Path:
    """Append one dictionary row, creating the file and header when absent."""
    if not isinstance(row, dict):
        raise TypeError("row must be a dict")
    target = Path(path)
    ensure_dir(target.parent)
    existed = target.exists()
    with target.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not existed:
            writer.writeheader()
        writer.writerow(row)
    return target
