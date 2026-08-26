from __future__ import annotations





























from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from data.shapes import to_b1hw


class PDEBenchDarcyFlowDataset(Dataset):


















    def __init__(self, h5_path: str | Path) -> None:





















        self.h5_path = Path(h5_path).resolve()
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

        
        with h5py.File(self.h5_path, "r") as f:
            if "nu" not in f:
                raise KeyError(f"'nu' not found in HDF5 file: {self.h5_path}")
            if "tensor" not in f:
                raise KeyError(f"'tensor' not found in HDF5 file: {self.h5_path}")
            self.length = int(f["nu"].shape[0])

    def __len__(self) -> int:



        return self.length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:























        if not (0 <= index < self.length):
            raise IndexError(f"Index out of range: index={index}, length={self.length}")

        with h5py.File(self.h5_path, "r") as f:
            
            
            a_np = np.array(f["nu"][index], dtype=np.float32)

            
            
            u_np = np.array(f["tensor"][index, 0], dtype=np.float32)

            
            x_np = np.array(f["x-coordinate"], dtype=np.float32)
            y_np = np.array(f["y-coordinate"], dtype=np.float32)

        
        a = torch.from_numpy(a_np)
        u = torch.from_numpy(u_np)

        
        
        
        a = to_b1hw(a, name="a_single")[0]
        u = to_b1hw(u, name="u_single")[0]

        meta: Dict[str, Any] = {
            "index": int(index),
            "x": x_np,
            "y": y_np,
        }

        return a, u, meta


def collate_a_u_meta(
    batch: List[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:




























    if len(batch) == 0:
        raise ValueError("collate_a_u_meta received an empty batch")

    a_list: List[torch.Tensor] = []
    u_list: List[torch.Tensor] = []
    meta_list: List[Dict[str, Any]] = []

    for item in batch:
        if len(item) != 3:
            raise ValueError(
                "Each dataset item must be a tuple of (a, u, meta), "
                f"but got item with length {len(item)}"
            )
        a, u, meta = item
        a_list.append(a)
        u_list.append(u)
        meta_list.append(meta)

    a_batch = torch.stack(a_list, dim=0)  
    u_batch = torch.stack(u_list, dim=0)  

    return a_batch, u_batch, meta_list


def peek_dataset_shapes(dataset: Dataset, n: int = 3) -> List[Dict[str, Any]]:














    if n <= 0:
        raise ValueError(f"n must be positive, but got {n}")

    results: List[Dict[str, Any]] = []
    n = min(n, len(dataset))

    for i in range(n):
        a, u, meta = dataset[i]
        results.append(
            {
                "index": int(meta["index"]),
                "a_shape": tuple(a.shape),
                "u_shape": tuple(u.shape),
            }
        )

    return results
