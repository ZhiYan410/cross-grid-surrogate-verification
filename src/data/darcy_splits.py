from __future__ import annotations



























from typing import Any, Dict, List
import numpy as np

from data.io import load_json, save_json


SplitDict = Dict[str, Any]


def make_split_indices(
    n_total: int,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
) -> SplitDict:







































    if n_total <= 0:
        raise ValueError(f"n_total must be positive, but got {n_total}")

    if n_train < 0 or n_val < 0 or n_test < 0:
        raise ValueError(
            f"n_train, n_val, n_test must all be non-negative, "
            f"but got n_train={n_train}, n_val={n_val}, n_test={n_test}"
        )

    if n_train + n_val + n_test != n_total:
        raise ValueError(
            "Split counts do not sum to n_total: "
            f"n_train({n_train}) + n_val({n_val}) + n_test({n_test}) != n_total({n_total})"
        )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total).tolist()

    train = perm[:n_train]
    val = perm[n_train : n_train + n_val]
    test = perm[n_train + n_val :]

    split: SplitDict = {
        "meta": {
            "seed": int(seed),
            "n_total": int(n_total),
            "n_train": int(n_train),
            "n_val": int(n_val),
            "n_test": int(n_test),
        },
        "train": train,
        "val": val,
        "test": test,
    }

    
    validate_split_dict(split)

    return split


def save_split_json(split: SplitDict, path: str) -> None:








    validate_split_dict(split)
    save_json(split, path)


def load_split_json(path: str) -> SplitDict:








    split = load_json(path)
    validate_split_dict(split)
    return split


def get_split_counts(split: SplitDict) -> Dict[str, int]:


















    validate_split_dict(split)

    n_train = len(split["train"])
    n_val = len(split["val"])
    n_test = len(split["test"])
    n_total = n_train + n_val + n_test

    return {
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "n_total": n_total,
    }


def validate_split_dict(split: SplitDict) -> None:
















    if not isinstance(split, dict):
        raise TypeError(f"split must be a dict, but got {type(split)}")

    required_keys = ["meta", "train", "val", "test"]
    for key in required_keys:
        if key not in split:
            raise KeyError(f"split is missing required key: '{key}'")

    meta = split["meta"]
    train = split["train"]
    val = split["val"]
    test = split["test"]

    if not isinstance(meta, dict):
        raise TypeError(f"split['meta'] must be a dict, but got {type(meta)}")

    for name, part in [("train", train), ("val", val), ("test", test)]:
        if not isinstance(part, list):
            raise TypeError(f"split['{name}'] must be a list, but got {type(part)}")

        for i, idx in enumerate(part):
            if not isinstance(idx, int):
                raise TypeError(
                    f"split['{name}'][{i}] must be int, but got {type(idx)} with value={idx}"
                )

    train_set = set(train)
    val_set = set(val)
    test_set = set(test)

    
    if len(train_set) != len(train):
        raise ValueError("Duplicate indices found inside split['train']")
    if len(val_set) != len(val):
        raise ValueError("Duplicate indices found inside split['val']")
    if len(test_set) != len(test):
        raise ValueError("Duplicate indices found inside split['test']")

    
    if len(train_set & val_set) > 0:
        raise ValueError("split['train'] and split['val'] overlap")
    if len(train_set & test_set) > 0:
        raise ValueError("split['train'] and split['test'] overlap")
    if len(val_set & test_set) > 0:
        raise ValueError("split['val'] and split['test'] overlap")

    
    n_train = len(train)
    n_val = len(val)
    n_test = len(test)
    n_total = n_train + n_val + n_test

    if "n_train" in meta and int(meta["n_train"]) != n_train:
        raise ValueError(
            f"meta['n_train']={meta['n_train']} but actual len(train)={n_train}"
        )
    if "n_val" in meta and int(meta["n_val"]) != n_val:
        raise ValueError(
            f"meta['n_val']={meta['n_val']} but actual len(val)={n_val}"
        )
    if "n_test" in meta and int(meta["n_test"]) != n_test:
        raise ValueError(
            f"meta['n_test']={meta['n_test']} but actual len(test)={n_test}"
        )
    if "n_total" in meta and int(meta["n_total"]) != n_total:
        raise ValueError(
            f"meta['n_total']={meta['n_total']} but actual total={n_total}"
        )

    
    if "seed" in meta:
        try:
            int(meta["seed"])
        except Exception as e:
            raise ValueError(f"meta['seed'] must be int-like, but got {meta['seed']}") from e
