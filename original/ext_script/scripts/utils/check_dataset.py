#!/usr/bin/env python3
import argparse
import os
from typing import Tuple, Union

import numpy as np
import torch


def load_any(path: str) -> torch.Tensor:
    """
    Load a single .pt file and return a 2D float32 tensor.
    Tries to handle a few common formats:
      - Tensor
      - dict with 'data', 'X', 'tensor', 'arr'
      - pandas DataFrame (if accidentally saved)
    Raises ValueError if it cannot interpret the file as a dataset.
    """
    obj = torch.load(path, map_location="cpu")
    print(f"  [INFO] raw type from torch.load: {type(obj)}")

    # 1) Direct tensor
    if isinstance(obj, torch.Tensor):
        t = obj

    # 2) dict-style checkpoint / dataset
    elif isinstance(obj, dict):
        # Try common keys in a sensible order
        for key in ["data", "X", "tensor", "arr", "dataset"]:
            if key in obj:
                print(f"  [INFO] using dict key '{key}' as dataset tensor")
                t = obj[key]
                break
        else:
            raise ValueError(
                f"{path}: dict does not contain a dataset-like key "
                f"(found keys: {list(obj.keys())})"
            )

        if not isinstance(t, torch.Tensor):
            # Could be numpy or pandas inside dict
            import pandas as pd

            if isinstance(t, np.ndarray):
                t = torch.from_numpy(t)
            elif isinstance(t, pd.DataFrame):
                print("  [WARN] value is DataFrame; converting .values to tensor")
                t = torch.from_numpy(t.values)
            else:
                raise ValueError(
                    f"{path}: value under dataset key is type {type(t)}, "
                    f"expected Tensor/ndarray/DataFrame."
                )

    else:
        # 3) Might be numpy / pandas directly
        import pandas as pd

        if isinstance(obj, np.ndarray):
            t = torch.from_numpy(obj)
        elif isinstance(obj, pd.DataFrame):
            print("  [WARN] object is DataFrame; converting .values to tensor")
            t = torch.from_numpy(obj.values)
        else:
            raise ValueError(
                f"{path}: unsupported top-level type {type(obj)}. "
                f"Expected Tensor, dict, ndarray, or DataFrame."
            )

    if t.ndim != 2:
        raise ValueError(f"{path}: tensor must be 2D [N, C], got shape {tuple(t.shape)}")

    # Ensure float32
    if not torch.is_floating_point(t):
        print(f"  [INFO] converting dtype {t.dtype} -> float32")
        t = t.float()
    else:
        t = t.to(torch.float32)

    # Check for NaN/Inf
    if not torch.isfinite(t).all():
        bad = (~torch.isfinite(t)).sum().item()
        print(f"  [WARN] tensor contains {bad} NaN/Inf entries; cleaning to 0")
        t[~torch.isfinite(t)] = 0.0

    return t


def process_file(path: str, save_fixed: bool, suffix: str) -> Tuple[int, int]:
    """
    Check one .pt file and optionally overwrite with a clean format:
        {'data': <2D float32 tensor>}
    Returns (num_rows, num_cols).
    """
    print(f"\n[CHECK] {path}")
    t = load_any(path)
    n_rows, n_cols = t.shape
    print(f"  [OK] shape = {t.shape}, dtype = {t.dtype}")

    if save_fixed:
        # Re-save as a simple dict with key 'data'
        base, ext = os.path.splitext(path)
        out_path = base + suffix + ext if suffix else path
        torch.save({"data": t}, out_path)
        print(f"  [SAVE] cleaned dataset -> {out_path}")

    return n_rows, n_cols


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Check and normalise .pt kinematics datasets.\n"
            "Ensures each file is a 2D float32 tensor and optionally rewrites "
            "it as {'data': tensor}."
        )
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Single .pt file or directory of .pt files.",
    )
    parser.add_argument(
        "--save_fixed",
        action="store_true",
        help="If set, re-save each file in a normalised format.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help=(
            "Suffix added before .pt when saving fixed files. "
            "Default '' overwrites in place. Example: '_fixed'."
        ),
    )
    args = parser.parse_args()

    path = args.path
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    total_rows = 0
    cols = None

    if os.path.isdir(path):
        files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.endswith(".pt") or f.endswith(".bin")
        ]
        files = sorted(files)
        if not files:
            raise ValueError(f"No .pt/.bin files found in directory {path}")

        for f in files:
            n_rows, n_cols = process_file(f, args.save_fixed, args.suffix)
            total_rows += n_rows
            if cols is None:
                cols = n_cols
            elif cols != n_cols:
                print(
                    f"  [WARN] {f} has {n_cols} columns; "
                    f"previous files had {cols} columns."
                )
    else:
        n_rows, n_cols = process_file(path, args.save_fixed, args.suffix)
        total_rows = n_rows
        cols = n_cols

    print("\n=== SUMMARY ===")
    print(f"Total rows:    {total_rows}")
    print(f"Columns/file:  {cols}")


if __name__ == "__main__":
    main()
