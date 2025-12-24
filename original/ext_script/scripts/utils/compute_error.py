#!/usr/bin/env python3
import argparse
import math
import os

import numpy as np
import pandas as pd
import torch


JOINT_COLS = [f"q{i}" for i in range(1, 8)]
POSE_COLS = ["x", "y", "z", "qw", "qx", "qy", "qz"]
ALL_COLS = JOINT_COLS + POSE_COLS


def std_from_csv(path: str) -> pd.Series:
    print(f"[INFO] Loading CSV: {path}")
    df = pd.read_csv(path)

    available = [c for c in ALL_COLS if c in df.columns]
    if not available:
        raise ValueError(
            f"No expected columns (q1..q7, x,y,z,qw,qx,qy,qz) found in CSV '{path}'."
        )

    df = df[available]
    # population std (ddof=0), consistent with training
    stds = df.std(ddof=0)
    return stds


def std_from_pt_file(path: str) -> pd.Series:
    print(f"[INFO] Loading tensor file: {path}")
    data = torch.load(path, map_location="cpu")

    if isinstance(data, torch.Tensor):
        arr = data.numpy()
    elif isinstance(data, dict) and "data" in data:
        arr = data["data"].numpy()
    else:
        raise ValueError(
            f"Unexpected format in '{path}'. Expected a Tensor or dict with key 'data'."
        )

    if arr.ndim != 2:
        raise ValueError(f"Tensor in '{path}' must be 2D, got shape {arr.shape}.")

    n_rows, n_cols = arr.shape
    print(f"[INFO] Tensor shape: {arr.shape}")

    if n_cols == len(ALL_COLS):
        cols = ALL_COLS
    else:
        # Fall back to generic col names
        cols = [f"c{i}" for i in range(n_cols)]
        print(f"[WARN] Non-standard column count ({n_cols}); using generic names.")

    df = pd.DataFrame(arr, columns=cols)
    available = [c for c in ALL_COLS if c in df.columns]
    if not available:
        print("[WARN] No standard kinematics columns found; computing std for all.")
        stds = df.std(ddof=0)
    else:
        stds = df[available].std(ddof=0)
    return stds


def std_from_pt_directory(path: str) -> pd.Series:
    """
    Compute per-column std from a directory of .pt/.bin shards
    using streaming sums, so we never load all shards at once.
    Assumes standard ordering: [q1..q7, x, y, z, qw, qx, qy, qz].
    """
    shard_files = [
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.endswith(".pt") or f.endswith(".bin")
    ]
    shard_files = sorted(shard_files)

    if not shard_files:
        raise ValueError(f"No .pt/.bin files found in directory: {path}")

    sum_x = None
    sum_x2 = None
    n_total = 0
    n_cols = None

    for sp in shard_files:
        print(f"[INFO] Loading shard: {sp}")
        data = torch.load(sp, map_location="cpu")

        if isinstance(data, torch.Tensor):
            arr = data.numpy()
        elif isinstance(data, dict) and "data" in data:
            arr = data["data"].numpy()
        else:
            raise ValueError(
                f"Unexpected format in shard '{sp}'. "
                f"Expected a Tensor or dict with key 'data'."
            )

        if arr.ndim != 2:
            raise ValueError(f"Tensor in '{sp}' must be 2D, got shape {arr.shape}.")

        if n_cols is None:
            n_cols = arr.shape[1]
            sum_x = np.zeros(n_cols, dtype=np.float64)
            sum_x2 = np.zeros(n_cols, dtype=np.float64)
        elif arr.shape[1] != n_cols:
            raise ValueError(
                f"Shard '{sp}' has different column count ({arr.shape[1]}) "
                f"from previous shards ({n_cols})."
            )

        sum_x += arr.sum(axis=0, dtype=np.float64)
        sum_x2 += np.square(arr, dtype=np.float64).sum(axis=0)
        n_total += arr.shape[0]

    print(f"[INFO] Total rows across shards: {n_total}")
    if n_total == 0:
        raise ValueError("No rows found across shards.")

    mean = sum_x / n_total
    var = sum_x2 / n_total - mean ** 2
    var = np.maximum(var, 0.0)  # avoid tiny negative from numerical error
    std = np.sqrt(var)

    if n_cols == len(ALL_COLS):
        cols = ALL_COLS
    else:
        cols = [f"c{i}" for i in range(n_cols)]
        print(f"[WARN] Non-standard column count ({n_cols}); using generic names.")

    stds = pd.Series(std, index=cols)
    return stds


def load_std(path: str) -> pd.Series:
    """
    Dispatch to CSV / .pt / directory-of-.pt loaders.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")

    if os.path.isdir(path):
        return std_from_pt_directory(path)

    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return std_from_csv(path)
    elif ext in [".pt", ".bin"]:
        return std_from_pt_file(path)
    else:
        raise ValueError(
            f"Unsupported file extension '{ext}'. Use .csv, .pt, .bin or a directory."
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute approximate FK/IK errors in meters/degrees "
            "from normalized validation MSE and dataset statistics.\n"
            "Dataset can be a CSV, a single .pt/.bin file, or a directory of .pt/.bin shards."
        )
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to dataset (CSV, .pt/.bin, or directory of .pt/.bin shards).",
    )
    parser.add_argument(
        "--fk_loss",
        type=float,
        default=None,
        help=(
            "Validation MSE for FK (normalized pose: [x,y,z,qw,qx,qy,qz]). "
            "If None, FK error is not computed."
        ),
    )
    parser.add_argument(
        "--ik_loss",
        type=float,
        default=None,
        help=(
            "Validation MSE for IK (normalized joints: q1..q7). "
            "If None, IK error is not computed."
        ),
    )
    args = parser.parse_args()

    # --- load stds ---
    stds = load_std(args.path)

    print("\n=== Column standard deviations (dataset scale) ===")
    for c in ALL_COLS:
        if c in stds:
            print(f"{c:4s}  std = {stds[c]: .6f}")

    # ---------------- FK error in meters ----------------
    if args.fk_loss is not None:
        missing_pos = [c for c in ["x", "y", "z"] if c not in stds.index]
        if missing_pos:
            print(
                "\n[FK] Cannot compute FK error: missing position columns:",
                ", ".join(missing_pos),
            )
        else:
            fk_loss = args.fk_loss
            e_norm = math.sqrt(fk_loss)

            sx = stds["x"]
            sy = stds["y"]
            sz = stds["z"]

            ex = e_norm * sx
            ey = e_norm * sy
            ez = e_norm * sz

            e_3d = math.sqrt(ex ** 2 + ey ** 2 + ez ** 2)

            print("\n=== FK error estimate (from normalized FK loss) ===")
            print(f"FK val MSE (normalized): {fk_loss:.6f}")
            print(f"RMS error x: {ex:.4f} m")
            print(f"RMS error y: {ey:.4f} m")
            print(f"RMS error z: {ez:.4f} m")
            print(f"RMS 3D position error: {e_3d:.4f} m")

    # ---------------- IK error in degrees ----------------
    if args.ik_loss is not None:
        missing_joints = [c for c in JOINT_COLS if c not in stds.index]
        if missing_joints:
            print(
                "\n[IK] Cannot compute IK error: missing joint columns:",
                ", ".join(missing_joints),
            )
        else:
            ik_loss = args.ik_loss
            e_norm = math.sqrt(ik_loss)

            print("\n=== IK error estimate (from normalized IK loss) ===")
            print(f"IK val MSE (normalized): {ik_loss:.6f}")

            joint_errors_deg = []
            for j in JOINT_COLS:
                sigma_rad = stds[j]
                e_rad = e_norm * sigma_rad
                e_deg = e_rad * 180.0 / math.pi
                joint_errors_deg.append(e_deg)
                print(f"RMS error {j}: {e_deg:.3f} deg")

            avg_deg = sum(joint_errors_deg) / len(joint_errors_deg)
            print(f"Average RMS joint error: {avg_deg:.3f} deg")


if __name__ == "__main__":
    main()
