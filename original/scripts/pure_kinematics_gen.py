#!/usr/bin/env python3
"""
generate_iiwa14_grid_fk_pin_mp.py

Multi-process FK-only grid dataset generator for a robot URDF using Pinocchio.

Each row in the CSV:
    q1..qN, x, y, z, qw, qx, qy, qz
"""

import argparse
import csv
import math
from typing import List, Tuple

import numpy as np
import pinocchio as pin
from multiprocessing import Pool, cpu_count


# ============================================================
# Worker globals
# ============================================================

_urdf_path = None
_ee_frame_name = None
_num_joints = None
_joint_grids: List[np.ndarray] = []
_grid_sizes: List[int] = []

_model: pin.Model = None
_data: pin.Data = None
_ee_frame_id: int = None
_num_cols: int = None


# ============================================================
# Utility functions
# ============================================================

def index_to_multi(index: int, grid_sizes: List[int]) -> List[int]:
    """
    Convert flat index -> per-joint indices for a Cartesian grid.
    """
    idx_list = []
    for size in reversed(grid_sizes):
        idx_list.append(index % size)
        index //= size
    idx_list.reverse()
    return idx_list


def build_joint_grids_from_model(
    model: pin.Model,
    num_joints: int,
    step_deg: float,
) -> Tuple[List[np.ndarray], List[int], int]:
    """
    Build a list of arrays of joint values for each joint, based on
    the model's position limits and the given step in degrees.
    """
    step_rad = math.radians(step_deg)

    lower = model.lowerPositionLimit[:num_joints]
    upper = model.upperPositionLimit[:num_joints]

    joint_grids: List[np.ndarray] = []
    grid_sizes: List[int] = []

    for j in range(num_joints):
        lo = float(lower[j])
        hi = float(upper[j])

        # Guard against insane limits in some URDFs
        if not np.isfinite(lo) or abs(lo) > 1e4:
            lo = -math.pi
        if not np.isfinite(hi) or abs(hi) > 1e4:
            hi = math.pi

        num_steps = int(math.floor((hi - lo) / step_rad)) + 1
        values = lo + np.arange(num_steps, dtype=np.float32) * step_rad

        joint_grids.append(values)
        grid_sizes.append(len(values))

        print(
            f"[INFO] Joint {j+1}: range [{lo:.3f}, {hi:.3f}] rad, "
            f"step={step_rad:.3f}, points={len(values)}"
        )

    total_combinations = 1
    for s in grid_sizes:
        total_combinations *= s

    print(f"[INFO] Theoretical total combinations: {total_combinations}")
    return joint_grids, grid_sizes, total_combinations


# ============================================================
# Worker init and chunk processing
# ============================================================

def init_worker(
    urdf_path: str,
    ee_frame_name: str,
    num_joints: int,
    joint_grids: List[np.ndarray],
    grid_sizes: List[int],
):
    """
    Initializer for each worker process.
    Builds its own Pinocchio model/data and stores shared info in globals.
    """
    global _urdf_path, _ee_frame_name, _num_joints
    global _joint_grids, _grid_sizes
    global _model, _data, _ee_frame_id, _num_cols

    _urdf_path = urdf_path
    _ee_frame_name = ee_frame_name
    _num_joints = num_joints
    _joint_grids = joint_grids
    _grid_sizes = grid_sizes

    # Build model & data inside the worker
    _model = pin.buildModelFromUrdf(_urdf_path)
    _data = _model.createData()

    _ee_frame_id = _model.getFrameId(_ee_frame_name)
    _num_cols = _num_joints + 3 + 4  # q1..qN, x y z, qw qx qy qz

    print(f"[WORKER] Initialized with URDF={_urdf_path}, EE='{_ee_frame_name}', "
          f"num_joints={_num_joints}, ee_frame_id={_ee_frame_id}")


def process_chunk(chunk_range):
    """
    Worker function: process a range of global indices [start, end).

    Returns:
        np.ndarray of shape (num_rows, num_cols), dtype float32
    """
    start, end = chunk_range
    rows = []

    for global_idx in range(start, end):
        idx_list = index_to_multi(global_idx, _grid_sizes)
        q_vals = [_joint_grids[j][idx_list[j]] for j in range(_num_joints)]
        q = np.array(q_vals, dtype=float)  # float64 for Pinocchio

        # Forward kinematics to EE frame
        pin.forwardKinematics(_model, _data, q)
        pin.updateFramePlacements(_model, _data)
        oMf = _data.oMf[_ee_frame_id]

        pos = np.array(oMf.translation).reshape(3)
        R = np.array(oMf.rotation)

        # Quaternion from rotation matrix (xyzw -> wxyz)
        quat_xyzw = pin.Quaternion(R).coeffs()
        quat_wxyz = np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
            dtype=np.float32,
        )

        row = np.concatenate(
            [q.astype(np.float32), pos.astype(np.float32), quat_wxyz]
        )
        rows.append(row)

    if not rows:
        return np.empty((0, _num_cols), dtype=np.float32)

    return np.vstack(rows)


# ============================================================
# Main generation function
# ============================================================

def generate_grid_dataset_fk_mp(
    urdf_path: str,
    ee_frame_name: str,
    step_deg: float,
    max_samples: int,
    output_csv: str,
    num_joints: int,
    num_workers: int,
    chunk_size: int,
):
    print(f"[INFO] Loading URDF (master model) from: {urdf_path}")
    model = pin.buildModelFromUrdf(urdf_path)

    if num_joints > model.nq:
        raise ValueError(
            f"Requested num_joints={num_joints}, but model.nq={model.nq}. "
            f"Decrease num_joints or use a URDF with more joints."
        )

    try:
        ee_frame_id = model.getFrameId(ee_frame_name)
    except Exception as e:
        raise ValueError(
            f"EE frame '{ee_frame_name}' not found in URDF. "
            f"Available frames: {[f.name for f in model.frames]}"
        ) from e

    print(f"[INFO] Using frame '{ee_frame_name}' (id {ee_frame_id}) as EE.")

    # Build grids and sizes using the master model's limits
    joint_grids, grid_sizes, total_combinations = build_joint_grids_from_model(
        model=model,
        num_joints=num_joints,
        step_deg=step_deg,
    )

    if max_samples is not None:
        max_global = min(max_samples, total_combinations)
        print(
            f"[INFO] max_samples={max_samples}, "
            f"effective combinations={max_global}"
        )
    else:
        max_global = total_combinations
        print(f"[INFO] No max_samples cap; using all {max_global} combinations.")

    if max_global <= 0:
        print("[WARN] No samples to generate (max_global <= 0). Exiting.")
        return

    # Determine workers
    if num_workers is None or num_workers <= 0:
        num_workers = cpu_count()
    print(f"[INFO] Using {num_workers} worker processes.")

    # Build chunk ranges
    chunks = []
    cur = 0
    while cur < max_global:
        chunks.append((cur, min(cur + chunk_size, max_global)))
        cur += chunk_size

    print(f"[INFO] Total chunks: {len(chunks)}, chunk_size={chunk_size}")

    # Prepare CSV header
    header = (
        [f"q{i+1}" for i in range(num_joints)]
        + ["x", "y", "z", "qw", "qx", "qy", "qz"]
    )

    # Open CSV and launch pool
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        with Pool(
            processes=num_workers,
            initializer=init_worker,
            initargs=(urdf_path, ee_frame_name, num_joints, joint_grids, grid_sizes),
        ) as pool:
            for i, chunk_rows in enumerate(pool.imap_unordered(process_chunk, chunks)):
                if chunk_rows.size == 0:
                    continue

                writer.writerows(chunk_rows.tolist())
                print(
                    f"[INFO] Finished chunk {i+1}/{len(chunks)}, "
                    f"rows={chunk_rows.shape[0]}"
                )

    print(f"[INFO] Done. Dataset saved to '{output_csv}'")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-process FK grid dataset using URDF + Pinocchio.",
    )
    parser.add_argument(
        "--urdf_path",
        type=str,
        required=True,
        help="Path to the robot URDF file.",
    )
    parser.add_argument(
        "--ee_frame",
        type=str,
        required=True,
        help="Name of the EE frame in the URDF (e.g. wrist link or gripper base).",
    )
    parser.add_argument(
        "--step_deg",
        type=float,
        default=5.0,
        help="Joint step (degrees) between grid samples.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional limit on number of samples (stops early if exceeded).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="iiwa_grid_fk_mp.csv",
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--num_joints",
        type=int,
        default=7,
        help="Number of arm joints from the URDF to use (e.g. 7 for iiwa14).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of worker processes (0 = use all CPU cores).",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10000,
        help="Number of samples per chunk per worker.",
    )

    args = parser.parse_args()

    generate_grid_dataset_fk_mp(
        urdf_path=args.urdf_path,
        ee_frame_name=args.ee_frame,
        step_deg=args.step_deg,
        max_samples=args.max_samples,
        output_csv=args.output_csv,
        num_joints=args.num_joints,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    # Needed on Windows for multiprocessing
    import multiprocessing as mp
    mp.freeze_support()
    main()
