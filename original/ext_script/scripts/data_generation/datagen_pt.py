#!/usr/bin/env python3
"""
datagen.py

Fast FK dataset generator using:

- URDF (via urdfpy) to derive kinematics & joint limits
- PyTorch (CPU/GPU) to do batched forward kinematics

Two modes:

1) RANDOM mode (num_samples > 0):
   - num_samples > 0
   - Samples joint angles uniformly in [lower, upper]

2) GRID mode ("ALL" combinations):
   - num_samples <= 0 (default -1)
   - Uses joint limits + step_deg to build a Cartesian grid per joint
   - Generates ALL combinations in that grid in batches

Outputs (you can choose any combination):
- CSV file with columns:
    q1..qN, x, y, z, qw, qx, qy, qz
- Sharded .pt tensors:
    each shard has shape (M, 7+7) with columns [q1..q7, x, y, z, qw, qx, qy, qz]
    filenames: <output_pt_prefix>_part000.pt, _part001.pt, ...
"""

import argparse
import csv
import math
from collections import deque
from typing import List, Tuple, Optional

import numpy as np
import torch
from urdfpy import URDF
from tqdm import tqdm


# ============================================================
# 1. URDF → kinematic constants
# ============================================================

def _find_joint_chain_urdfpy(robot: URDF, root_link: str, tip_link: str) -> List[str]:
    """
    Find the ordered list of joint names from root_link → tip_link using urdfpy.
    Breadth-first search over links; collects joints along the path.
    """
    parent_to_joints = {}
    for j in robot.joints:
        parent_to_joints.setdefault(j.parent, []).append(j)

    q = deque([(root_link, [])])  # (current_link, list_of_joint_objs)
    visited = set()

    while q:
        link, chain = q.popleft()
        if link == tip_link:
            return [j.name for j in chain]

        if link in visited:
            continue
        visited.add(link)

        for j in parent_to_joints.get(link, []):
            q.append((j.child, chain + [j]))

    raise ValueError(f"No kinematic chain from '{root_link}' to '{tip_link}' in URDF.")


def build_urdf_kinematics(
    urdf_path: str,
    base_link: str,
    ee_link: str,
) -> Tuple[
    np.ndarray,  # T_parent_joint_zero: (J, 4, 4)
    np.ndarray,  # joint_axes: (J, 3)
    np.ndarray,  # lower_limits: (J,)
    np.ndarray,  # upper_limits: (J,)
    np.ndarray,  # T_lastlink_to_ee: (4, 4)
]:
    """
    Parse URDF with urdfpy, extract chain from base_link to ee_link.

    For each joint in the chain:
      - T_parent_joint_zero: parent link → joint frame at q=0
      - joint_axes: axis of rotation in joint frame
      - lower/upper_limits: joint limits (revolute), or defaults [-pi, +pi]

    Also computes:
      - T_lastlink_to_ee: transform from last chain link → ee_link at zero config
    """
    robot = URDF.load(urdf_path)

    joint_names = _find_joint_chain_urdfpy(robot, base_link, ee_link)
    J = len(joint_names)
    print(f"[INFO] Joint chain from '{base_link}' to '{ee_link}': {joint_names}")

    T_parent_joint_zero = np.zeros((J, 4, 4), dtype=np.float64)
    joint_axes = np.zeros((J, 3), dtype=np.float64)
    lower_limits = np.zeros((J,), dtype=np.float64)
    upper_limits = np.zeros((J,), dtype=np.float64)

    for idx, jn in enumerate(joint_names):
        joint = robot.joint_map[jn]

        # urdfpy: joint.origin is a 4x4 transform matrix
        if joint.origin is not None:
            T = np.array(joint.origin, dtype=np.float64)
        else:
            T = np.eye(4, dtype=np.float64)

        T_parent_joint_zero[idx] = T

        if joint.joint_type not in ["revolute", "continuous"]:
            print(f"[WARN] Joint '{jn}' is type '{joint.joint_type}', treating as fixed.")
            axis = np.array([0.0, 0.0, 0.0])
        else:
            axis = np.array(joint.axis, dtype=np.float64)
        joint_axes[idx] = axis

        if joint.limit is not None and joint.joint_type == "revolute":
            lower_limits[idx] = joint.limit.lower
            upper_limits[idx] = joint.limit.upper
        else:
            lower_limits[idx] = -math.pi
            upper_limits[idx] = +math.pi

        print(
            f"[INFO] Joint {idx}: name='{jn}', type='{joint.joint_type}', "
            f"axis={axis}, limit=[{lower_limits[idx]:.3f}, {upper_limits[idx]:.3f}]"
        )

    # FK at zero to get last_link → ee_link transform
    q_zero = {j.name: 0.0 for j in robot.joints}
    link_fk = robot.link_fk(cfg=q_zero)

    last_joint = robot.joint_map[joint_names[-1]]
    last_link_name = last_joint.child
    last_link = robot.link_map[last_link_name]
    ee_link_obj = robot.link_map[ee_link]

    T_base_last = link_fk[last_link]
    T_base_ee = link_fk[ee_link_obj]
    T_last_to_ee = np.linalg.inv(T_base_last) @ T_base_ee

    print(f"[INFO] last_link='{last_link_name}', T_last_to_ee computed.")

    return (
        T_parent_joint_zero,
        joint_axes,
        lower_limits,
        upper_limits,
        T_last_to_ee,
    )


# ============================================================
# 2. FK in PyTorch using URDF-derived constants
# ============================================================

def rotation_matrix_from_axis_angle_batch(axis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Compute batch of rotation matrices for rotations about a fixed axis.

    axis  : (3,)   constant axis
    theta : (B,)   angles

    Returns:
        R : (B, 3, 3)
    """
    device = theta.device
    axis = axis.to(device=device, dtype=torch.float32)
    axis = axis / (axis.norm() + 1e-8)

    x, y, z = axis[0], axis[1], axis[2]
    B = theta.shape[0]

    t = theta.view(B, 1, 1)
    c = torch.cos(t)
    s = torch.sin(t)
    one_c = 1.0 - c

    R = torch.zeros(B, 3, 3, device=device, dtype=torch.float32)

    R[:, 0, 0] = c[:, 0, 0] + x * x * one_c[:, 0, 0]
    R[:, 1, 1] = c[:, 0, 0] + y * y * one_c[:, 0, 0]
    R[:, 2, 2] = c[:, 0, 0] + z * z * one_c[:, 0, 0]

    R[:, 0, 1] = x * y * one_c[:, 0, 0] - z * s[:, 0, 0]
    R[:, 1, 0] = y * x * one_c[:, 0, 0] + z * s[:, 0, 0]

    R[:, 0, 2] = x * z * one_c[:, 0, 0] + y * s[:, 0, 0]
    R[:, 2, 0] = z * x * one_c[:, 0, 0] - y * s[:, 0, 0]

    R[:, 1, 2] = y * z * one_c[:, 0, 0] - x * s[:, 0, 0]
    R[:, 2, 1] = z * y * one_c[:, 0, 0] + x * s[:, 0, 0]

    return R


def rotation_matrix_to_quaternion_batch(R: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of rotation matrices to quaternions [w, x, y, z].

    R : (B, 3, 3)
    """
    B = R.shape[0]
    quat = torch.empty(B, 4, device=R.device, dtype=torch.float32)

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    mask = trace > 0

    # Case 1
    t = torch.sqrt(trace[mask] + 1.0) * 2.0
    quat[mask, 0] = 0.25 * t
    quat[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / t
    quat[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / t
    quat[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / t

    # Case 2
    mask2 = (~mask) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    t2 = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2.0
    quat[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / t2
    quat[mask2, 1] = 0.25 * t2
    quat[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / t2
    quat[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / t2

    # Case 3
    mask3 = (~mask) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    t3 = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2.0
    quat[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / t3
    quat[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / t3
    quat[mask3, 2] = 0.25 * t3
    quat[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / t3

    # Case 4
    mask4 = (~mask) & (~mask2) & (~mask3)
    t4 = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2.0
    quat[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / t4
    quat[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / t4
    quat[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / t4
    quat[mask4, 3] = 0.25 * t4

    quat = quat / (quat.norm(dim=1, keepdim=True) + 1e-8)
    return quat


def batch_fk_urdf_torch(
    q: torch.Tensor,                 # (B, J)
    T_parent_joint: torch.Tensor,    # (J, 4, 4)
    joint_axes: torch.Tensor,        # (J, 3)
    T_last_to_ee: torch.Tensor,      # (4, 4)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched FK using URDF-derived transforms.
    """
    device = q.device
    B, J = q.shape

    T_parent_joint = T_parent_joint.to(device=device, dtype=torch.float32)
    joint_axes = joint_axes.to(device=device, dtype=torch.float32)
    T_last_to_ee = T_last_to_ee.to(device=device, dtype=torch.float32)

    T = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).expand(B, 4, 4).clone()

    for j in range(J):
        T = T @ T_parent_joint[j]

        axis = joint_axes[j]
        theta_j = q[:, j]
        Rj = rotation_matrix_from_axis_angle_batch(axis, theta_j)

        Tj_rot = torch.zeros(B, 4, 4, device=device, dtype=torch.float32)
        Tj_rot[:, :3, :3] = Rj
        Tj_rot[:, 3, 3] = 1.0

        T = T @ Tj_rot

    T = T @ T_last_to_ee

    pos = T[:, :3, 3]
    R = T[:, :3, :3]
    quat = rotation_matrix_to_quaternion_batch(R)

    return pos, quat


# ============================================================
# 3. Dataset generation helpers (with sharded .pt)
# ============================================================

def _maybe_flush_shard(
    shard_rows,
    rows_in_shard: int,
    shard_idx: int,
    output_pt_prefix: str,
):
    """
    If there are accumulated rows, save them as one shard and reset.
    Returns new (shard_rows, rows_in_shard, shard_idx).
    """
    if rows_in_shard == 0:
        return shard_rows, rows_in_shard, shard_idx

    big_tensor = torch.cat(shard_rows, dim=0)
    out_path = f"{output_pt_prefix}_part{shard_idx:03d}.pt"
    torch.save(big_tensor, out_path)
    print(f"[INFO] Saved shard {shard_idx} with {rows_in_shard} samples to '{out_path}'")

    shard_rows = []
    rows_in_shard = 0
    shard_idx += 1
    return shard_rows, rows_in_shard, shard_idx


def generate_random_fk_dataset(
    num_joints: int,
    num_samples: int,
    batch_size: int,
    output_csv: Optional[str],
    output_pt_prefix: Optional[str],
    pt_shard_size: int,
    device,
    T_parent_joint_zero,
    joint_axes,
    lower,
    upper,
    T_last_to_ee,
):
    """
    RANDOM mode: sample 'num_samples' joint vectors uniformly in [lower, upper].
    Optionally write CSV and/or sharded .pt tensors.
    """
    write_csv = bool(output_csv)
    save_pt = bool(output_pt_prefix)

    header = [f"q{i+1}" for i in range(num_joints)] + [
        "x", "y", "z", "qw", "qx", "qy", "qz"
    ]

    if write_csv:
        f = open(output_csv, "w", newline="")
        writer = csv.writer(f)
        writer.writerow(header)
    else:
        f = None
        writer = None

    if save_pt:
        shard_rows = []
        rows_in_shard = 0
        shard_idx = 0
    else:
        shard_rows = None
        rows_in_shard = 0
        shard_idx = 0

    span = upper - lower
    remaining = num_samples
    total_written = 0

    with tqdm(total=num_samples, desc="Random FK", unit="samples") as pbar:
        while remaining > 0:
            cur_bs = min(batch_size, remaining)

            q = lower + span * torch.rand(cur_bs, num_joints, device=device)
            pos, quat = batch_fk_urdf_torch(q, T_parent_joint_zero, joint_axes, T_last_to_ee)

            rows = torch.cat([q, pos, quat], dim=1)  # (B, 7+7)

            if writer is not None:
                writer.writerows(rows.cpu().numpy().tolist())

            if save_pt:
                shard_rows.append(rows.cpu())
                rows_in_shard += cur_bs
                if rows_in_shard >= pt_shard_size:
                    shard_rows, rows_in_shard, shard_idx = _maybe_flush_shard(
                        shard_rows,
                        rows_in_shard,
                        shard_idx,
                        output_pt_prefix,
                    )

            remaining -= cur_bs
            total_written += cur_bs
            pbar.update(cur_bs)

    if f is not None:
        f.close()
        print(f"[INFO] RANDOM mode done. Saved {total_written} samples to '{output_csv}'")
    else:
        print(f"[INFO] RANDOM mode done. Saved {total_written} samples (no CSV).")

    if save_pt:
        shard_rows, rows_in_shard, shard_idx = _maybe_flush_shard(
            shard_rows,
            rows_in_shard,
            shard_idx,
            output_pt_prefix,
        )


def generate_grid_fk_dataset(
    num_joints: int,
    step_deg: float,
    batch_size: int,
    output_csv: Optional[str],
    output_pt_prefix: Optional[str],
    pt_shard_size: int,
    device,
    T_parent_joint_zero,
    joint_axes,
    lower,
    upper,
    T_last_to_ee,
):
    """
    GRID mode: build joint grid using step_deg between lower/upper, then iterate
    over all combinations in batches, computing FK.

    Uses an index-based scheme to avoid materialising the full cartesian product
    at once in memory. Sharded .pt writing to avoid RAM blow-ups.
    """
    write_csv = bool(output_csv)
    save_pt = bool(output_pt_prefix)

    header = [f"q{i+1}" for i in range(num_joints)] + [
        "x", "y", "z", "qw", "qx", "qy", "qz"
    ]

    if write_csv:
        f = open(output_csv, "w", newline="")
        writer = csv.writer(f)
        writer.writerow(header)
    else:
        f = None
        writer = None

    if save_pt:
        shard_rows = []
        rows_in_shard = 0
        shard_idx = 0
    else:
        shard_rows = None
        rows_in_shard = 0
        shard_idx = 0

    step_rad = math.radians(step_deg)
    lower_np = lower.cpu().numpy()
    upper_np = upper.cpu().numpy()

    joint_values = []
    n_vals = []
    total_combos = 1

    for j in range(num_joints):
        vals_j = np.arange(lower_np[j], upper_np[j] + 1e-9, step_rad, dtype=np.float64)
        if vals_j.size == 0:
            raise ValueError(
                f"Empty grid for joint {j} with limits [{lower_np[j]}, {upper_np[j]}] and step_deg={step_deg}"
            )
        joint_values.append(torch.from_numpy(vals_j).float().to(device))
        n_vals.append(vals_j.size)
        total_combos *= vals_j.size
        print(f"[INFO] Joint {j}: {vals_j.size} grid points")

    print(f"[INFO] Total grid combinations: {total_combos}")

    n_vals_t = torch.tensor(n_vals, dtype=torch.long, device=device)
    strides_t = torch.empty(num_joints, dtype=torch.long, device=device)
    stride = 1
    for j in reversed(range(num_joints)):
        strides_t[j] = stride
        stride *= n_vals_t[j]

    remaining = total_combos
    start_idx = 0
    total_written = 0

    with tqdm(total=total_combos, desc="Grid FK", unit="samples") as pbar:
        while remaining > 0:
            cur_bs = min(batch_size, remaining)

            g = torch.arange(start_idx, start_idx + cur_bs, device=device, dtype=torch.long)
            idx_mat = torch.empty(cur_bs, num_joints, device=device, dtype=torch.long)

            for j in range(num_joints):
                idx_mat[:, j] = (g // strides_t[j]) % n_vals_t[j]

            q_batch = torch.empty(cur_bs, num_joints, device=device, dtype=torch.float32)
            for j in range(num_joints):
                q_batch[:, j] = joint_values[j][idx_mat[:, j]]

            pos, quat = batch_fk_urdf_torch(q_batch, T_parent_joint_zero, joint_axes, T_last_to_ee)

            rows = torch.cat([q_batch, pos, quat], dim=1)

            if writer is not None:
                writer.writerows(rows.cpu().numpy().tolist())

            if save_pt:
                shard_rows.append(rows.cpu())
                rows_in_shard += cur_bs
                if rows_in_shard >= pt_shard_size:
                    shard_rows, rows_in_shard, shard_idx = _maybe_flush_shard(
                        shard_rows,
                        rows_in_shard,
                        shard_idx,
                        output_pt_prefix,
                    )

            remaining -= cur_bs
            total_written += cur_bs
            start_idx += cur_bs
            pbar.update(cur_bs)

    if f is not None:
        f.close()
        print(f"[INFO] GRID mode done. Saved {total_written} samples to '{output_csv}'")
    else:
        print(f"[INFO] GRID mode done. Saved {total_written} samples (no CSV).")

    if save_pt:
        shard_rows, rows_in_shard, shard_idx = _maybe_flush_shard(
            shard_rows,
            rows_in_shard,
            shard_idx,
            output_pt_prefix,
        )


# ============================================================
# 4. Top-level entry
# ============================================================

def generate_fk_dataset_from_urdf(
    urdf_path: str,
    base_link: str,
    ee_link: str,
    num_joints: int,
    num_samples: int,
    step_deg: float,
    batch_size: int,
    output_csv: Optional[str],
    output_pt_prefix: Optional[str],
    pt_shard_size: int,
    use_cuda: bool,
):
    (
        T_parent_joint_zero_np,
        joint_axes_np,
        lower_np,
        upper_np,
        T_last_to_ee_np,
    ) = build_urdf_kinematics(urdf_path, base_link, ee_link)

    T_parent_joint_zero_np = T_parent_joint_zero_np[:num_joints]
    joint_axes_np = joint_axes_np[:num_joints]
    lower_np = lower_np[:num_joints]
    upper_np = upper_np[:num_joints]

    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    print(f"[INFO] Using device: {device}")

    T_parent_joint_zero = torch.from_numpy(T_parent_joint_zero_np).float().to(device)
    joint_axes = torch.from_numpy(joint_axes_np).float().to(device)
    lower = torch.from_numpy(lower_np).float().to(device)
    upper = torch.from_numpy(upper_np).float().to(device)
    T_last_to_ee = torch.from_numpy(T_last_to_ee_np).float().to(device)

    if num_samples is not None and num_samples > 0:
        print(f"[INFO] RANDOM mode (num_samples = {num_samples})")
        generate_random_fk_dataset(
            num_joints=num_joints,
            num_samples=num_samples,
            batch_size=batch_size,
            output_csv=output_csv,
            output_pt_prefix=output_pt_prefix,
            pt_shard_size=pt_shard_size,
            device=device,
            T_parent_joint_zero=T_parent_joint_zero,
            joint_axes=joint_axes,
            lower=lower,
            upper=upper,
            T_last_to_ee=T_last_to_ee,
        )
    else:
        print(f"[INFO] GRID mode (step_deg = {step_deg})")
        generate_grid_fk_dataset(
            num_joints=num_joints,
            step_deg=step_deg,
            batch_size=batch_size,
            output_csv=output_csv,
            output_pt_prefix=output_pt_prefix,
            pt_shard_size=pt_shard_size,
            device=device,
            T_parent_joint_zero=T_parent_joint_zero,
            joint_axes=joint_axes,
            lower=lower,
            upper=upper,
            T_last_to_ee=T_last_to_ee,
        )


# ============================================================
# 5. CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate FK dataset from a URDF using batched PyTorch kinematics (with sharded .pt)."
    )
    parser.add_argument("--urdf_path", type=str, required=True,
                        help="Path to URDF file.")
    parser.add_argument("--base_link", type=str, default="world",
                        help="Name of the base link in URDF.")
    parser.add_argument("--ee_link", type=str, default="iiwa_link_7",
                        help="Name of the end-effector link in URDF.")
    parser.add_argument("--num_joints", type=int, default=7,
                        help="Number of joints in the chain (first J joints).")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="> 0: RANDOM mode with this many samples; "
             "<= 0: GRID mode with ALL combinations from step_deg.",
    )
    parser.add_argument(
        "--step_deg",
        type=float,
        default=5.0,
        help="Joint step (degrees) used in GRID mode when num_samples <= 0.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=65536,
        help="Batch size for FK evaluation.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="Output CSV filename. Use empty string '' to disable CSV output.",
    )
    parser.add_argument(
        "--output_pt_prefix",
        type=str,
        default="",
        help="Prefix for sharded .pt tensors, e.g. './data/iiwa_7dof_30deg'. "
             "Shards will be '<prefix>_part000.pt', '<prefix>_part001.pt', ...",
    )
    parser.add_argument(
        "--pt_shard_size",
        type=int,
        default=1_000_000,
        help="Maximum number of samples per .pt shard (e.g. 5_000_000).",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use CUDA if available.",
    )

    args = parser.parse_args()

    output_csv = args.output_csv if args.output_csv.strip() else None
    output_pt_prefix = args.output_pt_prefix if args.output_pt_prefix.strip() else None

    if not output_csv and not output_pt_prefix:
        raise ValueError("At least one of --output_csv or --output_pt_prefix must be non-empty.")

    generate_fk_dataset_from_urdf(
        urdf_path=args.urdf_path,
        base_link=args.base_link,
        ee_link=args.ee_link,
        num_joints=args.num_joints,
        num_samples=args.num_samples,
        step_deg=args.step_deg,
        batch_size=args.batch_size,
        output_csv=output_csv,
        output_pt_prefix=output_pt_prefix,
        pt_shard_size=args.pt_shard_size,
        use_cuda=args.cuda,
    )


if __name__ == "__main__":
    main()
