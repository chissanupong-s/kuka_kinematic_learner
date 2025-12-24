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
   - num_samples <= 0  (default -1)
   - Uses joint limits + step_deg to build a Cartesian grid of joint values
   - Generates ALL combinations in that grid (be careful: can explode!)

Output CSV rows:
    q1..qN, x, y, z, qw, qx, qy, qz
"""

import argparse
import csv
import math
from collections import deque
from typing import List, Tuple

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

    Returns:
        T_parent_joint_zero : (J, 4, 4) parent link → joint frame at q=0
        joint_axes          : (J, 3)   joint axes in joint frame
        lower_limits        : (J,)     joint lower limits
        upper_limits        : (J,)     joint upper limits
        T_lastlink_to_ee    : (4, 4)   last chain link → ee_link transform
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

        # urdfpy: joint.origin is a 4x4 transform matrix (numpy array)
        if joint.origin is not None:
            T = np.array(joint.origin, dtype=np.float64)
        else:
            T = np.eye(4, dtype=np.float64)

        T_parent_joint_zero[idx] = T

        if joint.joint_type not in ["revolute", "continuous"]:
            print(f"[WARN] Joint '{jn}' is type '{joint.joint_type}', treating as fixed.")
            axis = np.array([0.0, 0.0, 0.0])
        else:
            axis = np.array(joint.axis)
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

    quat = quat / quat.norm(dim=1, keepdim=True)
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
# 3. Dataset generation helpers
# ============================================================

def generate_random_fk_dataset(
    num_joints: int,
    num_samples: int,
    batch_size: int,
    output_csv: str,
    device,
    T_parent_joint_zero,
    joint_axes,
    lower,
    upper,
    T_last_to_ee,
):
    """RANDOM mode: num_samples > 0."""
    span = upper - lower
    header = [f"q{i+1}" for i in range(num_joints)] + ["x", "y", "z", "qw", "qx", "qy", "qz"]

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        remaining = num_samples
        total_written = 0
        batch_idx = 0

        with tqdm(total=num_samples, desc="Random FK", unit="samples") as pbar:
            while remaining > 0:
                cur_bs = min(batch_size, remaining)
                batch_idx += 1

                q = lower + span * torch.rand(cur_bs, num_joints, device=device)
                pos, quat = batch_fk_urdf_torch(q, T_parent_joint_zero, joint_axes, T_last_to_ee)

                rows = torch.cat([q, pos, quat], dim=1)
                writer.writerows(rows.cpu().numpy().tolist())

                remaining -= cur_bs
                total_written += cur_bs
                pbar.update(cur_bs)

    print(f"[INFO] RANDOM mode done. Saved {total_written} samples to '{output_csv}'")


def generate_grid_fk_dataset(
    num_joints: int,
    step_deg: float,
    batch_size: int,
    output_csv: str,
    device,
    T_parent_joint_zero,
    joint_axes,
    lower,
    upper,
    T_last_to_ee,
):
    """GRID mode: num_samples <= 0 → ALL combinations."""
    step_rad = math.radians(step_deg)

    joint_values: List[torch.Tensor] = []
    n_vals: List[int] = []

    for j in range(num_joints):
        lo = float(lower[j].cpu())
        hi = float(upper[j].cpu())

        if not math.isfinite(lo) or abs(lo) > 1e4:
            lo = -math.pi
        if not math.isfinite(hi) or abs(hi) > 1e4:
            hi = math.pi

        n_j = int(math.floor((hi - lo) / step_rad)) + 1
        n_j = max(n_j, 1)

        vals = lo + torch.arange(n_j, device=device, dtype=torch.float32) * step_rad
        joint_values.append(vals)
        n_vals.append(n_j)

        print(f"[INFO] Joint {j+1}: range [{lo:.3f}, {hi:.3f}] rad, "
              f"step={step_rad:.3f}, points={n_j}")

    total_combinations = 1
    for n in n_vals:
        total_combinations *= n

    print(f"[INFO] GRID mode: total combinations = {total_combinations}")
    if total_combinations <= 0:
        print("[WARN] No combinations to generate.")
        return

    J = num_joints
    strides = [1] * J
    prod = 1
    for j in range(J - 1, -1, -1):
        strides[j] = prod
        prod *= n_vals[j]

    strides_t = torch.tensor(strides, device=device, dtype=torch.long)
    n_vals_t = torch.tensor(n_vals, device=device, dtype=torch.long)

    header = [f"q{i+1}" for i in range(num_joints)] + ["x", "y", "z", "qw", "qx", "qy", "qz"]

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        remaining = total_combinations
        total_written = 0
        batch_idx = 0
        start_idx = 0

        with tqdm(total=total_combinations, desc="Grid FK", unit="samples") as pbar:
            while remaining > 0:
                cur_bs = min(batch_size, remaining)
                batch_idx += 1

                g = torch.arange(start_idx, start_idx + cur_bs, device=device, dtype=torch.long)

                idx_mat = torch.empty(cur_bs, J, device=device, dtype=torch.long)
                for j in range(J):
                    idx_mat[:, j] = (g // strides_t[j]) % n_vals_t[j]

                q_batch = torch.empty(cur_bs, J, device=device, dtype=torch.float32)
                for j in range(J):
                    q_batch[:, j] = joint_values[j][idx_mat[:, j]]

                pos, quat = batch_fk_urdf_torch(q_batch, T_parent_joint_zero, joint_axes, T_last_to_ee)

                rows = torch.cat([q_batch, pos, quat], dim=1)
                writer.writerows(rows.cpu().numpy().tolist())

                remaining -= cur_bs
                total_written += cur_bs
                start_idx += cur_bs
                pbar.update(cur_bs)

    print(f"[INFO] GRID mode done. Saved {total_written} samples to '{output_csv}'")


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
    output_csv: str,
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
            device=device,
            T_parent_joint_zero=T_parent_joint_zero,
            joint_axes=joint_axes,
            lower=lower,
            upper=upper,
            T_last_to_ee=T_last_to_ee,
        )
    else:
        print(f"[INFO] GRID mode (ALL combinations) with step_deg = {step_deg}")
        print("[WARN] Total combinations might be huge for small step sizes!")
        generate_grid_fk_dataset(
            num_joints=num_joints,
            step_deg=step_deg,
            batch_size=batch_size,
            output_csv=output_csv,
            device=device,
            T_parent_joint_zero=T_parent_joint_zero,
            joint_axes=joint_axes,
            lower=lower,
            upper=upper,
            T_last_to_ee=T_last_to_ee,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Fast FK dataset from URDF using PyTorch (GPU/CPU), "
                    "with RANDOM or GRID (ALL) modes."
    )
    parser.add_argument("--urdf_path", type=str, required=True,
                        help="Path to the robot URDF.")
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
        default="fk_dataset_urdf_torch.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use CUDA if available.",
    )

    args = parser.parse_args()

    generate_fk_dataset_from_urdf(
        urdf_path=args.urdf_path,
        base_link=args.base_link,
        ee_link=args.ee_link,
        num_joints=args.num_joints,
        num_samples=args.num_samples,
        step_deg=args.step_deg,
        batch_size=args.batch_size,
        output_csv=args.output_csv,
        use_cuda=args.cuda,
    )


if __name__ == "__main__":
    main()
