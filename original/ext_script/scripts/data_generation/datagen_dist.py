#!/usr/bin/env python3
"""
datagen_dist.py (edited)

Generates FK datasets from URDF using batched PyTorch FK.

Distributions supported:
- GRID     : full cartesian grid (step_deg)
- UNIFORM  : uniform random in [lower, upper]
- NORMAL   : truncated normal around a mean, within [lower, upper]

NEW (for NORMAL):
- --normal_cover_limits : automatically choose per-joint sigma so that ±kσ spans limits
- --normal_cover_k      : k in "±kσ" (default 3)

Outputs:
- Optional CSV
- Optional sharded .pt tensors with rows [q1..qJ, x,y,z,qw,qx,qy,qz]
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
    parent_to_joints = {}
    for j in robot.joints:
        parent_to_joints.setdefault(j.parent, []).append(j)

    q = deque([(root_link, [])])
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        T = np.array(joint.origin, dtype=np.float64) if joint.origin is not None else np.eye(4, dtype=np.float64)
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

    # FK at zero to get last_link → ee_link
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
    return T_parent_joint_zero, joint_axes, lower_limits, upper_limits, T_last_to_ee


# ============================================================
# 2. FK in PyTorch
# ============================================================

def rotation_matrix_from_axis_angle_batch(axis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
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
    B = R.shape[0]
    quat = torch.empty(B, 4, device=R.device, dtype=torch.float32)

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    mask = trace > 0

    t = torch.sqrt(trace[mask] + 1.0) * 2.0
    quat[mask, 0] = 0.25 * t
    quat[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / t
    quat[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / t
    quat[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / t

    mask2 = (~mask) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    t2 = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2.0
    quat[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / t2
    quat[mask2, 1] = 0.25 * t2
    quat[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / t2
    quat[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / t2

    mask3 = (~mask) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    t3 = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2.0
    quat[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / t3
    quat[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / t3
    quat[mask3, 2] = 0.25 * t3
    quat[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / t3

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
# 3. Output helpers
# ============================================================

def _maybe_flush_shard(shard_rows, rows_in_shard: int, shard_idx: int, output_pt_prefix: str):
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


# ============================================================
# 4. Sampling methods
# ============================================================

def sample_uniform(lower: torch.Tensor, upper: torch.Tensor, B: int, device) -> torch.Tensor:
    span = (upper - lower)
    return lower + span * torch.rand(B, lower.shape[0], device=device)

def sample_truncated_normal(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    B: int,
    device,
    max_rounds: int = 8,
) -> torch.Tensor:
    """
    Truncated normal by resampling invalid dimensions.
    mu, sigma, lower, upper: (J,)
    Returns q: (B,J) within [lower, upper]
    """
    J = mu.shape[0]
    mu = mu.to(device=device, dtype=torch.float32)
    sigma = sigma.to(device=device, dtype=torch.float32).clamp_min(1e-8)
    lower = lower.to(device=device, dtype=torch.float32)
    upper = upper.to(device=device, dtype=torch.float32)

    q = mu.unsqueeze(0) + sigma.unsqueeze(0) * torch.randn(B, J, device=device)

    for _ in range(max_rounds):
        bad = (q < lower.unsqueeze(0)) | (q > upper.unsqueeze(0))
        if not bad.any():
            return q
        eps = torch.randn(B, J, device=device)
        q = torch.where(bad, mu.unsqueeze(0) + sigma.unsqueeze(0) * eps, q)

    # final safety clamp (rarely used if max_rounds is enough)
    return torch.max(torch.min(q, upper.unsqueeze(0)), lower.unsqueeze(0))


# ============================================================
# 5. Dataset generation (random: uniform/normal)
# ============================================================

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
    dist: str = "uniform",
    normal_mu: Optional[torch.Tensor] = None,     # (J,)
    normal_sigma: Optional[torch.Tensor] = None,  # (J,) or scalar expanded
):
    write_csv = bool(output_csv)
    save_pt = bool(output_pt_prefix)

    header = [f"q{i+1}" for i in range(num_joints)] + ["x", "y", "z", "qw", "qx", "qy", "qz"]

    if write_csv:
        f = open(output_csv, "w", newline="")
        writer = csv.writer(f)
        writer.writerow(header)
    else:
        f, writer = None, None

    if save_pt:
        shard_rows, rows_in_shard, shard_idx = [], 0, 0
    else:
        shard_rows, rows_in_shard, shard_idx = None, 0, 0

    remaining = num_samples
    total_written = 0

    if dist == "normal":
        if normal_mu is None or normal_sigma is None:
            raise ValueError("normal dist requires normal_mu and normal_sigma")
        normal_mu = normal_mu[:num_joints].to(device=device)
        if normal_sigma.numel() == 1:
            normal_sigma = normal_sigma.expand(num_joints)
        normal_sigma = normal_sigma[:num_joints].to(device=device)

    with tqdm(total=num_samples, desc=f"Random FK ({dist})", unit="samples") as pbar:
        while remaining > 0:
            cur_bs = min(batch_size, remaining)

            if dist == "uniform":
                q = sample_uniform(lower[:num_joints], upper[:num_joints], cur_bs, device)
            elif dist == "normal":
                q = sample_truncated_normal(
                    mu=normal_mu, sigma=normal_sigma,
                    lower=lower[:num_joints], upper=upper[:num_joints],
                    B=cur_bs, device=device
                )
            else:
                raise ValueError(f"Unknown dist '{dist}' (expected uniform|normal)")

            pos, quat = batch_fk_urdf_torch(q, T_parent_joint_zero, joint_axes, T_last_to_ee)
            rows = torch.cat([q, pos, quat], dim=1)

            if writer is not None:
                writer.writerows(rows.cpu().numpy().tolist())

            if save_pt:
                shard_rows.append(rows.cpu())
                rows_in_shard += cur_bs
                if rows_in_shard >= pt_shard_size:
                    shard_rows, rows_in_shard, shard_idx = _maybe_flush_shard(
                        shard_rows, rows_in_shard, shard_idx, output_pt_prefix
                    )

            remaining -= cur_bs
            total_written += cur_bs
            pbar.update(cur_bs)

    if f is not None:
        f.close()
        print(f"[INFO] RANDOM ({dist}) done. Saved {total_written} samples to '{output_csv}'")
    else:
        print(f"[INFO] RANDOM ({dist}) done. Saved {total_written} samples (no CSV).")

    if save_pt:
        shard_rows, rows_in_shard, shard_idx = _maybe_flush_shard(
            shard_rows, rows_in_shard, shard_idx, output_pt_prefix
        )


# ============================================================
# 6. GRID generation
# ============================================================

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
    write_csv = bool(output_csv)
    save_pt = bool(output_pt_prefix)

    header = [f"q{i+1}" for i in range(num_joints)] + ["x", "y", "z", "qw", "qx", "qy", "qz"]

    if write_csv:
        f = open(output_csv, "w", newline="")
        writer = csv.writer(f)
        writer.writerow(header)
    else:
        f, writer = None, None

    if save_pt:
        shard_rows, rows_in_shard, shard_idx = [], 0, 0
    else:
        shard_rows, rows_in_shard, shard_idx = None, 0, 0

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
                        shard_rows, rows_in_shard, shard_idx, output_pt_prefix
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
            shard_rows, rows_in_shard, shard_idx, output_pt_prefix
        )


# ============================================================
# 7. Top-level entry
# ============================================================

def parse_list_floats(s: str, expected_len: int) -> torch.Tensor:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != expected_len:
        raise ValueError(f"Expected {expected_len} comma-separated floats, got {len(parts)}: '{s}'")
    return torch.tensor([float(x) for x in parts], dtype=torch.float32)

def compute_sigma_cover_limits(mu: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, k: float) -> torch.Tensor:
    """
    Per-joint sigma so that mu ± k*sigma stays inside [lower, upper].
    If mu is centered, this makes ±kσ span the full range.
    """
    k = float(k)
    if k <= 0:
        raise ValueError("--normal_cover_k must be > 0")

    left = (mu - lower).clamp_min(0.0)
    right = (upper - mu).clamp_min(0.0)
    margin = torch.minimum(left, right)  # nearest limit distance
    sigma = (margin / k).clamp_min(1e-8)
    return sigma

def generate_fk_dataset_from_urdf(
    urdf_path: str,
    base_link: str,
    ee_link: str,
    num_joints: int,
    dist: str,
    num_samples: int,
    step_deg: float,
    batch_size: int,
    output_csv: Optional[str],
    output_pt_prefix: Optional[str],
    pt_shard_size: int,
    use_cuda: bool,
    seed: int,
    normal_sigma_deg: float,
    normal_sigma_rad: float,
    normal_mu_mode: str,
    normal_mu_deg: str,
    normal_mu_rad: str,
    normal_cover_limits: bool,
    normal_cover_k: float,
):
    (T_parent_joint_zero_np, joint_axes_np, lower_np, upper_np, T_last_to_ee_np) = build_urdf_kinematics(
        urdf_path, base_link, ee_link
    )

    # truncate to requested J
    T_parent_joint_zero_np = T_parent_joint_zero_np[:num_joints]
    joint_axes_np = joint_axes_np[:num_joints]
    lower_np = lower_np[:num_joints]
    upper_np = upper_np[:num_joints]

    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    print(f"[INFO] Using device: {device}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    T_parent_joint_zero = torch.from_numpy(T_parent_joint_zero_np).float().to(device)
    joint_axes = torch.from_numpy(joint_axes_np).float().to(device)
    lower = torch.from_numpy(lower_np).float().to(device)
    upper = torch.from_numpy(upper_np).float().to(device)
    T_last_to_ee = torch.from_numpy(T_last_to_ee_np).float().to(device)

    dist = dist.lower().strip()
    if dist == "grid":
        if num_samples > 0:
            print("[WARN] dist=grid ignores --num_samples; using full grid combos.")
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
        return

    if num_samples <= 0:
        raise ValueError("For dist=uniform/normal you must set --num_samples > 0")

    normal_mu = None
    normal_sigma = None

    if dist == "normal":
        # mean
        if normal_mu_mode == "midpoint":
            normal_mu = 0.5 * (lower + upper)
        elif normal_mu_mode == "zero":
            normal_mu = torch.zeros_like(lower)
        elif normal_mu_mode == "custom":
            if normal_mu_rad.strip():
                normal_mu = parse_list_floats(normal_mu_rad, num_joints).to(device)
            elif normal_mu_deg.strip():
                mu_deg = parse_list_floats(normal_mu_deg, num_joints).to(device)
                normal_mu = mu_deg * (math.pi / 180.0)
            else:
                raise ValueError("normal_mu_mode=custom requires --normal_mu_deg or --normal_mu_rad")
        else:
            raise ValueError("normal_mu_mode must be midpoint|zero|custom")

        # ensure mu inside limits
        normal_mu = torch.max(torch.min(normal_mu, upper), lower)

        # sigma
        if normal_cover_limits:
            normal_sigma = compute_sigma_cover_limits(normal_mu, lower, upper, normal_cover_k)
            print(f"[INFO] normal_cover_limits=ON (k={normal_cover_k}) -> per-joint sigma computed.")
        else:
            if normal_sigma_rad > 0:
                sigma = float(normal_sigma_rad)
            else:
                sigma = math.radians(float(normal_sigma_deg))
            normal_sigma = torch.tensor([sigma], dtype=torch.float32, device=device)

        # reporting
        sig = normal_sigma.detach().cpu().numpy()
        print(f"[INFO] normal_mu_mode={normal_mu_mode}")
        if sig.size == 1:
            print(f"[INFO] normal_sigma(rad)={float(sig.item()):.6f} (~{float(sig.item())*180/math.pi:.2f} deg)")
        else:
            print(f"[INFO] normal_sigma(rad) per-joint: min={sig.min():.6f}, max={sig.max():.6f} "
                  f"(~{sig.min()*180/math.pi:.2f} to {sig.max()*180/math.pi:.2f} deg)")

    print(f"[INFO] dist={dist}, num_samples={num_samples}")

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
        dist=dist,
        normal_mu=normal_mu,
        normal_sigma=normal_sigma,
    )


# ============================================================
# 8. CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate FK dataset from a URDF using batched PyTorch kinematics (grid / uniform / normal)."
    )
    parser.add_argument("--urdf_path", type=str, required=True)
    parser.add_argument("--base_link", type=str, default="world")
    parser.add_argument("--ee_link", type=str, default="iiwa_link_7")
    parser.add_argument("--num_joints", type=int, default=7)

    parser.add_argument("--dist", type=str, default="grid",
                        help="grid | uniform | normal")

    parser.add_argument("--num_samples", type=int, default=-1,
                        help="For uniform/normal: number of random samples (>0). For grid: ignored.")
    parser.add_argument("--step_deg", type=float, default=5.0,
                        help="Grid step (degrees) for dist=grid.")
    parser.add_argument("--batch_size", type=int, default=65536)

    parser.add_argument("--output_csv", type=str, default="")
    parser.add_argument("--output_pt_prefix", type=str, default="")
    parser.add_argument("--pt_shard_size", type=int, default=1_000_000)

    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # normal params
    parser.add_argument("--normal_sigma_deg", type=float, default=20.0,
                        help="Normal std in degrees (ignored if normal_sigma_rad > 0 or if --normal_cover_limits is set).")
    parser.add_argument("--normal_sigma_rad", type=float, default=0.0,
                        help="Normal std in radians. If >0, overrides normal_sigma_deg (unless --normal_cover_limits).")
    parser.add_argument("--normal_mu_mode", type=str, default="midpoint",
                        help="midpoint | zero | custom")
    parser.add_argument("--normal_mu_deg", type=str, default="",
                        help="Comma-separated mean in degrees, length=num_joints (only if mu_mode=custom).")
    parser.add_argument("--normal_mu_rad", type=str, default="",
                        help="Comma-separated mean in radians, length=num_joints (only if mu_mode=custom).")

    # NEW: cover limits with ±kσ
    parser.add_argument("--normal_cover_limits", action="store_true",
                        help="If set, compute per-joint sigma so that mu ± k*sigma stays within joint limits.")
    parser.add_argument("--normal_cover_k", type=float, default=3.0,
                        help="k in ±kσ when using --normal_cover_limits (default 3).")

    args = parser.parse_args()

    output_csv = args.output_csv.strip() or None
    output_pt_prefix = args.output_pt_prefix.strip() or None
    if not output_csv and not output_pt_prefix:
        raise ValueError("At least one of --output_csv or --output_pt_prefix must be non-empty.")

    generate_fk_dataset_from_urdf(
        urdf_path=args.urdf_path,
        base_link=args.base_link,
        ee_link=args.ee_link,
        num_joints=args.num_joints,
        dist=args.dist,
        num_samples=args.num_samples,
        step_deg=args.step_deg,
        batch_size=args.batch_size,
        output_csv=output_csv,
        output_pt_prefix=output_pt_prefix,
        pt_shard_size=args.pt_shard_size,
        use_cuda=args.cuda,
        seed=args.seed,
        normal_sigma_deg=args.normal_sigma_deg,
        normal_sigma_rad=args.normal_sigma_rad,
        normal_mu_mode=args.normal_mu_mode,
        normal_mu_deg=args.normal_mu_deg,
        normal_mu_rad=args.normal_mu_rad,
        normal_cover_limits=args.normal_cover_limits,
        normal_cover_k=args.normal_cover_k,
    )

if __name__ == "__main__":
    main()
