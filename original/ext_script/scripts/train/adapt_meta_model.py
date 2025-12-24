#!/usr/bin/env python3
# Adapt a meta-trained kinematics model (FK or IK) on a single DOF dataset
# and report errors in physical units, plus adaptation time for each step
# configuration.

import argparse
import math
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


POSE_COLS = ["x", "y", "z", "qw", "qx", "qy", "qz"]
JOINT_COLS = [f"q{i}" for i in range(1, 8)]


# ---------------------------------------------------------------------------
# Data utilities (same format as meta-training script)
# ---------------------------------------------------------------------------

def load_dataset_tensor(path: str) -> torch.Tensor:
    """
    Load the full dataset as a 2D float32 Tensor [N,14] from:
      - CSV file with columns [q1..q7,x,y,z,qw,qx,qy,qz]
      - single .pt/.bin tensor
      - directory of .pt/.bin shards (dict with 'data' or raw tensor)
    """
    print(f"[INFO] Loading dataset from {path}")
    cols = JOINT_COLS + POSE_COLS

    if os.path.isdir(path):
        shard_files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.endswith(".pt") or f.endswith(".bin")
        ]
        shard_files = sorted(shard_files)
        if not shard_files:
            raise ValueError(f"No .pt/.bin shards found in directory: {path}")

        arrs: List[np.ndarray] = []
        total_rows = 0
        for f_path in shard_files:
            print(f"[INFO]  shard: {f_path}")
            obj = torch.load(f_path)
            if isinstance(obj, dict) and "data" in obj:
                arr = obj["data"]
            else:
                arr = obj
            if not isinstance(arr, torch.Tensor):
                raise TypeError(f"Shard {f_path} must contain a Tensor or dict with 'data'.")
            if arr.ndim != 2 or arr.shape[1] != 14:
                raise ValueError(f"Shard {f_path} must have shape [N,14], got {tuple(arr.shape)}")
            arr_np = arr.cpu().float().numpy()
            arrs.append(arr_np)
            total_rows += arr_np.shape[0]

        print(f"[INFO]  total rows across shards: {total_rows}")
        full_np = np.concatenate(arrs, axis=0)
        full_tensor = torch.from_numpy(full_np).float()

    elif path.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(path)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"CSV {path} missing columns: {missing}")
        data_np = df[cols].values.astype(np.float32)
        full_tensor = torch.from_numpy(data_np).float()

    elif path.endswith(".pt") or path.endswith(".bin"):
        obj = torch.load(path)
        if isinstance(obj, dict) and "data" in obj:
            full_tensor = obj["data"]
        else:
            full_tensor = obj
        if not isinstance(full_tensor, torch.Tensor):
            raise TypeError(f"{path} must contain a Tensor or dict with 'data'.")
        if full_tensor.ndim != 2 or full_tensor.shape[1] != 14:
            raise ValueError(f"{path} tensor must be [N,14], got {tuple(full_tensor.shape)}")
        full_tensor = full_tensor.float()
    else:
        raise ValueError(f"Unsupported dataset path type: {path}")

    print(f"[INFO] Loaded tensor of shape {tuple(full_tensor.shape)}")
    return full_tensor


class KinematicsTensorDataset(Dataset):
    """
    Dataset for kinematics with dataset-specific normalisation.

    mode:
      - 'fk': input  = joints [q1..q7]
              target = pose   [x,y,z,qw,qx,qy,qz]
      - 'ik': input  = pose   [x,y,z,qw,qx,qy,qz]
              target = joints [q1..q7]
    """

    def __init__(self, full_tensor: torch.Tensor, mode: str):
        super().__init__()
        if full_tensor.ndim != 2 or full_tensor.shape[1] != 14:
            raise ValueError(f"Expected full_tensor [N,14], got {tuple(full_tensor.shape)}")

        if mode == "fk":
            X = full_tensor[:, 0:7]
            Y = full_tensor[:, 7:14]
        elif mode == "ik":
            X = full_tensor[:, 7:14]
            Y = full_tensor[:, 0:7]
        else:
            raise ValueError(f"Unknown mode {mode}, expected 'fk' or 'ik'.")

        X = X.cpu().numpy().astype(np.float32)
        Y = Y.cpu().numpy().astype(np.float32)

        input_mean = X.mean(axis=0, keepdims=True)
        input_std = X.std(axis=0, keepdims=True) + 1e-8
        target_mean = Y.mean(axis=0, keepdims=True)
        target_std = Y.std(axis=0, keepdims=True) + 1e-8

        self.inputs = (X - input_mean) / input_std
        self.targets = (Y - target_mean) / target_std

        self.input_mean = input_mean
        self.input_std = input_std
        self.target_mean = target_mean
        self.target_std = target_std

        self.mode = mode

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.inputs[idx]).float()
        y = torch.from_numpy(self.targets[idx]).float()
        return x, y


# ---------------------------------------------------------------------------
# Models: FK and IK (must match meta-training)
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, dim: int, p_drop: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        h = self.fc2(h)
        return self.act(h + x)


class ResidualMLP(nn.Module):
    """FK network: q -> pose."""
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 512, num_blocks: int = 4):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc_in(x))
        for blk in self.blocks:
            h = blk(h)
        return self.fc_out(h)


class IKResNetDualHead(nn.Module):
    """IK network: pose -> joints with aux pose/orientation heads."""
    def __init__(self, in_dim: int = 7, hidden_dim: int = 1024, num_blocks: int = 4, out_dim: int = 7):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])
        self.fc_joint = nn.Linear(hidden_dim, out_dim)
        self.fc_pos = nn.Linear(hidden_dim, 3)
        self.fc_ori = nn.Linear(hidden_dim, 4)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor):
        h = self.act(self.fc_in(x))
        for blk in self.blocks:
            h = blk(h)
        q_pred = self.fc_joint(h)
        pos_pred = self.fc_pos(h)
        ori_pred = self.fc_ori(h)
        return q_pred, pos_pred, ori_pred


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def eval_fk_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean,
    target_std,
) -> Tuple[float, float]:
    """
    Evaluate FK model on query set.

    Returns:
      - position RMSE in metres
      - orientation RMSE in degrees (quaternion angle)
    """
    model.eval()

    tgt_mean = torch.from_numpy(target_mean).float().to(device)  # [1,7]
    tgt_std = torch.from_numpy(target_std).float().to(device)    # [1,7]

    sum_sq_pos = 0.0
    sum_sq_ang = 0.0
    n_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)  # normalised pose

            pose_pred_norm = model(x)           # [B,7]
            pose_pred = pose_pred_norm * tgt_std + tgt_mean
            pose_true = y * tgt_std + tgt_mean

            pos_pred = pose_pred[:, :3]
            pos_true = pose_true[:, :3]

            # position error
            dist = torch.norm(pos_pred - pos_true, dim=1)   # [B]
            sum_sq_pos += torch.sum(dist ** 2).item()

            # orientation error (quaternion angle)
            quat_pred = pose_pred[:, 3:7]
            quat_true = pose_true[:, 3:7]

            # normalise quaternions to unit length
            quat_pred = quat_pred / (torch.norm(quat_pred, dim=1, keepdim=True) + 1e-8)
            quat_true = quat_true / (torch.norm(quat_true, dim=1, keepdim=True) + 1e-8)

            # q_delta = q_true^{-1} * q_pred
            w1, x1, y1, z1 = quat_true[:, 0], quat_true[:, 1], quat_true[:, 2], quat_true[:, 3]
            w2, x2, y2, z2 = quat_pred[:, 0], quat_pred[:, 1], quat_pred[:, 2], quat_pred[:, 3]

            # conj(q_true) = (w1, -x1, -y1, -z1)
            # q_delta = conj(q_true) * q_pred
            w = w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2
            x_d = w1 * x2 - x1 * w2 - y1 * z2 + z1 * y2
            y_d = w1 * y2 + x1 * z2 - y1 * w2 - z1 * x2
            z_d = w1 * z2 - x1 * y2 + y1 * x2 - z1 * w2

            # rotation angle: 2 * acos(|w|)
            w_clamped = torch.clamp(torch.abs(w), -1.0, 1.0)
            angle_rad = 2.0 * torch.acos(w_clamped)  # [B]
            sum_sq_ang += torch.sum(angle_rad ** 2).item()

            n_samples += x.size(0)

    if n_samples == 0:
        return float("nan"), float("nan")

    pos_rmse = math.sqrt(sum_sq_pos / n_samples)
    ang_rmse_rad = math.sqrt(sum_sq_ang / n_samples)
    ang_rmse_deg = ang_rmse_rad * 180.0 / math.pi
    return pos_rmse, ang_rmse_deg


def eval_ik_joint_rmse_deg(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean,
    target_std,
) -> float:
    """
    Evaluate IK model joint-space error as mean RMSE in degrees.
    """
    model.eval()
    tgt_mean = torch.from_numpy(target_mean).float().to(device)  # [1,7]
    tgt_std = torch.from_numpy(target_std).float().to(device)    # [1,7]

    sum_sq_err = torch.zeros(7, device=device)
    n_samples = 0

    with torch.no_grad():
        for x_norm, y_norm in loader:
            x_norm = x_norm.to(device, non_blocking=True)
            y_norm = y_norm.to(device, non_blocking=True)

            q_pred_norm, _, _ = model(x_norm)
            q_pred = q_pred_norm * tgt_std + tgt_mean
            q_true = y_norm * tgt_std + tgt_mean

            err = q_pred - q_true
            sum_sq_err += torch.sum(err ** 2, dim=0)
            n_samples += err.size(0)

    if n_samples == 0:
        return float("nan")

    mse_per_joint = sum_sq_err / n_samples
    rmse_rad = torch.sqrt(mse_per_joint)
    rmse_deg = rmse_rad * 180.0 / math.pi
    mean_rmse_deg = rmse_deg.mean().item()
    return mean_rmse_deg


# ---------------------------------------------------------------------------
# Adaptation
# ---------------------------------------------------------------------------

def adapt_on_task(
    meta_ckpt_path: str,
    data_path: str,
    device: torch.device,
    steps_list,
    support_size: int,
    query_size: int,
    batch_size: int,
    inner_lr: float,
    aux_loss_weight_override: float = None,
    num_workers: int = 0,
    seed: int = 42,
):
    """
    For each step count s in steps_list:
      - copy meta model
      - run s gradient steps on support set
      - evaluate on query set
      - print metrics AND wall-clock time.

    Returns:
      FK mode:  list of (steps, pos_rmse_m, ori_rmse_deg, time_sec)
      IK mode:  list of (steps, joint_rmse_deg, time_sec)
    """
    # load checkpoint
    print(f"[INFO] Loading meta checkpoint from {meta_ckpt_path}")
    try:
        ckpt = torch.load(meta_ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(meta_ckpt_path, map_location=device)

    mode = ckpt.get("mode", "ik")
    print(f"[INFO] Checkpoint mode: {mode}")

    hidden_dim = ckpt.get("hidden_dim", 1024)
    num_blocks = ckpt.get("num_blocks", 4)
    ckpt_aux = ckpt.get("aux_loss_weight", 0.1)
    aux_weight = ckpt_aux if aux_loss_weight_override is None else aux_loss_weight_override

    # load dataset
    full_tensor = load_dataset_tensor(data_path)
    ds = KinematicsTensorDataset(full_tensor, mode=mode)
    N = len(ds)
    if support_size + query_size > N:
        raise ValueError(f"support_size+query_size={support_size+query_size} > N={N}")

    # reproducible split
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(N, generator=g)
    support_idx = indices[:support_size]
    query_idx = indices[support_size:support_size + query_size]

    support_set = torch.utils.data.Subset(ds, support_idx.tolist())
    query_set = torch.utils.data.Subset(ds, query_idx.tolist())

    pin = device.type == "cuda"

    support_loader = DataLoader(
        support_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    query_loader = DataLoader(
        query_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    if len(support_loader) == 0:
        raise RuntimeError("Support loader has no batches. Reduce batch_size or increase support_size.")

    in_dim = ds.inputs.shape[1]
    out_dim = ds.targets.shape[1]

    # build base meta model
    if mode == "fk":
        base_model = ResidualMLP(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
        ).to(device)
    else:  # ik
        base_model = IKResNetDualHead(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            out_dim=out_dim,
        ).to(device)

    base_model.load_state_dict(ckpt["model_state_dict"], strict=True)
    base_model.eval()

    print(f"[INFO] Dataset: N={N}, support={support_size}, query={query_size}")
    print(f"[INFO] in_dim={in_dim}, out_dim={out_dim}, hidden_dim={hidden_dim}, num_blocks={num_blocks}")
    if mode == "ik":
        print(f"[INFO] IK aux_loss_weight = {aux_weight:.3f}")

    criterion = nn.MSELoss()

    # pre-cache support batches for deterministic behaviour
    support_batches = list(iter(support_loader))
    if len(support_batches) == 0:
        raise RuntimeError("No support batches after iter().")

    results = []

    for steps in steps_list:
        # wall-clock timer starts *per configuration*
        t0 = time.perf_counter()

        # create a fresh adapted model
        if mode == "fk":
            adapted = ResidualMLP(
                in_dim=in_dim,
                out_dim=out_dim,
                hidden_dim=hidden_dim,
                num_blocks=num_blocks,
            ).to(device)
        else:
            adapted = IKResNetDualHead(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                num_blocks=num_blocks,
                out_dim=out_dim,
            ).to(device)

        adapted.load_state_dict(base_model.state_dict(), strict=True)
        adapted.train()
        opt = torch.optim.Adam(adapted.parameters(), lr=inner_lr)

        # inner updates
        for step in range(steps):
            x, y = support_batches[step % len(support_batches)]
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad()
            if mode == "fk":
                y_hat = adapted(x)
                loss = criterion(y_hat, y)
            else:  # ik
                q_pred, pos_pred, ori_pred = adapted(x)
                loss_q = criterion(q_pred, y)
                pos_target = x[:, :3]
                ori_target = x[:, 3:7]
                loss_pos = criterion(pos_pred, pos_target)
                loss_ori = criterion(ori_pred, ori_target)
                loss = loss_q + aux_weight * (loss_pos + loss_ori)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapted.parameters(), max_norm=1.0)
            opt.step()

        # evaluation
        adapted.eval()
        if mode == "fk":
            pos_rmse_m, ori_rmse_deg = eval_fk_metrics(
                adapted,
                query_loader,
                device,
                ds.target_mean,
                ds.target_std,
            )
            t1 = time.perf_counter()
            elapsed = t1 - t0
            print(
                f"[FK RESULT] steps={steps:4d} | "
                f"pos_RMSE = {pos_rmse_m:.6f} m | "
                f"ori_RMSE = {ori_rmse_deg:.3f} deg | "
                f"time = {elapsed:.2f} s"
            )
            results.append((steps, pos_rmse_m, ori_rmse_deg, elapsed))
        else:
            joint_rmse_deg = eval_ik_joint_rmse_deg(
                adapted,
                query_loader,
                device,
                ds.target_mean,
                ds.target_std,
            )
            t1 = time.perf_counter()
            elapsed = t1 - t0
            print(
                f"[IK RESULT] steps={steps:4d} | "
                f"joint_RMSE = {joint_rmse_deg:.4f} deg | "
                f"time = {elapsed:.2f} s"
            )
            results.append((steps, joint_rmse_deg, elapsed))

        del adapted, opt  # free memory

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Adapt a meta-trained FK/IK model on a single DOF dataset "
                    "and report error vs. number of fine-tuning steps, "
                    "including adaptation time."
    )
    p.add_argument("--meta_checkpoint", type=str, required=True,
                   help="Path to meta_{mode}_5_6_7dof_best.pt from train_meta_kinematics_reptile.py")
    p.add_argument("--data", type=str, required=True,
                   help="Path to DOF dataset (.csv/.pt or dir of shards)")

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--support_size", type=int, default=2000,
                   help="Number of samples for adaptation (support set).")
    p.add_argument("--query_size", type=int, default=50000,
                   help="Number of samples for evaluation (query set).")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--inner_lr", type=float, default=1e-3)
    p.add_argument("--aux_loss_weight", type=float, default=None,
                   help="Optional override for IK aux loss weight; if None, use checkpoint value.")
    p.add_argument("--steps", type=str, default="0,1,5,10,20,50",
                   help="Comma-separated list of adaptation step counts.")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    steps_list = [int(s) for s in args.steps.split(",") if s.strip()]

    print(f"[INFO] Adapting on device {device} with steps {steps_list}")
    adapt_on_task(
        meta_ckpt_path=args.meta_checkpoint,
        data_path=args.data,
        device=device,
        steps_list=steps_list,
        support_size=args.support_size,
        query_size=args.query_size,
        batch_size=args.batch_size,
        inner_lr=args.inner_lr,
        aux_loss_weight_override=args.aux_loss_weight,
        num_workers=args.num_workers,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
