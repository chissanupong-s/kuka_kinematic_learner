#!/usr/bin/env python3
import argparse
import math
import os
import time
from typing import Optional, Tuple, List
import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset

# Adjust this import if your models / loader live elsewhere
from train_kinematics_nn_pol_pt_2 import ResidualMLP, IKResNetDualHead, load_dataset_tensor


# ---------------------------------------------------------------------------
# Dataset with optional precomputed normalisation stats
# ---------------------------------------------------------------------------

class KinematicsTensorDataset(Dataset):
    """
    Tensor dataset [N,14] with optional precomputed normalisation stats.

    mode = 'fk':
        X = [q1..q7], Y = [x,y,z,qw,qx,qy,qz]
    mode = 'ik':
        X = [x,y,z,qw,qx,qy,qz], Y = [q1..q7]
    """

    def __init__(
        self,
        full_tensor: torch.Tensor,
        mode: str,
        input_mean: Optional[torch.Tensor] = None,
        input_std: Optional[torch.Tensor] = None,
        target_mean: Optional[torch.Tensor] = None,
        target_std: Optional[torch.Tensor] = None,
    ):
        assert mode in ("fk", "ik")
        self.mode = mode

        full_tensor = full_tensor.float()
        if mode == "fk":
            X = full_tensor[:, 0:7]   # q
            Y = full_tensor[:, 7:14]  # pose
        else:  # ik
            X = full_tensor[:, 7:14]  # pose
            Y = full_tensor[:, 0:7]   # q

        # Compute stats if not provided (dataset-wide)
        if input_mean is None:
            input_mean = X.mean(dim=0, keepdim=True)
        if input_std is None:
            input_std = X.std(dim=0, unbiased=False, keepdim=True) + 1e-8
        if target_mean is None:
            target_mean = Y.mean(dim=0, keepdim=True)
        if target_std is None:
            target_std = Y.std(dim=0, unbiased=False, keepdim=True) + 1e-8

        self.input_mean = input_mean.detach().clone()
        self.input_std = input_std.detach().clone()
        self.target_mean = target_mean.detach().clone()
        self.target_std = target_std.detach().clone()

        self.inputs = (X - self.input_mean) / self.input_std
        self.targets = (Y - self.target_mean) / self.target_std

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_fk_pos_rmse_m(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> float:
    """
    Evaluate FK model: RMS error of (x,y,z) in metres.
    """
    model.eval()
    target_mean = target_mean.to(device)
    target_std = target_std.to(device)

    sum_sq = torch.zeros(3, device=device)
    n_samples = 0

    for x_norm, y_norm in loader:
        x_norm = x_norm.to(device)
        y_norm = y_norm.to(device)

        y_pred_norm = model(x_norm)

        # denormalise pose
        pose_pred = y_pred_norm * target_std + target_mean
        pose_true = y_norm * target_std + target_mean

        pos_pred = pose_pred[:, 0:3]
        pos_true = pose_true[:, 0:3]

        diff = pos_pred - pos_true  # [B,3] in metres
        sum_sq += (diff * diff).sum(dim=0)
        n_samples += diff.shape[0]

    mse_per_coord = sum_sq / max(n_samples, 1)
    rmse_per_coord = torch.sqrt(mse_per_coord)
    # 3D RMS (Euclidean) error
    rmse_3d = torch.sqrt((rmse_per_coord ** 2).sum() / 3.0)
    return rmse_3d.item()


@torch.no_grad()
def eval_ik_joint_rmse_deg(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> float:
    """
    Evaluate IK model: mean per-joint RMSE in degrees.

    Assumes targets are joint angles in radians normalised with target_mean/std.
    """
    model.eval()
    target_mean = target_mean.to(device)
    target_std = target_std.to(device)

    sum_sq = torch.zeros(7, device=device)
    n_samples = 0

    for x_norm, y_norm in loader:
        x_norm = x_norm.to(device)
        y_norm = y_norm.to(device)

        # model returns (q_pred_norm, pos_pred_norm, ori_pred_norm)
        q_pred_norm, _, _ = model(x_norm)

        # denormalise
        q_pred = q_pred_norm * target_std + target_mean
        q_true = y_norm * target_std + target_mean

        diff = q_pred - q_true  # radians
        sum_sq += (diff * diff).sum(dim=0)
        n_samples += diff.shape[0]

    mse_per_joint = sum_sq / max(n_samples, 1)
    rmse_rad = torch.sqrt(mse_per_joint)
    rmse_deg = rmse_rad * (180.0 / math.pi)
    return rmse_deg.mean().item()


# ---------------------------------------------------------------------------
# Adaptation
# ---------------------------------------------------------------------------

def build_model_from_checkpoint(ckpt: dict, mode: str, device: torch.device) -> Tuple[nn.Module, float]:
    """
    Rebuild model using info in checkpoint dict.
    Returns (model, aux_loss_weight).
    """
    hidden_dim = ckpt.get("hidden_dim", 1024)
    num_blocks = ckpt.get("num_blocks", 4)
    aux_loss_weight = float(ckpt.get("aux_loss_weight", 0.1))

    if mode == "fk":
        model = ResidualMLP(in_dim=7, out_dim=7, hidden_dim=hidden_dim, num_blocks=num_blocks)
    elif mode == "ik":
        model = IKResNetDualHead(in_dim=7, hidden_dim=hidden_dim, num_blocks=num_blocks)
    else:
        raise ValueError(f"Unknown mode '{mode}', expected 'fk' or 'ik'.")

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    return model, aux_loss_weight


def get_ckpt_norm_stats(ckpt: dict) -> Tuple[Optional[torch.Tensor], ...]:
    """
    Extract normalisation stats from checkpoint if present.
    Returns (input_mean, input_std, target_mean, target_std) or (None,...).
    """
    keys = ("input_mean", "input_std", "target_mean", "target_std")
    if not all(k in ckpt for k in keys):
        return None, None, None, None

    def to_tensor(v):
        if isinstance(v, torch.Tensor):
            return v.detach().clone().float()
        return torch.as_tensor(v, dtype=torch.float32)

    return tuple(to_tensor(ckpt[k]) for k in keys)


def make_support_query_splits(
    n_rows: int,
    support_size: int,
    query_size: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_rows)
    support_size = min(support_size, n_rows)
    query_size = min(query_size, n_rows - support_size)
    support_idx = perm[:support_size]
    query_idx = perm[support_size:support_size + query_size]
    return support_idx, query_idx


def cycle_batches(loader: DataLoader):
    """Infinite generator cycling over a loader."""
    while True:
        for batch in loader:
            yield batch


def adapt_on_task(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    # Load checkpoint (meta or single-task)
    try:
        ckpt = torch.load(args.meta_checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.meta_checkpoint, map_location=device)

    ckpt_mode = ckpt.get("mode", args.mode)
    if args.mode is None:
        mode = ckpt_mode
    else:
        if args.mode != ckpt_mode:
            print(f"[WARN] Overriding checkpoint mode {ckpt_mode} -> {args.mode}")
        mode = args.mode

    print(f"[INFO] Checkpoint mode: {mode}")

    base_model, aux_loss_weight = build_model_from_checkpoint(ckpt, mode, device)

    # Load dataset tensor
    full_tensor = load_dataset_tensor(args.data)
    n_rows = full_tensor.shape[0]
    print(f"[INFO] Loaded tensor of shape {tuple(full_tensor.shape)}")
    print(f"[INFO] Dataset: N={n_rows}, support={args.support_size}, query={args.query_size}")

    # Normalisation stats (prefer checkpoint's stats if present)
    ck_input_mean, ck_input_std, ck_target_mean, ck_target_std = get_ckpt_norm_stats(ckpt)

    ds = KinematicsTensorDataset(
        full_tensor,
        mode=mode,
        input_mean=ck_input_mean,
        input_std=ck_input_std,
        target_mean=ck_target_mean,
        target_std=ck_target_std,
    )

    support_idx, query_idx = make_support_query_splits(
        len(ds), args.support_size, args.query_size, args.seed
    )
    support_ds = Subset(ds, support_idx)
    query_ds = Subset(ds, query_idx)

    support_loader_for_cache = DataLoader(
        support_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    query_loader = DataLoader(
        query_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Pre-cache support batches (not strictly needed now, but cheap)
    support_batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for xb, yb in support_loader_for_cache:
        support_batches.append((xb, yb))
    if not support_batches:
        raise RuntimeError("Support set is empty.")

    # For evaluation, we need target stats from the dataset we actually used
    tgt_mean = ds.target_mean
    tgt_std = ds.target_std

    steps_list = [int(s) for s in args.steps.split(",") if s.strip()]

    for steps in steps_list:
        start_time = time.perf_counter()

        # fresh copy of base model for each steps setting
        model = copy.deepcopy(base_model)
        model.to(device)

        if steps > 0:
            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=args.inner_lr)
            step_loader = DataLoader(
                support_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            batch_cycle = cycle_batches(step_loader)

            for _ in range(steps):
                x_norm, y_norm = next(batch_cycle)
                x_norm = x_norm.to(device)
                y_norm = y_norm.to(device)

                opt.zero_grad()

                if mode == "fk":
                    y_pred_norm = model(x_norm)
                    loss = nn.functional.mse_loss(y_pred_norm, y_norm)
                else:  # ik
                    q_pred_norm, pos_pred_norm, ori_pred_norm = model(x_norm)
                    loss_q = nn.functional.mse_loss(q_pred_norm, y_norm)

                    pos_target = x_norm[:, 0:3]
                    ori_target = x_norm[:, 3:7]
                    loss_pos = nn.functional.mse_loss(pos_pred_norm, pos_target)
                    loss_ori = nn.functional.mse_loss(ori_pred_norm, ori_target)

                    loss = loss_q + aux_loss_weight * (loss_pos + loss_ori)

                loss.backward()
                opt.step()

        # Evaluate
        if mode == "fk":
            metric = eval_fk_pos_rmse_m(model, query_loader, device, tgt_mean, tgt_std)
            label = "pos_RMSE"
            unit = "m"
        else:
            metric = eval_ik_joint_rmse_deg(model, query_loader, device, tgt_mean, tgt_std)
            label = "joint_RMSE"
            unit = "deg"

        elapsed = time.perf_counter() - start_time
        print(
            f"[{mode.upper()} RESULT] steps={steps:4d} | {label} = {metric:.4f} {unit} | "
            f"time = {elapsed:6.2f} s"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adapt a (meta-trained or single-task) kinematics model.")
    p.add_argument("--mode", type=str, choices=["fk", "ik"], default=None,
                   help="Override checkpoint mode (fk/ik). If omitted, use checkpoint's mode.")
    p.add_argument("--meta_checkpoint", type=str, required=True,
                   help="Path to meta-trained or single-task checkpoint (.pt).")
    p.add_argument("--data", type=str, required=True,
                   help="Path to dataset (.pt/.csv or directory of shards).")
    p.add_argument("--support_size", type=int, default=2000,
                   help="Number of support samples for adaptation.")
    p.add_argument("--query_size", type=int, default=50000,
                   help="Number of query samples for evaluation.")
    p.add_argument("--batch_size", type=int, default=512,
                   help="Batch size for support/query loaders.")
    p.add_argument("--inner_lr", type=float, default=1e-3,
                   help="Inner loop learning rate for adaptation.")
    p.add_argument("--steps", type=str, default="0,5,10,20,50,200",
                   help="Comma-separated list of adaptation step counts, e.g. '0,10,50,200'.")
    p.add_argument("--device", type=str, default="cuda",
                   help="Device to use: 'cuda' or 'cpu'.")
    p.add_argument("--num_workers", type=int, default=8,
                   help="Number of DataLoader workers.")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for support/query split.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    adapt_on_task(args)


if __name__ == "__main__":
    main()
