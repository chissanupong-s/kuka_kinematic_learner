#!/usr/bin/env python3
"""
train_multitask_masked_decay.py

Single model across 5/6/7-DOF (iiwa) using:
- mask conditioning (additive projection; keeps old checkpoint compatibility)
- masked IK loss (inactive joints ignored)
- simple supervised multi-task training (not meta)

This version also:
- saves per-task normalization (task_norm) into checkpoints so eval scripts can
  normalize inputs exactly like training
- supports safe checkpoint loading with PyTorch 2.6+ (weights_only default)
- optionally supports averaging multiple init checkpoints (5/6/7)
- supports LR warmup + LR decay (step or cosine)

Resume support:
- --resume_ckpt loads model weights and continues global step counter
- Writes TensorBoard events into the SAME log_dir and uses purge_step to avoid step overlap
"""

import argparse
import math
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

POSE_COLS = ["x", "y", "z", "qw", "qx", "qy", "qz"]
JOINT_COLS = [f"q{i}" for i in range(1, 8)]


# -------------------------
# Safe torch.load (PyTorch 2.6+ compatibility)
# -------------------------
def safe_torch_load(path: str, map_location):
    """Best-effort torch.load that works across torch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)


def strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    keys = list(sd.keys())
    if not keys:
        return sd
    n_module = sum(k.startswith("module.") for k in keys)
    if n_module > len(keys) // 2:
        return {k[len("module.") :]: v for k, v in sd.items()}
    return sd


def extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj:
        sd = ckpt_obj["model_state_dict"]
    else:
        sd = ckpt_obj
    if not isinstance(sd, dict):
        raise ValueError("Checkpoint does not contain a state_dict dict.")
    return strip_module_prefix(sd)


def load_state_dict_any(path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    ckpt = safe_torch_load(path, map_location=device)
    return extract_state_dict(ckpt)


def average_state_dicts(state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Averages only keys present in ALL dicts and with identical tensor shapes."""
    if not state_dicts:
        return {}

    common = set(state_dicts[0].keys())
    for sd in state_dicts[1:]:
        common &= set(sd.keys())

    avg_sd: Dict[str, torch.Tensor] = {}
    for k in sorted(common):
        vals = [sd[k] for sd in state_dicts]
        if not all(isinstance(v, torch.Tensor) for v in vals):
            continue
        if not all(v.shape == vals[0].shape for v in vals):
            continue
        stacked = torch.stack([v.detach().float().cpu() for v in vals], dim=0).mean(dim=0)
        avg_sd[k] = stacked.to(dtype=vals[0].dtype)
    return avg_sd


def init_model_from_checkpoints(model: nn.Module, ckpt_paths: List[str], device: torch.device) -> bool:
    """Initialize model from a single checkpoint or an average over multiple."""
    ckpt_paths = [p for p in ckpt_paths if p and os.path.exists(p)]
    if not ckpt_paths:
        return False

    if len(ckpt_paths) == 1:
        sd = load_state_dict_any(ckpt_paths[0], device)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[INFO] Loaded init from {ckpt_paths[0]}")
        if missing:
            print(f"[INFO]  missing keys (ok for mask_proj): {missing[:8]}{'...' if len(missing) > 8 else ''}")
        if unexpected:
            print(f"[WARN]  unexpected keys: {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")
        return True

    sds = [load_state_dict_any(p, device) for p in ckpt_paths]
    avg_sd = average_state_dicts(sds)
    missing, unexpected = model.load_state_dict(avg_sd, strict=False)
    print(f"[INFO] Initialising from AVERAGED checkpoints ({len(ckpt_paths)}):")
    for p in ckpt_paths:
        print(f"       - {p}")
    if missing:
        print(f"[INFO]  missing keys (ok for mask_proj): {missing[:8]}{'...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"[WARN]  unexpected keys: {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")
    return True


def load_resume_checkpoint(path: str, device: torch.device) -> Tuple[Dict[str, torch.Tensor], Dict]:
    ckpt = safe_torch_load(path, map_location=device)
    if isinstance(ckpt, dict):
        sd = extract_state_dict(ckpt)
        meta = ckpt
    else:
        sd = extract_state_dict(ckpt)
        meta = {}
    return sd, meta


# -------------------------
# Data loading
# -------------------------
def load_dataset_tensor(path: str) -> torch.Tensor:
    """Loads dataset and returns Tensor [N,14] float32 on CPU."""
    print(f"[INFO] Loading dataset from {path}")
    cols = JOINT_COLS + POSE_COLS

    if os.path.isdir(path):
        shard_files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(".pt") or f.endswith(".bin")])
        if not shard_files:
            raise ValueError(f"No .pt/.bin shards found in directory: {path}")

        arrs = []
        total = 0
        for fp in shard_files:
            obj = safe_torch_load(fp, map_location="cpu")
            t = obj["data"] if isinstance(obj, dict) and "data" in obj else obj
            if not isinstance(t, torch.Tensor) or t.ndim != 2 or t.shape[1] != 14:
                raise ValueError(f"Shard {fp} must be Tensor [N,14]")
            arrs.append(t.cpu().float().numpy())
            total += t.shape[0]
        print(f"[INFO]  total rows across shards: {total}")
        return torch.from_numpy(np.concatenate(arrs, axis=0)).float()

    if path.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(path)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"CSV {path} missing columns: {missing}")
        return torch.from_numpy(df[cols].values.astype(np.float32)).float()

    if path.endswith(".pt") or path.endswith(".bin"):
        obj = safe_torch_load(path, map_location="cpu")
        t = obj["data"] if isinstance(obj, dict) and "data" in obj else obj
        if not isinstance(t, torch.Tensor) or t.ndim != 2 or t.shape[1] != 14:
            raise ValueError(f"{path} must contain Tensor [N,14]")
        return t.float()

    raise ValueError(f"Unsupported dataset path: {path}")


class MaskedKinematicsDataset(Dataset):
    """
    Returns: (x_norm, y_norm, mask7)

    mode:
      fk: x = q(7),    y = pose(7)
      ik: x = pose(7), y = q(7)

    Normalization is per-task (dataset-specific).
    """
    def __init__(self, full_tensor: torch.Tensor, mode: str, mask7: torch.Tensor, std_floor_q_rad: float = 0.0):
        super().__init__()
        assert full_tensor.ndim == 2 and full_tensor.shape[1] == 14
        assert mask7.shape == (7,)
        self.mode = mode
        self.mask7 = mask7.float()

        if mode == "fk":
            X = full_tensor[:, 0:7]
            Y = full_tensor[:, 7:14]
        elif mode == "ik":
            X = full_tensor[:, 7:14]
            Y = full_tensor[:, 0:7]
        else:
            raise ValueError("mode must be 'fk' or 'ik'")

        X = X.cpu().numpy().astype(np.float32)
        Y = Y.cpu().numpy().astype(np.float32)

        x_mean = X.mean(axis=0, keepdims=True)
        x_std = X.std(axis=0, keepdims=True) + 1e-8

        y_mean = Y.mean(axis=0, keepdims=True)
        y_std = Y.std(axis=0, keepdims=True) + 1e-8

        if mode == "ik" and std_floor_q_rad > 0.0:
            y_std = np.maximum(y_std, std_floor_q_rad)

        self.x_norm = (X - x_mean) / x_std
        self.y_norm = (Y - y_mean) / y_std

        self.x_mean, self.x_std = x_mean, x_std
        self.y_mean, self.y_std = y_mean, y_std

    def __len__(self):
        return self.x_norm.shape[0]

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.x_norm[idx]).float()
        y = torch.from_numpy(self.y_norm[idx]).float()
        return x, y, self.mask7


def stats_dict_from_dataset(ds: MaskedKinematicsDataset) -> Dict[str, list]:
    return {"x_mean": ds.x_mean.tolist(), "x_std": ds.x_std.tolist(), "y_mean": ds.y_mean.tolist(), "y_std": ds.y_std.tolist()}


def make_infinite_loader(ds: Dataset, batch_size: int, num_workers: int, device: torch.device):
    pin = device.type == "cuda"
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )

    def gen():
        while True:
            for b in loader:
                yield b

    return gen()


# -------------------------
# Models (mask-conditioned)
# -------------------------
class ResBlock(nn.Module):
    def __init__(self, dim: int, p_drop: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.drop(h)
        h = self.fc2(h)
        return self.act(h + x)


class ResidualMLP_Mask(nn.Module):
    def __init__(self, in_dim: int = 7, out_dim: int = 7, hidden_dim: int = 512, num_blocks: int = 8):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.mask_proj = nn.Linear(7, hidden_dim, bias=False)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ReLU()
        nn.init.zeros_(self.mask_proj.weight)

    def forward(self, x7: torch.Tensor, mask7: torch.Tensor):
        h = self.act(self.fc_in(x7) + self.mask_proj(mask7))
        for blk in self.blocks:
            h = blk(h)
        return self.fc_out(h)


class IKResNetDualHead_Mask(nn.Module):
    def __init__(self, in_dim: int = 7, out_dim: int = 7, hidden_dim: int = 1024, num_blocks: int = 8):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.mask_proj = nn.Linear(7, hidden_dim, bias=False)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])
        self.fc_joint = nn.Linear(hidden_dim, out_dim)
        self.fc_pos = nn.Linear(hidden_dim, 3)
        self.fc_ori = nn.Linear(hidden_dim, 4)
        self.act = nn.ReLU()
        nn.init.zeros_(self.mask_proj.weight)

    def forward(self, x7: torch.Tensor, mask7: torch.Tensor):
        h = self.act(self.fc_in(x7) + self.mask_proj(mask7))
        for blk in self.blocks:
            h = blk(h)
        return self.fc_joint(h), self.fc_pos(h), self.fc_ori(h)


# -------------------------
# Loss
# -------------------------
def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    se = (pred - target) ** 2
    se = se * mask
    denom = mask.sum(dim=1).clamp_min(eps)
    per = se.sum(dim=1) / denom
    return per.mean()


# -------------------------
# LR schedule
# -------------------------
def parse_milestones(s: str) -> List[int]:
    if not s:
        return []
    out = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return sorted(set(out))


def set_optimizer_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for pg in opt.param_groups:
        pg["lr"] = lr


def compute_lr(
    base_lr: float,
    lr_min: float,
    step: int,
    total_steps: int,
    schedule: str,
    warmup_steps: int,
    step_milestones: List[int],
    step_gamma: float,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)

    if schedule == "none":
        return base_lr

    after = step - warmup_steps
    denom = max(1, total_steps - warmup_steps)

    if schedule == "cosine":
        t = min(max(after / denom, 0.0), 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * t))
        return lr_min + (base_lr - lr_min) * cos

    if schedule == "step":
        k = 0
        for m in step_milestones:
            if step >= m:
                k += 1
        return max(lr_min, base_lr * (step_gamma ** k))

    raise ValueError(f"Unknown lr_schedule: {schedule}")


# -------------------------
# Train loop
# -------------------------
def train_multitask(
    mode: str,
    model: nn.Module,
    task_gens: Dict[str, object],
    aux_weight: float,
    base_lr: float,
    extra_steps: int,
    start_step: int,
    total_steps_for_schedule: int,
    device: torch.device,
    log_dir: str,
    out_dir: str,
    task_norm: Dict[str, Dict[str, list]],
    lr_schedule: str = "none",
    lr_min: float = 1e-5,
    warmup_steps: int = 0,
    step_milestones: Optional[List[int]] = None,
    step_gamma: float = 0.5,
    save_every: int = 2000,
    grad_clip: float = 1.0,
    print_every: int = 50,
    eval_every: int = 500,
    log_lr_every: int = 50,
):
    os.makedirs(out_dir, exist_ok=True)
    if start_step > 0:
        writer = SummaryWriter(log_dir=log_dir, purge_step=start_step)
    else:
        writer = SummaryWriter(log_dir=log_dir)

    opt = torch.optim.Adam(model.parameters(), lr=base_lr)
    mse = nn.MSELoss()

    task_names = list(task_gens.keys())
    print(f"[INFO] Training mode={mode} on tasks={task_names}")
    print(f"[INFO] start_step={start_step} | extra_steps={extra_steps} | end_step={start_step + extra_steps}")
    print(f"[INFO] LR schedule={lr_schedule} | base_lr={base_lr} | lr_min={lr_min} | warmup={warmup_steps} | total_steps_for_schedule={total_steps_for_schedule}")

    if step_milestones is None:
        step_milestones = []

    best = float("inf")
    best_path = os.path.join(out_dir, f"multitask_{mode}_best.pt")

    model.to(device)
    device_type = device.type

    pbar = tqdm(
        range(start_step, start_step + extra_steps),
        desc=f"multitask-train({mode})",
        ncols=140,
        dynamic_ncols=True,
        leave=True,
        file=sys.stdout,
        mininterval=0.5,
    )

    for gstep in pbar:
        cur_lr = compute_lr(
            base_lr=base_lr,
            lr_min=lr_min,
            step=gstep,
            total_steps=total_steps_for_schedule,
            schedule=lr_schedule,
            warmup_steps=warmup_steps,
            step_milestones=step_milestones,
            step_gamma=step_gamma,
        )
        set_optimizer_lr(opt, cur_lr)

        tname = random.choice(task_names)
        x, y, mask = next(task_gens[tname])

        x = x.to(device, non_blocking=(device_type == "cuda"))
        y = y.to(device, non_blocking=(device_type == "cuda"))
        mask = mask.to(device, non_blocking=(device_type == "cuda"))

        model.train()
        opt.zero_grad(set_to_none=True)

        if mode == "fk":
            yhat = model(x, mask)
            loss = mse(yhat, y)
        else:
            q_pred, pos_pred, ori_pred = model(x, mask)
            loss_q = masked_mse(q_pred, y, mask)
            pos_t = x[:, :3]
            ori_t = x[:, 3:7]
            loss_pos = mse(pos_pred, pos_t)
            loss_ori = mse(ori_pred, ori_t)
            loss = loss_q + aux_weight * (loss_pos + loss_ori)

        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        writer.add_scalar(f"train/{tname}_loss", loss.item(), gstep)
        if log_lr_every and (gstep + 1) % log_lr_every == 0:
            writer.add_scalar("train/lr", cur_lr, gstep)

        if (gstep + 1) % max(1, print_every) == 0:
            pbar.set_postfix(step=gstep + 1, task=tname, loss=f"{loss.item():.4f}", lr=f"{cur_lr:.2e}")

        if (gstep + 1) % max(1, eval_every) == 0:
            model.eval()
            with torch.no_grad():
                losses = []
                for tn in task_names:
                    xb, yb, mb = next(task_gens[tn])
                    xb = xb.to(device, non_blocking=(device_type == "cuda"))
                    yb = yb.to(device, non_blocking=(device_type == "cuda"))
                    mb = mb.to(device, non_blocking=(device_type == "cuda"))

                    if mode == "fk":
                        l = mse(model(xb, mb), yb).item()
                    else:
                        qp, pp, op = model(xb, mb)
                        lq = masked_mse(qp, yb, mb).item()
                        lp = mse(pp, xb[:, :3]).item()
                        lo = mse(op, xb[:, 3:7]).item()
                        l = lq + aux_weight * (lp + lo)
                    losses.append(l)

                avg = float(np.mean(losses))
                writer.add_scalar("eval/avg_loss_1batch_each_task", avg, gstep)

                if avg < best:
                    best = avg
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "mode": mode,
                            "aux_loss_weight": aux_weight,
                            "task_norm": task_norm,
                            "step": gstep + 1,
                            "total_steps": total_steps_for_schedule,
                            "lr_schedule": lr_schedule,
                            "base_lr": base_lr,
                            "lr_min": lr_min,
                            "warmup_steps": warmup_steps,
                            "step_milestones": step_milestones,
                            "step_gamma": step_gamma,
                        },
                        best_path,
                    )
                    print(f"[INFO] New best avg loss {best:.6f} -> {best_path}")

        if save_every and (gstep + 1) % save_every == 0:
            path = os.path.join(out_dir, f"multitask_{mode}_step{gstep+1}.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "mode": mode,
                    "aux_loss_weight": aux_weight,
                    "task_norm": task_norm,
                    "step": gstep + 1,
                    "total_steps": total_steps_for_schedule,
                    "lr_schedule": lr_schedule,
                    "base_lr": base_lr,
                    "lr_min": lr_min,
                    "warmup_steps": warmup_steps,
                    "step_milestones": step_milestones,
                    "step_gamma": step_gamma,
                },
                path,
            )

    writer.close()
    print(f"[INFO] Done. Best avg loss (quick-eval): {best:.6f}")
    print(f"[INFO] Best ckpt: {best_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["fk", "ik"], required=True)

    p.add_argument("--task_5dof", type=str, required=True)
    p.add_argument("--task_6dof", type=str, required=True)
    p.add_argument("--task_7dof", type=str, required=True)

    p.add_argument("--hidden_dim", type=int, default=1024)
    p.add_argument("--num_blocks", type=int, default=8)

    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--num_workers", type=int, default=8)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--steps", type=int, default=1_000_000,
                   help="From scratch: total steps. With --resume_ckpt: additional steps.")

    p.add_argument("--aux_loss_weight", type=float, default=0.03)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--log_dir", type=str, default="runs/multitask_masked")
    p.add_argument("--out_dir", type=str, default="runs/multitask_masked_ckpts")  # kept for compatibility

    p.add_argument("--max_samples_per_task", type=int, default=15_000_000)

    # Init (averaged warm start)
    p.add_argument("--init_ckpt", type=str, default=None)
    p.add_argument("--init_ckpt_5dof", type=str, default=None)
    p.add_argument("--init_ckpt_6dof", type=str, default=None)
    p.add_argument("--init_ckpt_7dof", type=str, default=None)

    # Resume
    p.add_argument("--resume_ckpt", type=str, default=None, help="Resume from multitask_*_step*.pt or *_best.pt")
    p.add_argument("--resume_strict", action="store_true", help="Use strict=True when loading resume ckpt.")
    p.add_argument("--total_steps", type=int, default=0,
                   help="Total steps for LR schedule when resuming. If 0, uses max(ckpt.total_steps, start_step + --steps).")

    p.add_argument("--std_floor_q_deg", type=float, default=1.0)

    p.add_argument("--print_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--save_every", type=int, default=5000)

    p.add_argument("--lr_schedule", choices=["none", "step", "cosine"], default="cosine")
    p.add_argument("--lr_min", type=float, default=1e-5)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--step_milestones", type=str, default="100000,200000,400000")
    p.add_argument("--step_gamma", type=float, default=0.5)
    p.add_argument("--log_lr_every", type=int, default=50)

    return p.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    print(f"[INFO] Using device: {device}")

    mask_5 = torch.tensor([1, 1, 1, 1, 1, 0, 0], dtype=torch.float32)
    mask_6 = torch.tensor([1, 1, 1, 1, 1, 1, 0], dtype=torch.float32)
    mask_7 = torch.tensor([1, 1, 1, 1, 1, 1, 1], dtype=torch.float32)

    std_floor_q_rad = float(args.std_floor_q_deg) * np.pi / 180.0

    def load_task(path: str) -> torch.Tensor:
        t = load_dataset_tensor(path)
        if args.max_samples_per_task and t.shape[0] > args.max_samples_per_task:
            idx = torch.randperm(t.shape[0])[: args.max_samples_per_task]
            t = t[idx]
            print(f"[INFO]  subsampled to {t.shape[0]} rows")
        return t

    t5 = load_task(args.task_5dof)
    t6 = load_task(args.task_6dof)
    t7 = load_task(args.task_7dof)

    ds5 = MaskedKinematicsDataset(t5, mode=args.mode, mask7=mask_5, std_floor_q_rad=std_floor_q_rad)
    ds6 = MaskedKinematicsDataset(t6, mode=args.mode, mask7=mask_6, std_floor_q_rad=std_floor_q_rad)
    ds7 = MaskedKinematicsDataset(t7, mode=args.mode, mask7=mask_7, std_floor_q_rad=std_floor_q_rad)

    task_norm = {
        "5dof": stats_dict_from_dataset(ds5),
        "6dof": stats_dict_from_dataset(ds6),
        "7dof": stats_dict_from_dataset(ds7),
    }

    task_gens = {
        "5dof": make_infinite_loader(ds5, args.batch_size, args.num_workers, device),
        "6dof": make_infinite_loader(ds6, args.batch_size, args.num_workers, device),
        "7dof": make_infinite_loader(ds7, args.batch_size, args.num_workers, device),
    }

    if args.mode == "fk":
        model = ResidualMLP_Mask(in_dim=7, out_dim=7, hidden_dim=args.hidden_dim, num_blocks=args.num_blocks)
    else:
        model = IKResNetDualHead_Mask(in_dim=7, out_dim=7, hidden_dim=args.hidden_dim, num_blocks=args.num_blocks)

    start_step = 0
    resume_meta: Dict = {}

    if args.resume_ckpt:
        if not os.path.exists(args.resume_ckpt):
            raise FileNotFoundError(f"--resume_ckpt not found: {args.resume_ckpt}")
        sd, resume_meta = load_resume_checkpoint(args.resume_ckpt, device)
        missing, unexpected = model.load_state_dict(sd, strict=args.resume_strict)
        start_step = int(resume_meta.get("step", 0)) if isinstance(resume_meta, dict) else 0
        print(f"[INFO] Resumed model from {args.resume_ckpt} at step={start_step}")
        if missing:
            print(f"[INFO]  missing keys (ok for mask_proj): {missing[:8]}{'...' if len(missing) > 8 else ''}")
        if unexpected:
            print(f"[WARN]  unexpected keys: {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")
    else:
        init_paths: List[str] = []
        if args.init_ckpt_5dof: init_paths.append(args.init_ckpt_5dof)
        if args.init_ckpt_6dof: init_paths.append(args.init_ckpt_6dof)
        if args.init_ckpt_7dof: init_paths.append(args.init_ckpt_7dof)
        if not init_paths and args.init_ckpt:
            init_paths = [args.init_ckpt]

        if init_paths:
            ok = init_model_from_checkpoints(model, init_paths, device)
            if not ok:
                print("[WARN] No valid init checkpoints found. Training from random init.")
        else:
            print("[INFO] No init checkpoints provided. Training from random init.")

    log_dir = os.path.join(args.log_dir, f"{args.mode}_iiwa_5_6_7")
    out_dir = log_dir
    os.makedirs(out_dir, exist_ok=True)

    milestones = parse_milestones(args.step_milestones) if args.lr_schedule == "step" else []

    if start_step > 0:
        extra_steps = args.steps
        ck_total = int(resume_meta.get("total_steps", 0)) if isinstance(resume_meta, dict) else 0
        inferred_total = max(ck_total, start_step + extra_steps)
        total_steps_for_schedule = args.total_steps if args.total_steps > 0 else inferred_total

        if ck_total == 0 and args.total_steps == 0 and args.lr_schedule != "none":
            print("[WARN] Resume ckpt has no total_steps. If you want identical LR schedule as before, pass --total_steps <old_total>.")

    else:
        extra_steps = args.steps
        total_steps_for_schedule = args.steps

    train_multitask(
        mode=args.mode,
        model=model,
        task_gens=task_gens,
        aux_weight=args.aux_loss_weight,
        base_lr=args.lr,
        extra_steps=extra_steps,
        start_step=start_step,
        total_steps_for_schedule=total_steps_for_schedule,
        device=device,
        log_dir=log_dir,
        out_dir=out_dir,
        task_norm=task_norm,
        lr_schedule=args.lr_schedule,
        lr_min=args.lr_min,
        warmup_steps=args.warmup_steps,
        step_milestones=milestones,
        step_gamma=args.step_gamma,
        save_every=args.save_every,
        grad_clip=args.grad_clip,
        print_every=args.print_every,
        eval_every=args.eval_every,
        log_lr_every=args.log_lr_every,
    )


if __name__ == "__main__":
    main()
