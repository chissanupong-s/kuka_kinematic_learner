"""
Stage 2: Shared meta-kinematics training of the MetaKinFormer over the
union of all three DoF datasets (5, 6, 7-DoF).

Drop-in equivalent of `train_kinematics_nn_pol_pt_2.py` Stage-2 mode but
using the MetaKinFormer architecture instead of ResMLP_Mask.

Each training step samples a task k uniformly from {5, 6, 7}, then samples
a minibatch from D_k. The model sees all three configurations through a
single set of parameters, with the active-DoF mask passed through to the
attention computation.

Usage:
    python train_multitask_transformer.py \\
        --data_5 data/narrowed/5DOF_8deg.pt_part000.pt \\
        --data_6 data/narrowed/6DOF_12deg.pt_part000.pt \\
        --data_7 data/narrowed/7DOF_15deg/7DOF_15deg_part000.pt \\
        --init_from_stage1 \\
            runs/singletask_transformer_5dof/ckpts/best.pt \\
            runs/singletask_transformer_6dof/ckpts/best.pt \\
            runs/singletask_transformer_7dof/ckpts/best.pt \\
        --d_model 128 --n_layers 4 --n_heads 4 \\
        --batch_size 4096 --steps 1000000 --lr 3e-4 --warmup 2000 \\
        --device cuda --enable_tf32 \\
        --tb_logdir runs/multitask_transformer \\
        --save_dir runs/multitask_transformer/ckpts
"""
from __future__ import annotations
import argparse
import math
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from meta_kin_former import MetaKinFormer
from train_singletask_transformer import safe_torch_load, FKDataset


# --------------- Multi-task data sampler ---------------

class TaskSampler:
    """Holds three FKDataset instances and samples uniformly across them.

    Per call to `sample_batch(rng)`:
      1. Pick a task k ~ Uniform({5, 6, 7})
      2. Sample a minibatch of size `batch_size` from D_k (with replacement)
      3. Return (q, mask, p, dof) — dof identifies the task
    """

    def __init__(self, datasets: Dict[int, FKDataset], batch_size: int, device):
        self.datasets = datasets  # {5: FKDataset, 6: FKDataset, 7: FKDataset}
        self.dofs = sorted(datasets.keys())
        self.batch_size = batch_size
        self.device = device
        # Precompute mask tensors for each task on device
        self.masks_dev = {dof: ds.mask.to(device) for dof, ds in datasets.items()}
        # Pre-stage data on GPU for fast random sampling — only for small datasets!
        # For 50M samples it's too big. Instead, keep on CPU and index there.
        self.q_cpu = {dof: ds.q for dof, ds in datasets.items()}
        self.p_cpu = {dof: ds.p for dof, ds in datasets.items()}
        self.sizes = {dof: len(ds) for dof, ds in datasets.items()}

    def sample_batch(self, rng: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        # Pick a task uniformly
        idx = torch.randint(0, len(self.dofs), (1,), generator=rng).item()
        dof = self.dofs[idx]
        # Sample batch indices with replacement
        idxs = torch.randint(0, self.sizes[dof], (self.batch_size,), generator=rng)
        q = self.q_cpu[dof][idxs].to(self.device, non_blocking=True)
        p = self.p_cpu[dof][idxs].to(self.device, non_blocking=True)
        mask = self.masks_dev[dof].unsqueeze(0).expand(self.batch_size, -1)
        return q, mask, p, dof


# --------------- Init from Stage-1 average ---------------

def merge_stage1_ckpts(paths: List[str]) -> Dict[str, torch.Tensor]:
    """Average parameters from a list of Stage-1 single-task checkpoints.

    Each .pt file contains 'model_state_dict' from a MetaKinFormer.
    All checkpoints must have the same architecture/shape.
    """
    states = []
    for p in paths:
        ck = safe_torch_load(p, map_location="cpu")
        if "model_state_dict" not in ck:
            raise ValueError(f"{p}: no 'model_state_dict' key")
        states.append(ck["model_state_dict"])
    if not states:
        raise ValueError("No Stage-1 checkpoints provided")

    avg = {}
    for k in states[0].keys():
        if states[0][k].dtype.is_floating_point:
            avg[k] = sum(s[k] for s in states) / float(len(states))
        else:
            # Keep integer/bool buffers from the first checkpoint
            avg[k] = states[0][k].clone()
    return avg


# --------------- Cosine schedule with warmup ---------------

def cosine_schedule(step: int, total_steps: int, warmup: int, base_lr: float, min_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    progress = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# --------------- Quick eval ---------------

@torch.no_grad()
def quick_eval(model, sampler: TaskSampler, criterion, n_batches: int = 32, device="cuda"):
    """Eval each task by drawing a few minibatches; report per-task and mean loss."""
    model.eval()
    losses = {dof: [] for dof in sampler.dofs}
    rng_eval = torch.Generator().manual_seed(0)
    for _ in range(n_batches):
        for dof in sampler.dofs:
            idxs = torch.randint(0, sampler.sizes[dof], (sampler.batch_size,), generator=rng_eval)
            q = sampler.q_cpu[dof][idxs].to(device, non_blocking=True)
            p = sampler.p_cpu[dof][idxs].to(device, non_blocking=True)
            mask = sampler.masks_dev[dof].unsqueeze(0).expand(sampler.batch_size, -1)
            pred = model(q, mask)
            losses[dof].append(criterion(pred, p).item())
    model.train()
    return {dof: float(np.mean(losses[dof])) for dof in sampler.dofs}


# --------------- Main ---------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_5", type=str, required=True)
    p.add_argument("--data_6", type=str, required=True)
    p.add_argument("--data_7", type=str, required=True)
    p.add_argument("--init_from_stage1", nargs=3, default=None,
                   help="3 Stage-1 best.pt paths (5/6/7-DoF). If absent: random init.")
    # Model
    p.add_argument("--n_joints", type=int, default=7)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--dim_feedforward", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    # Training
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # Eval / logging
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--save_every", type=int, default=10000)
    # System
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--enable_tf32", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    # Output
    p.add_argument("--tb_logdir", type=str, default="runs/multitask_transformer")
    p.add_argument("--tb_name", type=str, default="run")
    p.add_argument("--save_dir", type=str, default="runs/multitask_transformer/ckpts")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.tb_logdir, args.tb_name))

    # Load datasets
    print("[INFO] Loading datasets")
    raw_5 = safe_torch_load(args.data_5, map_location="cpu")
    raw_6 = safe_torch_load(args.data_6, map_location="cpu")
    raw_7 = safe_torch_load(args.data_7, map_location="cpu")
    if isinstance(raw_5, dict): raw_5 = next(iter(raw_5.values()))
    if isinstance(raw_6, dict): raw_6 = next(iter(raw_6.values()))
    if isinstance(raw_7, dict): raw_7 = next(iter(raw_7.values()))

    ds_5 = FKDataset(raw_5, dof=5, n_joints_max=args.n_joints)
    ds_6 = FKDataset(raw_6, dof=6, n_joints_max=args.n_joints)
    ds_7 = FKDataset(raw_7, dof=7, n_joints_max=args.n_joints)
    print(f"  5-DoF: {len(ds_5):,} samples")
    print(f"  6-DoF: {len(ds_6):,} samples")
    print(f"  7-DoF: {len(ds_7):,} samples")

    # Save merged stats
    stats = {
        "dof_5": {"q_mean": ds_5.q_mean, "q_std": ds_5.q_std, "p_mean": ds_5.p_mean, "p_std": ds_5.p_std, "mask": ds_5.mask},
        "dof_6": {"q_mean": ds_6.q_mean, "q_std": ds_6.q_std, "p_mean": ds_6.p_mean, "p_std": ds_6.p_std, "mask": ds_6.mask},
        "dof_7": {"q_mean": ds_7.q_mean, "q_std": ds_7.q_std, "p_mean": ds_7.p_mean, "p_std": ds_7.p_std, "mask": ds_7.mask},
        "n_joints": args.n_joints,
    }
    torch.save(stats, os.path.join(args.save_dir, "stats.pt"))

    sampler = TaskSampler({5: ds_5, 6: ds_6, 7: ds_7}, batch_size=args.batch_size, device=device)

    # Model
    model = MetaKinFormer(
        n_joints=args.n_joints, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, dim_feedforward=args.dim_feedforward, dropout=args.dropout,
        out_dim=7,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] MetaKinFormer params: {n_params:,}")

    # Optional warm-start from Stage-1 averaged checkpoints
    if args.init_from_stage1:
        merged = merge_stage1_ckpts(args.init_from_stage1)
        missing, unexpected = model.load_state_dict(merged, strict=False)
        print(f"[INFO] Warm-started from Stage-1 average. Missing: {missing}, Unexpected: {unexpected}")
    else:
        print("[INFO] No Stage-1 init given — using random initialisation")

    # Optimiser
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                             betas=(0.9, 0.999), eps=1e-8)
    criterion = nn.MSELoss()

    rng = torch.Generator(device="cpu").manual_seed(args.seed)

    # Training loop
    best_eval = float("inf")
    best_step = -1
    print(f"[INFO] Starting Stage-2 training for {args.steps} steps")
    t0 = time.time()
    for step in tqdm(range(args.steps), desc="steps"):
        # LR schedule (cosine with warmup)
        lr = cosine_schedule(step, args.steps, args.warmup, args.lr, args.min_lr)
        for pg in optim.param_groups:
            pg["lr"] = lr

        q, mask, p_target, dof = sampler.sample_batch(rng)
        optim.zero_grad(set_to_none=True)
        pred = model(q, mask)
        loss = criterion(pred, p_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optim.step()

        if (step + 1) % args.log_every == 0:
            writer.add_scalar(f"train/{dof}dof_loss", loss.item(), step)
            writer.add_scalar("train/lr", lr, step)

        if (step + 1) % args.eval_every == 0:
            eval_losses = quick_eval(model, sampler, criterion, n_batches=8, device=device)
            mean_eval = float(np.mean(list(eval_losses.values())))
            writer.add_scalar("eval/avg_loss_1batch_each_task", mean_eval, step)
            for dof_k, l in eval_losses.items():
                writer.add_scalar(f"eval/dof_{dof_k}_loss", l, step)

            if mean_eval < best_eval:
                best_eval = mean_eval
                best_step = step
                torch.save({
                    "step": step, "model_state_dict": model.state_dict(),
                    "args": vars(args), "eval_loss": mean_eval, "stats": stats,
                }, os.path.join(args.save_dir, "multitask_fk_best.pt"))

        if (step + 1) % args.save_every == 0:
            torch.save({
                "step": step, "model_state_dict": model.state_dict(),
                "args": vars(args), "stats": stats,
            }, os.path.join(args.save_dir, f"multitask_fk_step{step+1}.pt"))

    elapsed = time.time() - t0
    torch.save({
        "step": args.steps - 1, "model_state_dict": model.state_dict(),
        "args": vars(args), "stats": stats,
    }, os.path.join(args.save_dir, "multitask_fk_last.pt"))

    print(f"\n[INFO] Stage-2 done in {elapsed/60:.1f} min")
    print(f"[INFO] best eval = {best_eval:.6e} at step {best_step}")
    writer.close()


if __name__ == "__main__":
    main()
