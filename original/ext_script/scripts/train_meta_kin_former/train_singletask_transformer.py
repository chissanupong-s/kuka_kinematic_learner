"""
Stage 1: Single-task forward-kinematics training with the MetaKinFormer
(joint-as-token Transformer) architecture.

Drop-in replacement for `train_kinematics_nn_pol_pt_2.py` Stage-1 mode, but
trains a MetaKinFormer instead of a ResidualMLP.

Usage example (5-DoF):
    python train_singletask_transformer.py \\
        --data data/narrowed/5DOF_8deg.pt_part000.pt \\
        --dof 5 --mode fk \\
        --d_model 128 --n_layers 4 --n_heads 4 \\
        --batch_size 4096 --epochs 200 --lr 5e-4 \\
        --device cuda --enable_tf32 \\
        --tb_logdir runs/singletask_transformer_5dof \\
        --save_dir runs/singletask_transformer_5dof/ckpts

Outputs:
    save_dir/best.pt   — best checkpoint by validation loss
    save_dir/last.pt   — final checkpoint
    save_dir/stats.pt  — per-task standardisation statistics

Conventions matched to the ResMLP baseline:
    - 70/10/20 train/val/test split (deterministic by seed)
    - Adam optimiser, ReduceLROnPlateau scheduler
    - per-task standardisation of joint angles + pose vector
    - validation-loss-based early-stop checkpoint
"""
from __future__ import annotations
import argparse
import os
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from meta_kin_former import MetaKinFormer


# --------------- Data utilities ---------------

def safe_torch_load(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        return torch.load(path, map_location=map_location)


class FKDataset(Dataset):
    """Wraps an (N, 14) tensor: 7 joint angles + 7 pose values.

    Standardises both q and p per task during __init__. Stores raw means/stds
    so they can be persisted and re-used at evaluation/adaptation time.
    """

    def __init__(self, raw: torch.Tensor, dof: int, n_joints_max: int = 7):
        assert raw.dim() == 2 and raw.shape[1] == 14, f"Expected (N,14), got {raw.shape}"
        self.dof = dof
        self.n_joints_max = n_joints_max

        # Build active mask: first `dof` joints are active, rest clamped/inactive
        self.mask = torch.zeros(n_joints_max, dtype=torch.float32)
        self.mask[:dof] = 1.0

        # Split q (cols 0:7) and p (cols 7:14)
        q_raw = raw[:, :n_joints_max].clone()  # (N, 7)
        p_raw = raw[:, n_joints_max:].clone()  # (N, 7)

        # Clamp inactive joints to zero (defensive — data should already have this)
        if dof < n_joints_max:
            q_raw[:, dof:] = 0.0

        # Per-task standardisation (compute on ALL of this task's data;
        # we don't have a held-out fit-set issue here because q is already
        # uniformly distributed within the workspace bounds)
        self.q_mean = q_raw.mean(dim=0)              # (7,)
        self.q_std = q_raw.std(dim=0).clamp_min(1e-8) # (7,)
        # For inactive joint columns std might be 0 — clamp avoids div-by-zero
        if dof < n_joints_max:
            self.q_std[dof:] = 1.0  # keep inactive at 0 after standardisation

        self.p_mean = p_raw.mean(dim=0)
        self.p_std = p_raw.std(dim=0).clamp_min(1e-8)

        self.q = (q_raw - self.q_mean) / self.q_std
        self.p = (p_raw - self.p_mean) / self.p_std

    def __len__(self) -> int:
        return self.q.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.q[idx], self.mask, self.p[idx]


# --------------- Training step ---------------

def train_one_epoch(model, loader, optim, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    total_n = 0
    for q, mask, p in loader:
        q = q.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        p = p.to(device, non_blocking=True)

        optim.zero_grad(set_to_none=True)
        # We always train in FP32 here (the model is small enough that mixed
        # precision gives little speedup). If you want AMP, wrap with autocast.
        pred = model(q, mask)
        loss = criterion(pred, p)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()

        total_loss += loss.item() * q.shape[0]
        total_n += q.shape[0]
    return total_loss / max(1, total_n)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_n = 0
    for q, mask, p in loader:
        q = q.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        p = p.to(device, non_blocking=True)
        pred = model(q, mask)
        loss = criterion(pred, p)
        total_loss += loss.item() * q.shape[0]
        total_n += q.shape[0]
    return total_loss / max(1, total_n)


# --------------- Main ---------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="path to (N, 14) .pt tensor")
    p.add_argument("--dof", type=int, required=True, choices=[5, 6, 7])
    p.add_argument("--mode", choices=["fk"], default="fk",
                   help="this script is FK-only; IK would need a different head")
    # Model hyperparameters
    p.add_argument("--n_joints", type=int, default=7)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--dim_feedforward", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    # Training hyperparameters
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--lr_patience", type=int, default=10)
    p.add_argument("--lr_factor", type=float, default=0.5)
    # System
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--enable_tf32", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    # Output
    p.add_argument("--tb_logdir", type=str, default="runs/singletask_transformer")
    p.add_argument("--tb_name", type=str, default="run")
    p.add_argument("--save_dir", type=str, default="runs/singletask_transformer/ckpts")
    args = p.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # TF32
    if args.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device(args.device)

    # Output dirs
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.tb_logdir, args.tb_name))

    # Load data
    print(f"[INFO] Loading data from {args.data}")
    raw = safe_torch_load(args.data, map_location="cpu")
    if isinstance(raw, dict):
        raw = raw["data"] if "data" in raw else next(iter(raw.values()))
    print(f"[INFO] Data shape: {raw.shape}, dtype: {raw.dtype}")

    full = FKDataset(raw, dof=args.dof, n_joints_max=args.n_joints)
    n = len(full)
    n_train = int(0.7 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        full, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"[INFO] Splits: train={n_train:,}  val={n_val:,}  test={n_test:,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # Save standardisation stats so adapt_*.py can re-use them
    stats = {
        "q_mean": full.q_mean, "q_std": full.q_std,
        "p_mean": full.p_mean, "p_std": full.p_std,
        "dof": args.dof, "n_joints": args.n_joints,
    }
    torch.save(stats, os.path.join(args.save_dir, "stats.pt"))

    # Model
    model = MetaKinFormer(
        n_joints=args.n_joints,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        out_dim=7,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] MetaKinFormer params: {n_params:,}")

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                             betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=args.lr_factor, patience=args.lr_patience
    )
    criterion = nn.MSELoss()

    # Train loop
    best_val = float("inf")
    best_epoch = -1
    print(f"[INFO] Starting training for {args.epochs} epochs")
    t0 = time.time()
    for epoch in tqdm(range(args.epochs), desc="epochs"):
        train_loss = train_one_epoch(model, train_loader, optim, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        sched.step(val_loss)

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("train/lr", optim.param_groups[0]["lr"], epoch)

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "val_loss": val_loss,
                "stats": stats,
            }, os.path.join(args.save_dir, "best.pt"))

    # Final evaluation on test set
    test_loss = evaluate(model, test_loader, criterion, device)
    elapsed = time.time() - t0

    torch.save({
        "epoch": args.epochs - 1,
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "val_loss": best_val,
        "stats": stats,
    }, os.path.join(args.save_dir, "last.pt"))

    print(f"\n[INFO] Done in {elapsed/60:.1f} min")
    print(f"[INFO] best val_loss = {best_val:.6e} at epoch {best_epoch}")
    print(f"[INFO] test_loss      = {test_loss:.6e}")
    writer.add_scalar("test/loss", test_loss, args.epochs)
    writer.close()


if __name__ == "__main__":
    main()
