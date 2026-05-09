"""
Stage 3: Per-DoF adaptation of the shared MetaKinFormer to a target DoF.

Drop-in equivalent of `adapt_multitask_newest.py` but uses the MetaKinFormer
architecture instead of ResMLP_Mask. Same CLI surface and same loss form.

Loss (Eq. 3.3b in the paper):
    L_adapt = E[ lambda_p * ||t_hat - t||^2 + lambda_o * geo(r_hat, r)^2 ]
              + lambda_reg * ||theta - theta_init||^2

Score for BEST step selection: s = E_pos + 0.01 * E_ori (mm + degrees scale)

Usage example (7-DoF):
    python adapt_multitask_transformer.py \\
        --ckpt runs/multitask_transformer/ckpts/multitask_fk_best.pt \\
        --dof 7 --data data/narrowed/7DOF_15deg/7DOF_15deg_part000.pt \\
        --support_size 50000 --query_size 2000000 --adapt_steps 100000 \\
        --inner_lr 1e-6 --batch_size 8192 --query_batch_size 8192 \\
        --pos_weight 1.0 --ori_weight 0.05 \\
        --device cuda --enable_tf32 \\
        --tb_logdir runs/adapt_transformer --tb_name dof7_seed42 \\
        --save_dir runs/adapt_transformer/ckpts/dof7_seed42 \\
        --seed 42
"""
from __future__ import annotations
import argparse
import math
import os
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from meta_kin_former import MetaKinFormer
from train_singletask_transformer import safe_torch_load


# --------------- Helpers ---------------

def quat_geodesic_deg(r_hat: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Geodesic angle (degrees) between unit quaternions, robust to sign.

    r_hat, r: (B, 4) tensors of [w, x, y, z] quaternions, NOT necessarily unit.
    Returns: (B,) angles in degrees.
    """
    r_hat = r_hat / r_hat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    r = r / r.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    dot = (r_hat * r).sum(dim=-1).abs().clamp(0.0, 1.0)
    angle_rad = 2.0 * torch.acos(dot)
    return angle_rad * (180.0 / math.pi)


class AdaptDataset(Dataset):
    """A simple wrapper around the pre-standardised q and p tensors + mask."""

    def __init__(self, q_std: torch.Tensor, p_std: torch.Tensor, mask: torch.Tensor):
        self.q = q_std
        self.p = p_std
        self.mask = mask

    def __len__(self):
        return self.q.shape[0]

    def __getitem__(self, idx):
        return self.q[idx], self.mask, self.p[idx]


# --------------- Main ---------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Stage-2 multitask checkpoint .pt")
    p.add_argument("--dof", type=int, required=True, choices=[5, 6, 7])
    p.add_argument("--mode", default="fk", choices=["fk"])
    p.add_argument("--data", required=True, help="(N,14) tensor for the target DoF")
    # Adaptation
    p.add_argument("--support_size", type=int, default=50000)
    p.add_argument("--query_size", type=int, default=2_000_000)
    p.add_argument("--adapt_steps", type=int, default=100_000)
    p.add_argument("--inner_lr", type=float, default=1e-6)
    p.add_argument("--batch_size", type=int, default=8192)
    p.add_argument("--query_batch_size", type=int, default=8192)
    p.add_argument("--num_workers", type=int, default=4)
    # Loss weights (matching Eq. 3.3b in the paper)
    p.add_argument("--pos_weight", type=float, default=1.0)
    p.add_argument("--ori_weight", type=float, default=0.05)
    p.add_argument("--l2_reg", type=float, default=1e-6,
                   help="L2-to-init coefficient lambda_reg in Eq. 3.3b")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--adam_eps", type=float, default=1e-7)
    # Score weights (used to pick BEST step)
    p.add_argument("--score_pos_w", type=float, default=1.0)
    p.add_argument("--score_ori_w", type=float, default=0.01)
    # System
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--enable_tf32", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    # Eval / logging
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=5000)
    p.add_argument("--eval_pbar", action="store_true")
    # Output
    p.add_argument("--tb_logdir", type=str, default="runs/adapt_transformer")
    p.add_argument("--tb_name", type=str, default="run")
    p.add_argument("--save_dir", type=str, default="runs/adapt_transformer/ckpts")
    # Optional flag for logging-only — not used here, kept for CLI parity
    p.add_argument("--adapt", default="all", choices=["all", "head"])
    p.add_argument("--std_floor_q_deg", type=float, default=1.0)
    p.add_argument("--eval_steps_list", type=str, default="")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.tb_logdir, args.tb_name))

    print(f"[INFO] device={device}")
    print(f"[INFO] Loading Stage-2 checkpoint {args.ckpt}")
    ck = safe_torch_load(args.ckpt, map_location="cpu")

    # Recover model architecture from saved args; fall back to defaults
    saved_args = ck.get("args", {})
    n_joints = saved_args.get("n_joints", 7)
    d_model = saved_args.get("d_model", 128)
    n_layers = saved_args.get("n_layers", 4)
    n_heads = saved_args.get("n_heads", 4)
    dim_ff = saved_args.get("dim_feedforward", 512)
    dropout = saved_args.get("dropout", 0.1)

    model = MetaKinFormer(
        n_joints=n_joints, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, dim_feedforward=dim_ff, dropout=dropout, out_dim=7,
    ).to(device)
    model.load_state_dict(ck["model_state_dict"], strict=True)

    # Snapshot init params for L2-to-init regulariser
    init_state = {k: v.clone().detach().to(device) for k, v in model.state_dict().items()
                  if v.dtype.is_floating_point}

    # Recover standardisation stats for this DoF from the multitask checkpoint
    stats_all = ck.get("stats", None)
    if stats_all is None or f"dof_{args.dof}" not in stats_all:
        raise ValueError(f"Stage-2 ckpt missing stats for dof_{args.dof}")
    stats = stats_all[f"dof_{args.dof}"]
    q_mean = stats["q_mean"].to(device)
    q_std = stats["q_std"].to(device)
    p_mean = stats["p_mean"].to(device)
    p_std = stats["p_std"].to(device)
    mask = stats["mask"].to(device)

    # Load and standardise data
    print(f"[INFO] Loading data {args.data}")
    raw = safe_torch_load(args.data, map_location="cpu")
    if isinstance(raw, dict): raw = next(iter(raw.values()))
    if raw.dtype != torch.float32:
        raw = raw.float()

    q_raw = raw[:, :n_joints].clone()
    p_raw = raw[:, n_joints:].clone()
    if args.dof < n_joints:
        q_raw[:, args.dof:] = 0.0

    # Move stats to CPU for the standardisation step (data is on CPU)
    q_std_data = (q_raw - stats["q_mean"]) / stats["q_std"]
    p_std_data = (p_raw - stats["p_mean"]) / stats["p_std"]

    # Random shuffle for support/query split (deterministic by seed)
    rng = np.random.RandomState(args.seed)
    perm = torch.from_numpy(rng.permutation(q_std_data.shape[0]))
    n_total = q_std_data.shape[0]
    n_support = min(args.support_size, n_total)
    n_query = min(args.query_size, n_total - n_support)

    sup_idx = perm[:n_support]
    qry_idx = perm[n_support:n_support + n_query]

    q_sup = q_std_data[sup_idx]
    p_sup = p_std_data[sup_idx]
    q_qry = q_std_data[qry_idx]
    p_qry = p_std_data[qry_idx]
    p_qry_raw = p_raw[qry_idx]  # raw units for physical-error eval
    print(f"[INFO] support: {n_support:,}  query: {n_query:,}  total: {n_total:,}")

    # Stage-3 dataloaders
    sup_ds = AdaptDataset(q_sup, p_sup, mask.cpu())
    qry_ds = AdaptDataset(q_qry, p_qry, mask.cpu())
    sup_loader = DataLoader(sup_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)
    qry_loader = DataLoader(qry_ds, batch_size=args.query_batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Optimiser (only the model's parameters; init_state is just for L2 regularisation)
    optim = torch.optim.Adam(model.parameters(), lr=args.inner_lr,
                             betas=(0.9, 0.999), eps=args.adam_eps)
    sup_iter = iter(sup_loader)

    # Eval helper — computes physical errors on raw-units query split
    @torch.no_grad()
    def eval_query() -> Tuple[float, float]:
        model.eval()
        pos_errs = []
        ori_errs = []
        loader = qry_loader
        if args.eval_pbar:
            loader = tqdm(loader, desc="eval", leave=False)
        idx = 0
        for q, m, p in loader:
            q = q.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)
            pred_std = model(q, m)
            # De-standardise prediction
            pred_raw = pred_std * p_std + p_mean
            pred_t = pred_raw[:, :3]
            pred_r = pred_raw[:, 3:]
            # Ground truth raw
            B = q.shape[0]
            gt_raw = p_qry_raw[idx: idx + B].to(device, non_blocking=True)
            idx += B
            gt_t = gt_raw[:, :3]
            gt_r = gt_raw[:, 3:]
            pos_err = (pred_t - gt_t).norm(dim=-1)  # (B,)
            ori_err = quat_geodesic_deg(pred_r, gt_r)
            pos_errs.append(pos_err.cpu())
            ori_errs.append(ori_err.cpu())
        model.train()
        pos_all = torch.cat(pos_errs)
        ori_all = torch.cat(ori_errs)
        return float(pos_all.mean()), float(ori_all.mean())

    # Adaptation loop
    print(f"[INFO] Starting Stage-3 adaptation for {args.adapt_steps} steps")
    best_score = float("inf")
    best_step = -1
    best_metrics = None
    t0 = time.time()

    for step in tqdm(range(args.adapt_steps), desc="adapt(fk,all)", smoothing=0.0):
        # Get next support batch
        try:
            q, m, p_target = next(sup_iter)
        except StopIteration:
            sup_iter = iter(sup_loader)
            q, m, p_target = next(sup_iter)

        q = q.to(device, non_blocking=True)
        m = m.to(device, non_blocking=True)
        p_target_std = p_target.to(device, non_blocking=True)

        optim.zero_grad(set_to_none=True)
        pred_std = model(q, m)
        # De-standardise both pred and target into raw units to apply Eq. 3.3b
        pred_raw = pred_std * p_std + p_mean
        gt_raw = p_target_std * p_std + p_mean
        pred_t, pred_r = pred_raw[:, :3], pred_raw[:, 3:]
        gt_t, gt_r = gt_raw[:, :3], gt_raw[:, 3:]

        loss_pos = ((pred_t - gt_t) ** 2).sum(dim=-1).mean()
        # Squared geodesic angle in radians
        pred_r_norm = pred_r / pred_r.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        gt_r_norm = gt_r / gt_r.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        dot = (pred_r_norm * gt_r_norm).sum(dim=-1).abs().clamp(0.0, 1.0)
        angle_rad = 2.0 * torch.acos(dot)
        loss_ori = (angle_rad ** 2).mean()

        # L2 to init
        l2_to_init = 0.0
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point and k in init_state:
                l2_to_init = l2_to_init + ((v - init_state[k]) ** 2).sum()

        loss = (
            args.pos_weight * loss_pos
            + args.ori_weight * loss_ori
            + args.l2_reg * l2_to_init
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optim.step()

        if (step + 1) % args.log_every == 0:
            writer.add_scalar("train/loss", loss.item(), step + 1)
            writer.add_scalar("train/loss_pos", loss_pos.item(), step + 1)
            writer.add_scalar("train/loss_ori", loss_ori.item(), step + 1)

        if (step + 1) % args.eval_every == 0 or (step + 1) == args.adapt_steps:
            pos_mae, ori_deg = eval_query()
            score = args.score_pos_w * pos_mae + args.score_ori_w * ori_deg
            writer.add_scalar("eval/pos_mae_m", pos_mae, step + 1)
            writer.add_scalar("eval/ori_deg", ori_deg, step + 1)
            writer.add_scalar("eval/score", score, step + 1)
            if score < best_score:
                best_score = score
                best_step = step + 1
                best_metrics = {"pos_mae_m": pos_mae, "ori_deg": ori_deg}
                torch.save({
                    "step": step + 1, "model_state_dict": model.state_dict(),
                    "args": vars(args), "score": score, "metrics": best_metrics,
                }, os.path.join(args.save_dir, "best.pt"))

    elapsed = time.time() - t0
    torch.save({
        "step": args.adapt_steps, "model_state_dict": model.state_dict(),
        "args": vars(args),
    }, os.path.join(args.save_dir, "last.pt"))

    print(f"\n[INFO] Stage-3 adaptation done in {elapsed/60:.2f} min")
    print(f"[INFO] BEST step={best_step} metrics={best_metrics} score={best_score}")
    writer.close()


if __name__ == "__main__":
    main()
