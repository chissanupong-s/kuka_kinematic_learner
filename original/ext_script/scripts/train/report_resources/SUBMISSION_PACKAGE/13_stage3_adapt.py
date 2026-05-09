#!/usr/bin/env python3
"""adapt_multitask_fk_weighted.py

Adapt a multitask FK/IK model to a target DOF using a small support set and
evaluate on a query set.

This is a **FK-focused** variant of `adapt_multitask_newest.py` with improved
control over the FK loss balance:
  - FK support loss:  pos_weight * pos_mse  +  ori_weight * ori_geodesic_rad2
  - FK best-step scoring: score_pos_w * pos_mae_m + score_ori_w * ori_deg

Why this matters:
  - Position and orientation live in different numeric units (meters vs radians/deg).
  - Without explicit weights, adaptation may optimize position while ignoring
    orientation (or vice versa), especially across DOFs.

Notes:
  - FK uses **raw** pose units for metrics (denormalized by support stats).
  - Orientation error is computed as geodesic angle between quaternions.
  - IK path is kept for compatibility (unchanged from the original script).
"""

import argparse, os, math, time
from typing import Dict, Tuple, Set, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def safe_torch_load(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)


def load_dataset_tensor(path: str) -> torch.Tensor:
    """Returns CPU float32 torch Tensor [N,14]."""
    if os.path.isdir(path):
        files = sorted([os.path.join(path, f) for f in os.listdir(path)
                        if f.endswith(".pt") or f.endswith(".bin")])
        if not files:
            raise ValueError(f"No shards found in {path}")
        arrs = []
        for fp in files:
            obj = safe_torch_load(fp, map_location="cpu")
            t = obj["data"] if isinstance(obj, dict) and "data" in obj else obj
            if not isinstance(t, torch.Tensor) or t.ndim != 2 or t.shape[1] != 14:
                raise ValueError(f"{fp} must be Tensor [N,14]")
            arrs.append(t.float().cpu())
        return torch.cat(arrs, dim=0).contiguous()

    if path.endswith(".pt") or path.endswith(".bin"):
        obj = safe_torch_load(path, map_location="cpu")
        t = obj["data"] if isinstance(obj, dict) and "data" in obj else obj
        if not isinstance(t, torch.Tensor) or t.ndim != 2 or t.shape[1] != 14:
            raise ValueError(f"{path} must be Tensor [N,14]")
        return t.float().cpu().contiguous()

    if path.endswith(".csv"):
        import pandas as pd
        cols = [f"q{i}" for i in range(1, 8)] + ["x", "y", "z", "qw", "qx", "qy", "qz"]
        df = pd.read_csv(path)
        return torch.from_numpy(df[cols].values.astype(np.float32)).contiguous()

    raise ValueError(f"Unsupported data path: {path}")


def get_mask_for_dof(dof: int) -> torch.Tensor:
    if dof == 4:
        return torch.tensor([1,1,1,1,0,0,0], dtype=torch.float32)
    if dof == 5:
        return torch.tensor([1,1,1,1,1,0,0], dtype=torch.float32)
    if dof == 6:
        return torch.tensor([1,1,1,1,1,1,0], dtype=torch.float32)
    if dof == 7:
        return torch.tensor([1,1,1,1,1,1,1], dtype=torch.float32)
    raise ValueError("--dof must be 4/5/6/7")


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
    def __init__(self, hidden_dim: int = 1024, num_blocks: int = 8):
        super().__init__()
        self.fc_in = nn.Linear(7, hidden_dim)
        self.mask_proj = nn.Linear(7, hidden_dim, bias=False)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])
        self.fc_out = nn.Linear(hidden_dim, 7)
        self.act = nn.ReLU()

    def forward(self, x7, mask7):
        h = self.act(self.fc_in(x7) + self.mask_proj(mask7))
        for b in self.blocks:
            h = b(h)
        return self.fc_out(h)


class IKResNetDualHead_Mask(nn.Module):
    def __init__(self, hidden_dim: int = 1024, num_blocks: int = 8):
        super().__init__()
        self.fc_in = nn.Linear(7, hidden_dim)
        self.mask_proj = nn.Linear(7, hidden_dim, bias=False)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])
        self.fc_joint = nn.Linear(hidden_dim, 7)
        self.fc_pos = nn.Linear(hidden_dim, 3)
        self.fc_ori = nn.Linear(hidden_dim, 4)
        self.act = nn.ReLU()

    def forward(self, x7, mask7):
        h = self.act(self.fc_in(x7) + self.mask_proj(mask7))
        for b in self.blocks:
            h = b(h)
        return self.fc_joint(h), self.fc_pos(h), self.fc_ori(h)


def infer_arch_from_state_dict(sd: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    hidden_dim = sd["fc_in.weight"].shape[0]
    block_ids = set()
    for k in sd.keys():
        if k.startswith("blocks.") and k.endswith(".fc1.weight"):
            idx = int(k.split(".")[1])
            block_ids.add(idx)
    num_blocks = (max(block_ids) + 1) if block_ids else 0
    return hidden_dim, num_blocks


def infer_mode_from_state_dict(sd: Dict[str, torch.Tensor]) -> str:
    if any(k.startswith("fc_joint.") for k in sd.keys()):
        return "ik"
    if any(k.startswith("fc_out.") for k in sd.keys()):
        return "fk"
    return "fk"


def compute_norm_from_support(full: torch.Tensor, mode: str, support_idx: np.ndarray,
                              std_floor_q_deg: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sup = full[support_idx]
    if mode == "fk":
        X = sup[:, 0:7]
        Y = sup[:, 7:14]
    else:
        X = sup[:, 7:14]
        Y = sup[:, 0:7]

    x_mean = X.mean(dim=0)
    x_std  = X.std(dim=0).clamp_min(1e-8)
    y_mean = Y.mean(dim=0)
    y_std  = Y.std(dim=0).clamp_min(1e-8)

    if mode == "ik":
        std_floor_q_rad = std_floor_q_deg * math.pi / 180.0
        y_std = torch.maximum(y_std, torch.full_like(y_std, std_floor_q_rad))

    return x_mean, x_std, y_mean, y_std


class AdaptDataset(Dataset):
    def __init__(self, full: torch.Tensor, mode: str, mask7: torch.Tensor,
                 x_mean: torch.Tensor, x_std: torch.Tensor,
                 y_mean: torch.Tensor, y_std: torch.Tensor,
                 indices: np.ndarray):
        self.full = full
        self.mode = mode
        self.mask7 = mask7.float()
        self.idx = indices
        self.x_mean = x_mean.float()
        self.x_std  = x_std.float()
        self.y_mean = y_mean.float()
        self.y_std  = y_std.float()

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        row = self.full[self.idx[i]]
        if self.mode == "fk":
            x = row[0:7]
            y = row[7:14]
        else:
            x = row[7:14]
            y = row[0:7]
        x = (x - self.x_mean) / (self.x_std + 1e-8)
        y = (y - self.y_mean) / (self.y_std + 1e-8)
        return x.float(), y.float(), self.mask7


def quat_angle_rad(q1: torch.Tensor, q2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Geodesic angle between quaternions in radians. Sign-invariant."""
    q1 = q1 / q1.norm(dim=1, keepdim=True).clamp_min(eps)
    q2 = q2 / q2.norm(dim=1, keepdim=True).clamp_min(eps)
    dot = (q1 * q2).sum(dim=1).abs().clamp(0.0, 1.0)
    return 2.0 * torch.acos(dot.clamp(1e-7, 1.0 - 1e-7))


def quat_angle_error_deg(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    return (quat_angle_rad(q1, q2) * 180.0 / math.pi).mean()


def masked_joint_rmse_deg(q_pred: torch.Tensor, q_true: torch.Tensor, mask: torch.Tensor,
                          angles_in_degrees: bool) -> float:
    err = (q_pred - q_true) * mask
    denom = mask.sum(dim=1).clamp_min(1e-8)
    mse = (err ** 2).sum(dim=1) / denom
    rmse = torch.sqrt(mse).mean()
    if angles_in_degrees:
        return float(rmse.item())
    return float((rmse * (180.0 / math.pi)).item())


@torch.no_grad()
def eval_query_metrics(model: nn.Module, mode: str, loader: DataLoader, device: torch.device,
                       y_mean: torch.Tensor, y_std: torch.Tensor,
                       angles_in_degrees: bool, use_pbar: bool) -> Dict[str, float]:
    model.eval()
    it_loader = tqdm(loader, desc="eval", leave=False, ncols=120) if use_pbar else loader

    if mode == "ik":
        vals = []
        for x, y, m in it_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)
            qn, _, _ = model(x, m)
            q_pred = qn * y_std + y_mean
            q_true = y  * y_std + y_mean
            vals.append(masked_joint_rmse_deg(q_pred, q_true, m, angles_in_degrees))
        return {"joint_rmse_deg": float(np.mean(vals))}

    pos_l2_all = []
    ori_deg_all = []
    for x, y, m in it_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        m = m.to(device, non_blocking=True)

        pn = model(x, m)
        pose_pred = pn * y_std + y_mean
        pose_true = y  * y_std + y_mean

        dp = pose_pred[:, :3] - pose_true[:, :3]
        pos_l2_all.append(dp.norm(dim=1))
        ori_deg_all.append(quat_angle_error_deg(pose_pred[:, 3:7], pose_true[:, 3:7]))

    pos_l2_all = torch.cat(pos_l2_all, dim=0)
    pos_mae = pos_l2_all.mean().item()
    pos_rmse = torch.sqrt((pos_l2_all ** 2).mean()).item()
    ori_deg = torch.stack(ori_deg_all).mean().item()
    return {"pos_mae_m": float(pos_mae), "pos_rmse_m": float(pos_rmse), "ori_deg": float(ori_deg)}


def set_trainable(model: nn.Module, adapt: str):
    for p in model.parameters():
        p.requires_grad = False
    if adapt == "all":
        for p in model.parameters():
            p.requires_grad = True
        return
    if hasattr(model, "mask_proj"):
        for p in model.mask_proj.parameters():
            p.requires_grad = True
    if hasattr(model, "fc_out"):
        for p in model.fc_out.parameters():
            p.requires_grad = True
    if hasattr(model, "fc_joint"):
        for p in model.fc_joint.parameters():
            p.requires_grad = True
    if hasattr(model, "fc_pos"):
        for p in model.fc_pos.parameters():
            p.requires_grad = True
    if hasattr(model, "fc_ori"):
        for p in model.fc_ori.parameters():
            p.requires_grad = True


def copy_state_dict_to_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}


def l2_to_init_trainables(trainable_params: List[torch.Tensor], trainable_init: List[torch.Tensor]) -> torch.Tensor:
    loss = 0.0
    for p, p0 in zip(trainable_params, trainable_init):
        loss = loss + (p - p0).pow(2).mean()
    return loss


def build_eval_steps(adapt_steps: int, eval_every: int, eval_steps_list: str) -> List[int]:
    if eval_steps_list.strip():
        xs = [int(x.strip()) for x in eval_steps_list.split(",") if x.strip()]
        xs = [x for x in xs if 0 <= x <= adapt_steps]
        if 0 not in xs:
            xs = [0] + xs
        if adapt_steps not in xs:
            xs = xs + [adapt_steps]
        return sorted(set(xs))

    out = [0]
    if eval_every > 0:
        k = eval_every
        while k < adapt_steps:
            out.append(k)
            k += eval_every
    out.append(adapt_steps)
    return out


def score_fk(metrics: Dict[str, float], score_pos_w: float, score_ori_w: float) -> float:
    return float(score_pos_w * metrics["pos_mae_m"] + score_ori_w * metrics["ori_deg"])


def main():
    ap = argparse.ArgumentParser("Stable accurate adaptation with TensorBoard + best-step selection")

    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", choices=["auto", "fk", "ik"], default="auto")
    ap.add_argument("--dof", type=int, required=True)
    ap.add_argument("--data", required=True)

    ap.add_argument("--support_size", type=int, default=2000)
    ap.add_argument("--query_size", type=int, default=100000)
    ap.add_argument("--adapt_steps", type=int, default=10000)

    ap.add_argument("--inner_lr", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--query_batch_size", type=int, default=8192)
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--adapt", choices=["all", "head"], default="all")
    ap.add_argument("--l2_reg", type=float, default=1e-6)
    ap.add_argument("--std_floor_q_deg", type=float, default=1.0)
    ap.add_argument("--angles_in_degrees", action="store_true")

    # FK weights (main change)
    ap.add_argument("--pos_weight", type=float, default=1.0,
                    help="FK: weight for position loss (raw meters, per-dim MSE)")
    ap.add_argument("--ori_weight", type=float, default=0.1,
                    help="FK: weight for orientation loss (geodesic radians^2)")
    ap.add_argument("--score_pos_w", type=float, default=1.0,
                    help="FK: weight for position term in best-step score")
    ap.add_argument("--score_ori_w", type=float, default=0.01,
                    help="FK: weight for orientation term (degrees) in best-step score")

    # IK (kept for compatibility)
    ap.add_argument("--aux_weight", type=float, default=0.03)

    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--adam_eps", type=float, default=1e-7)

    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--eval_steps_list", type=str, default="")
    ap.add_argument("--eval_pbar", action="store_true")
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--enable_tf32", action="store_true")

    ap.add_argument("--tb_logdir", type=str, default="./runs/adapt_tb_fixed")
    ap.add_argument("--tb_name", type=str, default="")
    ap.add_argument("--save_dir", type=str, default="")

    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else "cpu")
    print(f"[INFO] device={device}")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(args.enable_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.enable_tf32)
        try:
            torch.set_float32_matmul_precision("high" if args.enable_tf32 else "highest")
        except Exception:
            pass

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.tb_logdir, exist_ok=True)
    run_dir = os.path.join(
        args.tb_logdir,
        args.tb_name.strip() if args.tb_name.strip() else
        f"{args.mode}_dof{args.dof}_K{args.support_size}_Q{args.query_size}_S{args.adapt_steps}_lr{args.inner_lr:g}_bs{args.batch_size}_{args.adapt}"
    )
    os.makedirs(run_dir, exist_ok=True)

    save_dir = args.save_dir.strip() if args.save_dir.strip() else run_dir
    os.makedirs(save_dir, exist_ok=True)

    writer = SummaryWriter(run_dir)
    writer.add_text("config", str(vars(args)))

    ckpt = safe_torch_load(args.ckpt, map_location=device)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError("Checkpoint must contain model_state_dict")
    sd = ckpt["model_state_dict"]

    inferred_mode = infer_mode_from_state_dict(sd)
    mode = inferred_mode if args.mode == "auto" else args.mode
    if args.mode != "auto" and args.mode != inferred_mode:
        raise ValueError(f"Checkpoint looks like '{inferred_mode}' but you passed --mode {args.mode}")

    hidden_dim, num_blocks = infer_arch_from_state_dict(sd)
    model = ResidualMLP_Mask(hidden_dim, num_blocks) if mode == "fk" else IKResNetDualHead_Mask(hidden_dim, num_blocks)
    model.load_state_dict(sd, strict=True)
    model.to(device)

    full = load_dataset_tensor(args.data)
    if torch.isnan(full).any() or torch.isinf(full).any():
        raise ValueError("Dataset contains NaN/Inf. Fix datagen or filter those rows.")

    N = full.shape[0]
    if args.support_size + args.query_size > N:
        raise ValueError(f"support+query exceeds N ({args.support_size}+{args.query_size}>{N})")

    perm = rng.permutation(N)
    support_idx = perm[:args.support_size]
    query_idx   = perm[args.support_size: args.support_size + args.query_size]

    mask_cpu = get_mask_for_dof(args.dof)
    _ = mask_cpu.to(device)

    x_mean, x_std, y_mean, y_std = compute_norm_from_support(full, mode, support_idx, args.std_floor_q_deg)
    y_mean_t = y_mean.to(device).float()
    y_std_t  = y_std.to(device).float()

    ds_sup = AdaptDataset(full, mode, mask_cpu, x_mean, x_std, y_mean, y_std, support_idx)
    ds_qry = AdaptDataset(full, mode, mask_cpu, x_mean, x_std, y_mean, y_std, query_idx)

    pin = (device.type == "cuda")
    persistent = (args.num_workers > 0)
    sup_loader = DataLoader(ds_sup, batch_size=args.batch_size, shuffle=True, drop_last=False,
                            num_workers=args.num_workers, pin_memory=pin, persistent_workers=persistent)
    qry_loader = DataLoader(ds_qry, batch_size=args.query_batch_size, shuffle=False, drop_last=False,
                            num_workers=args.num_workers, pin_memory=pin, persistent_workers=persistent)

    eval_steps = build_eval_steps(args.adapt_steps, args.eval_every, args.eval_steps_list)
    eval_set: Set[int] = set(eval_steps)

    m0 = eval_query_metrics(model, mode, qry_loader, device, y_mean_t, y_std_t,
                            angles_in_degrees=args.angles_in_degrees, use_pbar=args.eval_pbar)
    print(f"[EVAL] step=0 | {m0}")
    for k, v in m0.items():
        writer.add_scalar(f"query/{k}", v, 0)
    if mode == "fk":
        writer.add_scalar("query/score", score_fk(m0, args.score_pos_w, args.score_ori_w), 0)
    writer.flush()

    init_sd_cpu = copy_state_dict_to_cpu(model)

    set_trainable(model, args.adapt)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=args.inner_lr, eps=args.adam_eps)
    mse = nn.MSELoss()
    trainable_init = [p.detach().clone() for p in trainable]

    best_step = 0
    best_metrics = dict(m0)
    best_score = score_fk(m0, args.score_pos_w, args.score_ori_w) if mode == "fk" else float(m0["joint_rmse_deg"])

    torch.save({"model_state_dict": init_sd_cpu, "mode": mode, "dof": args.dof,
                "best_step": best_step, "best_metrics": best_metrics, "note": "init"},
               os.path.join(save_dir, "best.pt"))

    pbar = tqdm(total=args.adapt_steps, desc=f"adapt({mode},{args.adapt})", ncols=120, unit="step")
    running = 0.0
    seen = 0
    it = 0
    t0 = time.time()

    model.train()
    while it < args.adapt_steps:
        for x, y, m in sup_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            if mode == "fk":
                pn = model(x, m)
                pose_pred = pn * y_std_t + y_mean_t
                pose_true = y  * y_std_t + y_mean_t

                dp = pose_pred[:, :3] - pose_true[:, :3]
                loss_pos = (dp ** 2).mean()  # per-dim MSE (meters^2)

                ang = quat_angle_rad(pose_pred[:, 3:7], pose_true[:, 3:7])
                loss_ori = (ang ** 2).mean()  # rad^2

                loss = args.pos_weight * loss_pos + args.ori_weight * loss_ori
            else:
                qn, posn, orin = model(x, m)
                q_pred = qn * y_std_t + y_mean_t
                q_true = y  * y_std_t + y_mean_t
                se = ((q_pred - q_true) ** 2) * m
                denom = m.sum(dim=1).clamp_min(1e-8)
                loss_q = (se.sum(dim=1) / denom).mean()
                loss_aux = mse(posn, x[:, :3]) + mse(orin, x[:, 3:7])
                loss = loss_q + args.aux_weight * loss_aux

            if args.l2_reg > 0:
                loss = loss + args.l2_reg * l2_to_init_trainables(trainable, trainable_init)

            if torch.isnan(loss) or torch.isinf(loss):
                print("\n[ERROR] loss became NaN/Inf. Stopping.")
                return

            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=args.grad_clip)
            opt.step()

            it += 1
            pbar.update(1)

            running += float(loss.item())
            seen += 1

            if mode == "fk":
                writer.add_scalar("support/loss_pos", float(loss_pos.item()), it)
                writer.add_scalar("support/loss_ori", float(loss_ori.item()), it)
                writer.add_scalar("support/pos_weight", float(args.pos_weight), it)
                writer.add_scalar("support/ori_weight", float(args.ori_weight), it)

            if args.log_every > 0 and (it % args.log_every) == 0:
                avg_loss = running / max(1, seen)
                lr_now = opt.param_groups[0]["lr"]
                pbar.set_postfix(loss=f"{avg_loss:.6f}", lr=f"{lr_now:.2e}")
                writer.add_scalar("support/loss", avg_loss, it)
                writer.add_scalar("support/lr", lr_now, it)
                running = 0.0
                seen = 0

            if it in eval_set:
                mq = eval_query_metrics(model, mode, qry_loader, device, y_mean_t, y_std_t,
                                        angles_in_degrees=args.angles_in_degrees, use_pbar=args.eval_pbar)
                print(f"[EVAL] step={it} | {mq}")
                for k, v in mq.items():
                    writer.add_scalar(f"query/{k}", v, it)
                if mode == "fk":
                    sc = score_fk(mq, args.score_pos_w, args.score_ori_w)
                    writer.add_scalar("query/score", sc, it)
                writer.flush()

                sc = score_fk(mq, args.score_pos_w, args.score_ori_w) if mode == "fk" else float(mq["joint_rmse_deg"])
                if sc < best_score:
                    best_score = sc
                    best_step = it
                    best_metrics = dict(mq)
                    best_sd = copy_state_dict_to_cpu(model)
                    torch.save({"model_state_dict": best_sd, "mode": mode, "dof": args.dof,
                                "best_step": best_step, "best_metrics": best_metrics, "score": best_score,
                                "note": "best"},
                               os.path.join(save_dir, "best.pt"))

            if it >= args.adapt_steps:
                break

    pbar.close()
    dt = time.time() - t0

    last_sd = copy_state_dict_to_cpu(model)
    torch.save({"model_state_dict": last_sd, "mode": mode, "dof": args.dof,
                "adapt_steps": args.adapt_steps, "time_s": dt, "note": "last"},
               os.path.join(save_dir, "last.pt"))

    writer.add_scalar("meta/adapt_time_s", dt, args.adapt_steps)
    writer.add_text("best", f"best_step={best_step} best_metrics={best_metrics} best_score={best_score}")
    writer.flush()
    writer.close()

    print(f"[INFO] Done. time={dt/60:.2f} min")
    print(f"[INFO] BEST step={best_step} metrics={best_metrics} score={best_score}")
    print(f"[INFO] Saved best.pt + last.pt in: {save_dir}")
    print(f"[INFO] TensorBoard: tensorboard serve --logdir {args.tb_logdir}")
    print(f"[INFO] Run dir: {run_dir}")


if __name__ == "__main__":
    main()