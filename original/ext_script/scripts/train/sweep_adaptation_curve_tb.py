#!/usr/bin/env python3
import argparse, os, math, time
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

JOINT_COLS = [f"q{i}" for i in range(1, 8)]
POSE_COLS  = ["x","y","z","qw","qx","qy","qz"]

# -------------------------
# Safe load (PyTorch 2.6+)
# -------------------------
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
    # returns Tensor [N,14] on CPU
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
        return torch.cat(arrs, dim=0).float().cpu()

    if path.endswith(".pt") or path.endswith(".bin"):
        obj = safe_torch_load(path, map_location="cpu")
        t = obj["data"] if isinstance(obj, dict) and "data" in obj else obj
        if not isinstance(t, torch.Tensor) or t.ndim != 2 or t.shape[1] != 14:
            raise ValueError(f"{path} must be Tensor [N,14]")
        return t.float().cpu()

    if path.endswith(".csv"):
        import pandas as pd
        cols = JOINT_COLS + POSE_COLS
        df = pd.read_csv(path)
        return torch.from_numpy(df[cols].values.astype(np.float32)).float()

    raise ValueError(f"Unsupported data path: {path}")

# -------------------------
# Mask helper
# -------------------------
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

# -------------------------
# Models (must match training)
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

# -------------------------
# Norm from SUPPORT only (Torch, fast)
# -------------------------
def split_xy(full: torch.Tensor, mode: str, idx_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    sub = full.index_select(0, idx_t)  # [K,14]
    if mode == "fk":
        x = sub[:, 0:7]
        y = sub[:, 7:14]
    else:
        x = sub[:, 7:14]
        y = sub[:, 0:7]
    return x, y

def compute_norm_from_support_torch(full: torch.Tensor, mode: str, support_idx_t: torch.Tensor,
                                   std_floor_q_deg: float):
    xs, ys = split_xy(full, mode, support_idx_t)
    x_mean = xs.mean(dim=0)
    x_std  = xs.std(dim=0).clamp_min(1e-8)
    y_mean = ys.mean(dim=0)
    y_std  = ys.std(dim=0).clamp_min(1e-8)

    if mode == "ik":
        std_floor_q_rad = std_floor_q_deg * math.pi / 180.0
        y_std = torch.clamp(y_std, min=std_floor_q_rad)

    return x_mean, x_std, y_mean, y_std

def norm_xy(xs: torch.Tensor, ys: torch.Tensor, x_mean, x_std, y_mean, y_std):
    xs_n = (xs - x_mean) / (x_std + 1e-8)
    ys_n = (ys - y_mean) / (y_std + 1e-8)
    return xs_n, ys_n

# -------------------------
# Metrics
# -------------------------
def masked_joint_rmse_deg(q_pred: torch.Tensor, q_true: torch.Tensor, mask: torch.Tensor, angles_in_degrees: bool):
    err = (q_pred - q_true) * mask
    denom = mask.sum(dim=1).clamp_min(1e-8)
    mse = (err ** 2).sum(dim=1) / denom
    rmse = torch.sqrt(mse).mean().item()
    if angles_in_degrees:
        return rmse
    return rmse * (180.0 / math.pi)

def quat_angle_error_deg(q1: torch.Tensor, q2: torch.Tensor, eps: float = 1e-9):
    q1 = q1 / q1.norm(dim=1, keepdim=True).clamp_min(eps)
    q2 = q2 / q2.norm(dim=1, keepdim=True).clamp_min(eps)
    dot = (q1*q2).sum(dim=1).abs().clamp(-1.0, 1.0)
    ang = 2.0 * torch.acos(dot)
    return (ang * 180.0 / math.pi).mean().item()

@torch.no_grad()
def eval_model_tensor(model, mode: str,
                      xq_n: torch.Tensor, yq_n: torch.Tensor,
                      mask: torch.Tensor,
                      device: torch.device,
                      y_mean: torch.Tensor, y_std: torch.Tensor,
                      angles_in_degrees: bool,
                      eval_bs: int,
                      use_pbar: bool):
    model.eval()
    n = xq_n.shape[0]
    it = range(0, n, eval_bs)
    if use_pbar:
        it = tqdm(it, total=(n + eval_bs - 1)//eval_bs, desc="eval", position=2, leave=False, ncols=120, dynamic_ncols=True)

    y_mean_d = y_mean.to(device)
    y_std_d  = y_std.to(device)

    if mode == "ik":
        rmses = []
        for s in it:
            e = min(s + eval_bs, n)
            x = xq_n[s:e].to(device, non_blocking=True)
            y = yq_n[s:e].to(device, non_blocking=True)
            m = mask.expand(e - s, -1)

            qn, _, _ = model(x, m)
            q_pred = qn * y_std_d + y_mean_d
            q_true = y  * y_std_d + y_mean_d
            rmses.append(masked_joint_rmse_deg(q_pred, q_true, m, angles_in_degrees))
        return {"joint_rmse_deg": float(np.mean(rmses))}

    pos_errs, ori_errs = [], []
    for s in it:
        e = min(s + eval_bs, n)
        x = xq_n[s:e].to(device, non_blocking=True)
        y = yq_n[s:e].to(device, non_blocking=True)
        m = mask.expand(e - s, -1)

        pn = model(x, m)
        pose_pred = pn * y_std_d + y_mean_d
        pose_true = y  * y_std_d + y_mean_d
        pos_errs.append((pose_pred[:, :3] - pose_true[:, :3]).norm(dim=1).mean().item())
        ori_errs.append(quat_angle_error_deg(pose_pred[:, 3:7], pose_true[:, 3:7]))

    return {"pos_mae_m": float(np.mean(pos_errs)), "ori_mean_deg": float(np.mean(ori_errs))}

# -------------------------
# Adaptation controls
# -------------------------
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

def parse_sizes_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", choices=["fk","ik"], required=True)
    ap.add_argument("--dof", type=int, required=True)
    ap.add_argument("--data", required=True)

    # sweep sizes
    ap.add_argument("--support_sizes", type=str, default="1000,2000,5000,10000,20000")
    ap.add_argument("--support_min", type=int, default=-1)
    ap.add_argument("--support_max", type=int, default=-1)
    ap.add_argument("--support_step", type=int, default=1000)

    # fixed adaptation steps
    ap.add_argument("--query_size", type=int, default=100000)
    ap.add_argument("--adapt_steps", type=int, default=10000)

    ap.add_argument("--inner_lr", type=float, default=5e-5)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--eval_batch", type=int, default=8192)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--adapt", choices=["all","head"], default="all")
    ap.add_argument("--l2_reg", type=float, default=1e-6)
    ap.add_argument("--aux_weight", type=float, default=0.03)
    ap.add_argument("--std_floor_q_deg", type=float, default=1.0)
    ap.add_argument("--angles_in_degrees", action="store_true")

    # pbar / logging
    ap.add_argument("--log_every", type=int, default=200, help="Update adapt bar postfix every N steps.")
    ap.add_argument("--eval_pbar", action="store_true", help="Show eval progress bar (position=2).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeat", type=int, default=1)

    # TensorBoard
    ap.add_argument("--tb_logdir", type=str, default="./runs/adapt_sweep")
    ap.add_argument("--tb_name", type=str, default="", help="Optional run name. Default auto.")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    tqdm.write(f"[INFO] device={device}")

    # Speed knobs
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # support sizes
    if args.support_min > 0 and args.support_max > 0:
        support_sizes = list(range(args.support_min, args.support_max + 1, args.support_step))
    else:
        support_sizes = parse_sizes_list(args.support_sizes)
    support_sizes = sorted(list(set(support_sizes)))
    max_support = max(support_sizes)

    # TensorBoard writer
    run_name = args.tb_name.strip()
    if not run_name:
        run_name = f"{args.mode}_dof{args.dof}_steps{args.adapt_steps}_adapt{args.adapt}_lr{args.inner_lr:g}"
    tb_dir = os.path.join(args.tb_logdir, run_name)
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(tb_dir)
    writer.add_text("config", str(vars(args)))

    # load ckpt + model
    ckpt = safe_torch_load(args.ckpt, map_location=device)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError("Checkpoint must contain model_state_dict")
    sd = ckpt["model_state_dict"]

    # guard: fk vs ik mismatch
    has_fk_head = any(k.startswith("fc_out.") for k in sd.keys())
    has_ik_head = any(k.startswith("fc_joint.") for k in sd.keys())
    if args.mode == "fk" and not has_fk_head:
        raise ValueError("You set --mode fk but checkpoint looks like IK (missing fc_out.*).")
    if args.mode == "ik" and not has_ik_head:
        raise ValueError("You set --mode ik but checkpoint looks like FK (missing fc_joint/fc_pos/fc_ori.*).")

    hidden_dim, num_blocks = infer_arch_from_state_dict(sd)
    if args.mode == "fk":
        model = ResidualMLP_Mask(hidden_dim=hidden_dim, num_blocks=num_blocks)
    else:
        model = IKResNetDualHead_Mask(hidden_dim=hidden_dim, num_blocks=num_blocks)
    model.load_state_dict(sd, strict=True)
    model.to(device)

    # keep init weights on CPU to reset quickly/reliably
    init_sd_cpu = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}

    mask = get_mask_for_dof(args.dof).to(device)

    # load data once
    full = load_dataset_tensor(args.data)  # CPU [N,14]
    N = full.shape[0]
    tqdm.write(f"[INFO] loaded N={N}")
    if max_support + args.query_size > N:
        raise ValueError(f"Need at least max_support+query_size={max_support+args.query_size}, but N={N}")

    tag_prefix = f"sweep_steps{args.adapt_steps}"

    # OUTER pbar (position=0)
    outer = tqdm(support_sizes, desc="support sweep", ncols=120, position=0, leave=True, dynamic_ncols=True)

    for K_support in outer:
        rep_metrics = []
        rep_times = []

        for r in range(args.repeat):
            seed = args.seed + r
            rng = np.random.RandomState(seed)

            # deterministic seeds for torch sampling too
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

            perm = rng.permutation(N)
            support_pool = perm[:max_support]
            query_idx_np = perm[max_support:max_support + args.query_size]
            support_idx_np = support_pool[:K_support]

            support_idx_t = torch.from_numpy(support_idx_np).long()
            query_idx_t   = torch.from_numpy(query_idx_np).long()

            # norm from support only (CPU torch)
            x_mean, x_std, y_mean, y_std = compute_norm_from_support_torch(full, args.mode, support_idx_t, args.std_floor_q_deg)

            # build normalized support/query tensors (CPU)
            xs, ys = split_xy(full, args.mode, support_idx_t)
            xq, yq = split_xy(full, args.mode, query_idx_t)
            xs_n, ys_n = norm_xy(xs, ys, x_mean, x_std, y_mean, y_std)
            xq_n, yq_n = norm_xy(xq, yq, x_mean, x_std, y_mean, y_std)

            # move support to device once (FAST inner loop)
            xs_n = xs_n.to(device, non_blocking=True)
            ys_n = ys_n.to(device, non_blocking=True)

            # y stats to device for eval denorm
            y_mean_d = y_mean.to(device, dtype=torch.float32)
            y_std_d  = y_std.to(device, dtype=torch.float32)

            # reset model
            model.load_state_dict(init_sd_cpu, strict=True)
            model.to(device)
            set_trainable(model, args.adapt)

            trainable = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.Adam(trainable, lr=args.inner_lr)
            mse = nn.MSELoss()

            # snapshot trainable params for L2 (fast version, no state_dict loop)
            trainable_init = [p.detach().clone() for p in trainable]

            # INNER pbar for adaptation (position=1)
            model.train()
            t0 = time.time()
            running = 0.0
            seen = 0

            adapt_bar = tqdm(
                total=args.adapt_steps,
                desc=f"adapt K={K_support} rep={r+1}/{args.repeat}",
                ncols=120,
                position=1,
                leave=False,
                dynamic_ncols=True,
                unit="step"
            )

            for it in range(1, args.adapt_steps + 1):
                bidx = torch.randint(0, K_support, (args.batch_size,), device=device)
                x = xs_n[bidx]
                y = ys_n[bidx]
                m = mask.expand(args.batch_size, -1)

                opt.zero_grad(set_to_none=True)

                if args.mode == "fk":
                    yhat = model(x, m)
                    loss = mse(yhat, y)
                else:
                    qn, posn, orin = model(x, m)
                    se = ((qn - y) ** 2) * m
                    denom = m.sum(dim=1).clamp_min(1e-8)
                    loss_q = (se.sum(dim=1) / denom).mean()
                    loss_aux = mse(posn, x[:, :3]) + mse(orin, x[:, 3:7])
                    loss = loss_q + args.aux_weight * loss_aux

                if args.l2_reg > 0:
                    l2 = 0.0
                    for p, p0 in zip(trainable, trainable_init):
                        l2 = l2 + (p - p0).pow(2).mean()
                    loss = loss + args.l2_reg * l2

                loss.backward()
                opt.step()

                running += float(loss.item())
                seen += 1
                adapt_bar.update(1)

                if args.log_every > 0 and (it % args.log_every) == 0:
                    adapt_bar.set_postfix(loss=f"{running/max(1,seen):.6f}", lr=f"{opt.param_groups[0]['lr']:.2e}")

            adapt_bar.close()
            dt = time.time() - t0

            # eval on query (batched to device)
            met = eval_model_tensor(
                model=model,
                mode=args.mode,
                xq_n=xq_n, yq_n=yq_n,
                mask=mask,
                device=device,
                y_mean=y_mean_d, y_std=y_std_d,
                angles_in_degrees=args.angles_in_degrees,
                eval_bs=args.eval_batch,
                use_pbar=args.eval_pbar
            )

            rep_metrics.append(met)
            rep_times.append(dt)

        # log to TensorBoard with x-axis = support size
        step_x = K_support
        writer.add_scalar(f"{tag_prefix}/time_mean_s", float(np.mean(rep_times)), step_x)

        if args.mode == "ik":
            vals = [m["joint_rmse_deg"] for m in rep_metrics]
            mean_v = float(np.mean(vals))
            std_v  = float(np.std(vals))
            outer.set_postfix(rmse=f"{mean_v:.3f}±{std_v:.3f}")

            writer.add_scalar(f"{tag_prefix}/joint_rmse_deg_mean", mean_v, step_x)
            writer.add_scalar(f"{tag_prefix}/joint_rmse_deg_std",  std_v,  step_x)
            for i, v in enumerate(vals):
                writer.add_scalar(f"{tag_prefix}/joint_rmse_deg_rep{i}", float(v), step_x)
        else:
            pos_vals = [m["pos_mae_m"] for m in rep_metrics]
            ori_vals = [m["ori_mean_deg"] for m in rep_metrics]
            pos_mean, pos_std = float(np.mean(pos_vals)), float(np.std(pos_vals))
            ori_mean, ori_std = float(np.mean(ori_vals)), float(np.std(ori_vals))
            outer.set_postfix(pos=f"{pos_mean:.4f}±{pos_std:.4f}")

            writer.add_scalar(f"{tag_prefix}/pos_mae_m_mean", pos_mean, step_x)
            writer.add_scalar(f"{tag_prefix}/pos_mae_m_std",  pos_std,  step_x)
            writer.add_scalar(f"{tag_prefix}/ori_mean_deg_mean", ori_mean, step_x)
            writer.add_scalar(f"{tag_prefix}/ori_mean_deg_std",  ori_std,  step_x)
            for i, v in enumerate(pos_vals):
                writer.add_scalar(f"{tag_prefix}/pos_mae_m_rep{i}", float(v), step_x)

        writer.flush()

    writer.close()
    tqdm.write(f"[INFO] TensorBoard logs saved to: {tb_dir}")
    tqdm.write(f"[INFO] Launch: tensorboard serve --logdir {args.tb_logdir}")

if __name__ == "__main__":
    main()
