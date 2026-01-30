#!/usr/bin/env python3
import argparse, os, math, time
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

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
    # returns Tensor [N,14]
    if os.path.isdir(path):
        files = sorted([os.path.join(path,f) for f in os.listdir(path) if f.endswith(".pt") or f.endswith(".bin")])
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

def set_trainable(model: nn.Module, adapt: str):
    for p in model.parameters():
        p.requires_grad = False
    if adapt == "all":
        for p in model.parameters():
            p.requires_grad = True
        return
    # head-only
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

def l2_to_init(model: nn.Module, init_sd: Dict[str, torch.Tensor]) -> torch.Tensor:
    loss = 0.0
    msd = model.state_dict()
    for k, v in msd.items():
        if k in init_sd and torch.is_tensor(v):
            loss = loss + (v - init_sd[k].to(v.device)).pow(2).mean()
    return loss

# -------------------------
# Torch-only normalization
# -------------------------
def compute_norm_from_support(full: torch.Tensor, mode: str, support_idx: np.ndarray, std_floor_q_deg: float):
    sup = full[support_idx]  # CPU torch [K,14]
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
        y_std = torch.maximum(y_std, torch.tensor(std_floor_q_rad, dtype=torch.float32))
    return x_mean, x_std, y_mean, y_std

def split_xy(full: torch.Tensor, mode: str, idx: np.ndarray):
    sub = full[idx]
    if mode == "fk":
        x = sub[:, 0:7]
        y = sub[:, 7:14]
    else:
        x = sub[:, 7:14]
        y = sub[:, 0:7]
    return x, y

def norm_xy(x: torch.Tensor, y: torch.Tensor, x_mean, x_std, y_mean, y_std):
    x_n = (x - x_mean) / (x_std + 1e-8)
    y_n = (y - y_mean) / (y_std + 1e-8)
    return x_n, y_n

def denorm(y_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return y_norm * std + mean

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
def eval_on_query(model, mode, xq_n, yq_n, mask, device, y_mean, y_std, angles_in_degrees: bool, eval_bs: int):
    model.eval()
    n = xq_n.shape[0]
    pos_errs = []
    ori_errs = []
    rmses = []

    # move denorm params to device once
    y_mean_d = y_mean.to(device)
    y_std_d  = y_std.to(device)

    for s in range(0, n, eval_bs):
        e = min(s + eval_bs, n)
        x = xq_n[s:e].to(device, non_blocking=True)
        y = yq_n[s:e].to(device, non_blocking=True)
        m = mask.expand(e - s, -1).to(device)

        if mode == "ik":
            qn, _, _ = model(x, m)
            q_pred = denorm(qn, y_mean_d, y_std_d)
            q_true = denorm(y,  y_mean_d, y_std_d)
            rmses.append(masked_joint_rmse_deg(q_pred, q_true, m, angles_in_degrees))
        else:
            pn = model(x, m)
            pose_pred = denorm(pn, y_mean_d, y_std_d)
            pose_true = denorm(y,  y_mean_d, y_std_d)
            pos_errs.append((pose_pred[:, :3] - pose_true[:, :3]).norm(dim=1).mean().item())
            ori_errs.append(quat_angle_error_deg(pose_pred[:, 3:7], pose_true[:, 3:7]))

    if mode == "ik":
        return {"joint_rmse_deg": float(np.mean(rmses))}
    return {"pos_mae_m": float(np.mean(pos_errs)), "ori_mean_deg": float(np.mean(ori_errs))}

# -------------------------
# Parse supports "name=path"
# -------------------------
def parse_supports(support_args: List[str]) -> Dict[str, str]:
    out = {}
    for item in support_args:
        if "=" not in item:
            raise ValueError(f"--supports items must be name=path, got: {item}")
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Bad support spec: {item}")
        out[name] = path
    if not out:
        raise ValueError("No supports provided.")
    return out

def main():
    ap = argparse.ArgumentParser("Experiment A: fixed budget compare support distributions")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", choices=["fk","ik"], required=True)
    ap.add_argument("--dof", type=int, required=True)

    ap.add_argument("--query_data", required=True, help="Fixed query distribution dataset (recommend uniform).")
    ap.add_argument("--supports", nargs="+", required=True, help="List of name=path for support datasets.")

    ap.add_argument("--support_size", type=int, default=10000)
    ap.add_argument("--query_size", type=int, default=100000)
    ap.add_argument("--adapt_steps", type=int, default=10000)

    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--eval_batch", type=int, default=8192)
    ap.add_argument("--inner_lr", type=float, default=5e-5)
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--adapt", choices=["all","head"], default="all")
    ap.add_argument("--l2_reg", type=float, default=1e-6)
    ap.add_argument("--aux_weight", type=float, default=0.03)
    ap.add_argument("--std_floor_q_deg", type=float, default=1.0)
    ap.add_argument("--angles_in_degrees", action="store_true")

    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--tb_logdir", type=str, default="./runs/expA")
    ap.add_argument("--run_name", type=str, default="")

    args = ap.parse_args()
    supports = parse_supports(args.supports)

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    print(f"[INFO] device={device}")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # tensorboard
    run_name = args.run_name.strip()
    if not run_name:
        run_name = f"{args.mode}_dof{args.dof}_steps{args.adapt_steps}_K{args.support_size}_Q{args.query_size}"
    tb_dir = os.path.join(args.tb_logdir, run_name)
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(tb_dir)
    writer.add_text("config", str(vars(args)))

    # load ckpt & build model
    ckpt = safe_torch_load(args.ckpt, map_location=device)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError("Checkpoint must contain model_state_dict")
    sd = ckpt["model_state_dict"]

    hidden_dim, num_blocks = infer_arch_from_state_dict(sd)
    if args.mode == "fk":
        model = ResidualMLP_Mask(hidden_dim=hidden_dim, num_blocks=num_blocks)
    else:
        model = IKResNetDualHead_Mask(hidden_dim=hidden_dim, num_blocks=num_blocks)
    model.load_state_dict(sd, strict=True)
    model.to(device)

    init_sd = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}  # keep init on CPU for reset
    mask = get_mask_for_dof(args.dof).to(device)  # [7]

    # load query dataset once
    query_full = load_dataset_tensor(args.query_data)  # CPU [Nq,14]
    Nq = query_full.shape[0]
    if args.query_size > Nq:
        raise ValueError(f"query_size {args.query_size} > N_query {Nq}")
    print(f"[INFO] query_data N={Nq}")

    # load all support datasets once
    support_fulls = {}
    for name, path in supports.items():
        t = load_dataset_tensor(path)
        if args.support_size > t.shape[0]:
            raise ValueError(f"support_size {args.support_size} > N_support({name}) {t.shape[0]}")
        support_fulls[name] = t
        print(f"[INFO] support '{name}' N={t.shape[0]}")

    # store per-dist results over repeats
    all_results = {name: [] for name in supports.keys()}
    all_times   = {name: [] for name in supports.keys()}

    outer = tqdm(range(args.repeat), desc="repeats", ncols=120, position=0, leave=True, dynamic_ncols=True)
    for r in outer:
        seed = args.seed + r
        rng = np.random.RandomState(seed)

        # fixed query indices for this repeat (shared across all support dists)
        qperm = rng.permutation(Nq)
        query_idx = qperm[:args.query_size]

        # materialize query x/y on CPU
        xq, yq = split_xy(query_full, args.mode, query_idx)

        # loop support distributions
        dist_loop = tqdm(list(supports.keys()), desc=f"support dists (rep {r+1}/{args.repeat})",
                         ncols=120, position=1, leave=False, dynamic_ncols=True)

        for dist_name in dist_loop:
            sup_full = support_fulls[dist_name]
            Ns = sup_full.shape[0]
            sperm = rng.permutation(Ns)
            support_idx = sperm[:args.support_size]

            # compute support-only norm (CPU torch)
            x_mean, x_std, y_mean, y_std = compute_norm_from_support(
                sup_full, args.mode, support_idx, args.std_floor_q_deg
            )

            # build support + normalized tensors
            xs, ys = split_xy(sup_full, args.mode, support_idx)
            xs_n, ys_n = norm_xy(xs, ys, x_mean, x_std, y_mean, y_std)
            xq_n, yq_n = norm_xy(xq, yq, x_mean, x_std, y_mean, y_std)

            # move support to device once (FAST)
            xs_n = xs_n.to(device, non_blocking=True)
            ys_n = ys_n.to(device, non_blocking=True)

            # reset model to init
            model.load_state_dict(init_sd, strict=True)
            model.to(device)
            set_trainable(model, args.adapt)

            params = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.Adam(params, lr=args.inner_lr)
            mse = nn.MSELoss()

            # adapt pbar (position=2)
            t0 = time.time()
            running = 0.0
            seen = 0

            adapt_bar = tqdm(total=args.adapt_steps,
                             desc=f"adapt {dist_name}",
                             ncols=120, position=2, leave=False, dynamic_ncols=True, unit="step")

            for it in range(1, args.adapt_steps + 1):
                bidx = torch.randint(0, args.support_size, (args.batch_size,), device=device)
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
                    loss = loss + args.l2_reg * l2_to_init(model, init_sd)

                loss.backward()
                opt.step()

                running += float(loss.item())
                seen += 1
                if (it % 500) == 0:
                    adapt_bar.set_postfix(loss=f"{running/max(1,seen):.6f}", lr=f"{opt.param_groups[0]['lr']:.2e}")
                adapt_bar.update(1)

            adapt_bar.close()
            dt = time.time() - t0

            # eval on fixed query set (CPU -> GPU batches)
            metrics = eval_on_query(
                model=model,
                mode=args.mode,
                xq_n=xq_n, yq_n=yq_n,
                mask=mask,
                device=device,
                y_mean=y_mean, y_std=y_std,
                angles_in_degrees=args.angles_in_degrees,
                eval_bs=args.eval_batch
            )

            all_results[dist_name].append(metrics)
            all_times[dist_name].append(dt)

            # TB per-repeat
            step = r
            writer.add_scalar(f"{dist_name}/adapt_time_s", float(dt), step)
            if args.mode == "ik":
                writer.add_scalar(f"{dist_name}/joint_rmse_deg", float(metrics["joint_rmse_deg"]), step)
            else:
                writer.add_scalar(f"{dist_name}/pos_mae_m", float(metrics["pos_mae_m"]), step)
                writer.add_scalar(f"{dist_name}/ori_mean_deg", float(metrics["ori_mean_deg"]), step)
            writer.flush()

            # update postfix
            if args.mode == "ik":
                dist_loop.set_postfix(rmse=f"{metrics['joint_rmse_deg']:.3f}", t=f"{dt/60:.2f}m")
            else:
                dist_loop.set_postfix(pos=f"{metrics['pos_mae_m']:.4f}", t=f"{dt/60:.2f}m")

    # summary logging
    print("\n========== Experiment A Summary ==========")
    for dist_name in supports.keys():
        ts = np.array(all_times[dist_name], dtype=np.float64)
        if args.mode == "ik":
            vals = np.array([m["joint_rmse_deg"] for m in all_results[dist_name]], dtype=np.float64)
            mean_v, std_v = vals.mean(), vals.std()
            print(f"{dist_name:>12s}: joint_rmse_deg = {mean_v:.3f} ± {std_v:.3f} | time = {ts.mean()/60:.2f} min")
            writer.add_scalar(f"{dist_name}/SUMMARY_joint_rmse_deg_mean", float(mean_v), 0)
            writer.add_scalar(f"{dist_name}/SUMMARY_joint_rmse_deg_std",  float(std_v),  0)
        else:
            pos = np.array([m["pos_mae_m"] for m in all_results[dist_name]], dtype=np.float64)
            ori = np.array([m["ori_mean_deg"] for m in all_results[dist_name]], dtype=np.float64)
            print(f"{dist_name:>12s}: pos_mae_m = {pos.mean():.5f} ± {pos.std():.5f} | "
                  f"ori_deg = {ori.mean():.3f} ± {ori.std():.3f} | time = {ts.mean()/60:.2f} min")
            writer.add_scalar(f"{dist_name}/SUMMARY_pos_mae_m_mean", float(pos.mean()), 0)
            writer.add_scalar(f"{dist_name}/SUMMARY_pos_mae_m_std",  float(pos.std()),  0)
            writer.add_scalar(f"{dist_name}/SUMMARY_ori_deg_mean", float(ori.mean()), 0)
            writer.add_scalar(f"{dist_name}/SUMMARY_ori_deg_std",  float(ori.std()),  0)

        writer.add_scalar(f"{dist_name}/SUMMARY_time_mean_s", float(ts.mean()), 0)
        writer.add_scalar(f"{dist_name}/SUMMARY_time_std_s",  float(ts.std()),  0)

    writer.close()
    print(f"\n[INFO] TensorBoard: tensorboard --logdir {args.tb_logdir}")
    print(f"[INFO] Run dir: {tb_dir}")

if __name__ == "__main__":
    main()
