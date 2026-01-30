#!/usr/bin/env python3
import argparse, os, math, time
from typing import Dict, Tuple, Set, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
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
            arrs.append(t.float().cpu().numpy())
        return torch.from_numpy(np.concatenate(arrs, axis=0)).float()

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
# Normalization from SUPPORT only
# -------------------------
def compute_norm_from_support(full: torch.Tensor, mode: str, support_idx: np.ndarray,
                              std_floor_q_deg: float = 1.0):
    sup = full[support_idx]  # [K,14]
    if mode == "fk":
        X = sup[:, 0:7].numpy()
        Y = sup[:, 7:14].numpy()
    else:
        X = sup[:, 7:14].numpy()
        Y = sup[:, 0:7].numpy()

    x_mean = X.mean(axis=0, keepdims=True).astype(np.float32)
    x_std  = (X.std(axis=0, keepdims=True) + 1e-8).astype(np.float32)
    y_mean = Y.mean(axis=0, keepdims=True).astype(np.float32)
    y_std  = (Y.std(axis=0, keepdims=True) + 1e-8).astype(np.float32)

    if mode == "ik":
        std_floor_q_rad = std_floor_q_deg * math.pi / 180.0
        y_std = np.maximum(y_std, std_floor_q_rad).astype(np.float32)

    return x_mean, x_std, y_mean, y_std

class AdaptDataset(Dataset):
    def __init__(self, full: torch.Tensor, mode: str, mask7: torch.Tensor,
                 x_mean, x_std, y_mean, y_std, indices: np.ndarray):
        self.full = full
        self.mode = mode
        self.mask7 = mask7.float()
        self.idx = indices
        self.x_mean = x_mean; self.x_std = x_std
        self.y_mean = y_mean; self.y_std = y_std

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        row = self.full[self.idx[i]]
        if self.mode == "fk":
            x = row[0:7].numpy().astype(np.float32)
            y = row[7:14].numpy().astype(np.float32)
        else:
            x = row[7:14].numpy().astype(np.float32)
            y = row[0:7].numpy().astype(np.float32)

        x = (x - self.x_mean[0]) / (self.x_std[0] + 1e-8)
        y = (y - self.y_mean[0]) / (self.y_std[0] + 1e-8)

        return torch.from_numpy(x).float(), torch.from_numpy(y).float(), self.mask7

# -------------------------
# Metrics
# -------------------------
def masked_joint_rmse_deg(q_pred: torch.Tensor, q_true: torch.Tensor, mask: torch.Tensor,
                          angles_in_degrees: bool):
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
def eval_model_and_tb(model, mode, loader, device,
                      y_mean_t: torch.Tensor, y_std_t: torch.Tensor,
                      angles_in_degrees: bool,
                      use_pbar: bool):
    model.eval()
    it_loader = tqdm(loader, desc="eval", leave=False, ncols=120) if use_pbar else loader

    if mode == "ik":
        rmses = []
        for x, y, m in it_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)
            qn, _, _ = model(x, m)
            q_pred = qn * y_std_t + y_mean_t
            q_true = y  * y_std_t + y_mean_t
            rmses.append(masked_joint_rmse_deg(q_pred, q_true, m, angles_in_degrees))
        return {"joint_rmse_deg": float(np.mean(rmses))}

    # FK: meters + degrees
    pos_rmse_list = []
    pos_mae_list  = []
    ori_deg_list  = []

    for x, y, m in it_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        m = m.to(device, non_blocking=True)

        pn = model(x, m)
        pose_pred = pn * y_std_t + y_mean_t
        pose_true = y  * y_std_t + y_mean_t

        dp = pose_pred[:, :3] - pose_true[:, :3]              # meters
        pos_rmse = torch.sqrt((dp ** 2).mean(dim=1)).mean()   # meters
        pos_mae  = dp.norm(dim=1).mean()                      # meters

        ori_deg = quat_angle_error_deg(pose_pred[:, 3:7], pose_true[:, 3:7])  # degrees

        pos_rmse_list.append(pos_rmse.item())
        pos_mae_list.append(pos_mae.item())
        ori_deg_list.append(ori_deg)

    return {
        "pos_rmse_m": float(np.mean(pos_rmse_list)),
        "pos_mae_m": float(np.mean(pos_mae_list)),
        "ori_angle_deg": float(np.mean(ori_deg_list)),
    }

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

def copy_params(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}

def l2_to_init_trainables(trainable_params: List[torch.Tensor], trainable_init: List[torch.Tensor]) -> torch.Tensor:
    loss = 0.0
    for p, p0 in zip(trainable_params, trainable_init):
        loss = loss + (p - p0).pow(2).mean()
    return loss

def build_eval_steps(adapt_steps: int, eval_every: int, eval_steps_list: str) -> List[int]:
    if eval_steps_list.strip():
        xs = [int(x.strip()) for x in eval_steps_list.split(",") if x.strip()]
        xs = [x for x in xs if 0 <= x <= adapt_steps]
        if 0 not in xs: xs = [0] + xs
        if adapt_steps not in xs: xs = xs + [adapt_steps]
        return sorted(set(xs))

    out = [0]
    if eval_every > 0:
        k = eval_every
        while k < adapt_steps:
            out.append(k)
            k += eval_every
    out.append(adapt_steps)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", choices=["fk","ik"], required=True)
    ap.add_argument("--dof", type=int, required=True)  # 4/5/6/7
    ap.add_argument("--data", required=True)

    # fixed setting (no sweep)
    ap.add_argument("--support_size", type=int, default=2000)
    ap.add_argument("--query_size", type=int, default=100000)
    ap.add_argument("--adapt_steps", type=int, default=10000)

    ap.add_argument("--inner_lr", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=512)         # support batch size
    ap.add_argument("--query_batch_size", type=int, default=8192)  # query batch size
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--adapt", choices=["all","head"], default="all")
    ap.add_argument("--l2_reg", type=float, default=1e-6)
    ap.add_argument("--std_floor_q_deg", type=float, default=1.0)
    ap.add_argument("--angles_in_degrees", action="store_true")

    # logging controls
    ap.add_argument("--log_every", type=int, default=200, help="Log support loss every N inner steps.")
    ap.add_argument("--eval_every", type=int, default=1000, help="Eval query metrics every N steps (if eval_steps_list empty).")
    ap.add_argument("--eval_steps_list", type=str, default="", help="Comma list, e.g. '0,100,500,1000,2000'. Overrides eval_every.")
    ap.add_argument("--eval_pbar", action="store_true", help="Show progress bar during evaluation.")
    ap.add_argument("--num_workers", type=int, default=4)

    # TensorBoard
    ap.add_argument("--tb_logdir", type=str, default="./runs/adapt_tb_fixed")
    ap.add_argument("--tb_name", type=str, default="", help="Run name; if empty, auto name is used.")

    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    print(f"[INFO] device={device}")

    # speed knobs
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # TensorBoard writer
    os.makedirs(args.tb_logdir, exist_ok=True)
    if args.tb_name.strip():
        run_dir = os.path.join(args.tb_logdir, args.tb_name.strip())
    else:
        run_dir = os.path.join(
            args.tb_logdir,
            f"{args.mode}_dof{args.dof}_K{args.support_size}_Q{args.query_size}_S{args.adapt_steps}_lr{args.inner_lr:g}_bs{args.batch_size}_{args.adapt}"
        )
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(run_dir)
    writer.add_text("config", str(vars(args)))

    # Build model
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

    # Load data
    full = load_dataset_tensor(args.data)
    N = full.shape[0]
    if args.support_size + args.query_size > N:
        raise ValueError(f"support+query exceeds N ({args.support_size}+{args.query_size}>{N})")

    perm = rng.permutation(N)
    support_idx = perm[:args.support_size]
    query_idx   = perm[args.support_size: args.support_size + args.query_size]

    mask = get_mask_for_dof(args.dof)

    # Normalize using SUPPORT ONLY
    x_mean, x_std, y_mean, y_std = compute_norm_from_support(full, args.mode, support_idx, args.std_floor_q_deg)
    y_mean_t = torch.tensor(y_mean, device=device, dtype=torch.float32)
    y_std_t  = torch.tensor(y_std,  device=device, dtype=torch.float32)

    ds_sup = AdaptDataset(full, args.mode, mask, x_mean, x_std, y_mean, y_std, support_idx)
    ds_qry = AdaptDataset(full, args.mode, mask, x_mean, x_std, y_mean, y_std, query_idx)

    pin = (device.type == "cuda")
    sup_loader = DataLoader(ds_sup, batch_size=args.batch_size, shuffle=True, drop_last=True,
                            num_workers=args.num_workers, pin_memory=pin, persistent_workers=(args.num_workers > 0))
    qry_loader = DataLoader(ds_qry, batch_size=args.query_batch_size, shuffle=False, drop_last=False,
                            num_workers=args.num_workers, pin_memory=pin, persistent_workers=(args.num_workers > 0))

    # Eval schedule (steps on x-axis)
    eval_steps = build_eval_steps(args.adapt_steps, args.eval_every, args.eval_steps_list)
    eval_set: Set[int] = set(eval_steps)

    # Step=0 query eval
    m0 = eval_model_and_tb(model, args.mode, qry_loader, device, y_mean_t, y_std_t,
                           angles_in_degrees=args.angles_in_degrees, use_pbar=args.eval_pbar)
    print(f"[EVAL] step=0 | {m0}")
    for k, v in m0.items():
        writer.add_scalar(f"query/{k}", v, 0)
    writer.flush()

    # Adaptation
    init_sd = copy_params(model)
    model.load_state_dict(init_sd, strict=True)
    set_trainable(model, args.adapt)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=args.inner_lr)
    mse = nn.MSELoss()

    # fast L2 against init (only for trainable params)
    trainable_init = [p.detach().clone() for p in trainable]

    pbar = tqdm(total=args.adapt_steps, desc=f"adapt({args.adapt})", ncols=120, unit="step")
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

            if args.mode == "fk":
                yhat = model(x, m)
                loss = mse(yhat, y)
            else:
                qn, posn, orin = model(x, m)
                se = ((qn - y) ** 2) * m
                denom = m.sum(dim=1).clamp_min(1e-8)
                loss_q = (se.sum(dim=1) / denom).mean()
                loss_aux = mse(posn, x[:, :3]) + mse(orin, x[:, 3:7])
                loss = loss_q + 0.03 * loss_aux

            if args.l2_reg > 0:
                loss = loss + args.l2_reg * l2_to_init_trainables(trainable, trainable_init)

            loss.backward()
            opt.step()

            it += 1
            pbar.update(1)

            running += float(loss.item())
            seen += 1

            if args.log_every > 0 and (it % args.log_every) == 0:
                avg_loss = running / max(1, seen)
                lr_now = opt.param_groups[0]["lr"]
                pbar.set_postfix(loss=f"{avg_loss:.6f}", lr=f"{lr_now:.2e}")
                writer.add_scalar("support/loss_norm", avg_loss, it)
                writer.add_scalar("support/lr", lr_now, it)

            if it in eval_set:
                mq = eval_model_and_tb(model, args.mode, qry_loader, device, y_mean_t, y_std_t,
                                       angles_in_degrees=args.angles_in_degrees, use_pbar=args.eval_pbar)
                print(f"[EVAL] step={it} | {mq}")
                for k, v in mq.items():
                    writer.add_scalar(f"query/{k}", v, it)
                writer.flush()

            if it >= args.adapt_steps:
                break

    pbar.close()
    dt = time.time() - t0
    print(f"[INFO] Adaptation finished: steps={args.adapt_steps} time={dt:.2f}s")
    writer.add_scalar("meta/adapt_time_s", dt, args.adapt_steps)
    writer.flush()
    writer.close()
    print(f"[INFO] TensorBoard run dir: {run_dir}")
    print(f"[INFO] Launch: tensorboard serve --logdir {args.tb_logdir}")

if __name__ == "__main__":
    main()
