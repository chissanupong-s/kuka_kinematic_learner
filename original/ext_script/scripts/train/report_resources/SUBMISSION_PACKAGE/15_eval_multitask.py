#!/usr/bin/env python3
import argparse, os, math
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


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


# -------------------------
# Data loader (Tensor [N,14])
# columns: [q1..q7, x,y,z, qw,qx,qy,qz]
# -------------------------
def load_dataset_tensor(path: str) -> torch.Tensor:
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
# Model (must match training)
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
# Dataset applying task_norm
# -------------------------
class FKEvalDataset(Dataset):
    def __init__(self, full: torch.Tensor, mask7: torch.Tensor,
                 x_mean: np.ndarray, x_std: np.ndarray,
                 y_mean: np.ndarray, y_std: np.ndarray,
                 indices: np.ndarray):
        self.full = full
        self.mask7 = mask7.float()
        self.idx = indices

        self.x_mean = x_mean.astype(np.float32)  # [1,7]
        self.x_std  = x_std.astype(np.float32)   # [1,7]
        self.y_mean = y_mean.astype(np.float32)  # [1,7]
        self.y_std  = y_std.astype(np.float32)   # [1,7]

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        row = self.full[self.idx[i]]  # [14]
        q = row[0:7].numpy().astype(np.float32)
        pose = row[7:14].numpy().astype(np.float32)

        qn = (q - self.x_mean[0]) / (self.x_std[0] + 1e-8)
        yn = (pose - self.y_mean[0]) / (self.y_std[0] + 1e-8)

        return (torch.from_numpy(qn).float(),
                torch.from_numpy(yn).float(),
                torch.from_numpy(pose).float(),  # raw pose for metric
                self.mask7)


# -------------------------
# Quaternion + metrics
# -------------------------
def quat_normalize(q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return q / q.norm(dim=1, keepdim=True).clamp_min(eps)

def quat_align_sign(q_pred: torch.Tensor, q_true: torch.Tensor) -> torch.Tensor:
    dot = (q_pred * q_true).sum(dim=1, keepdim=True)
    sign = torch.where(dot < 0.0,
                       torch.tensor(-1.0, device=q_pred.device),
                       torch.tensor(1.0, device=q_pred.device))
    return q_pred * sign

@torch.no_grad()
def quat_angle_error_deg(q_pred: torch.Tensor, q_true: torch.Tensor) -> torch.Tensor:
    q_pred = quat_normalize(q_pred)
    q_true = quat_normalize(q_true)
    q_pred = quat_align_sign(q_pred, q_true)
    dot = (q_pred * q_true).sum(dim=1).abs().clamp(-1.0, 1.0)
    ang = 2.0 * torch.acos(dot)
    return ang * (180.0 / math.pi)

@torch.no_grad()
def eval_fk(model: nn.Module, loader, device: torch.device,
            y_mean: np.ndarray, y_std: np.ndarray,
            use_pbar: bool):
    model.eval()
    y_mean_t = torch.tensor(y_mean, device=device, dtype=torch.float32)  # [1,7]
    y_std_t  = torch.tensor(y_std,  device=device, dtype=torch.float32)  # [1,7]

    pos_rmse_list = []
    pos_mae_list  = []
    ori_deg_list  = []

    it = tqdm(loader, desc="eval_fk", ncols=120) if use_pbar else loader
    for qn, yn, pose_raw, m in it:
        qn = qn.to(device, non_blocking=True)
        m  = m.to(device, non_blocking=True)

        yn_pred = model(qn, m)                      # normalized pose
        pose_pred = yn_pred * y_std_t + y_mean_t    # denorm -> meters + quaternion
        pose_true = pose_raw.to(device, non_blocking=True)

        # position (meters)
        dp = pose_pred[:, :3] - pose_true[:, :3]
        pos_rmse = torch.sqrt((dp ** 2).mean(dim=1)).mean()   # meters
        pos_mae  = dp.norm(dim=1).mean()                      # meters

        # orientation (degrees)
        ori_deg = quat_angle_error_deg(pose_pred[:, 3:7], pose_true[:, 3:7]).mean()

        pos_rmse_list.append(pos_rmse.item())
        pos_mae_list.append(pos_mae.item())
        ori_deg_list.append(ori_deg.item())

    return {
        "pos_rmse_m": float(np.mean(pos_rmse_list)),
        "pos_mae_m":  float(np.mean(pos_mae_list)),
        "ori_angle_deg": float(np.mean(ori_deg_list)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="multitask_fk_best.pt")
    ap.add_argument("--dof", type=int, required=True, help="5/6/7 (or 4 if you have it)")
    ap.add_argument("--data", required=True, help="dataset .pt or directory of shards")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_samples", type=int, default=0, help="0 = all; else random subset")
    ap.add_argument("--eval_pbar", action="store_true")

    # optional: log into tensorboard at ckpt step
    ap.add_argument("--tb_logdir", default="", help="if set, appends eval scalars here")
    ap.add_argument("--tb_tag", default="", help="prefix tag, e.g. 'eval_fk'")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    print(f"[INFO] device={device}")

    ckpt = safe_torch_load(args.ckpt, map_location=device)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError("Checkpoint must contain model_state_dict")
    if "task_norm" not in ckpt:
        raise ValueError("Checkpoint missing task_norm (needed to evaluate in real units)")

    sd = ckpt["model_state_dict"]
    hidden_dim, num_blocks = infer_arch_from_state_dict(sd)
    model = ResidualMLP_Mask(hidden_dim=hidden_dim, num_blocks=num_blocks)
    model.load_state_dict(sd, strict=True)
    model.to(device)

    step = int(ckpt.get("step", 0))
    print(f"[INFO] ckpt step={step}")

    task_key = f"{args.dof}dof"
    if task_key not in ckpt["task_norm"]:
        raise ValueError(f"task_norm missing key '{task_key}'. Available: {list(ckpt['task_norm'].keys())}")

    tn = ckpt["task_norm"][task_key]
    x_mean = np.array(tn["x_mean"], dtype=np.float32)  # [1,7]
    x_std  = np.array(tn["x_std"],  dtype=np.float32)
    y_mean = np.array(tn["y_mean"], dtype=np.float32)  # [1,7]
    y_std  = np.array(tn["y_std"],  dtype=np.float32)

    # Load data
    full = load_dataset_tensor(args.data)
    N = full.shape[0]

    # Select indices
    rng = np.random.RandomState(42)
    if args.max_samples and args.max_samples < N:
        idx = rng.choice(N, size=args.max_samples, replace=False)
        print(f"[INFO] Using subset: {args.max_samples}/{N}")
    else:
        idx = np.arange(N)
        print(f"[INFO] Using all samples: {N}")

    mask = get_mask_for_dof(args.dof)

    ds = FKEvalDataset(full, mask, x_mean, x_std, y_mean, y_std, idx)
    pin = (device.type == "cuda")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=pin, persistent_workers=(args.num_workers > 0))

    metrics = eval_fk(model, loader, device, y_mean, y_std, use_pbar=args.eval_pbar)
    print(f"[FK EVAL] dof={args.dof} | {metrics}")

    # TensorBoard logging (optional)
    if args.tb_logdir.strip():
        os.makedirs(args.tb_logdir, exist_ok=True)
        writer = SummaryWriter(args.tb_logdir)
        prefix = args.tb_tag.strip() or f"eval_fk/{task_key}"
        for k, v in metrics.items():
            writer.add_scalar(f"{prefix}/{k}", v, step)
        writer.flush()
        writer.close()
        print(f"[INFO] Wrote TensorBoard scalars to: {args.tb_logdir} (step={step})")


if __name__ == "__main__":
    main()
