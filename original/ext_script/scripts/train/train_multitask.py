
import argparse, os, random
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

POSE_COLS = ["x", "y", "z", "qw", "qx", "qy", "qz"]
JOINT_COLS = [f"q{i}" for i in range(1, 8)]


# -------------------------
# Data loading (same format)
# -------------------------
def load_dataset_tensor(path: str) -> torch.Tensor:
    print(f"[INFO] Loading dataset from {path}")
    cols = JOINT_COLS + POSE_COLS

    if os.path.isdir(path):
        shard_files = sorted(
            [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".pt") or f.endswith(".bin")]
        )
        if not shard_files:
            raise ValueError(f"No .pt/.bin shards found in directory: {path}")

        arrs = []
        total = 0
        for fp in shard_files:
            obj = torch.load(fp)
            t = obj["data"] if isinstance(obj, dict) and "data" in obj else obj
            if not isinstance(t, torch.Tensor) or t.ndim != 2 or t.shape[1] != 14:
                raise ValueError(f"Shard {fp} must be Tensor [N,14]")
            arrs.append(t.cpu().float().numpy())
            total += t.shape[0]
        print(f"[INFO]  total rows across shards: {total}")
        full = torch.from_numpy(np.concatenate(arrs, axis=0)).float()
        return full

    if path.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(path)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"CSV {path} missing columns: {missing}")
        return torch.from_numpy(df[cols].values.astype(np.float32)).float()

    if path.endswith(".pt") or path.endswith(".bin"):
        obj = torch.load(path)
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

    Normalization is per-task (dataset-specific), mask is not normalized.
    """

    def __init__(
        self,
        full_tensor: torch.Tensor,
        mode: str,
        mask7: torch.Tensor,
        std_floor_q_rad: float = 0.0,
    ):
        super().__init__()
        assert full_tensor.ndim == 2 and full_tensor.shape[1] == 14
        assert mask7.shape == (7,)
        self.mode = mode
        self.mask7 = mask7.float()

        if mode == "fk":
            X = full_tensor[:, 0:7]   # q
            Y = full_tensor[:, 7:14]  # pose
        elif mode == "ik":
            X = full_tensor[:, 7:14]  # pose
            Y = full_tensor[:, 0:7]   # q
        else:
            raise ValueError("mode must be 'fk' or 'ik'")

        X = X.cpu().numpy().astype(np.float32)
        Y = Y.cpu().numpy().astype(np.float32)

        x_mean = X.mean(axis=0, keepdims=True)
        x_std = X.std(axis=0, keepdims=True) + 1e-8

        y_mean = Y.mean(axis=0, keepdims=True)
        y_std = Y.std(axis=0, keepdims=True) + 1e-8

        # If IK, protect against near-constant joints (inactive joints often have tiny std)
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


def make_infinite_loader(ds: Dataset, batch_size: int, num_workers: int, device: torch.device):
    pin = device.type == "cuda"
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=pin
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
    """FK: q -> pose, conditioned by mask via additive projection."""
    def __init__(self, in_dim: int = 7, out_dim: int = 7, hidden_dim: int = 512, num_blocks: int = 4):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.mask_proj = nn.Linear(7, hidden_dim, bias=False)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ReLU()

        # start "mask has no effect" so old checkpoints behave the same
        nn.init.zeros_(self.mask_proj.weight)

    def forward(self, x7: torch.Tensor, mask7: torch.Tensor):
        h = self.act(self.fc_in(x7) + self.mask_proj(mask7))
        for blk in self.blocks:
            h = blk(h)
        return self.fc_out(h)


class IKResNetDualHead_Mask(nn.Module):
    """IK: pose -> q, conditioned by mask via additive projection; aux pose heads."""
    def __init__(self, in_dim: int = 7, out_dim: int = 7, hidden_dim: int = 1024, num_blocks: int = 4):
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
        q_pred = self.fc_joint(h)
        pos_pred = self.fc_pos(h)
        ori_pred = self.fc_ori(h)
        return q_pred, pos_pred, ori_pred


# -------------------------
# Loss helpers
# -------------------------
def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # pred/target: [B,7], mask: [B,7] (0/1)
    se = (pred - target) ** 2
    se = se * mask
    denom = mask.sum(dim=1).clamp_min(eps)  # [B]
    per = se.sum(dim=1) / denom             # [B]
    return per.mean()


# -------------------------
# Checkpoint load (optional)
# -------------------------
def try_load_state(model: nn.Module, ckpt_path: Optional[str], device: torch.device):
    if ckpt_path is None or not os.path.exists(ckpt_path):
        return False
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[INFO] Loaded init from {ckpt_path}")
    if missing:   print(f"[INFO]  missing keys (ok for mask_proj): {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:print(f"[WARN]  unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    return True


# -------------------------
# Train loop (multi-task supervised)
# -------------------------
def train_multitask(
    mode: str,
    model: nn.Module,
    task_gens: Dict[str, object],
    aux_weight: float,
    lr: float,
    steps: int,
    device: torch.device,
    log_dir: str,
    out_dir: str,
    save_every: int = 2000,
    grad_clip: float = 1.0,
):
    os.makedirs(out_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    task_names = list(task_gens.keys())
    print(f"[INFO] Training mode={mode} on tasks={task_names}")
    best = float("inf")
    best_path = os.path.join(out_dir, f"multitask_{mode}_best.pt")

    model.to(device)
    device_type = device.type

    pbar = tqdm(range(steps), desc=f"multitask-train({mode})", ncols=140)
    for it in pbar:
        tname = random.choice(task_names)
        x, y, mask = next(task_gens[tname])

        x = x.to(device, non_blocking=(device_type == "cuda"))
        y = y.to(device, non_blocking=(device_type == "cuda"))
        mask = mask.to(device, non_blocking=(device_type == "cuda"))

        model.train()
        opt.zero_grad()

        if mode == "fk":
            yhat = model(x, mask)
            loss = mse(yhat, y)
        else:
            q_pred, pos_pred, ori_pred = model(x, mask)
            loss_q = masked_mse(q_pred, y, mask)  # <--- masked joints loss
            pos_t = x[:, :3]
            ori_t = x[:, 3:7]
            loss_pos = mse(pos_pred, pos_t)
            loss_ori = mse(ori_pred, ori_t)
            loss = loss_q + aux_weight * (loss_pos + loss_ori)

        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        writer.add_scalar(f"train/{tname}_loss", loss.item(), it)
        pbar.set_postfix(task=tname, loss=f"{loss.item():.4f}")

        # periodic quick eval: average one batch from each task
        if (it + 1) % 500 == 0:
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
                writer.add_scalar("eval/avg_loss_1batch_each_task", avg, it)

                if avg < best:
                    best = avg
                    torch.save({"model_state_dict": model.state_dict(), "mode": mode, "aux_loss_weight": aux_weight}, best_path)
                    print(f"[INFO] New best avg loss {best:.6f} -> {best_path}")

        if save_every and (it + 1) % save_every == 0:
            path = os.path.join(out_dir, f"multitask_{mode}_step{it+1}.pt")
            torch.save({"model_state_dict": model.state_dict(), "mode": mode, "aux_loss_weight": aux_weight}, path)

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
    p.add_argument("--num_blocks", type=int, default=4)

    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--aux_loss_weight", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--log_dir", type=str, default="runs/multitask_masked")
    p.add_argument("--out_dir", type=str, default="runs/multitask_masked_ckpts")

    p.add_argument("--max_samples_per_task", type=int, default=10_000_000)

    # optional warm-starts
    p.add_argument("--init_ckpt", type=str, default=None, help="Optional single-task checkpoint to warm-start (e.g., 7DOF).")

    # std floor for IK target (deg)
    p.add_argument("--std_floor_q_deg", type=float, default=1.0)

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

    # masks (assumes first k joints are active)
    mask_5 = torch.tensor([1,1,1,1,1,0,0], dtype=torch.float32)
    mask_6 = torch.tensor([1,1,1,1,1,1,0], dtype=torch.float32)
    mask_7 = torch.tensor([1,1,1,1,1,1,1], dtype=torch.float32)

    std_floor_q_rad = float(args.std_floor_q_deg) * np.pi / 180.0

    # load + (optional) subsample
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

    task_gens = {
        "5dof": make_infinite_loader(ds5, args.batch_size, args.num_workers, device),
        "6dof": make_infinite_loader(ds6, args.batch_size, args.num_workers, device),
        "7dof": make_infinite_loader(ds7, args.batch_size, args.num_workers, device),
    }

    # build model
    if args.mode == "fk":
        model = ResidualMLP_Mask(in_dim=7, out_dim=7, hidden_dim=args.hidden_dim, num_blocks=args.num_blocks)
    else:
        model = IKResNetDualHead_Mask(in_dim=7, out_dim=7, hidden_dim=args.hidden_dim, num_blocks=args.num_blocks)

    # warm-start (loads old weights; mask_proj stays zero-init)
    if args.init_ckpt:
        try_load_state(model, args.init_ckpt, device)

    log_dir = os.path.join(args.log_dir, f"{args.mode}_iiwa_5_6_7")
    out_dir = os.path.join(args.out_dir, f"{args.mode}_iiwa_5_6_7")

    train_multitask(
        mode=args.mode,
        model=model,
        task_gens=task_gens,
        aux_weight=args.aux_loss_weight,
        lr=args.lr,
        steps=args.steps,
        device=device,
        log_dir=log_dir,
        out_dir=out_dir,
        save_every=5000,
        grad_clip=args.grad_clip,
    )


if __name__ == "__main__":
    main()
