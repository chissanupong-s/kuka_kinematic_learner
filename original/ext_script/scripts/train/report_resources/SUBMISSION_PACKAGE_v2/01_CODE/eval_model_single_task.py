#!/usr/bin/env python3
# eval — load ckpt, run held-out, print mean+rmse pos err + ori err

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


POSE_COLS = ["x", "y", "z", "qw", "qx", "qy", "qz"]
JOINT_COLS = [f"q{i}" for i in range(1, 8)]


def get_args():
    p = argparse.ArgumentParser("Evaluate FK/IK model with metric pose errors")
    p.add_argument(
        "--csv",
        type=str,
        required=True,
        help="CSV, single .pt/.bin, or directory of .pt/.bin shards.",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to *_pose_best.pt checkpoint saved by the training script.",
    )
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def load_dataset_tensor(path: str) -> torch.Tensor:
    """Return [N,14] float32 tensor [q1..q7,x,y,z,qw,qx,qy,qz]."""
    print(f"[INFO] Loading dataset from {path}")
    cols = JOINT_COLS + POSE_COLS

    if os.path.isdir(path):
        shard_files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.endswith(".pt") or f.endswith(".bin")
        ]
        shard_files = sorted(shard_files)
        if not shard_files:
            raise ValueError(f"No .pt/.bin shards found in {path}")

        arrs = []
        total_rows = 0
        for sp in shard_files:
            print(f"[INFO]  - loading shard '{sp}'")
            obj = torch.load(sp, map_location="cpu")
            if isinstance(obj, torch.Tensor):
                t = obj
            elif isinstance(obj, dict) and "data" in obj:
                t = obj["data"]
            else:
                raise ValueError(
                    f"Shard '{sp}' must be Tensor or dict with 'data'."
                )

            if t.ndim != 2 or t.shape[1] != len(cols):
                raise ValueError(
                    f"Shard '{sp}' has shape {tuple(t.shape)}, expected [N,{len(cols)}]."
                )

            arrs.append(t.numpy())
            total_rows += t.shape[0]

        print(f"[INFO] Total rows across shards: {total_rows}")
        arr = np.concatenate(arrs, axis=0).astype(np.float32)
        return torch.from_numpy(arr)

    if path.endswith(".pt") or path.endswith(".bin"):
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            t = obj
        elif isinstance(obj, dict) and "data" in obj:
            t = obj["data"]
        else:
            raise ValueError(
                f"File '{path}' must be Tensor or dict with 'data' when using .pt/.bin."
            )

        if t.ndim != 2 or t.shape[1] != len(cols):
            raise ValueError(
                f"Tensor in '{path}' has shape {tuple(t.shape)}, expected [N,{len(cols)}]."
            )

        return t.float()

    import pandas as pd

    df = pd.read_csv(path)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    arr = df[cols].values.astype(np.float32)
    return torch.from_numpy(arr)


class EvalKinematicsDataset(Dataset):
    """
    Dataset using stats from checkpoint so normalization matches training.

    mode:
      fk: X=q     Y=pose
      ik: X=pose  Y=q
    """

    def __init__(self, full_tensor: torch.Tensor, mode: str,
                 input_mean, input_std, target_mean, target_std):
        super().__init__()

        if full_tensor.ndim != 2 or full_tensor.shape[1] != 14:
            raise ValueError(f"full_tensor must be [N,14], got {tuple(full_tensor.shape)}")

        self.mode = mode

        if mode == "fk":
            X = full_tensor[:, 0:7]
            Y = full_tensor[:, 7:14]
        elif mode == "ik":
            X = full_tensor[:, 7:14]
            Y = full_tensor[:, 0:7]
        else:
            raise ValueError(f"Unsupported mode {mode}")

        X = X.cpu().numpy().astype(np.float32)
        Y = Y.cpu().numpy().astype(np.float32)

        self.inputs = (X - input_mean) / input_std
        self.targets = (Y - target_mean) / target_std

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.inputs[idx]).float()
        y = torch.from_numpy(self.targets[idx]).float()
        return x, y


class ResBlock(nn.Module):
    def __init__(self, dim: int, p_drop: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.fc2(h)
        return self.act(h + x)


class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 512, num_blocks: int = 4):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.act(self.fc_in(x))
        for b in self.blocks:
            h = b(h)
        return self.fc_out(h)


class IKResNetDualHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512, num_blocks: int = 4, out_dim: int = 7):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])
        self.fc_joint = nn.Linear(hidden_dim, out_dim)
        self.fc_pos = nn.Linear(hidden_dim, 3)
        self.fc_ori = nn.Linear(hidden_dim, 4)
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.act(self.fc_in(x))
        for b in self.blocks:
            h = b(h)
        q_pred = self.fc_joint(h)
        pos_pred = self.fc_pos(h)
        ori_pred = self.fc_ori(h)
        return q_pred, pos_pred, ori_pred


def quat_angle_deg(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    q1, q2: [B,4] quaternions in [qw,qx,qy,qz] order
    returns: [B] angular distance in degrees
    """
    q1 = q1 / (torch.linalg.norm(q1, dim=1, keepdim=True) + 1e-8)
    q2 = q2 / (torch.linalg.norm(q2, dim=1, keepdim=True) + 1e-8)

    dot = torch.sum(q1 * q2, dim=1).abs()
    dot = dot.clamp(0.0, 1.0)

    return 2.0 * torch.rad2deg(torch.acos(dot))


def main():
    args = get_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"[INFO] Using device: {device}")

    print(f"[INFO] Loading checkpoint from {args.checkpoint}")
    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location=device)

    mode = ckpt.get("mode", "fk")
    hidden_dim = ckpt.get("hidden_dim", 512)
    num_blocks = ckpt.get("num_blocks", 4)
    aux_weight = ckpt.get("aux_loss_weight", 0.1)

    in_mean = np.asarray(ckpt["input_mean"], dtype=np.float32)
    in_std = np.asarray(ckpt["input_std"], dtype=np.float32)
    tgt_mean = np.asarray(ckpt["target_mean"], dtype=np.float32)
    tgt_std = np.asarray(ckpt["target_std"], dtype=np.float32)

    print(f"[INFO] mode={mode}, hidden_dim={hidden_dim}, num_blocks={num_blocks}")
    print(f"[INFO] aux_loss_weight={aux_weight}")

    full_tensor = load_dataset_tensor(args.csv)

    dataset = EvalKinematicsDataset(
        full_tensor=full_tensor,
        mode=mode,
        input_mean=in_mean,
        input_std=in_std,
        target_mean=tgt_mean,
        target_std=tgt_std,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    input_dim = dataset.inputs.shape[1]
    output_dim = dataset.targets.shape[1]

    if mode == "fk":
        model = ResidualMLP(
            input_dim,
            output_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
        ).to(device)
    else:
        model = IKResNetDualHead(
            input_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            out_dim=output_dim,
        ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    in_mean_t = torch.tensor(in_mean, dtype=torch.float32, device=device)
    in_std_t = torch.tensor(in_std, dtype=torch.float32, device=device)
    tgt_mean_t = torch.tensor(tgt_mean, dtype=torch.float32, device=device)
    tgt_std_t = torch.tensor(tgt_std, dtype=torch.float32, device=device)

    print("[INFO] Running evaluation...")

    bar = tqdm(total=len(loader), ncols=120)

    tot = 0.0
    tot_q = 0.0
    tot_pos = 0.0
    tot_ori = 0.0
    seen = 0

    sum_pos_err = 0.0
    sum_pos_err_sq = 0.0
    sum_ori_deg = 0.0
    sum_ori_deg_sq = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if mode == "fk":
                y_hat = model(x)

                loss = nn.functional.mse_loss(y_hat, y)
                loss_q = loss
                loss_pos = torch.tensor(0.0, device=device)
                loss_ori = torch.tensor(0.0, device=device)

                pose_pred = y_hat * tgt_std_t + tgt_mean_t
                pose_true = y * tgt_std_t + tgt_mean_t

                pos_pred = pose_pred[:, :3]
                pos_true = pose_true[:, :3]
                quat_pred = pose_pred[:, 3:7]
                quat_true = pose_true[:, 3:7]

            else:
                q_pred, pos_pred_n, ori_pred_n = model(x)

                loss_q = nn.functional.mse_loss(q_pred, y)

                pos_tgt_n = x[:, :3]
                ori_tgt_n = x[:, 3:7]

                loss_pos = nn.functional.mse_loss(pos_pred_n, pos_tgt_n)
                loss_ori = nn.functional.mse_loss(ori_pred_n, ori_tgt_n)
                loss = loss_q + aux_weight * (loss_pos + loss_ori)

                pos_pred = pos_pred_n * in_std_t[:3] + in_mean_t[:3]
                pos_true = pos_tgt_n * in_std_t[:3] + in_mean_t[:3]

                quat_pred = ori_pred_n * in_std_t[3:7] + in_mean_t[3:7]
                quat_true = ori_tgt_n * in_std_t[3:7] + in_mean_t[3:7]

            pos_err = torch.linalg.norm(pos_pred - pos_true, dim=1)
            ori_err_deg = quat_angle_deg(quat_pred, quat_true)

            bs = x.size(0)

            tot += loss.item() * bs
            tot_q += loss_q.item() * bs
            tot_pos += loss_pos.item() * bs
            tot_ori += loss_ori.item() * bs
            seen += bs

            sum_pos_err += pos_err.sum().item()
            sum_pos_err_sq += (pos_err ** 2).sum().item()
            sum_ori_deg += ori_err_deg.sum().item()
            sum_ori_deg_sq += (ori_err_deg ** 2).sum().item()

            bar.update(1)

    bar.close()

    total_mse = tot / max(seen, 1)
    q_mse = tot_q / max(seen, 1)
    pos_mse = tot_pos / max(seen, 1)
    ori_mse = tot_ori / max(seen, 1)

    mean_pos_m = sum_pos_err / max(seen, 1)
    rmse_pos_m = (sum_pos_err_sq / max(seen, 1)) ** 0.5
    mean_ori_deg = sum_ori_deg / max(seen, 1)
    rmse_ori_deg = (sum_ori_deg_sq / max(seen, 1)) ** 0.5

    print("\n=== Evaluation results ===")
    print(f"Total MSE: {total_mse:.6f}")
    print(f"Joint MSE (q): {q_mse:.6f}")

    if mode == "ik":
        print(f"Aux position MSE: {pos_mse:.6f}")
        print(f"Aux orientation MSE: {ori_mse:.6f}")

    print(f"Mean position error: {mean_pos_m:.6f} m")
    print(f"RMSE position error: {rmse_pos_m:.6f} m")
    print(f"Mean orientation error: {mean_ori_deg:.6f} deg")
    print(f"RMSE orientation error: {rmse_ori_deg:.6f} deg")


if __name__ == "__main__":
    main()