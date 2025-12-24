#!/usr/bin/env python3
# Meta-learning (Reptile-style) for kinematics across 5, 6, 7 DOF tasks.
# Supports:
#   --mode fk : meta-learn FK model  (q -> pose)
#   --mode ik : meta-learn IK model  (pose -> q  with aux heads)
#
# Datasets: CSV or .pt/.bin (or dir of .pt shards) with columns:
#   [q1..q7, x, y, z, qw, qx, qy, qz]
#
# For IK/FK mode, you can warm-start from:
#   - a single init checkpoint (--init_checkpoint), OR
#   - three single-task checkpoints for 5/6/7 DOF (--init_ckpt_5dof/6dof/7dof),
#     which will be parameter-wise averaged to form a balanced meta init.

import argparse
import os
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

POSE_COLS = ["x", "y", "z", "qw", "qx", "qy", "qz"]
JOINT_COLS = [f"q{i}" for i in range(1, 8)]


# ---------------------------------------------------------------------------
# Data utilities (same format as your single-task script)
# ---------------------------------------------------------------------------

def load_dataset_tensor(path: str) -> torch.Tensor:
    """
    Load the full dataset as a 2D float32 Tensor [N,14] from:
      - CSV file with columns [q1..q7,x,y,z,qw,qx,qy,qz]
      - single .pt/.bin tensor
      - directory of .pt/.bin shards (dict with 'data' or raw tensor)
    """
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
            raise ValueError(f"No .pt/.bin shards found in directory: {path}")

        arrs: List[np.ndarray] = []
        total_rows = 0
        for f_path in shard_files:
            print(f"[INFO]  shard: {f_path}")
            obj = torch.load(f_path)
            if isinstance(obj, dict) and "data" in obj:
                arr = obj["data"]
            else:
                arr = obj
            if not isinstance(arr, torch.Tensor):
                raise TypeError(f"Shard {f_path} must contain a Tensor or dict with 'data'.")
            if arr.ndim != 2 or arr.shape[1] != 14:
                raise ValueError(f"Shard {f_path} must have shape [N,14], got {tuple(arr.shape)}")
            arr_np = arr.cpu().float().numpy()
            arrs.append(arr_np)
            total_rows += arr_np.shape[0]

        print(f"[INFO]  total rows across shards: {total_rows}")
        full_np = np.concatenate(arrs, axis=0)
        full_tensor = torch.from_numpy(full_np).float()

    elif path.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(path)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"CSV {path} missing columns: {missing}")
        data_np = df[cols].values.astype(np.float32)
        full_tensor = torch.from_numpy(data_np).float()

    elif path.endswith(".pt") or path.endswith(".bin"):
        obj = torch.load(path)
        if isinstance(obj, dict) and "data" in obj:
            full_tensor = obj["data"]
        else:
            full_tensor = obj
        if not isinstance(full_tensor, torch.Tensor):
            raise TypeError(f"{path} must contain a Tensor or dict with 'data'.")
        if full_tensor.ndim != 2 or full_tensor.shape[1] != 14:
            raise ValueError(f"{path} tensor must be [N,14], got {tuple(full_tensor.shape)}")
        full_tensor = full_tensor.float()
    else:
        raise ValueError(f"Unsupported dataset path type: {path}")

    print(f"[INFO] Loaded tensor of shape {tuple(full_tensor.shape)}")
    return full_tensor


class KinematicsTensorDataset(Dataset):
    """
    Dataset for kinematics with dataset-specific normalisation.

    mode:
      - 'fk': input  = joints [q1..q7]
              target = pose   [x,y,z,qw,qx,qy,qz]
      - 'ik': input  = pose   [x,y,z,qw,qx,qy,qz]
              target = joints [q1..q7]
    """

    def __init__(self, full_tensor: torch.Tensor, mode: str):
        super().__init__()
        if full_tensor.ndim != 2 or full_tensor.shape[1] != 14:
            raise ValueError(f"Expected full_tensor [N,14], got {tuple(full_tensor.shape)}")

        if mode == "fk":
            X = full_tensor[:, 0:7]   # q
            Y = full_tensor[:, 7:14]  # pose
        elif mode == "ik":
            X = full_tensor[:, 7:14]  # pose
            Y = full_tensor[:, 0:7]   # q
        else:
            raise ValueError(f"Unknown mode {mode}, expected 'fk' or 'ik'.")

        X = X.cpu().numpy().astype(np.float32)
        Y = Y.cpu().numpy().astype(np.float32)

        input_mean = X.mean(axis=0, keepdims=True)
        input_std = X.std(axis=0, keepdims=True) + 1e-8
        target_mean = Y.mean(axis=0, keepdims=True)
        target_std = Y.std(axis=0, keepdims=True) + 1e-8

        self.inputs = (X - input_mean) / input_std
        self.targets = (Y - target_mean) / target_std

        self.input_mean = input_mean
        self.input_std = input_std
        self.target_mean = target_mean
        self.target_std = target_std

        self.mode = mode

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.inputs[idx]).float()
        y = torch.from_numpy(self.targets[idx]).float()
        return x, y


# ---------------------------------------------------------------------------
# Models: FK and IK
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, dim: int, p_drop: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        h = self.fc2(h)
        return self.act(h + x)


class ResidualMLP(nn.Module):
    """
    FK network: in_dim -> hidden_dim -> num_blocks -> out_dim.
    Used for FK meta-learning (q -> pose).
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 512, num_blocks: int = 4):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc_in(x))
        for blk in self.blocks:
            h = blk(h)
        return self.fc_out(h)


class IKResNetDualHead(nn.Module):
    """
    IK network: pose -> joints with aux heads for pose reconstruction.

      input:  pose [x,y,z,qw,qx,qy,qz] (normalised)
      outputs:
        - q_pred   (7)
        - pos_pred (3)
        - ori_pred (4)

    Loss: L = L_q + aux_weight * (L_pos + L_ori)
    """

    def __init__(self, in_dim: int = 7, hidden_dim: int = 1024, num_blocks: int = 4, out_dim: int = 7):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])
        self.fc_joint = nn.Linear(hidden_dim, out_dim)
        self.fc_pos = nn.Linear(hidden_dim, 3)
        self.fc_ori = nn.Linear(hidden_dim, 4)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor):
        h = self.act(self.fc_in(x))
        for blk in self.blocks:
            h = blk(h)
        q_pred = self.fc_joint(h)
        pos_pred = self.fc_pos(h)
        ori_pred = self.fc_ori(h)
        return q_pred, pos_pred, ori_pred


# ---------------------------------------------------------------------------
# Meta utilities
# ---------------------------------------------------------------------------

def make_infinite_loader(dataset: Dataset, batch_size: int, num_workers: int, device: torch.device):
    """Create an infinite generator of (x, y) batches from a Dataset."""
    pin = device.type == "cuda"
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin,
    )

    def _gen():
        while True:
            for batch in loader:
                yield batch

    return _gen()


def reptile_meta_train(
    model: nn.Module,
    mode: str,
    task_loaders: Dict[str, object],
    criterion: nn.Module,
    aux_weight: float,
    meta_lr: float,
    inner_lr: float,
    inner_steps: int,
    meta_iters: int,
    log_dir: str,
    out_dir: str,
    device: torch.device,
    grad_clip: float = 1.0,
):
    """
    Reptile meta-training for FK or IK.

    - mode == 'fk':
        y_hat = model(x), loss = MSE(y_hat, y)
    - mode == 'ik':
        q_pred, pos_pred, ori_pred = model(x),
        loss = L_q + aux_weight * (L_pos + L_ori)

    Reuses a single model object, updating base_state with no_grad().
    """
    os.makedirs(out_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    task_names = list(task_loaders.keys())
    print(f"[INFO] Meta tasks: {task_names}")
    writer.add_text(
        "config",
        "\n".join([
            f"mode: {mode}",
            f"meta_lr: {meta_lr}",
            f"inner_lr: {inner_lr}",
            f"inner_steps: {inner_steps}",
            f"meta_iters: {meta_iters}",
            f"aux_weight: {aux_weight}",
            f"tasks: {task_names}",
        ]),
        0,
    )

    model.to(device)
    device_type = device.type

    # base/meta params
    base_state = {k: v.clone().detach() for k, v in model.state_dict().items()}
    best_query_loss = float("inf")
    best_path = os.path.join(out_dir, f"meta_{mode}_5_6_7dof_best.pt")

    with tqdm(total=meta_iters, desc=f"Meta-training ({mode})", ncols=150) as pbar:
        for meta_iter in range(meta_iters):
            # reset accumulator
            delta = {k: torch.zeros_like(v, device=device) for k, v in base_state.items()}
            total_query_loss = 0.0
            n_query = 0

            for tname in task_names:
                batch_gen = task_loaders[tname]

                # reset model to base/meta params
                with torch.no_grad():
                    model.load_state_dict(base_state)
                model.train()
                inner_opt = torch.optim.Adam(model.parameters(), lr=inner_lr)

                # ----- inner adaptation -----
                for step in range(inner_steps):
                    x, y = next(batch_gen)
                    x = x.to(device, non_blocking=(device_type == "cuda"))
                    y = y.to(device, non_blocking=(device_type == "cuda"))

                    inner_opt.zero_grad()
                    if mode == "fk":
                        y_hat = model(x)
                        loss = criterion(y_hat, y)
                    else:  # ik
                        q_pred, pos_pred, ori_pred = model(x)
                        loss_q = criterion(q_pred, y)
                        pos_target = x[:, :3]
                        ori_target = x[:, 3:7]
                        loss_pos = criterion(pos_pred, pos_target)
                        loss_ori = criterion(ori_pred, ori_target)
                        loss = loss_q + aux_weight * (loss_pos + loss_ori)

                    loss.backward()
                    if grad_clip is not None and grad_clip > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                    inner_opt.step()

                # ----- query loss for logging -----
                model.eval()
                with torch.no_grad():
                    xq, yq = next(batch_gen)
                    xq = xq.to(device, non_blocking=(device_type == "cuda"))
                    yq = yq.to(device, non_blocking=(device_type == "cuda"))

                    if mode == "fk":
                        y_hat_q = model(xq)
                        query_loss = criterion(y_hat_q, yq).item()
                    else:
                        q_pred_q, pos_pred_q, ori_pred_q = model(xq)
                        loss_q_q = criterion(q_pred_q, yq)
                        pos_tq = xq[:, :3]
                        ori_tq = xq[:, 3:7]
                        loss_pos_q = criterion(pos_pred_q, pos_tq)
                        loss_ori_q = criterion(ori_pred_q, ori_tq)
                        query_loss = (loss_q_q + aux_weight * (loss_pos_q + loss_ori_q)).item()

                total_query_loss += query_loss
                n_query += 1

                # ----- accumulate parameter differences -----
                with torch.no_grad():
                    adapted_state = model.state_dict()
                    for k in base_state.keys():
                        delta[k] += adapted_state[k] - base_state[k]

                del inner_opt  # free Adam state

            # ----- meta update -----
            avg_query_loss = total_query_loss / max(n_query, 1)
            num_tasks = len(task_names)
            step_size = meta_lr / max(num_tasks, 1)

            with torch.no_grad():
                for k in base_state.keys():
                    base_state[k].add_(step_size * delta[k])

            with torch.no_grad():
                model.load_state_dict(base_state)

            # logging
            writer.add_scalar("meta/query_loss", avg_query_loss, meta_iter)

            if avg_query_loss < best_query_loss:
                best_query_loss = avg_query_loss
                torch.save(
                    {
                        "model_state_dict": base_state,
                        "mode": mode,
                        "hidden_dim": model.fc_in.out_features,
                        "num_blocks": len(model.blocks),
                        "aux_loss_weight": aux_weight,
                        "meta_lr": meta_lr,
                        "inner_lr": inner_lr,
                        "inner_steps": inner_steps,
                        "meta_iters": meta_iters,
                        "tasks": task_names,
                    },
                    best_path,
                )
                tqdm.write(
                    f"[Meta iter {meta_iter}] New best meta query loss "
                    f"{avg_query_loss:.6f}, saved to {best_path}"
                )

            pbar.set_postfix(query_loss=f"{avg_query_loss:.4f}", best=f"{best_query_loss:.4f}")
            pbar.update(1)

    writer.close()

    final_path = os.path.join(out_dir, f"meta_{mode}_5_6_7dof_final.pt")
    torch.save(
        {
            "model_state_dict": base_state,
            "mode": mode,
            "hidden_dim": model.fc_in.out_features,
            "num_blocks": len(model.blocks),
            "aux_loss_weight": aux_weight,
            "meta_lr": meta_lr,
            "inner_lr": inner_lr,
            "inner_steps": inner_steps,
            "meta_iters": meta_iters,
            "tasks": task_names,
        },
        final_path,
    )
    print(f"[INFO] Saved final meta {mode} model to {final_path}")
    print(f"[INFO] Best meta query loss: {best_query_loss:.6f}")


# ---------------------------------------------------------------------------
# Helper: load and average multiple init checkpoints
# ---------------------------------------------------------------------------

def load_and_average_checkpoints(
    paths: List[Optional[str]],
    expected_mode: str,
    device: torch.device,
) -> Optional[dict]:
    """
    Load 1-3 checkpoints, keep those that exist and match expected_mode,
    and return a dict with averaged model_state_dict and metadata.

    Returns:
        None if no valid checkpoints were found.
    """
    valid_ckpts = []
    for pth in paths:
        if pth is None:
            continue
        if not os.path.exists(pth):
            print(f"[WARN] init checkpoint {pth} not found, skipping.")
            continue
        try:
            ckpt = torch.load(pth, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(pth, map_location=device)
        mode = ckpt.get("mode", expected_mode)
        if mode != expected_mode:
            print(f"[WARN] init checkpoint {pth} has mode={mode}, expected {expected_mode}; skipping.")
            continue
        valid_ckpts.append(ckpt)
        print(f"[INFO] Included init checkpoint: {pth}")

    if not valid_ckpts:
        return None

    # Average state dicts
    ref_sd = valid_ckpts[0]["model_state_dict"]
    keys = ref_sd.keys()
    avg_sd = {}
    n = float(len(valid_ckpts))
    for k in keys:
        tensors = [ck["model_state_dict"][k].to(device) for ck in valid_ckpts]
        avg = sum(tensors) / n
        avg_sd[k] = avg

    # Metadata: just take from first (they should match)
    hidden_dim = valid_ckpts[0].get("hidden_dim", None)
    num_blocks = valid_ckpts[0].get("num_blocks", None)
    aux_w = valid_ckpts[0].get("aux_loss_weight", 0.1)

    print(f"[INFO] Averaged {len(valid_ckpts)} init checkpoints.")
    return {
        "model_state_dict": avg_sd,
        "mode": expected_mode,
        "hidden_dim": hidden_dim,
        "num_blocks": num_blocks,
        "aux_loss_weight": aux_w,
    }


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Reptile meta-training for FK/IK across 5/6/7 DOF."
    )
    p.add_argument("--mode", type=str, choices=["fk", "ik"], required=True,
                   help="fk: q->pose, ik: pose->q")

    p.add_argument("--task_5dof", type=str, required=True)
    p.add_argument("--task_6dof", type=str, required=True)
    p.add_argument("--task_7dof", type=str, required=True)

    p.add_argument("--hidden_dim", type=int, default=1024)
    p.add_argument("--num_blocks", type=int, default=4)

    p.add_argument("--meta_iters", type=int, default=2000)
    p.add_argument("--inner_steps", type=int, default=20)
    p.add_argument("--inner_lr", type=float, default=1e-3)
    p.add_argument("--meta_lr", type=float, default=1e-4)

    p.add_argument("--batch_size", type=int, default=8192)
    p.add_argument("--aux_loss_weight", type=float, default=0.1,
                   help="Only used in IK mode.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--log_dir", type=str, default="runs/meta")
    p.add_argument("--out_dir", type=str, default="runs/meta_ckpts")

    p.add_argument(
        "--max_samples_per_task",
        type=int,
        default=25_000_000,
        help="Optional cap on samples per task for meta-training.",
    )

    # Single init checkpoint (old behaviour, still supported)
    p.add_argument(
        "--init_checkpoint",
        type=str,
        default=None,
        help="Optional single-task checkpoint (.pt) used to initialise meta model "
             "(FK or IK, depending on --mode).",
    )

    # NEW: three DOF-specific init checkpoints to be averaged
    p.add_argument(
        "--init_ckpt_5dof",
        type=str,
        default=None,
        help="Optional single-task 5DOF checkpoint to include in averaged init.",
    )
    p.add_argument(
        "--init_ckpt_6dof",
        type=str,
        default=None,
        help="Optional single-task 6DOF checkpoint to include in averaged init.",
    )
    p.add_argument(
        "--init_ckpt_7dof",
        type=str,
        default=None,
        help="Optional single-task 7DOF checkpoint to include in averaged init.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    # reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Meta mode: {args.mode}")

    # ---- load datasets for 5/6/7 DOF ----
    task_paths = {
        "5dof": args.task_5dof,
        "6dof": args.task_6dof,
        "7dof": args.task_7dof,
    }

    task_datasets: Dict[str, KinematicsTensorDataset] = {}
    for name, path in task_paths.items():
        print(f"[INFO] Loading task '{name}' from {path}")
        full_tensor = load_dataset_tensor(path)
        if args.max_samples_per_task is not None and full_tensor.shape[0] > args.max_samples_per_task:
            idx = torch.randperm(full_tensor.shape[0])[: args.max_samples_per_task]
            full_tensor = full_tensor[idx]
            print(f"       -> subsampled to {full_tensor.shape[0]} rows for meta-training.")
        ds = KinematicsTensorDataset(full_tensor, mode=args.mode)
        print(f"       -> {len(ds)} samples, in_dim={ds.inputs.shape[1]}, out_dim={ds.targets.shape[1]}")
        task_datasets[name] = ds

    # sanity check dims
    first_ds = next(iter(task_datasets.values()))
    in_dim = first_ds.inputs.shape[1]
    out_dim = first_ds.targets.shape[1]
    for name, ds in task_datasets.items():
        if ds.inputs.shape[1] != in_dim or ds.targets.shape[1] != out_dim:
            raise RuntimeError(f"Task {name} has mismatched dims: "
                               f"in={ds.inputs.shape[1]}, out={ds.targets.shape[1]}")

    # ---- build infinite loaders ----
    task_loaders = {
        name: make_infinite_loader(ds, args.batch_size, args.num_workers, device)
        for name, ds in task_datasets.items()
    }

    # ---- create meta model (warm-start logic) ----
    # First try to average 5/6/7 DOF checkpoints if any given.
    avg_ckpt = load_and_average_checkpoints(
        [args.init_ckpt_5dof, args.init_ckpt_6dof, args.init_ckpt_7dof],
        expected_mode=args.mode,
        device=device,
    )

    if args.mode == "fk":
        if avg_ckpt is not None:
            print("[INFO] Initialising FK meta model from AVERAGED 5/6/7DOF checkpoints.")
            ckpt_hidden = avg_ckpt.get("hidden_dim", args.hidden_dim)
            ckpt_blocks = avg_ckpt.get("num_blocks", args.num_blocks)
            model = ResidualMLP(
                in_dim=in_dim,
                out_dim=out_dim,
                hidden_dim=ckpt_hidden,
                num_blocks=ckpt_blocks,
            ).to(device)
            model.load_state_dict(avg_ckpt["model_state_dict"], strict=True)
            args.hidden_dim = ckpt_hidden
            args.num_blocks = ckpt_blocks

        elif args.init_checkpoint is not None and os.path.exists(args.init_checkpoint):
            print(f"[INFO] Initialising FK meta model from {args.init_checkpoint}")
            try:
                ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
            except TypeError:
                ckpt = torch.load(args.init_checkpoint, map_location=device)

            ckpt_mode = ckpt.get("mode", "fk")
            if ckpt_mode != "fk":
                print(f"[WARN] init_checkpoint mode={ckpt_mode}, expected 'fk'. Starting from random.")
                model = ResidualMLP(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    hidden_dim=args.hidden_dim,
                    num_blocks=args.num_blocks,
                ).to(device)
            else:
                ckpt_hidden = ckpt.get("hidden_dim", args.hidden_dim)
                ckpt_blocks = ckpt.get("num_blocks", args.num_blocks)
                model = ResidualMLP(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    hidden_dim=ckpt_hidden,
                    num_blocks=ckpt_blocks,
                ).to(device)
                model.load_state_dict(ckpt["model_state_dict"], strict=True)
                args.hidden_dim = ckpt_hidden
                args.num_blocks = ckpt_blocks
                print(f"[INFO] Loaded FK init: hidden_dim={ckpt_hidden}, num_blocks={ckpt_blocks}")
        else:
            if args.init_checkpoint is not None and avg_ckpt is None:
                print(f"[WARN] FK init_checkpoint {args.init_checkpoint} not found; starting from random.")
            model = ResidualMLP(
                in_dim=in_dim,
                out_dim=out_dim,
                hidden_dim=args.hidden_dim,
                num_blocks=args.num_blocks,
            ).to(device)

    else:  # IK meta: IKResNetDualHead
        if avg_ckpt is not None:
            print("[INFO] Initialising IK meta model from AVERAGED 5/6/7DOF checkpoints.")
            ckpt_hidden = avg_ckpt.get("hidden_dim", args.hidden_dim)
            ckpt_blocks = avg_ckpt.get("num_blocks", args.num_blocks)
            aux_w = avg_ckpt.get("aux_loss_weight", args.aux_loss_weight)
            model = IKResNetDualHead(
                in_dim=in_dim,
                hidden_dim=ckpt_hidden,
                num_blocks=ckpt_blocks,
                out_dim=out_dim,
            ).to(device)
            model.load_state_dict(avg_ckpt["model_state_dict"], strict=True)
            args.hidden_dim = ckpt_hidden
            args.num_blocks = ckpt_blocks
            args.aux_loss_weight = aux_w
            print(f"[INFO] Loaded IK averaged init: hidden_dim={ckpt_hidden}, num_blocks={ckpt_blocks}, aux={aux_w}")

        elif args.init_checkpoint is not None and os.path.exists(args.init_checkpoint):
            print(f"[INFO] Initialising IK meta model from {args.init_checkpoint}")
            try:
                ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
            except TypeError:
                ckpt = torch.load(args.init_checkpoint, map_location=device)

            ckpt_mode = ckpt.get("mode", "ik")
            if ckpt_mode != "ik":
                print(f"[WARN] init_checkpoint mode={ckpt_mode}, expected 'ik'. Starting from random.")
                model = IKResNetDualHead(
                    in_dim=in_dim,
                    hidden_dim=args.hidden_dim,
                    num_blocks=args.num_blocks,
                    out_dim=out_dim,
                ).to(device)
            else:
                ckpt_hidden = ckpt.get("hidden_dim", args.hidden_dim)
                ckpt_blocks = ckpt.get("num_blocks", args.num_blocks)
                aux_w = ckpt.get("aux_loss_weight", args.aux_loss_weight)
                model = IKResNetDualHead(
                    in_dim=in_dim,
                    hidden_dim=ckpt_hidden,
                    num_blocks=ckpt_blocks,
                    out_dim=out_dim,
                ).to(device)
                model.load_state_dict(ckpt["model_state_dict"], strict=True)
                args.hidden_dim = ckpt_hidden
                args.num_blocks = ckpt_blocks
                args.aux_loss_weight = aux_w
                print(f"[INFO] Loaded IK init: hidden_dim={ckpt_hidden}, num_blocks={ckpt_blocks}, aux={aux_w}")
        else:
            if args.init_checkpoint is not None and avg_ckpt is None:
                print(f"[WARN] IK init_checkpoint {args.init_checkpoint} not found; starting from random.")
            model = IKResNetDualHead(
                in_dim=in_dim,
                hidden_dim=args.hidden_dim,
                num_blocks=args.num_blocks,
                out_dim=out_dim,
            ).to(device)

    criterion = nn.MSELoss()

    log_dir = os.path.join(args.log_dir, f"{args.mode}_reptile_5_6_7dof")
    out_dir = log_dir

    reptile_meta_train(
        model,
        mode=args.mode,
        task_loaders=task_loaders,
        criterion=criterion,
        aux_weight=args.aux_loss_weight,
        meta_lr=args.meta_lr,
        inner_lr=args.inner_lr,
        inner_steps=args.inner_steps,
        meta_iters=args.meta_iters,
        log_dir=log_dir,
        out_dir=out_dir,
        device=device,
        grad_clip=args.grad_clip,
    )


if __name__ == "__main__":
    main()
