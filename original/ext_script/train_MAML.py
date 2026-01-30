import os
import copy
import math
import argparse
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ------------------------------
#  IK residual network (same style as chapter 2)
# ------------------------------

class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.act(out)
        return out


class IKResNetDualHead(nn.Module):
    """
    Input: pose (7D)  [x, y, z, qw, qx, qy, qz]
    Outputs:
      - joints (7D)
      - aux position (3D)
      - aux orientation (4D)
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        dropout: float = 0.0,
        output_joints: int = 7,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.output_joints = output_joints

        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)]
        )
        self.act = nn.ReLU()

        # Main head: joints
        self.fc_out_q = nn.Linear(hidden_dim, output_joints)

        # Aux heads
        self.fc_out_pos = nn.Linear(hidden_dim, 3)
        self.fc_out_ori = nn.Linear(hidden_dim, 4)

    def forward(self, x):
        h = self.fc_in(x)
        h = self.act(h)
        for block in self.blocks:
            h = block(h)
        q_pred = self.fc_out_q(h)
        pos_pred = self.fc_out_pos(h)
        ori_pred = self.fc_out_ori(h)
        return q_pred, pos_pred, ori_pred


# ------------------------------
#  Dataset
# ------------------------------

class KinematicsTensorDataset(Dataset):
    """
    Takes a [N,14] tensor with columns:
      [q1..q7, x, y, z, qw, qx, qy, qz]
    For IK: input = pose (7D), target = joints (7D).
    """

    def __init__(self, data: torch.Tensor, mode: str = "ik"):
        super().__init__()
        assert data.dim() == 2 and data.size(1) == 14, "Expected [N,14] tensor"
        self.mode = mode

        q = data[:, 0:7]
        pose = data[:, 7:14]

        if mode == "ik":
            X = pose
            Y = q
        elif mode == "fk":
            X = q
            Y = pose
        else:
            raise ValueError(f"Unknown mode {mode}")

        # Compute stats per task
        self.input_mean = X.mean(dim=0)
        self.input_std = X.std(dim=0) + 1e-8
        self.target_mean = Y.mean(dim=0)
        self.target_std = Y.std(dim=0) + 1e-8

        self.X = (X - self.input_mean) / self.input_std
        self.Y = (Y - self.target_mean) / self.target_std

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def _load_single_tensor_file(path: str) -> torch.Tensor:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, torch.Tensor):
        data = ckpt
    elif isinstance(ckpt, dict) and "data" in ckpt:
        data = ckpt["data"]
    else:
        raise RuntimeError(f"Unsupported format in {path}, expected Tensor or dict with 'data'")
    if data.dim() != 2 or data.size(1) != 14:
        raise RuntimeError(f"Tensor in {path} has shape {tuple(data.shape)}, expected [N,14]")
    return data.float()


def load_tensor_from_path(path: str) -> torch.Tensor:
    """
    Load a single [N,14] tensor from:
      - CSV file
      - .pt/.pth/.bin file (tensor or dict with 'data')
      - directory of .pt/.pth/.bin shards
    """
    path = os.path.expanduser(path)
    if os.path.isdir(path):
        parts: List[torch.Tensor] = []
        for fname in sorted(os.listdir(path)):
            if not (fname.endswith(".pt") or fname.endswith(".pth") or fname.endswith(".bin")):
                continue
            full = os.path.join(path, fname)
            t = _load_single_tensor_file(full)
            parts.append(t)
        if not parts:
            raise RuntimeError(f"No .pt/.pth/.bin files found in directory {path}")
        return torch.cat(parts, dim=0)

    # Single file
    if path.endswith(".csv"):
        import pandas as pd

        df = pd.read_csv(path)
        data = torch.tensor(df.values, dtype=torch.float32)
        if data.size(1) != 14:
            raise RuntimeError(f"CSV {path} has {data.size(1)} columns, expected 14")
        return data

    if path.endswith(".pt") or path.endswith(".pth") or path.endswith(".bin"):
        return _load_single_tensor_file(path)

    raise RuntimeError(f"Don't know how to load file type: {path}")


# ------------------------------
#  Task wrapper for meta-learning
# ------------------------------

class IKTask:
    def __init__(self, name: str, data: torch.Tensor, batch_size: int = 256, mode: str = "ik"):
        self.name = name
        self.dataset = KinematicsTensorDataset(data, mode=mode)
        self.batch_size = batch_size
        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        self.iterator = iter(self.loader)

    def sample_batch(self, device: torch.device):
        try:
            x, y = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            x, y = next(self.iterator)
        return x.to(device), y.to(device)


# ------------------------------
#  Reptile-style "MAML-ish" meta-training loop
# ------------------------------

def meta_train_reptile(
    task_dict: Dict[str, IKTask],
    hidden_dim: int = 256,
    num_blocks: int = 4,
    inner_lr: float = 1e-2,
    meta_lr: float = 1e-3,
    inner_steps: int = 5,
    meta_batch_size: int = 3,
    meta_iters: int = 1000,
    aux_loss_weight: float = 0.1,
    device: str = "cuda",
    save_path: str = "meta_ik_reptile.pt",
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = IKResNetDualHead(
        input_dim=7,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout=0.0,
        output_joints=7,
    ).to(device)

    # Initialise meta-parameters
    meta_state = copy.deepcopy(model.state_dict())
    loss_fn = nn.MSELoss()

    task_names = list(task_dict.keys())
    print(f"Meta-training over tasks: {task_names}")

    for it in range(1, meta_iters + 1):
        # Keep a frozen copy of the current meta-params
        base_state = copy.deepcopy(meta_state)

        # Sum of parameter deltas across tasks
        sum_delta = {k: torch.zeros_like(v) for k, v in base_state.items()}

        # Sample a meta-batch of tasks (with replacement if needed)
        chosen = []
        for _ in range(meta_batch_size):
            idx = torch.randint(0, len(task_names), (1,)).item()
            chosen.append(task_names[idx])

        meta_loss_estimate = 0.0

        for task_name in chosen:
            task = task_dict[task_name]

            # Start each task from the same base meta-params
            model.load_state_dict(base_state)
            model.to(device)

            inner_opt = torch.optim.SGD(model.parameters(), lr=inner_lr, momentum=0.9)

            # Inner-loop adaptation on this task
            for _ in range(inner_steps):
                x_s, y_s = task.sample_batch(device)
                q_pred, pos_pred, ori_pred = model(x_s)

                # Main joint loss
                loss_q = loss_fn(q_pred, y_s)

                # For simplicity here we only use joint loss for meta-training.
                loss = loss_q

                inner_opt.zero_grad()
                loss.backward()
                inner_opt.step()

            # After adaptation, measure query loss on a fresh batch
            x_q, y_q = task.sample_batch(device)
            with torch.no_grad():
                q_pred_q, _, _ = model(x_q)
                task_query_loss = loss_fn(q_pred_q, y_q).item()
            meta_loss_estimate += task_query_loss

            # Accumulate parameter deltas
            adapted_state = model.state_dict()
            for k in sum_delta.keys():
                sum_delta[k] += adapted_state[k] - base_state[k]

        # Average meta loss over tasks
        meta_loss_estimate /= meta_batch_size

        # Meta-update: move meta-params slightly towards adapted params
        for k in meta_state.keys():
            meta_state[k] = meta_state[k] + (meta_lr / meta_batch_size) * sum_delta[k]

        model.load_state_dict(meta_state)

        if it % 10 == 0:
            print(f"[Meta iter {it}/{meta_iters}] approx query loss: {meta_loss_estimate:.6f}")

        # (Optional) save intermediate checkpoints
        if it % 100 == 0:
            torch.save(
                {
                    "model_state_dict": meta_state,
                    "hidden_dim": hidden_dim,
                    "num_blocks": num_blocks,
                    "meta_iters": it,
                },
                save_path,
            )

    # Final save
    torch.save(
        {
            "model_state_dict": meta_state,
            "hidden_dim": hidden_dim,
            "num_blocks": num_blocks,
            "meta_iters": meta_iters,
        },
        save_path,
    )
    print(f"Saved meta-trained IK model to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="MAML-style (Reptile) meta-training for IK across DOFs")

    parser.add_argument("--task_5dof", type=str, required=True,
                        help="Path to 5 DOF dataset (csv / .pt / dir of shards)")
    parser.add_argument("--task_6dof", type=str, required=True,
                        help="Path to 6 DOF dataset (csv / .pt / dir of shards)")
    parser.add_argument("--task_7dof", type=str, required=True,
                        help="Path to 7 DOF dataset (csv / .pt / dir of shards)")

    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--inner_lr", type=float, default=1e-2)
    parser.add_argument("--meta_lr", type=float, default=1e-3)
    parser.add_argument("--inner_steps", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--meta_batch_size", type=int, default=3)
    parser.add_argument("--meta_iters", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_path", type=str, default="meta_ik_reptile.pt")

    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading datasets...")
    data_5 = load_tensor_from_path(args.task_5dof)
    data_6 = load_tensor_from_path(args.task_6dof)
    data_7 = load_tensor_from_path(args.task_7dof)

    print(f"5 DOF data: {data_5.shape}")
    print(f"6 DOF data: {data_6.shape}")
    print(f"7 DOF data: {data_7.shape}")

    task_5 = IKTask("5dof", data_5, batch_size=args.batch_size, mode="ik")
    task_6 = IKTask("6dof", data_6, batch_size=args.batch_size, mode="ik")
    task_7 = IKTask("7dof", data_7, batch_size=args.batch_size, mode="ik")

    tasks = {
        "5dof": task_5,
        "6dof": task_6,
        "7dof": task_7,
    }

    meta_train_reptile(
        task_dict=tasks,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        inner_lr=args.inner_lr,
        meta_lr=args.meta_lr,
        inner_steps=args.inner_steps,
        meta_batch_size=args.meta_batch_size,
        meta_iters=args.meta_iters,
        device=args.device,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
