# train_kinematics_nn_pose.py
#
# Train neural kinematics model with full pose (position + quaternion orientation).
# Modes:
#   - fk: q -> [x, y, z, qw, qx, qy, qz]
#   - ik: [x, y, z, qw, qx, qy, qz] -> q
#

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# ----------------------
# 1. Config / CLI
# ----------------------

parser = argparse.ArgumentParser(
    description="Train FK / IK neural network (with position + quaternion) from iiwa14 dataset."
)
parser.add_argument("--csv", type=str, required=True, help="Path to dataset CSV.")
parser.add_argument(
    "--mode", type=str, choices=["fk", "ik"], default="fk", help="fk: q->pose, ik: pose->q."
)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--hidden_dim", type=int, default=512)
parser.add_argument(
    "--num_blocks",
    type=int,
    default=3,
    help="Number of residual blocks in the MLP.",
)
parser.add_argument(
    "--train_split", type=float, default=0.9, help="Fraction of data for training."
)
parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu.")
parser.add_argument(
    "--out_dir", type=str, default="models", help="Folder to save models + scalers."
)
parser.add_argument(
    "--log_dir",
    type=str,
    default="runs/kinematics_pose",
    help="TensorBoard log directory.",
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=1e-5,
    help="L2 weight decay for Adam optimizer.",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
    help="DataLoader workers (use 0 on Windows if issues).",
)
parser.add_argument(
    "--scheduler_patience",
    type=int,
    default=10,
    help="Epochs of no val improvement before reducing LR.",
)
parser.add_argument(
    "--grad_clip",
    type=float,
    default=1.0,
    help="Max norm for gradient clipping (0 to disable).",
)

args = parser.parse_args()
device = torch.device(args.device if torch.cuda.is_available() else "cpu")


# ----------------------
# 2. Dataset
# ----------------------

class KinematicsPoseDataset(Dataset):
    """
    Dataset for kinematics with full pose.

    mode:
      - 'fk':  input = q1..q7                       (7)
               target = x,y,z,qw,qx,qy,qz          (7)
      - 'ik':  input = x,y,z,qw,qx,qy,qz           (7)
               target = q1..q7                     (7)

    Both inputs and targets are standardized (z-scored) inside this class.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        mode: str,
        input_mean=None,
        input_std=None,
        target_mean=None,
        target_std=None,
    ):
        self.mode = mode

        pose_cols = ["x", "y", "z", "qw", "qx", "qy", "qz"]
        joint_cols = [f"q{i}" for i in range(1, 8)]

        if not set(pose_cols).issubset(df.columns):
            missing = set(pose_cols) - set(df.columns)
            raise ValueError(f"Dataset is missing pose columns: {missing}")
        if not set(joint_cols).issubset(df.columns):
            missing = set(joint_cols) - set(df.columns)
            raise ValueError(f"Dataset is missing joint columns: {missing}")

        if mode == "fk":
            input_cols = joint_cols
            target_cols = pose_cols
        elif mode == "ik":
            input_cols = pose_cols
            target_cols = joint_cols
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        self.input_cols = input_cols
        self.target_cols = target_cols

        inputs = df[input_cols].values.astype(np.float32)
        targets = df[target_cols].values.astype(np.float32)

        # compute stats if not provided (for full dataset)
        if input_mean is None:
            input_mean = inputs.mean(axis=0, keepdims=True)
            input_std = inputs.std(axis=0, keepdims=True) + 1e-8
        if target_mean is None:
            target_mean = targets.mean(axis=0, keepdims=True)
            target_std = targets.std(axis=0, keepdims=True) + 1e-8

        self.inputs = (inputs - input_mean) / input_std
        self.targets = (targets - target_mean) / target_std

        # store stats for later use (saving in checkpoint)
        self.input_mean = input_mean
        self.input_std = input_std
        self.target_mean = target_mean
        self.target_std = target_std

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.targets[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


# ----------------------
# 3. Model
# ----------------------

class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        h = self.fc2(h)
        return self.act(h + x)


class MLP(nn.Module):
    """
    Residual MLP used for both FK and IK.

    - First projects input to hidden_dim
    - Then applies num_blocks residual blocks
    - Finally projects to output_dim
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 512, num_blocks: int = 3):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc_in(x))
        for block in self.blocks:
            h = block(h)
        return self.fc_out(h)


# ----------------------
# 4. Load data + split
# ----------------------

print(f"[INFO] Loading CSV from {args.csv}")
df = pd.read_csv(args.csv)
# Shuffle to break time correlations
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

full_dataset = KinematicsPoseDataset(df, mode=args.mode)

N = len(full_dataset)
N_train = int(args.train_split * N)
N_val = N - N_train
train_dataset, val_dataset = random_split(full_dataset, [N_train, N_val])

pin_mem = device.type == "cuda"

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=False,
    num_workers=args.num_workers,
    pin_memory=pin_mem,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=args.num_workers,
    pin_memory=pin_mem,
)

input_dim = full_dataset.inputs.shape[1]
output_dim = full_dataset.targets.shape[1]

model = MLP(
    input_dim,
    output_dim,
    hidden_dim=args.hidden_dim,
    num_blocks=args.num_blocks,
).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=args.scheduler_patience,
)

print(f"[INFO] Mode: {args.mode}")
print(f"[INFO] Input dim: {input_dim}, Output dim: {output_dim}")
print(f"[INFO] Train size: {N_train}, Val size: {N_val}")
print(f"[INFO] Device: {device}, pin_memory={pin_mem}, num_workers={args.num_workers}")


# ----------------------
# 5. TensorBoard writer
# ----------------------

log_dir = os.path.join(args.log_dir, args.mode)
writer = SummaryWriter(log_dir=log_dir)

hparams_text = (
    f"mode: {args.mode}\n"
    f"csv: {args.csv}\n"
    f"epochs: {args.epochs}\n"
    f"batch_size: {args.batch_size}\n"
    f"lr: {args.lr}\n"
    f"hidden_dim: {args.hidden_dim}\n"
    f"num_blocks: {args.num_blocks}\n"
    f"train_split: {args.train_split}\n"
    f"weight_decay: {args.weight_decay}\n"
    f"num_workers: {args.num_workers}\n"
    f"scheduler_patience: {args.scheduler_patience}\n"
    f"grad_clip: {args.grad_clip}\n"
)
writer.add_text("config", hparams_text, global_step=0)


# ----------------------
# 6. Training loop with ONE global progress bar
# ----------------------

best_val_loss = float("inf")
last_val_loss = float("nan")  # updated after first epoch

num_train_batches = len(train_loader)
total_steps = args.epochs * num_train_batches

global_bar = tqdm(
    total=total_steps,
    desc="Training (pose)",
    ncols=150,
)

for epoch in range(1, args.epochs + 1):
    model.train()
    running_train_loss = 0.0
    seen_train_samples = 0

    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        if args.grad_clip is not None and args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optimizer.step()

        batch_size = x.size(0)
        running_train_loss += loss.item() * batch_size
        seen_train_samples += batch_size
        train_mse_epoch = running_train_loss / max(seen_train_samples, 1)

        # update global progress bar
        global_bar.update(1)
        current_lr = optimizer.param_groups[0]["lr"]
        global_bar.set_postfix(
            epoch=f"{epoch}/{args.epochs}",
            batch=f"{batch_idx+1}/{num_train_batches}",
            train_mse=f"{train_mse_epoch:.4f}",
            val_mse=f"{last_val_loss:.4f}" if not np.isnan(last_val_loss) else "N/A",
            lr=f"{current_lr:.2e}",
        )

    # finalize train loss for epoch
    train_loss_epoch = running_train_loss / max(seen_train_samples, 1)

    # ---- validation ----
    model.eval()
    val_loss_epoch = 0.0
    val_seen = 0
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val = x_val.to(device, non_blocking=True)
            y_val = y_val.to(device, non_blocking=True)
            y_hat = model(x_val)
            loss = criterion(y_hat, y_val)
            bs = x_val.size(0)
            val_loss_epoch += loss.item() * bs
            val_seen += bs
    val_loss_epoch /= max(val_seen, 1)
    last_val_loss = val_loss_epoch

    # scheduler step on val loss
    scheduler.step(val_loss_epoch)

    # log to TensorBoard
    writer.add_scalar("loss/train_mse", train_loss_epoch, epoch)
    writer.add_scalar("loss/val_mse", val_loss_epoch, epoch)
    writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

    # update bar postfix with final epoch metrics
    global_bar.set_postfix(
        epoch=f"{epoch}/{args.epochs}",
        batch=f"{num_train_batches}/{num_train_batches}",
        train_mse=f"{train_loss_epoch:.4f}",
        val_mse=f"{val_loss_epoch:.4f}",
        lr=f"{optimizer.param_groups[0]['lr']:.2e}",
    )

    # optional: param histograms every 10 epochs
    if epoch % 10 == 0:
        for name, param in model.named_parameters():
            writer.add_histogram(f"params/{name}", param.data.cpu().numpy(), epoch)

    # save best model
    if val_loss_epoch < best_val_loss:
        best_val_loss = val_loss_epoch
        os.makedirs(args.out_dir, exist_ok=True)
        model_path = os.path.join(log_dir, f"{args.mode}_pose_best.pt")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_mean": full_dataset.input_mean,
                "input_std": full_dataset.input_std,
                "target_mean": full_dataset.target_mean,
                "target_std": full_dataset.target_std,
                "mode": args.mode,
                "input_cols": full_dataset.input_cols,
                "target_cols": full_dataset.target_cols,
            },
            model_path,
        )
        tqdm.write(
            f"Saved best model to {model_path} (epoch={epoch}, val MSE={val_loss_epoch:.6f})"
        )

global_bar.close()
writer.close()
print("[INFO] Training finished.")
