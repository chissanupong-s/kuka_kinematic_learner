#!/usr/bin/env python3
# fk/ik trainer — q <-> pose with ResMLP backbone

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


POSE_COLS = ["x", "y", "z", "qw", "qx", "qy", "qz"]
JOINT_COLS = [f"q{i}" for i in range(1, 8)]



def get_args():
    parser = argparse.ArgumentParser(
        description="Train FK / IK neural network (position + quaternion)."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help=(
            "Path to dataset: either\n"
            " - CSV file (.csv), or\n"
            " - single .pt/.bin Tensor with columns [q1..q7,x,y,z,qw,qx,qy,qz], or\n"
            " - directory containing multiple .pt/.bin shards."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["fk", "ik"],
        default="fk",
        help="fk: q->pose, ik: pose->q.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=4,
        help="Number of residual blocks (hidden layers) in the MLP trunk.",
    )
    parser.add_argument(
        "--train_frac",
        type=float,
        default=0.5,
        help="Fraction of data used for training (default 0.5).",
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.2,
        help="Fraction of data used for validation (default 0.2). "
             "Test fraction is 1 - train_frac - val_frac.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu.")
    parser.add_argument(
        "--out_dir", type=str, default="models",
        help="Folder to save models + scalers.",
    )
    parser.add_argument(
        "--log_dir", type=str, default="runs/kinematics_pose_pt",
        help="TensorBoard log directory root (mode is appended).",
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
    parser.add_argument(
        "--aux_loss_weight",
        type=float,
        default=0.1,
        help=(
            "Weight for auxiliary IK pose reconstruction losses "
            "(only used in ik mode: loss = L_q + w*(L_pos + L_ori))."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the train/val/test split + parameter init + minibatch order.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="If > 0, sub-sample the dataset to this many rows (for matching the report's 15M cap). 0 = use all.",
    )
    return parser.parse_args()



def load_dataset_tensor(path: str) -> torch.Tensor:
    """
    Load the full dataset as a 2D float32 Tensor [N,14] from:

      - CSV file with columns [q1..q7,x,y,z,qw,qx,qy,qz]
      - Single .pt/.bin Tensor
      - Directory of .pt/.bin shards

    No pandas objects are passed to the DataLoader; everything is a Tensor.
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
                    f"Unexpected shard format in '{sp}'. "
                    f"Expected Tensor or dict with key 'data'."
                )
            if t.ndim != 2 or t.shape[1] != len(cols):
                raise ValueError(
                    f"Shard '{sp}' has shape {tuple(t.shape)}, expected [N,{len(cols)}]."
                )
            arrs.append(t.numpy())
            total_rows += t.shape[0]

        print(f"[INFO] Total rows across shards: {total_rows}")
        arr_all = np.concatenate(arrs, axis=0).astype(np.float32)
        tensor = torch.from_numpy(arr_all)
        return tensor

    # single file path
    if path.endswith(".pt") or path.endswith(".bin"):
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            t = obj
        elif isinstance(obj, dict) and "data" in obj:
            t = obj["data"]
        else:
            raise ValueError(
                f"Unexpected .pt format in '{path}'. "
                f"Expected Tensor or dict with key 'data'."
            )
        if t.ndim != 2 or t.shape[1] != len(cols):
            raise ValueError(
                f"Tensor in '{path}' has shape {tuple(t.shape)}, expected [N,{len(cols)}]."
            )
        return t.float()

    # otherwise assume CSV
    import pandas as pd

    df = pd.read_csv(path)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    arr = df[cols].values.astype(np.float32)
    return torch.from_numpy(arr)


class KinematicsTensorDataset(Dataset):
    """
    Dataset for kinematics with full pose, based on a single 2D tensor [N,14].

    mode:
      - 'fk':  input = q1..q7                       (7)
               target = x,y,z,qw,qx,qy,qz          (7)
      - 'ik':  input = x,y,z,qw,qx,qy,qz           (7)
               target = q1..q7                     (7)

    Inputs and targets are standardized (z-scored) using stats computed on
    the full dataset (shared across train/val/test).
    """

    def __init__(
        self,
        full_tensor: torch.Tensor,
        mode: str,
        input_mean=None,
        input_std=None,
        target_mean=None,
        target_std=None,
    ):
        super().__init__()

        if full_tensor.ndim != 2 or full_tensor.shape[1] != 14:
            raise ValueError(
                f"full_tensor must be [N,14], got shape {tuple(full_tensor.shape)}"
            )

        self.mode = mode
        self.full_tensor = full_tensor  # [N,14]

        if mode == "fk":
            # q -> pose
            X = full_tensor[:, 0:7]   # q1..q7
            Y = full_tensor[:, 7:14]  # x,y,z,qw,qx,qy,qz
        elif mode == "ik":
            # pose -> q
            X = full_tensor[:, 7:14]
            Y = full_tensor[:, 0:7]
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        X = X.cpu().numpy().astype(np.float32)
        Y = Y.cpu().numpy().astype(np.float32)

        # compute stats if not provided
        if input_mean is None:
            input_mean = X.mean(axis=0, keepdims=True)
            input_std = X.std(axis=0, keepdims=True) + 1e-8
        if target_mean is None:
            target_mean = Y.mean(axis=0, keepdims=True)
            target_std = Y.std(axis=0, keepdims=True) + 1e-8

        self.inputs = (X - input_mean) / input_std
        self.targets = (Y - target_mean) / target_std

        # store stats for saving/checkpointing
        self.input_mean = input_mean
        self.input_std = input_std
        self.target_mean = target_mean
        self.target_std = target_std

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        h = self.fc2(h)
        return self.act(h + x)


class ResidualMLP(nn.Module):
    """
    Generic residual MLP: in_dim -> hidden_dim -> num_blocks -> out_dim.
    Used for FK (q -> pose).
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 512, num_blocks: int = 4):
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


class IKResNetDualHead(nn.Module):
    """
    Residual MLP with dual-head + auxiliary loss for IK.

    - Shared trunk maps pose input to a hidden representation.
    - Main head: predicts joints q1..q7 (primary IK output).
    - Aux heads: reconstruct position (x,y,z) and orientation (qw,qx,qy,qz)
      in normalized space as an auxiliary regularization task.

    During training:
      loss = L_q + aux_weight * (L_pos + L_ori)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,
        num_blocks: int = 4,
        out_dim: int = 7,
    ):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])
        self.fc_joint = nn.Linear(hidden_dim, out_dim)
        self.fc_pos = nn.Linear(hidden_dim, 3)   # x, y, z
        self.fc_ori = nn.Linear(hidden_dim, 4)   # qw, qx, qy, qz
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor):
        h = self.act(self.fc_in(x))
        for block in self.blocks:
            h = block(h)
        q_pred = self.fc_joint(h)
        pos_pred = self.fc_pos(h)
        ori_pred = self.fc_ori(h)
        return q_pred, pos_pred, ori_pred



def evaluate_on_loader(model, loader, device, criterion, aux_weight, mode: str):
    model.eval()
    total_loss = 0.0
    total_loss_q = 0.0
    total_loss_pos = 0.0
    total_loss_ori = 0.0
    seen = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if mode == "fk":
                y_hat = model(x)
                loss = criterion(y_hat, y)
                loss_q = loss
                loss_pos = torch.tensor(0.0, device=device)
                loss_ori = torch.tensor(0.0, device=device)
            else:
                q_pred, pos_pred, ori_pred = model(x)
                loss_q = criterion(q_pred, y)
                pos_target = x[:, :3]
                ori_target = x[:, 3:7]
                loss_pos = criterion(pos_pred, pos_target)
                loss_ori = criterion(ori_pred, ori_target)
                loss = loss_q + aux_weight * (loss_pos + loss_ori)

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_loss_q += loss_q.item() * bs
            total_loss_pos += loss_pos.item() * bs
            total_loss_ori += loss_ori.item() * bs
            seen += bs

    total_loss /= max(seen, 1)
    total_loss_q /= max(seen, 1)
    total_loss_pos /= max(seen, 1)
    total_loss_ori /= max(seen, 1)
    return total_loss, total_loss_q, total_loss_pos, total_loss_ori



def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- load data tensor and shuffle rows ----
    full_tensor = load_dataset_tensor(args.csv)  # [N,14]
    N_total = full_tensor.shape[0]

    # Optional sub-sample (e.g., to match the report's stated 15M cap)
    if args.max_samples > 0 and args.max_samples < N_total:
        # Sub-sample reproducibly using the seed (so all DoFs at the same seed
        # see a consistent subset, but different seeds see different subsets)
        sub_perm = torch.Generator().manual_seed(args.seed)
        sub_idx = torch.randperm(N_total, generator=sub_perm)[: args.max_samples]
        full_tensor = full_tensor[sub_idx]
        N_total = full_tensor.shape[0]
        print(f"[INFO] Sub-sampled to max_samples={args.max_samples}: N={N_total}")

    # shuffle rows reproducibly (seeded by args.seed)
    perm = torch.Generator().manual_seed(args.seed + 1)
    full_tensor = full_tensor[torch.randperm(N_total, generator=perm)]

    # ---- build normalized dataset ----
    full_dataset = KinematicsTensorDataset(full_tensor, mode=args.mode)

    N = len(full_dataset)
    train_frac = args.train_frac
    val_frac = args.val_frac

    if train_frac <= 0 or val_frac < 0 or train_frac + val_frac >= 1.0:
        raise ValueError(
            f"Invalid splits: train_frac={train_frac}, val_frac={val_frac}. "
            f"Require train_frac > 0, val_frac >= 0 and train_frac + val_frac < 1."
        )

    N_train = int(train_frac * N)
    N_val = int(val_frac * N)
    N_test = N - N_train - N_val

    print(f"[INFO] Total samples: {N}")
    print(f"[INFO] Train: {N_train}, Val: {N_val}, Test: {N_test}")

    # Seed everything (parameter init, dataloader shuffle, split) for reproducibility + multi-seed runs
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    import random as _random
    _random.seed(args.seed)
    np.random.seed(args.seed)

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [N_train, N_val, N_test], generator=generator
    )

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
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )

    input_dim = full_dataset.inputs.shape[1]
    output_dim = full_dataset.targets.shape[1]

    # ---- build model ----
    if args.mode == "fk":
        model = ResidualMLP(
            input_dim,
            output_dim,
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks,
        ).to(device)
    else:
        model = IKResNetDualHead(
            in_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks,
            out_dim=output_dim,
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
    print(f"[INFO] Device: {device}, pin_memory={pin_mem}, num_workers={args.num_workers}")

    # ---- TensorBoard ----
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
        f"train_frac: {args.train_frac}\n"
        f"val_frac: {args.val_frac}\n"
        f"weight_decay: {args.weight_decay}\n"
        f"num_workers: {args.num_workers}\n"
        f"scheduler_patience: {args.scheduler_patience}\n"
        f"grad_clip: {args.grad_clip}\n"
        f"aux_loss_weight: {args.aux_loss_weight}\n"
    )
    writer.add_text("config", hparams_text, global_step=0)

    best_val_loss = float("inf")
    last_val_loss = float("nan")

    num_train_batches = len(train_loader)
    total_steps = args.epochs * max(num_train_batches, 1)

    global_bar = tqdm(
        total=total_steps,
        desc="Training (pose)",
        ncols=150,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_train_loss = 0.0
        running_train_loss_q = 0.0
        running_train_loss_pos = 0.0
        running_train_loss_ori = 0.0
        seen_train_samples = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()

            if args.mode == "fk":
                y_pred = model(x)
                loss = criterion(y_pred, y)
                loss_q = loss
                loss_pos = torch.tensor(0.0, device=device)
                loss_ori = torch.tensor(0.0, device=device)
            else:
                q_pred, pos_pred, ori_pred = model(x)
                loss_q = criterion(q_pred, y)
                pos_target = x[:, :3]
                ori_target = x[:, 3:7]
                loss_pos = criterion(pos_pred, pos_target)
                loss_ori = criterion(ori_pred, ori_target)
                loss = loss_q + args.aux_loss_weight * (loss_pos + loss_ori)

            loss.backward()
            if args.grad_clip is not None and args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            batch_size = x.size(0)
            running_train_loss += loss.item() * batch_size
            running_train_loss_q += loss_q.item() * batch_size
            running_train_loss_pos += loss_pos.item() * batch_size
            running_train_loss_ori += loss_ori.item() * batch_size
            seen_train_samples += batch_size

            train_mse_epoch = running_train_loss / max(seen_train_samples, 1)
            current_lr = optimizer.param_groups[0]["lr"]

            global_bar.update(1)
            global_bar.set_postfix(
                epoch=f"{epoch}/{args.epochs}",
                batch=f"{batch_idx+1}/{num_train_batches}",
                train_mse=f"{train_mse_epoch:.4f}",
                val_mse=f"{last_val_loss:.4f}" if not np.isnan(last_val_loss) else "N/A",
                lr=f"{current_lr:.2e}",
            )

        train_loss_epoch = running_train_loss / max(seen_train_samples, 1)
        train_loss_q_epoch = running_train_loss_q / max(seen_train_samples, 1)
        train_loss_pos_epoch = running_train_loss_pos / max(seen_train_samples, 1)
        train_loss_ori_epoch = running_train_loss_ori / max(seen_train_samples, 1)

        val_loss_epoch, val_loss_q_epoch, val_loss_pos_epoch, val_loss_ori_epoch = \
            evaluate_on_loader(model, val_loader, device, criterion, args.aux_loss_weight, args.mode)
        last_val_loss = val_loss_epoch

        scheduler.step(val_loss_epoch)

        writer.add_scalar("loss/train_total_mse", train_loss_epoch, epoch)
        writer.add_scalar("loss/val_total_mse", val_loss_epoch, epoch)
        writer.add_scalar("loss/train_q_mse", train_loss_q_epoch, epoch)
        writer.add_scalar("loss/val_q_mse", val_loss_q_epoch, epoch)
        # if args.mode == "ik":
        #     writer.add_scalar("loss/train_pos_aux_mse", train_loss_pos_epoch, epoch)
        #     writer.add_scalar("loss/train_ori_aux_mse", train_loss_ori_epoch, epoch)
        #     writer.add_scalar("loss/val_pos_aux_mse", val_loss_pos_epoch, epoch)
        #     writer.add_scalar("loss/val_ori_aux_mse", val_loss_ori_epoch, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        global_bar.set_postfix(
            epoch=f"{epoch}/{args.epochs}",
            batch=f"{num_train_batches}/{num_train_batches}",
            train_mse=f"{train_loss_epoch:.4f}",
            val_mse=f"{val_loss_epoch:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

        # Save histograms every 10 epochs
        if epoch % 10 == 0:
            for name, param in model.named_parameters():
                writer.add_histogram(f"params/{name}", param.data.cpu().numpy(), epoch)

        # Save best model
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
                    "hidden_dim": args.hidden_dim,
                    "num_blocks": args.num_blocks,
                    "aux_loss_weight": args.aux_loss_weight,
                },
                model_path,
            )
            tqdm.write(
                f"Saved best model to {model_path} (epoch={epoch}, val MSE={val_loss_epoch:.6f})"
            )

    global_bar.close()

    # ---- evaluate best model on test set ----
    best_model_path = os.path.join(log_dir, f"{args.mode}_pose_best.pt")
    if os.path.exists(best_model_path):
        print(f"[INFO] Loading best model from {best_model_path} for test evaluation")
        # PyTorch 2.6+ default weights_only=True, so override
        try:
            ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
        except TypeError:
            # for older PyTorch versions without weights_only argument
            ckpt = torch.load(best_model_path, map_location=device)

        model.load_state_dict(ckpt["model_state_dict"])
        test_loss, test_loss_q, test_loss_pos, test_loss_ori = evaluate_on_loader(
            model, test_loader, device, criterion, args.aux_loss_weight, args.mode
        )
        writer.add_scalar("loss/test_total_mse", test_loss, args.epochs)
        writer.add_scalar("loss/test_q_mse", test_loss_q, args.epochs)
        if args.mode == "ik":
            writer.add_scalar("loss/test_pos_aux_mse", test_loss_pos, args.epochs)
            writer.add_scalar("loss/test_ori_aux_mse", test_loss_ori, args.epochs)
        print(
            f"[INFO] Test MSE - total: {test_loss:.6f}, "
            f"q: {test_loss_q:.6f}, pos_aux: {test_loss_pos:.6f}, "
            f"ori_aux: {test_loss_ori:.6f}"
        )
    else:
        print(
            f"[WARN] Best model checkpoint {best_model_path} not found; "
            f"skipping test evaluation."
        )

    writer.close()
    print("[INFO] Training finished.")


if __name__ == "__main__":
    main()
