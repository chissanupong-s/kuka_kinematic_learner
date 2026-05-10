#!/usr/bin/env python3
"""
Generalisation-focused single-task FK trainer.

Adapted from train_kinematics_nn_pol_pt_2.py — original is left untouched.
The goal is a 7-DoF model that does NOT overfit to a single angle range.

Differences from the baseline trainer:
  1. --csv accepts a comma-separated list of dataset paths; they are loaded
     and concatenated. Combining multiple joint-angle ranges (e.g. 5deg +
     10deg + 15deg + 20deg) gives the network a much broader joint-config
     distribution to learn the FK transform from, instead of memorising one
     range's specific samples.
  2. --dropout (default 0.2). The original ResBlock had a Dropout layer
     defined but never applied in forward() — effectively zero. This
     trainer uses a fixed ResBlock that actually applies dropout.
  3. --early_stopping_patience N (default 20). Stops training if val MSE
     hasn't improved by --early_stopping_min_delta for N consecutive
     epochs. Saves wall-clock and prevents the late-epoch overfitting tail.
  4. --ood_csv <comma-sep>. Optional held-out distribution(s) (e.g. an
     angle range NOT in --csv). Logged each epoch as ood_val_total_mse so
     you can see if generalisation is improving / collapsing without it
     leaking into early-stopping or best-checkpoint selection.
  5. --weight_decay default raised from 1e-5 to 1e-4.

Output is the same .pt format as the baseline so eval_model_single_task.py
works on the resulting checkpoints unchanged.
"""

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
    p = argparse.ArgumentParser(description="Single-task FK trainer with generalisation knobs.")
    p.add_argument("--csv", type=str, required=True,
                   help="Comma-separated dataset paths. Each can be a .csv, .pt, .bin, or directory of shards.")
    p.add_argument("--mode", type=str, choices=["fk", "ik"], default="fk")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--num_blocks", type=int, default=4)
    p.add_argument("--train_frac", type=float, default=0.7)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out_dir", type=str, default="models")
    p.add_argument("--log_dir", type=str, default="runs/kinematics_pose_generalize")
    p.add_argument("--weight_decay", type=float, default=1e-4,
                   help="L2 weight decay for Adam (default 1e-4, was 1e-5 in baseline trainer).")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--scheduler_patience", type=int, default=10)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--aux_loss_weight", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_samples", type=int, default=0,
                   help="Cap total combined dataset to N rows (0 = use all).")
    # ----- generalisation knobs -----
    p.add_argument("--dropout", type=float, default=0.2,
                   help="Dropout probability inside each residual block (default 0.2).")
    p.add_argument("--early_stopping_patience", type=int, default=20,
                   help="Stop after N epochs with no val improvement. 0 disables.")
    p.add_argument("--early_stopping_min_delta", type=float, default=1e-5,
                   help="Minimum val MSE improvement to reset the patience counter.")
    p.add_argument("--ood_csv", type=str, default="",
                   help="Optional comma-separated held-out distributions (logged only).")
    p.add_argument("--ood_max_samples", type=int, default=200000,
                   help="Cap on OOD val set size (200k by default, kept small for speed).")
    return p.parse_args()


def _load_one(path: str) -> torch.Tensor:
    """Load a single dataset path and return [N,14] float tensor."""
    cols = JOINT_COLS + POSE_COLS
    print(f"[INFO] Loading {path}")
    if os.path.isdir(path):
        shard_files = sorted(
            os.path.join(path, f) for f in os.listdir(path)
            if f.endswith(".pt") or f.endswith(".bin")
        )
        if not shard_files:
            raise ValueError(f"No .pt/.bin shards in directory: {path}")
        arrs = []
        for sp in shard_files:
            obj = torch.load(sp, map_location="cpu")
            t = obj if isinstance(obj, torch.Tensor) else obj["data"]
            arrs.append(t.numpy())
        arr_all = np.concatenate(arrs, axis=0).astype(np.float32)
        return torch.from_numpy(arr_all)
    if path.endswith(".pt") or path.endswith(".bin"):
        obj = torch.load(path, map_location="cpu")
        t = obj if isinstance(obj, torch.Tensor) else obj["data"]
        if t.ndim != 2 or t.shape[1] != len(cols):
            raise ValueError(f"Tensor in '{path}' has shape {tuple(t.shape)}, expected [N,{len(cols)}].")
        return t.float()
    import pandas as pd
    df = pd.read_csv(path)
    arr = df[cols].values.astype(np.float32)
    return torch.from_numpy(arr)


def load_dataset_tensor_multi(paths_csv: str) -> torch.Tensor:
    """Comma-separated paths → concatenated [N,14] tensor."""
    paths = [p.strip() for p in paths_csv.split(",") if p.strip()]
    if not paths:
        raise ValueError("--csv contained no paths after splitting on commas.")
    if len(paths) == 1:
        return _load_one(paths[0])
    parts = [_load_one(p) for p in paths]
    sizes = [t.shape[0] for t in parts]
    print(f"[INFO] Concatenating {len(parts)} datasets, sizes={sizes}, total={sum(sizes):,} rows")
    return torch.cat(parts, dim=0)


class KinematicsTensorDataset(Dataset):
    def __init__(self, full_tensor, mode, input_mean=None, input_std=None,
                 target_mean=None, target_std=None):
        super().__init__()
        if full_tensor.ndim != 2 or full_tensor.shape[1] != 14:
            raise ValueError(f"full_tensor must be [N,14], got {tuple(full_tensor.shape)}")
        self.mode = mode
        if mode == "fk":
            X = full_tensor[:, 0:7]
            Y = full_tensor[:, 7:14]
        else:
            X = full_tensor[:, 7:14]
            Y = full_tensor[:, 0:7]
        X = X.cpu().numpy().astype(np.float32)
        Y = Y.cpu().numpy().astype(np.float32)
        if input_mean is None:
            input_mean = X.mean(axis=0, keepdims=True)
            input_std = X.std(axis=0, keepdims=True) + 1e-8
        if target_mean is None:
            target_mean = Y.mean(axis=0, keepdims=True)
            target_std = Y.std(axis=0, keepdims=True) + 1e-8
        self.inputs = (X - input_mean) / input_std
        self.targets = (Y - target_mean) / target_std
        self.input_mean = input_mean
        self.input_std = input_std
        self.target_mean = target_mean
        self.target_std = target_std

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.inputs[idx]).float(), torch.from_numpy(self.targets[idx]).float()


# -- Fixed ResBlock that actually applies dropout (the baseline ResBlock
# defined a Dropout module but never used it in forward).
class ResBlock(nn.Module):
    def __init__(self, dim: int, p_drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.drop(h)
        h = self.fc2(h)
        h = self.drop(h)
        return self.act(h + x)


class ResidualMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=512, num_blocks=4, dropout=0.0):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim, p_drop=dropout) for _ in range(num_blocks)])
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.act(self.fc_in(x))
        for b in self.blocks:
            h = b(h)
        return self.fc_out(h)


class IKResNetDualHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, num_blocks=4, out_dim=7, dropout=0.0):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim, p_drop=dropout) for _ in range(num_blocks)])
        self.fc_joint = nn.Linear(hidden_dim, out_dim)
        self.fc_pos = nn.Linear(hidden_dim, 3)
        self.fc_ori = nn.Linear(hidden_dim, 4)
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.act(self.fc_in(x))
        for b in self.blocks:
            h = b(h)
        return self.fc_joint(h), self.fc_pos(h), self.fc_ori(h)


def evaluate_on_loader(model, loader, device, criterion, aux_weight, mode):
    model.eval()
    total_loss = total_q = total_pos = total_ori = seen = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            if mode == "fk":
                yh = model(x)
                loss = criterion(yh, y); lq = loss
                lp = torch.tensor(0.0, device=device); lo = torch.tensor(0.0, device=device)
            else:
                qp, pp, op = model(x)
                lq = criterion(qp, y)
                lp = criterion(pp, x[:, :3]); lo = criterion(op, x[:, 3:7])
                loss = lq + aux_weight * (lp + lo)
            bs = x.size(0)
            total_loss += loss.item()*bs; total_q += lq.item()*bs
            total_pos += lp.item()*bs; total_ori += lo.item()*bs
            seen += bs
    s = max(seen, 1)
    return total_loss/s, total_q/s, total_pos/s, total_ori/s


def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- load + optional cap + shuffle the combined training data ----
    full_tensor = load_dataset_tensor_multi(args.csv)
    N_total = full_tensor.shape[0]
    if args.max_samples > 0 and args.max_samples < N_total:
        sub_perm = torch.Generator().manual_seed(args.seed)
        sub_idx = torch.randperm(N_total, generator=sub_perm)[: args.max_samples]
        full_tensor = full_tensor[sub_idx]
        N_total = full_tensor.shape[0]
        print(f"[INFO] Capped to {N_total:,} samples (seed={args.seed})")
    perm = torch.Generator().manual_seed(args.seed + 1)
    full_tensor = full_tensor[torch.randperm(N_total, generator=perm)]

    full_dataset = KinematicsTensorDataset(full_tensor, mode=args.mode)
    N = len(full_dataset)
    if args.train_frac <= 0 or args.val_frac < 0 or args.train_frac + args.val_frac >= 1.0:
        raise ValueError(f"Invalid splits: train={args.train_frac}, val={args.val_frac}")
    N_train = int(args.train_frac * N); N_val = int(args.val_frac * N); N_test = N - N_train - N_val
    print(f"[INFO] Train: {N_train:,}, Val: {N_val:,}, Test: {N_test:,}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    import random as _random
    _random.seed(args.seed); np.random.seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds, test_ds = random_split(full_dataset, [N_train, N_val, N_test], generator=gen)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin)

    # ---- optional OOD validation set(s) ----
    ood_loader = None
    if args.ood_csv.strip():
        ood_tensor = load_dataset_tensor_multi(args.ood_csv)
        if args.ood_max_samples > 0 and args.ood_max_samples < ood_tensor.shape[0]:
            ood_perm = torch.Generator().manual_seed(args.seed + 7)
            ood_idx = torch.randperm(ood_tensor.shape[0], generator=ood_perm)[: args.ood_max_samples]
            ood_tensor = ood_tensor[ood_idx]
        # IMPORTANT: normalise using the TRAIN dataset's stats so the OOD val measures
        # generalisation under the same input/target standardisation, not its own.
        ood_dataset = KinematicsTensorDataset(
            ood_tensor, mode=args.mode,
            input_mean=full_dataset.input_mean, input_std=full_dataset.input_std,
            target_mean=full_dataset.target_mean, target_std=full_dataset.target_std,
        )
        ood_loader = DataLoader(ood_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=pin)
        print(f"[INFO] OOD val set: {len(ood_dataset):,} rows")

    in_dim = full_dataset.inputs.shape[1]; out_dim = full_dataset.targets.shape[1]
    if args.mode == "fk":
        model = ResidualMLP(in_dim, out_dim, hidden_dim=args.hidden_dim,
                            num_blocks=args.num_blocks, dropout=args.dropout).to(device)
    else:
        model = IKResNetDualHead(in_dim, hidden_dim=args.hidden_dim, num_blocks=args.num_blocks,
                                 out_dim=out_dim, dropout=args.dropout).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model: hidden={args.hidden_dim}, blocks={args.num_blocks}, dropout={args.dropout}, params={n_params:,}")
    print(f"[INFO] Optim:  lr={args.lr}, weight_decay={args.weight_decay}, early_stop_patience={args.early_stopping_patience}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                                           patience=args.scheduler_patience)

    log_dir = os.path.join(args.log_dir, args.mode)
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text("config", "\n".join(f"{k}: {v}" for k, v in sorted(vars(args).items())), 0)

    best_val = float("inf"); last_val = float("nan")
    patience_counter = 0
    nb = max(len(train_loader), 1)
    bar = tqdm(total=args.epochs * nb, desc="Training (generalize)", ncols=150)
    stopped_epoch = args.epochs

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = run_q = run_pos = run_ori = seen = 0
        for bi, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            if args.mode == "fk":
                yh = model(x); loss = criterion(yh, y); lq = loss
                lp = torch.tensor(0.0, device=device); lo = torch.tensor(0.0, device=device)
            else:
                qp, pp, op = model(x)
                lq = criterion(qp, y); lp = criterion(pp, x[:, :3]); lo = criterion(op, x[:, 3:7])
                loss = lq + args.aux_loss_weight*(lp + lo)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            bs = x.size(0)
            run_loss += loss.item()*bs; run_q += lq.item()*bs; run_pos += lp.item()*bs; run_ori += lo.item()*bs
            seen += bs
            bar.update(1)
            bar.set_postfix(epoch=f"{epoch}/{args.epochs}", batch=f"{bi+1}/{nb}",
                            train_mse=f"{run_loss/max(seen,1):.4f}",
                            val_mse=f"{last_val:.4f}" if not np.isnan(last_val) else "N/A",
                            lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        train_loss = run_loss/max(seen, 1)
        val_loss, val_q, val_pos, val_ori = evaluate_on_loader(model, val_loader, device, criterion, args.aux_loss_weight, args.mode)
        last_val = val_loss
        scheduler.step(val_loss)

        writer.add_scalar("loss/train_total_mse", train_loss, epoch)
        writer.add_scalar("loss/val_total_mse", val_loss, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        if ood_loader is not None:
            ood_loss, _, _, _ = evaluate_on_loader(model, ood_loader, device, criterion, args.aux_loss_weight, args.mode)
            writer.add_scalar("loss/ood_val_total_mse", ood_loss, epoch)
            tqdm.write(f"  [epoch {epoch}] train={train_loss:.5f}  val={val_loss:.5f}  ood_val={ood_loss:.5f}")

        improved = val_loss < best_val - args.early_stopping_min_delta
        if improved:
            best_val = val_loss
            patience_counter = 0
            os.makedirs(args.out_dir, exist_ok=True)
            ckpt_path = os.path.join(log_dir, f"{args.mode}_pose_best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_mean": full_dataset.input_mean, "input_std": full_dataset.input_std,
                "target_mean": full_dataset.target_mean, "target_std": full_dataset.target_std,
                "mode": args.mode, "hidden_dim": args.hidden_dim, "num_blocks": args.num_blocks,
                "aux_loss_weight": args.aux_loss_weight,
            }, ckpt_path)
            tqdm.write(f"  Saved best at epoch={epoch}, val_mse={val_loss:.6f}")
        else:
            patience_counter += 1
            if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
                tqdm.write(f"  [early-stop] no improvement for {args.early_stopping_patience} epochs (best val={best_val:.6f}). Stopping at epoch {epoch}.")
                stopped_epoch = epoch
                break

    bar.close()

    # ---- final test evaluation on the in-distribution test split ----
    best_path = os.path.join(log_dir, f"{args.mode}_pose_best.pt")
    if os.path.exists(best_path):
        try:
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        test_loss, test_q, test_pos, test_ori = evaluate_on_loader(model, test_loader, device, criterion, args.aux_loss_weight, args.mode)
        writer.add_scalar("loss/test_total_mse", test_loss, stopped_epoch)
        print(f"[INFO] Test MSE - total: {test_loss:.6f}, q: {test_q:.6f}")
    writer.close()
    print(f"[INFO] Training finished. Best val MSE: {best_val:.6f} (stopped at epoch {stopped_epoch})")


if __name__ == "__main__":
    main()
