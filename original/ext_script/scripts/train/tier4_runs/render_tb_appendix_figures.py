#!/usr/bin/env python3
"""
Render TensorBoard scalar logs as publication-quality PNG + PDF figures
for the FYP appendix.

Three sets of figures are produced:

  Stage 1 (single-task FK training):
    - For each DoF (5/6/7): training loss + validation loss vs epoch
    - Combined comparison across DoFs

  Stage 2 (shared multitask training, seed=42):
    - Per-task training loss vs step (5/6/7 DoF on one panel)
    - Eval loss vs step (the 1-batch quick-eval used for best-checkpoint selection)
    - LR schedule (cosine + warmup) for context

  Stage 3 (per-DoF adaptation, headline runs):
    - Support-set loss vs adapt step (per-DoF)
    - Query-set position/orientation error vs adapt step (per-DoF)

All figures are saved to:
  report_resources/figures/appendix_tb_plots/<TS>/

Usage:
  python render_tb_appendix_figures.py
"""

from pathlib import Path
import csv
import re

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TIMESTAMP = "20260507_045433"
TRAIN_DIR = Path("/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train")

OUT_DIR = TRAIN_DIR / "report_resources" / "figures" / "appendix_tb_plots" / TIMESTAMP
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --------- Source event files ---------
EVENT_FILES = {
    # Stage 1 — newest single-task runs (these match the headline numbers)
    "stage1_5dof": TRAIN_DIR / "runs/single_task_fk_20260313_144312/5dof/tb/fk/events.out.tfevents.1773387806.rai4090.99736.0",
    "stage1_6dof": TRAIN_DIR / "runs/single_task_fk_20260313_144312/6dof/tb/fk/events.out.tfevents.1773394655.rai4090.115435.0",
    "stage1_7dof": TRAIN_DIR / "runs/single_task_fk_20260313_144312/7dof/fk/events.out.tfevents.1766291992.wish-ROG-Strix-G834JZ-G834JZ.276802.0",

    # Stage 2 — shared meta-kinematics (seed=42 best.pt)
    "stage2_seed42": TRAIN_DIR / "runs/multitask/seperate_weight/1/fk_iiwa_5_6_7/events.out.tfevents.1772718883.rai4090.84404.0",

    # Stage 3 — adaptation runs that produced the headline Table 5.1 numbers
    "stage3_5dof_adapt": TRAIN_DIR / "runs/adapt_fk_weighted_5_6_7_20260306_004556/tb/dof5/events.out.tfevents.*",
    "stage3_6dof_adapt": TRAIN_DIR / "runs/adapt_fk_weighted_5_6_7_20260306_014642/tb/dof6/events.out.tfevents.*",
    # 7-DoF: use the sweep run that produced the headline (lr1e-6_l21e-6_S100000_ow0.05)
    "stage3_7dof_adapt": TRAIN_DIR / "runs/sweep_fk_dof7_only_one20260313_015503/tb/lr1e-6_l21e-6_S100000_ow0.05/events.out.tfevents.*",
}


def resolve(path_pattern):
    if "*" in str(path_pattern):
        matches = sorted(Path(path_pattern).parent.glob(Path(path_pattern).name))
        return matches[-1] if matches else None
    p = Path(path_pattern)
    return p if p.exists() else None


def load_scalars(event_path, tags=None):
    """Returns {tag: np.array of shape (N, 2) [step, value]}."""
    ea = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    ea.Reload()
    avail = ea.Tags().get("scalars", [])
    out = {}
    for t in avail:
        if tags is not None and t not in tags:
            continue
        rows = ea.Scalars(t)
        if rows:
            arr = np.array([(s.step, s.value) for s in rows])
            out[t] = arr
    return out, avail


def smooth(y, alpha=0.6):
    if len(y) == 0: return y
    out = np.empty_like(y, dtype=float)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * out[i-1] + (1 - alpha) * y[i]
    return out


def save_fig(fig, basename):
    png_path = OUT_DIR / f"{basename}.png"
    pdf_path = OUT_DIR / f"{basename}.pdf"
    fig.tight_layout()
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {basename}.png + {basename}.pdf")


def main():
    print(f"Output directory: {OUT_DIR}")

    # =================================================================
    # STAGE 1 — Single-task training curves
    # =================================================================
    print("\n[Stage 1] Single-task FK training curves")
    stage1_data = {}
    for dof in (5, 6, 7):
        ev = resolve(EVENT_FILES[f"stage1_{dof}dof"])
        if ev is None:
            print(f"  DoF {dof}: event file missing"); continue
        scalars, avail = load_scalars(ev, tags={"loss/train_total_mse", "loss/val_total_mse"})
        if "loss/train_total_mse" in scalars and "loss/val_total_mse" in scalars:
            stage1_data[dof] = scalars
            print(f"  DoF {dof}: {len(scalars['loss/train_total_mse'])} train epochs, {len(scalars['loss/val_total_mse'])} val epochs")

    # Per-DoF plots
    for dof, sc in stage1_data.items():
        fig, ax = plt.subplots(figsize=(6.5, 4.4))
        tr = sc["loss/train_total_mse"]; va = sc["loss/val_total_mse"]
        ax.plot(tr[:,0], tr[:,1], color="#1f77b4", alpha=0.35, linewidth=0.8, label="train (raw)")
        ax.plot(tr[:,0], smooth(tr[:,1]), color="#1f77b4", linewidth=1.6, label="train (smoothed)")
        ax.plot(va[:,0], va[:,1], color="#d62728", linewidth=1.6, label="val")
        ax.set_xlabel("Training epoch"); ax.set_ylabel("Standardised MSE loss")
        ax.set_yscale("log")
        ax.set_title(f"Stage 1 — Single-task {dof}-DoF FK training")
        ax.legend(loc="upper right", frameon=True)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        save_fig(fig, f"stage1_singletask_{dof}dof")

    # Combined comparison
    if stage1_data:
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        colors = {5: "#1f77b4", 6: "#ff7f0e", 7: "#2ca02c"}
        for dof, sc in stage1_data.items():
            va = sc["loss/val_total_mse"]
            ax.plot(va[:,0], va[:,1], color=colors[dof], linewidth=1.6, label=f"{dof} DoF (val)")
            tr = sc["loss/train_total_mse"]
            ax.plot(tr[:,0], smooth(tr[:,1]), color=colors[dof], linewidth=0.9, linestyle="--", alpha=0.7, label=f"{dof} DoF (train, smoothed)")
        ax.set_xlabel("Training epoch"); ax.set_ylabel("Standardised MSE loss")
        ax.set_yscale("log"); ax.set_title("Stage 1 — Single-task FK training across DoF configurations")
        ax.legend(loc="upper right", frameon=True, ncol=2, fontsize=8)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        save_fig(fig, "stage1_singletask_all_dofs")

    # =================================================================
    # STAGE 2 — Multitask shared training (seed=42)
    # =================================================================
    print("\n[Stage 2] Shared meta-kinematics training")
    ev = resolve(EVENT_FILES["stage2_seed42"])
    if ev is not None:
        scalars, avail = load_scalars(
            ev,
            tags={"train/5dof_loss", "train/6dof_loss", "train/7dof_loss",
                  "train/lr", "eval/avg_loss_1batch_each_task"})
        # Per-task training loss (all on one figure)
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        colors = {5: "#1f77b4", 6: "#ff7f0e", 7: "#2ca02c"}
        for dof in (5, 6, 7):
            key = f"train/{dof}dof_loss"
            if key in scalars:
                arr = scalars[key]
                # Subsample for plotting (10000 points → 500)
                if len(arr) > 500:
                    idx = np.linspace(0, len(arr)-1, 500).astype(int); arr = arr[idx]
                ax.plot(arr[:,0], arr[:,1], color=colors[dof], alpha=0.25, linewidth=0.6)
                ax.plot(arr[:,0], smooth(arr[:,1], alpha=0.95), color=colors[dof], linewidth=1.6, label=f"{dof} DoF (smoothed)")
        ax.set_xlabel("Training step"); ax.set_ylabel("Per-task standardised MSE loss")
        ax.set_yscale("log"); ax.set_title("Stage 2 — Shared meta-kinematics training (seed=42, 300 000 steps)")
        ax.legend(loc="upper right", frameon=True)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        save_fig(fig, "stage2_multitask_per_task_loss")

        # Eval (avg across 3 tasks, used for best-checkpoint)
        if "eval/avg_loss_1batch_each_task" in scalars:
            arr = scalars["eval/avg_loss_1batch_each_task"]
            fig, ax = plt.subplots(figsize=(7.5, 4.4))
            ax.plot(arr[:,0], arr[:,1], color="#9467bd", linewidth=1.6, label="avg quick-eval loss across 3 tasks")
            best_idx = int(np.argmin(arr[:,1]))
            ax.scatter([arr[best_idx,0]], [arr[best_idx,1]], marker="*", s=160, color="#9467bd", edgecolors="black", linewidths=0.6, zorder=5, label=f"best @ step {int(arr[best_idx,0]):,}")
            ax.set_xlabel("Training step"); ax.set_ylabel("Average standardised MSE loss")
            ax.set_yscale("log"); ax.set_title("Stage 2 — Quick-eval loss used for best-checkpoint selection")
            ax.legend(loc="upper right", frameon=True)
            ax.grid(True, which="both", linestyle="--", alpha=0.3)
            save_fig(fig, "stage2_multitask_eval_loss")

        # LR schedule
        if "train/lr" in scalars:
            arr = scalars["train/lr"]
            fig, ax = plt.subplots(figsize=(7.5, 3.4))
            ax.plot(arr[:,0], arr[:,1], color="#17becf", linewidth=1.6)
            ax.set_xlabel("Training step"); ax.set_ylabel("Learning rate")
            ax.set_title("Stage 2 — Cosine LR schedule with linear warmup of 2 000 steps")
            ax.grid(True, linestyle="--", alpha=0.3)
            save_fig(fig, "stage2_multitask_lr_schedule")

    # =================================================================
    # STAGE 3 — Adaptation (per-DoF, headline runs)
    # =================================================================
    print("\n[Stage 3] Per-DoF adaptation curves (TB)")
    for dof in (5, 6, 7):
        ev = resolve(EVENT_FILES[f"stage3_{dof}dof_adapt"])
        if ev is None:
            print(f"  DoF {dof}: event file missing"); continue
        scalars, avail = load_scalars(
            ev,
            tags={"support/loss_pos", "support/loss_ori", "support/loss",
                  "query/pos_mae_m", "query/ori_deg", "query/score"})

        # Combined: support loss + query metrics
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
        ax0, ax1 = axes

        # support losses
        if "support/loss_pos" in scalars:
            arr = scalars["support/loss_pos"]
            if len(arr) > 800:
                idx = np.linspace(0, len(arr)-1, 800).astype(int); arr = arr[idx]
            ax0.plot(arr[:,0], arr[:,1], color="#1f77b4", linewidth=0.6, alpha=0.4)
            ax0.plot(arr[:,0], smooth(arr[:,1], 0.97), color="#1f77b4", linewidth=1.6, label="position MSE (m²)")
        if "support/loss_ori" in scalars:
            arr = scalars["support/loss_ori"]
            if len(arr) > 800:
                idx = np.linspace(0, len(arr)-1, 800).astype(int); arr = arr[idx]
            ax0.plot(arr[:,0], arr[:,1], color="#d62728", linewidth=0.6, alpha=0.4)
            ax0.plot(arr[:,0], smooth(arr[:,1], 0.97), color="#d62728", linewidth=1.6, label="orientation geodesic² (rad²)")
        ax0.set_xlabel("Adaptation step"); ax0.set_ylabel("Support-set loss component")
        ax0.set_yscale("log"); ax0.set_title(f"(a) Support-set losses — {dof} DoF")
        ax0.legend(loc="upper right", frameon=True)
        ax0.grid(True, which="both", linestyle="--", alpha=0.3)

        # query metrics
        if "query/pos_mae_m" in scalars:
            arr = scalars["query/pos_mae_m"]
            ax1.plot(arr[:,0], arr[:,1], color="#1f77b4", marker="o", markersize=3, linewidth=1.2, label="position MAE (m)")
        if "query/ori_deg" in scalars:
            arr = scalars["query/ori_deg"]
            ax2 = ax1.twinx()
            ax2.plot(arr[:,0], arr[:,1], color="#ff7f0e", marker="s", markersize=3, linewidth=1.2, label="orientation (deg)")
            ax2.set_ylabel("Orientation error (deg)", color="#ff7f0e")
            ax2.tick_params(axis="y", labelcolor="#ff7f0e")
        ax1.set_xlabel("Adaptation step"); ax1.set_ylabel("Position error (m)", color="#1f77b4")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")
        ax1.set_yscale("log"); ax1.set_title(f"(b) Query-set held-out metrics — {dof} DoF")
        ax1.grid(True, which="both", linestyle="--", alpha=0.3)
        save_fig(fig, f"stage3_adapt_{dof}dof_curves")

    print(f"\n[DONE] All appendix figures saved under: {OUT_DIR}")


if __name__ == "__main__":
    main()
