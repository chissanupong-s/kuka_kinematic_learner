#!/usr/bin/env python3
"""
Option 1 — Adaptation convergence curves from existing logs (FREE, no GPU).
Generated: 2026-05-07.

Pulls per-step [EVAL] lines from the existing 5/6/7-DoF adapt logs and renders
a 2-panel figure showing how fast per-DoF adaptation converges. The data is
already there; this script just visualises it.

Output: tier4_runs/option1_adaptation_curves_<TS>/fig_adaptation_curves_<TS>.png (+ .pdf)
"""
import re, sys, time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


TIMESTAMP = "20260507_045433"
TRAIN_DIR = Path("/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train")
OUT_DIR   = TRAIN_DIR / f"tier4_runs/option1_adaptation_curves_{TIMESTAMP}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Use the LARGEST log we have for each DoF — picked from the user's headline run
# (these are the same logs that produced the Table 5.1 adapted-row numbers).
LOG_PATHS = {
    5: TRAIN_DIR / "runs/Used_runs/Adaptation/adapt_5DOF_separate_weight/logs/adapt_dof5.log",
    6: TRAIN_DIR / "runs/Used_runs/Adaptation/adapt_6DOF_separate_weight/logs/adapt_dof6.log",
}
# 7-DoF: the headline configuration from the sweep csv corresponds to
# inner_lr=1e-6, l2_reg=1e-6, adapt_steps=100000, ori_weight=0.05
# whose BEST line is pos_mae_m=0.009946, ori_deg=1.7116 (rounds to the headline 0.0099 m / 1.71° in Table 5.1)
LOG_PATHS[7] = TRAIN_DIR / "runs/sweep_fk_dof7_only_one20260313_015503/logs/lr1e-6_l21e-6_S100000_ow0.05.log"

print("Using logs:")
for d, p in LOG_PATHS.items():
    print(f"  DoF {d}: {p}  (exists={p.exists()}, size={p.stat().st_size if p.exists() else 0} B)")


_RE_EVAL = re.compile(r"\[EVAL\]\s+step=(\d+)\s*\|\s*\{[^}]*'pos_mae_m':\s*([0-9.eE+-]+)[^}]*'pos_rmse_m':\s*([0-9.eE+-]+)[^}]*'ori_deg':\s*([0-9.eE+-]+)")

def parse_curve(log_path):
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="ignore")
    rows = []
    for m in _RE_EVAL.finditer(text):
        step = int(m.group(1))
        rows.append((step, float(m.group(2)), float(m.group(3)), float(m.group(4))))
    if not rows:
        return None
    rows.sort(key=lambda r: r[0])
    rows = list({r[0]: r for r in rows}.values())   # dedup by step
    rows.sort(key=lambda r: r[0])
    return np.array(rows)   # cols: step, pos_mae, pos_rmse, ori_deg


curves = {}
for dof, path in LOG_PATHS.items():
    arr = parse_curve(path)
    if arr is not None and len(arr) > 1:
        curves[dof] = arr
        print(f"  DoF {dof}: parsed {len(arr)} eval points (step range {int(arr[0,0])}..{int(arr[-1,0])})")

# Plot
colors  = {5: "#1f77b4", 6: "#ff7f0e", 7: "#2ca02c"}
markers = {5: "o", 6: "s", 7: "^"}

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
ax_pos, ax_ori = axes

for dof in (5, 6, 7):
    if dof not in curves:
        continue
    arr = curves[dof]
    steps = arr[:, 0]
    pos   = arr[:, 1]   # pos_mae_m
    ori   = arr[:, 3]   # ori_deg

    # Drop the step=0 huge spike for log-scale clarity if it dominates the y-axis
    # (the eval at step=0 is the shared model on this DoF, before adapt).
    ax_pos.plot(steps, pos, color=colors[dof], marker=markers[dof], markersize=4,
                linewidth=1.2, label=f"{dof} DoF", alpha=0.9)
    ax_ori.plot(steps, ori, color=colors[dof], marker=markers[dof], markersize=4,
                linewidth=1.2, label=f"{dof} DoF", alpha=0.9)

    # Mark the BEST step with a star
    # (BEST = step that minimises the scalarised score 1.0*pos_mae + 0.01*ori_deg)
    score = pos + 0.01 * ori
    best_idx = int(np.argmin(score))
    ax_pos.scatter([steps[best_idx]], [pos[best_idx]],
                   marker="*", s=140, color=colors[dof], edgecolors="black", linewidths=0.6, zorder=5)
    ax_ori.scatter([steps[best_idx]], [ori[best_idx]],
                   marker="*", s=140, color=colors[dof], edgecolors="black", linewidths=0.6, zorder=5)

ax_pos.set_xlabel("Adaptation step")
ax_pos.set_ylabel("Mean Euclidean position error (m)")
ax_pos.set_title("(a) Position error during per-DoF adaptation")
ax_pos.set_yscale("log")
ax_pos.set_xlim(left=0)
ax_pos.grid(True, which="both", linestyle="--", alpha=0.3)
ax_pos.legend(loc="upper right", frameon=True)

ax_ori.set_xlabel("Adaptation step")
ax_ori.set_ylabel("Mean orientation error (deg)")
ax_ori.set_title("(b) Orientation error during per-DoF adaptation")
ax_ori.set_yscale("log")
ax_ori.set_xlim(left=0)
ax_ori.grid(True, which="both", linestyle="--", alpha=0.3)
ax_ori.legend(loc="upper right", frameon=True)

fig.suptitle("Stage-3 adaptation convergence on the held-out query split for each DoF configuration",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])

out_png = OUT_DIR / f"fig_adaptation_curves_{TIMESTAMP}.png"
out_pdf = OUT_DIR / f"fig_adaptation_curves_{TIMESTAMP}.pdf"
fig.savefig(out_png, dpi=200, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")

# Also save the parsed curves as csvs so they can be re-plotted any time
import csv
for dof, arr in curves.items():
    csv_path = OUT_DIR / f"curve_dof{dof}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "pos_mae_m", "pos_rmse_m", "ori_deg"])
        for row in arr:
            w.writerow([int(row[0])] + [float(x) for x in row[1:]])
print(f"Saved CSVs: {[str(p.name) for p in OUT_DIR.glob('curve_dof*.csv')]}")

# Also print convergence diagnostics — when each DoF first reached within 5% of its best
print("\n--- CONVERGENCE DIAGNOSTICS (steps to within 5% of best score) ---")
for dof in (5, 6, 7):
    if dof not in curves:
        continue
    arr = curves[dof]
    steps = arr[:, 0]; pos = arr[:, 1]; ori = arr[:, 3]
    score = pos + 0.01 * ori
    best = score.min()
    threshold = best * 1.05
    within = np.where(score <= threshold)[0]
    if len(within) > 0:
        first_within = int(steps[within[0]])
        best_step = int(steps[int(np.argmin(score))])
        print(f"  DoF {dof}: best step = {best_step}, first within 5% of best = step {first_within}; "
              f"final pos_mae = {pos[-1]:.4f} m, final ori = {ori[-1]:.4f}°")
