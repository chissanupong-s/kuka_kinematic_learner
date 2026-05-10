#!/usr/bin/env python3
"""
Render Figure 5.1 in 'Option C' style:
  • All 3 DoFs in a single panel (poster-like layout)
  • Both train and val curves shown (transparency-honest)
  • Best-val epoch marked with vertical line + 'test evaluated here' annotation
  • Log y-axis (highlights the gap and the early-stopping logic)
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt


TS = "20260507_045433"
TRAIN_DIR = Path("/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train")
TB_BASE   = TRAIN_DIR / f"report_resources/results/tb_extracted/{TS}"
OUT_DIR   = TRAIN_DIR / f"report_resources/figures/appendix_tb_plots/{TS}"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(path):
    if not path.exists(): return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return np.array([(int(r["step"]), float(r["value"])) for r in rows])


def smooth(y, alpha=0.7):
    if len(y) == 0: return y
    out = np.empty_like(y, dtype=float)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * out[i-1] + (1 - alpha) * y[i]
    return out


# Per-DoF data
data = {}
for dof in (5, 6, 7):
    run_dir = TB_BASE / f"stage1_{dof}dof_singletask"
    train = load_csv(run_dir / "loss__train_total_mse.csv")
    val   = load_csv(run_dir / "loss__val_total_mse.csv")
    if train is None or val is None:
        print(f"missing for DoF {dof}")
        continue
    # Find best-val step
    best_idx = int(np.argmin(val[:, 1]))
    best_step = int(val[best_idx, 0])
    best_val = float(val[best_idx, 1])
    data[dof] = {"train": train, "val": val, "best_step": best_step, "best_val": best_val}
    print(f"DoF {dof}: train epochs {len(train)}, val epochs {len(val)}, best-val @ epoch {best_step} = {best_val:.6g}")


# Plot
fig, ax = plt.subplots(figsize=(8.5, 5.5))
colors = {5: "#1f77b4", 6: "#d62687", 7: "#2ca02c"}     # poster-ish
markers = {5: "o", 6: "s", 7: "^"}

for dof in (5, 6, 7):
    if dof not in data: continue
    tr = data[dof]["train"]
    va = data[dof]["val"]
    c = colors[dof]
    # Train: faded raw underneath, smoothed solid on top
    ax.plot(tr[:, 0], tr[:, 1], color=c, alpha=0.18, linewidth=0.7)
    ax.plot(tr[:, 0], smooth(tr[:, 1], 0.85), color=c, linewidth=1.6,
            label=f"{dof} DoF train")
    # Val: dashed overlay
    ax.plot(va[:, 0], va[:, 1], color=c, linewidth=1.4, linestyle="--", alpha=0.85,
            label=f"{dof} DoF val")
    # Best-val marker: vertical line at best-val epoch
    bs = data[dof]["best_step"]; bv = data[dof]["best_val"]
    ax.axvline(bs, color=c, linewidth=0.9, linestyle=":", alpha=0.55)
    ax.scatter([bs], [bv], color=c, marker=markers[dof], s=80,
               edgecolors="black", linewidths=0.7, zorder=6)

# Single annotation arrow for best-val (use the 7-DoF position as anchor since it's furthest right)
if 7 in data:
    bs7 = data[7]["best_step"]
    bv7 = data[7]["best_val"]
    ax.annotate(
        "test evaluated here\n(val-min checkpoint, per DoF)",
        xy=(bs7, bv7), xytext=(bs7 - 80, bv7 * 0.35),
        fontsize=9, ha="center",
        arrowprops=dict(arrowstyle="->", color="#444", lw=1.0, connectionstyle="arc3,rad=0.2")
    )

ax.set_yscale("log")
ax.set_xlabel("Training epoch")
ax.set_ylabel("Standardised MSE loss  (log scale)")
ax.set_title("Figure 5.1 — Single-task FK training curves for the 5, 6 and 7 DoF configurations")
ax.grid(True, which="both", linestyle="--", alpha=0.3)
ax.legend(loc="lower left", frameon=True, fontsize=9, ncol=3)

caption_note = (
    "Solid lines = training loss (smoothed; faded raw underneath); dashed lines = validation loss.\n"
    "Vertical dotted lines and markers indicate the per-DoF validation-loss minimum, the checkpoint\n"
    "saved by the training script and used for the test-set evaluation reported in Table 5.1.\n"
    "The widening train–val gap after the val minimum reflects the optimiser fitting training-set\n"
    "detail under a small learning rate, but this over-fit tail is not deployed."
)
fig.text(0.05, -0.01, caption_note, fontsize=8, color="#444", ha="left", va="top")

fig.tight_layout()
out_png = OUT_DIR / "fig_5_1_singletask_option_c.png"
out_pdf = OUT_DIR / "fig_5_1_singletask_option_c.pdf"
fig.savefig(out_png, dpi=200, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {out_png}\nSaved {out_pdf}")
