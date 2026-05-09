#!/usr/bin/env python3
"""
Render the multi-seed comparison figure: per-DoF Adapted (best) values across
3 seeds with mean ± std error bars. Highlights the variance pattern uncovered
by the multi-seed sweep — small for 5/6 DoF, larger for 7 DoF.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


TIMESTAMP = "20260507_045433"
TRAIN_DIR = Path("/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train")
OUT_DIR   = TRAIN_DIR / f"report_resources/figures/appendix_tb_plots/{TIMESTAMP}"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Hard-coded from the multi-seed BEST lines (verified above)
DATA = {
    5: {
        "seed42": (0.006214, 0.8003),
        "seed1":  (0.006202, 0.8092),
        "seed2":  (0.006244, 0.8120),
    },
    6: {
        "seed42": (0.008876, 1.3294),
        "seed1":  (0.008927, 1.3316),
        "seed2":  (0.008891, 1.3277),
    },
    7: {
        "seed42": (0.009946, 1.7116),
        "seed1":  (0.013317, 2.5679),
        "seed2":  (0.013301, 2.5493),
    },
}

# Single-task baselines (for reference line)
SINGLE_TASK = {5: (0.0093, 1.2039), 6: (0.0110, 1.7136), 7: (0.0101, 2.0853)}


fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
ax_pos, ax_ori = axes

dofs = [5, 6, 7]
colours = {"seed42": "#1f77b4", "seed1": "#ff7f0e", "seed2": "#2ca02c"}
markers = {"seed42": "o", "seed1": "s", "seed2": "^"}
x_offsets = {"seed42": -0.1, "seed1": 0.0, "seed2": 0.1}

# Plot per-seed scatter + mean±std
for seed, c in colours.items():
    pos = [DATA[d][seed][0] for d in dofs]
    ori = [DATA[d][seed][1] for d in dofs]
    ax_pos.scatter([d + x_offsets[seed] for d in dofs], pos,
                   color=c, marker=markers[seed], s=70, edgecolors="black", linewidths=0.6,
                   label=f"{seed}", zorder=4)
    ax_ori.scatter([d + x_offsets[seed] for d in dofs], ori,
                   color=c, marker=markers[seed], s=70, edgecolors="black", linewidths=0.6,
                   label=f"{seed}", zorder=4)

# Mean ± std bars (n=3)
mean_pos = []; std_pos = []; mean_ori = []; std_ori = []
for d in dofs:
    p = [DATA[d][s][0] for s in ("seed42","seed1","seed2")]
    o = [DATA[d][s][1] for s in ("seed42","seed1","seed2")]
    mean_pos.append(np.mean(p)); std_pos.append(np.std(p, ddof=1))
    mean_ori.append(np.mean(o)); std_ori.append(np.std(o, ddof=1))

ax_pos.errorbar(dofs, mean_pos, yerr=std_pos, fmt="D", color="#9467bd", markersize=8,
                capsize=6, elinewidth=1.6, label="mean ± std (n=3)", zorder=5)
ax_ori.errorbar(dofs, mean_ori, yerr=std_ori, fmt="D", color="#9467bd", markersize=8,
                capsize=6, elinewidth=1.6, label="mean ± std (n=3)", zorder=5)

# Single-task baseline reference
for d in dofs:
    ax_pos.axhline(SINGLE_TASK[d][0], xmin=(d-4.5-0.6)/4.0, xmax=(d-4.5+0.6)/4.0,
                   color="#d62728", linestyle="--", linewidth=1.2, alpha=0.7)
    ax_ori.axhline(SINGLE_TASK[d][1], xmin=(d-4.5-0.6)/4.0, xmax=(d-4.5+0.6)/4.0,
                   color="#d62728", linestyle="--", linewidth=1.2, alpha=0.7)
# Add a single legend entry for the reference line
ax_pos.plot([], [], color="#d62728", linestyle="--", linewidth=1.2, label="single-task baseline")
ax_ori.plot([], [], color="#d62728", linestyle="--", linewidth=1.2, label="single-task baseline")

ax_pos.set_xticks(dofs); ax_pos.set_xlabel("DoF configuration")
ax_pos.set_ylabel("Mean Euclidean position error (m)")
ax_pos.set_title("(a) Adapted (best) position error across 3 seeds")
ax_pos.legend(loc="upper left", frameon=True, fontsize=9)
ax_pos.grid(True, linestyle="--", alpha=0.3)

ax_ori.set_xticks(dofs); ax_ori.set_xlabel("DoF configuration")
ax_ori.set_ylabel("Mean orientation error (deg)")
ax_ori.set_title("(b) Adapted (best) orientation error across 3 seeds")
ax_ori.legend(loc="upper left", frameon=True, fontsize=9)
ax_ori.grid(True, linestyle="--", alpha=0.3)

fig.suptitle("Multi-seed reproducibility of the per-DoF adaptation result", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])

png = OUT_DIR / f"fig_multiseed_reproducibility_{TIMESTAMP}.png"
pdf = OUT_DIR / f"fig_multiseed_reproducibility_{TIMESTAMP}.pdf"
fig.savefig(png, dpi=200, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
print(f"Saved: {png}\nSaved: {pdf}")
