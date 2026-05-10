#!/usr/bin/env python3
# bar charts for fig 5.3 5.4 5.5
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = Path("/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/report_resources/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Position error (mm) — values from latest authoritative Table 5.1
POS = {
    "Single-task":                       [9.308, 11.052, 10.137],
    "Shared meta-kinematics":            [6.821,  9.235, 10.913],
    "Adapted (best, n=3 mean)":          [6.203,  8.900,  9.901],
}
# Orientation error (deg)
ORI = {
    "Single-task":                       [1.204, 1.714, 2.085],
    "Shared meta-kinematics":            [0.910, 1.369, 1.995],
    "Adapted (best, n=3 mean)":          [0.809, 1.329, 1.722],
}
# Wall-clock training time (hours), shown on log y-axis
TIME = {
    "Single-task":                       [1.873, 4.973, 22.120],
    "Shared meta-kinematics":            [1.159, 1.159,  1.159],
    "Adapted (best, n=3 mean)":          [0.365, 0.363,  0.111],
}

DOF_LABELS = ["5 DoF", "6 DoF", "7 DoF"]
COLORS = ["#4D7FBF", "#E89A4B", "#5BB668"]


def render(data, ylabel, fmt, outfile, log_y=False, title_pad=8):
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=200)

    n_groups = len(DOF_LABELS)
    n_bars = len(data)
    bar_w = 0.27
    x = np.arange(n_groups)

    all_vals = [v for vals in data.values() for v in vals]
    vmax = max(all_vals)
    vmin = min(all_vals)

    for i, (label, vals) in enumerate(data.items()):
        offsets = x + (i - (n_bars - 1) / 2) * bar_w
        bars = ax.bar(offsets, vals, bar_w, label=label, color=COLORS[i],
                       edgecolor="#222", linewidth=0.6)
        for off, v in zip(offsets, vals):
            if log_y:
                # In log scale, offset labels by a multiplicative factor
                y_label = v * 1.10
            else:
                y_label = v + vmax * 0.015
            txt = fmt.format(v) if not log_y else f"{v:.2f} h" if v >= 1 else f"{v:.3f} h"
            ax.text(off, y_label, txt,
                     ha="center", va="bottom", fontsize=8.5, color="#222")

    ax.set_xticks(x)
    ax.set_xticklabels(DOF_LABELS, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)

    if log_y:
        ax.set_yscale("log")
        # Extra headroom in log space for legend + labels
        ax.set_ylim(max(0.05, vmin * 0.7), vmax * 2.5)
    else:
        # Extra headroom in linear space so the legend doesn't overlap bars/labels
        ax.set_ylim(0, vmax * 1.32)

    ax.legend(fontsize=9, loc="upper left", frameon=True,
              framealpha=0.95, edgecolor="#666",
              borderpad=0.6, handlelength=1.5)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    plt.tight_layout()
    # Use pad_inches=0.4 to ensure top of legend is not cropped when saved
    fig.savefig(outfile, dpi=200, bbox_inches="tight", pad_inches=0.4)
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)
    print(f"  wrote {outfile.name}")


print("=== Figure 5.3 — Position error (mm) ===")
render(POS,
       ylabel="Mean Euclidean position error (mm)",
       fmt="{:.3f}",
       outfile=OUT_DIR / "fig_5_3_position_error.png")

print("\n=== Figure 5.4 — Orientation error (°) ===")
render(ORI,
       ylabel="Mean orientation error (°)",
       fmt="{:.3f}",
       outfile=OUT_DIR / "fig_5_4_orientation_error.png")

print("\n=== Figure 5.5 — Wall-clock training time (hours, log scale) ===")
render(TIME,
       ylabel="Wall-clock training time (hours, log scale)",
       fmt="{:.3f} h",
       outfile=OUT_DIR / "fig_5_5_training_time.png",
       log_y=True)

print("\nDone.")
