#!/usr/bin/env python3
"""
Render Figures 5.3 (position error) and 5.4 (orientation error) for the FYP
report. Uses the final n=3 numbers from expB + expH + expK on part000 data.

Per user decision: no error bars on the Adapted bars (the 5/6-DoF stds are
~0.02-0.03 mm and 7-DoF is 0.014 mm — all invisible at this scale anyway, so
removing the single visible bar that was on the old 7-DoF Adapted gives a
cleaner figure. Tables 5.1 and Appendix E.7 still carry the n=3 mean ± std.)
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = Path("/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/report_resources/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- final numbers from v2.docx Table 5.1 ---
# Position error (mm)
POS = {
    "Single-task":             [9.3,    11.0,   10.1],
    "Shared meta-kinematics":  [6.8,    9.2,    10.9],
    "Adapted (best, n=3 mean)":[6.203,  8.900,  9.901],
}
# Orientation error (deg)
ORI = {
    "Single-task":             [1.2039, 1.7136, 2.0853],
    "Shared meta-kinematics":  [0.9096, 1.3689, 1.9953],
    "Adapted (best, n=3 mean)":[0.809,  1.329,  1.722],
}
DOF_LABELS = ["5 DoF", "6 DoF", "7 DoF"]
COLORS = ["#4D7FBF", "#E89A4B", "#5BB668"]    # blue, orange, green (close to the old style)


def render(data: dict, ylabel: str, fmt: str, outfile: Path, title: str = ""):
    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=200)

    n_groups = len(DOF_LABELS)
    n_bars = len(data)
    bar_w = 0.27
    x = np.arange(n_groups)

    for i, (label, vals) in enumerate(data.items()):
        offsets = x + (i - (n_bars - 1) / 2) * bar_w
        bars = ax.bar(offsets, vals, bar_w, label=label, color=COLORS[i],
                       edgecolor="#222", linewidth=0.6)
        # value labels on top of each bar (3 decimal places)
        for off, v in zip(offsets, vals):
            ax.text(off, v + max(vals) * 0.012, fmt.format(v),
                     ha="center", va="bottom", fontsize=8.5, color="#222")

    ax.set_xticks(x)
    ax.set_xticklabels(DOF_LABELS, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(0, max(max(v) for v in data.values()) * 1.15)
    ax.legend(fontsize=9, loc="upper left", frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if title:
        ax.set_title(title, fontsize=11, pad=8)
    plt.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outfile}  +  {outfile.with_suffix('.pdf')}")


print("=== Figure 5.3 — Position error comparison ===")
render(POS,
       ylabel="Mean Euclidean position error (mm)",
       fmt="{:.3f}",
       outfile=OUT_DIR / "fig_5_3_position_error.png")

print("\n=== Figure 5.4 — Orientation error comparison ===")
render(ORI,
       ylabel="Mean orientation error (°)",
       fmt="{:.3f}",
       outfile=OUT_DIR / "fig_5_4_orientation_error.png")

print("\nDone. Embed these into v2.docx in place of the existing Figures 5.3 / 5.4.")
