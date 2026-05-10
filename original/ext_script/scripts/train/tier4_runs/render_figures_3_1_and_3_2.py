#!/usr/bin/env python3
"""
Render Figure 3.1 (pipeline) and Figure 3.2 (architecture) as polished PNG/PDF
diagrams using matplotlib. Drop-in replacements for the docx.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np


TS = "20260507_045433"
OUT = Path(f"/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/report_resources/figures/appendix_tb_plots/{TS}")
OUT.mkdir(parents=True, exist_ok=True)


def styled_box(ax, x, y, w, h, text, fc="#e8f1ff", ec="#1f4e8a", lw=1.2, fs=9, ha="center", va="center"):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                         facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha=ha, va=va, fontsize=fs)


def arrow(ax, x1, y1, x2, y2, color="#333333", lw=1.4, style="->", text=None, text_offset=(0, 0.05), fs=8):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                        color=color, lw=lw)
    ax.add_patch(a)
    if text is not None:
        midx = (x1 + x2) / 2 + text_offset[0]
        midy = (y1 + y2) / 2 + text_offset[1]
        ax.text(midx, midy, text, fontsize=fs, ha="center", va="center",
                color=color, style="italic")


# ============================================================================
# FIGURE 3.1 — Pipeline
# ============================================================================
def render_pipeline():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13); ax.set_ylim(0, 7); ax.axis("off")

    # Title bar over each stage
    stage_titles = [(1.5, "Stage 1\nSingle-task initialisation"),
                    (5.5, "Stage 2\nShared meta-kinematics training"),
                    (10.5, "Stage 3\nPer-DoF adaptation")]
    for cx, t in stage_titles:
        ax.text(cx, 6.6, t, ha="center", va="center", fontsize=11, fontweight="bold", color="#1f4e8a")

    # Vertical separators
    for x in (3.5, 8.5):
        ax.axvline(x, ymin=0.05, ymax=0.92, color="#cccccc", linewidth=0.8, linestyle="--")

    # ---- Stage 1: 3 single-task models ----
    dof_data = [(5, 5.4, "16.2 M smp"), (6, 3.7, "34.9 M smp"), (7, 2.0, "3.9 M smp")]
    s1_ckpts_y = []
    for dof, y, smp in dof_data:
        styled_box(ax, 0.1, y - 0.35, 1.1, 0.7,
                   f"{dof}-DoF data\n{smp}", fc="#fff3d6", ec="#a07b1f", fs=9)
        styled_box(ax, 1.5, y - 0.35, 1.4, 0.7,
                   f"ResMLP\n(no mask)\nθ_{dof}", fc="#e8f1ff", ec="#1f4e8a", fs=9)
        arrow(ax, 1.21, y, 1.5, y, lw=1.0)
        s1_ckpts_y.append(y)

    # Arrow from Stage 1 box outputs to Stage 2 input (average)
    # Group lines into a small junction at x=3.4
    for y in s1_ckpts_y:
        arrow(ax, 2.91, y, 3.4, 4.0, lw=1.0, color="#666")
    ax.text(3.4, 4.4, "average\n(zero-init mask_proj)", ha="left", va="center", fontsize=8, style="italic", color="#666")

    # ---- Stage 2: shared model + union of datasets ----
    styled_box(ax, 4.0, 3.4, 3.9, 1.2,
               "ResMLP_Mask  (shared backbone)\nhidden=1024, blocks=8, mask_proj added\ntrained on union of 5+6+7 DoF datasets\nloss = MSE on per-task standardised pose",
               fc="#d8efd8", ec="#226622", fs=9)
    # Output arrow from Stage 2
    styled_box(ax, 4.6, 1.6, 2.7, 0.6,
               "θ_shared  (Stage-2 best.pt)",
               fc="#d8efd8", ec="#226622", fs=10)
    arrow(ax, 5.95, 3.4, 5.95, 2.2, lw=1.4, color="#226622")

    # Annotate dataset union arrow
    arrow(ax, 3.5, 4.0, 4.0, 4.0, color="#666", lw=1.0)

    # ---- Stage 3: 3 adapted models ----
    # Arrow from θ_shared to Stage 3
    arrow(ax, 7.3, 1.9, 8.7, 1.9, lw=1.4, color="#226622", text="init from θ_shared", text_offset=(0, 0.18))

    # Three adapted boxes
    adapt_y = [5.0, 3.5, 2.0]
    for dof, y in zip([5, 6, 7], adapt_y):
        styled_box(ax, 9.1, y - 0.35, 1.7, 0.7,
                   f"ResMLP_Mask  ({dof}-DoF\nsupport, fine-tune)",
                   fc="#fde8e8", ec="#992222", fs=9)
        # Arrow into adapted box from a junction
        arrow(ax, 8.7, 1.9, 9.1, y, lw=1.0, color="#992222")
        # Output checkpoint
        styled_box(ax, 11.0, y - 0.35, 1.8, 0.7,
                   f"θ_adapt_{dof}\n(deployable)",
                   fc="#fde8e8", ec="#992222", fs=10)
        arrow(ax, 10.81, y, 11.0, y, lw=1.0, color="#992222")

    # Title at the top
    fig.suptitle(
        "Figure 3.1 — Meta-kinematics three-stage pipeline (KUKA iiwa 14, 5 / 6 / 7 DoF)",
        fontsize=12, y=0.98)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "fig_3_1_pipeline.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig_3_1_pipeline.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}/fig_3_1_pipeline.png + .pdf")


# ============================================================================
# FIGURE 3.2 — ResidualMLP_Mask architecture
# ============================================================================
def render_architecture():
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.set_xlim(0, 11); ax.set_ylim(0, 8.5); ax.axis("off")

    cx = 5.5  # central x

    # --- Inputs ---
    # q vector input
    styled_box(ax, cx - 3.5, 7.6, 2.5, 0.6,
               "Joint vector  q ∈ ℝ⁷\n(inactive joints zero-clamped)",
               fc="#fff3d6", ec="#a07b1f", fs=9)
    # mask input
    styled_box(ax, cx + 1.0, 7.6, 2.5, 0.6,
               "Active-DoF mask  m_τ ∈ {0,1}⁷",
               fc="#fff3d6", ec="#a07b1f", fs=9)

    # --- fc_in and mask_proj ---
    styled_box(ax, cx - 2.7, 6.5, 1.8, 0.6,
               "fc_in: Linear 7→1024",
               fc="#e8f1ff", ec="#1f4e8a", fs=9)
    styled_box(ax, cx + 0.9, 6.5, 1.8, 0.6,
               "mask_proj: Linear 7→1024\n(bias=False, weight init=0)",
               fc="#e8f1ff", ec="#1f4e8a", fs=8)

    arrow(ax, cx - 2.25, 7.6, cx - 1.8, 7.1, lw=1.0)
    arrow(ax, cx + 2.25, 7.6, cx + 1.8, 7.1, lw=1.0)

    # --- Additive sum ---
    plus = plt.Circle((cx, 5.9), 0.18, fc="white", ec="#333333", linewidth=1.4)
    ax.add_patch(plus)
    ax.text(cx, 5.9, "+", ha="center", va="center", fontsize=14, fontweight="bold")
    arrow(ax, cx - 1.8, 6.5, cx - 0.15, 6.0, lw=1.0)
    arrow(ax, cx + 1.8, 6.5, cx + 0.15, 6.0, lw=1.0)

    # --- ReLU after additive sum ---
    styled_box(ax, cx - 0.6, 5.0, 1.2, 0.5, "ReLU", fc="#e8f8e8", ec="#226622", fs=10)
    arrow(ax, cx, 5.7, cx, 5.5, lw=1.0)

    # --- ResBlock × 8 ---
    rb_x = cx - 2.0; rb_y = 2.6; rb_w = 4.0; rb_h = 2.0
    block = FancyBboxPatch((rb_x, rb_y), rb_w, rb_h, boxstyle="round,pad=0.02,rounding_size=0.05",
                           facecolor="#f0e6f8", edgecolor="#5a3a8a", linewidth=1.6)
    ax.add_patch(block)
    ax.text(rb_x + rb_w/2, rb_y + rb_h - 0.25, "ResBlock × 8",
            ha="center", va="center", fontsize=11, fontweight="bold", color="#5a3a8a")
    inner = (
        "x → Linear 1024→1024 → ReLU → Dropout(0.1)\n"
        "  → Linear 1024→1024\n"
        "  → +  (skip from input x)\n"
        "  → ReLU"
    )
    ax.text(rb_x + rb_w/2, rb_y + 0.85, inner,
            ha="center", va="center", fontsize=9, family="monospace")
    arrow(ax, cx, 5.0, cx, 4.6, lw=1.0)

    # --- fc_out ---
    styled_box(ax, cx - 1.2, 1.5, 2.4, 0.6,
               "fc_out: Linear 1024→7",
               fc="#e8f1ff", ec="#1f4e8a", fs=10)
    arrow(ax, cx, 2.6, cx, 2.1, lw=1.0)

    # --- Output ---
    styled_box(ax, cx - 2.4, 0.4, 4.8, 0.7,
               "Output  [ t̂ ∈ ℝ³ , r̂ ∈ ℝ⁴ ]\n"
               "(quaternion unit-normalised in loss / evaluation, not in graph)",
               fc="#fde8e8", ec="#992222", fs=9)
    arrow(ax, cx, 1.5, cx, 1.1, lw=1.0)

    # Side note for Stage 1 difference
    ax.text(0.2, 0.2,
            "NOTE  Stage 1 uses ResidualMLP (same residual-block stack but no mask_proj branch and\n"
            "      no mask input). The Stage-1 ResBlock has identical layers but does NOT call dropout in forward.",
            ha="left", va="bottom", fontsize=8, style="italic", color="#555")

    fig.suptitle(
        "Figure 3.2 — ResidualMLP_Mask architecture (Stages 2 and 3)",
        fontsize=12, y=0.98)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "fig_3_2_architecture.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig_3_2_architecture.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}/fig_3_2_architecture.png + .pdf")


if __name__ == "__main__":
    render_pipeline()
    render_architecture()
