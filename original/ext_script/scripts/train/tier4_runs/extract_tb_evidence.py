#!/usr/bin/env python3
"""
Comprehensive TensorBoard evidence extractor.

For every relevant TB event file in this project, extract:
  • Every scalar tag — full (step, value) trace as CSV
  • Wall-clock duration of the run (from event timestamps)
  • Best step / best metric for each scalar (min for losses, min for errors)
  • LR schedule trace if present
  • Total number of steps logged

All outputs go to:
  report_resources/results/tb_extracted/<TS>/

Plus a single TB_KEY_NUMBERS.md summary that lists, for each run:
  - Run identity (stage, DoF, seed if applicable, hyperparams)
  - Wall-clock duration
  - Best metric values (with step at which they were attained)
  - Pointer to the CSVs

This file is the "evidence pack" for the report — every numerical claim in
Tables 5.1, 5.2, 5.3, 5.4, E.2 can be traced back to a row here.
"""

from __future__ import annotations
import csv, os, re, sys, glob, datetime
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TIMESTAMP = "20260507_045433"
TRAIN_DIR = Path("/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train")
OUT_DIR = TRAIN_DIR / "report_resources" / "results" / "tb_extracted" / TIMESTAMP
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Run definitions — friendly name → glob pattern + metadata
RUNS = {
    "stage1_5dof_singletask": {
        "glob": "runs/single_task_fk_20260313_144312/5dof/tb/fk/events.out.tfevents*",
        "stage": 1, "dof": 5, "role": "single-task FK",
        "hp": "lr=5e-4 (ReduceLROnPlateau), bs=8192, hidden=1024, num_blocks=8, weight_decay=1e-5",
    },
    "stage1_6dof_singletask": {
        "glob": "runs/single_task_fk_20260313_144312/6dof/tb/fk/events.out.tfevents*",
        "stage": 1, "dof": 6, "role": "single-task FK",
        "hp": "lr=5e-4 (ReduceLROnPlateau), bs=8192, hidden=1024, num_blocks=8, weight_decay=1e-5",
    },
    "stage1_7dof_singletask": {
        "glob": "runs/single_task_fk_20260313_144312/7dof/fk/events.out.tfevents*",
        "stage": 1, "dof": 7, "role": "single-task FK",
        "hp": "lr=5e-4 (ReduceLROnPlateau), bs=8192, hidden=1024, num_blocks=8, weight_decay=1e-5",
    },
    "stage2_seed42_multitask": {
        "glob": "runs/multitask/seperate_weight/1/fk_iiwa_5_6_7/events.out.tfevents*",
        "stage": 2, "dof": "5+6+7", "role": "shared meta-kinematics",
        "hp": "lr=3e-4 (cosine, warmup=2000, lr_min=1e-5), bs=4096, total_steps=300000, best_step=289500",
    },
    "stage3_5dof_seed42_adapt": {
        "glob": "runs/adapt_fk_weighted_5_6_7_20260306_004556/tb/dof5/events.out.tfevents*",
        "stage": 3, "dof": 5, "role": "adaptation (seed=42, headline)",
        "hp": "lr=1e-5, bs=2048, ori_w=0.30, l2=1e-6, support=50000, query=2000000, steps=100000",
    },
    "stage3_6dof_seed42_adapt": {
        "glob": "runs/adapt_fk_weighted_5_6_7_20260306_014642/tb/dof6/events.out.tfevents*",
        "stage": 3, "dof": 6, "role": "adaptation (seed=42, headline)",
        "hp": "lr=1e-6, bs=2048, ori_w=0.30, l2=1e-6, support=50000, query=2000000, steps=100000",
    },
    "stage3_7dof_seed42_adapt": {
        "glob": "runs/sweep_fk_dof7_only_one20260313_015503/tb/lr1e-6_l21e-6_S100000_ow0.05/events.out.tfevents*",
        "stage": 3, "dof": 7, "role": "adaptation (seed=42, headline)",
        "hp": "lr=1e-6, bs=8192, ori_w=0.05, l2=1e-6, support=50000, query=2000000, steps=100000",
    },
    # Tier-4 Experiment A: random-init 7-DoF adapt
    "stage3_7dof_random_init_expA": {
        "glob": f"tier4_runs/expA_random_init_7dof_{TIMESTAMP}/tb/dof7_random_init/events.out.tfevents*",
        "stage": "Ablation B", "dof": 7, "role": "adaptation from random init (seeds=42)",
        "hp": "lr=1e-5, bs=8192, ori_w=0.30, l2=1e-6, support=50000, query=2000000, steps=100000",
    },
}

# Multi-seed Tier-4 runs — added dynamically (these are still running for some DoFs).
MULTISEED_DIR = TRAIN_DIR / f"tier4_runs/expB_multiseed_smart_{TIMESTAMP}/tb"
for seed in (1, 2):
    for dof in (5, 6, 7):
        key = f"stage3_{dof}dof_seed{seed}_multiseed"
        RUNS[key] = {
            "glob": f"tier4_runs/expB_multiseed_smart_{TIMESTAMP}/tb/seed{seed}_dof{dof}/events.out.tfevents*",
            "stage": 3, "dof": dof, "role": f"adaptation (seed={seed}, multi-seed extra)",
            "hp": "(multi-seed sweep — see expB_multiseed_smart logs for exact params)",
        }


def resolve(rel_glob):
    matches = sorted((TRAIN_DIR / rel_glob).parent.glob((TRAIN_DIR / rel_glob).name))
    return matches[-1] if matches else None


def extract_run(name, meta, out_dir):
    fp = resolve(meta["glob"])
    if fp is None or not fp.exists():
        return {"status": "missing", "name": name, "meta": meta}
    try:
        ea = EventAccumulator(str(fp), size_guidance={"scalars": 0})
        ea.Reload()
    except Exception as e:
        return {"status": f"error: {e}", "name": name, "meta": meta}

    tags = ea.Tags().get("scalars", [])
    if not tags:
        return {"status": "no_scalars", "name": name, "meta": meta}

    run_dir = out_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save each scalar as a CSV
    summaries = {}
    wall_t_min, wall_t_max = float("inf"), float("-inf")
    for tag in tags:
        rows = ea.Scalars(tag)
        if not rows:
            continue
        steps = np.array([r.step for r in rows])
        vals  = np.array([r.value for r in rows])
        wts   = np.array([r.wall_time for r in rows])
        if len(wts) > 0:
            wall_t_min = min(wall_t_min, wts.min())
            wall_t_max = max(wall_t_max, wts.max())
        # CSV
        safe_tag = tag.replace("/", "__")
        csv_path = run_dir / f"{safe_tag}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "value", "wall_time"])
            for s, v, t in zip(steps.tolist(), vals.tolist(), wts.tolist()):
                w.writerow([int(s), float(v), float(t)])
        # Best (min) for losses/errors; (max) for some others
        is_descending = any(k in tag.lower() for k in ("loss", "err", "rmse", "mae", "score"))
        best_idx = int(np.argmin(vals)) if is_descending else int(np.argmax(vals))
        last_idx = int(len(vals) - 1)
        summaries[tag] = {
            "n_points": int(len(vals)),
            "first_step": int(steps[0]),
            "last_step": int(steps[-1]),
            "first_value": float(vals[0]),
            "last_value": float(vals[-1]),
            "best_step": int(steps[best_idx]),
            "best_value": float(vals[best_idx]),
            "best_min_or_max": "min" if is_descending else "max",
        }

    wall_clock_seconds = (wall_t_max - wall_t_min) if wall_t_max > wall_t_min else 0.0
    return {
        "status": "ok",
        "name": name,
        "meta": meta,
        "event_file": str(fp),
        "tags": tags,
        "summaries": summaries,
        "wall_clock_seconds": float(wall_clock_seconds),
        "wall_clock_human": f"{wall_clock_seconds/3600:.3f} hr ({wall_clock_seconds/60:.2f} min)",
        "first_event_iso": datetime.datetime.fromtimestamp(wall_t_min).isoformat() if wall_t_min != float('inf') else None,
        "last_event_iso": datetime.datetime.fromtimestamp(wall_t_max).isoformat() if wall_t_max != float('-inf') else None,
    }


def build_summary_md(results):
    lines = []
    lines.append(f"# TB-Extracted Evidence Pack — {TIMESTAMP}\n")
    lines.append(f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}\n")
    lines.append(f"Source: TensorBoard event files written by the project's training scripts.\n")
    lines.append(f"Output: every scalar tag of every run is saved as a CSV under this directory; this MD file is a navigation index plus key numbers.\n")
    lines.append("\n---\n")

    # Group by stage
    by_stage = {1: [], 2: [], 3: [], "Ablation B": []}
    for r in results:
        s = r["meta"].get("stage")
        by_stage.setdefault(s, []).append(r)

    for stage_label, items in by_stage.items():
        if not items: continue
        lines.append(f"\n## Stage {stage_label}\n")
        for r in items:
            lines.append(f"\n### {r['name']}\n")
            if r["status"] != "ok":
                lines.append(f"**STATUS:** {r['status']}\n")
                continue
            lines.append(f"- **Role:** {r['meta'].get('role')}")
            lines.append(f"- **DoF:** {r['meta'].get('dof')}")
            lines.append(f"- **Hyperparameters:** {r['meta'].get('hp')}")
            lines.append(f"- **Event file:** `{r['event_file']}`")
            lines.append(f"- **Wall-clock duration:** {r['wall_clock_human']}")
            if r["first_event_iso"]:
                lines.append(f"- **Run window:** {r['first_event_iso']}  →  {r['last_event_iso']}")
            lines.append(f"- **Scalar tags ({len(r['tags'])}):**")
            for tag in r["tags"]:
                s = r["summaries"][tag]
                lines.append(f"  - `{tag}` — {s['n_points']} points, steps {s['first_step']}–{s['last_step']}; "
                             f"last={s['last_value']:.6g}; **best ({s['best_min_or_max']}) = {s['best_value']:.6g} @ step {s['best_step']}**")

    # Cross-cutting key numbers section
    lines.append("\n\n---\n\n# CROSS-CUTTING KEY NUMBERS\n")
    # Stage 1 final test losses
    lines.append("\n## Stage 1 — Single-task FK final eval\n")
    lines.append("| DoF | Final val_total_mse | Final test_total_mse | Wall-clock |")
    lines.append("|---|---|---|---|")
    for r in results:
        if r["meta"].get("stage") != 1 or r["status"] != "ok": continue
        s = r["summaries"]
        val_last = s.get("loss/val_total_mse", {}).get("last_value")
        val_best = s.get("loss/val_total_mse", {}).get("best_value")
        test = s.get("loss/test_total_mse", {}).get("last_value")
        test_str = f"{test:.6g}" if test is not None else "n/a"
        val_last_str = f"{val_last:.6g}" if val_last is not None else "n/a"
        val_best_str = f"{val_best:.6g}" if val_best is not None else "n/a"
        lines.append(f"| {r['meta']['dof']} | {val_last_str} (best {val_best_str}) | {test_str} | {r['wall_clock_human']} |")

    # Stage 2 multitask
    lines.append("\n## Stage 2 — Shared meta-kinematics seed=42\n")
    for r in results:
        if r["meta"].get("stage") != 2 or r["status"] != "ok": continue
        s = r["summaries"]
        lines.append(f"- **Wall-clock:** {r['wall_clock_human']}")
        for k in ("eval/avg_loss_1batch_each_task", "train/5dof_loss", "train/6dof_loss", "train/7dof_loss"):
            if k in s:
                ss = s[k]
                lines.append(f"- `{k}`: best = {ss['best_value']:.6g} @ step {ss['best_step']}; last = {ss['last_value']:.6g}")

    # Stage 3 / Ablation B — query metrics
    lines.append("\n## Stage 3 + Ablation B — Adaptation query metrics\n")
    lines.append("| Run | Best pos_mae (m) | Best step | Best ori (deg) | Best step | Wall-clock |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        if r["meta"].get("stage") not in (3, "Ablation B"): continue
        if r["status"] != "ok": continue
        s = r["summaries"]
        pma = s.get("query/pos_mae_m")
        ori = s.get("query/ori_deg")
        if pma and ori:
            lines.append(f"| {r['name']} | {pma['best_value']:.6g} | {pma['best_step']} | {ori['best_value']:.6g} | {ori['best_step']} | {r['wall_clock_human']} |")

    # Files index
    lines.append("\n\n---\n\n# CSV INDEX\n")
    lines.append("Every `<tag>` for `<run>` is at `<run>/<tag-with-/-replaced-by-__>.csv`. Three columns: `step, value, wall_time`.\n")
    for r in results:
        if r["status"] != "ok": continue
        lines.append(f"\n- `{r['name']}/`")
        for t in r["tags"]:
            safe = t.replace("/", "__")
            lines.append(f"  - `{safe}.csv`")

    return "\n".join(lines)


def main():
    print(f"Output: {OUT_DIR}")
    results = []
    for name, meta in RUNS.items():
        print(f"\n[{name}]")
        r = extract_run(name, meta, OUT_DIR)
        if r["status"] == "ok":
            print(f"  OK — {len(r['tags'])} tags, wall-clock {r['wall_clock_human']}")
        else:
            print(f"  {r['status']}")
        results.append(r)

    md = build_summary_md(results)
    md_path = OUT_DIR / "TB_KEY_NUMBERS.md"
    md_path.write_text(md)
    print(f"\nWrote summary: {md_path}")

    # JSON dump for machine consumption
    import json
    json_path = OUT_DIR / "tb_evidence.json"
    json_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"Wrote JSON: {json_path}")


if __name__ == "__main__":
    main()
