#!/usr/bin/env python3
"""
Aggregate Stage-1 single-task multi-seed (n=3) training results from expG.

Usage:
    python aggregate_singletask_multiseed.py <expG_outroot>

Reports:
  - Per-DoF n=3 mean ± std on test-set position and orientation errors
  - 95% Student's t confidence intervals (df=2)
  - Comparison against the report's headline single-task baselines

Expects the following structure under <outroot>:
    dof5_seed42/logs/eval.log     -> contains "test pos_mae_m=... ori_deg=..."
    dof5_seed1/logs/eval.log
    dof5_seed2/logs/eval.log
    ...same for dof6_seed{42,1,2}, dof7_seed{42,1,2}
"""
from __future__ import annotations
import math
import re
import sys
from pathlib import Path

# Headline baselines from the report (Table 5.1)
HEADLINE = {
    5: {"pos_m": 0.0093, "ori_deg": 1.2039},
    6: {"pos_m": 0.0110, "ori_deg": 1.7136},
    7: {"pos_m": 0.0101, "ori_deg": 2.0853},
}

# Student's t two-sided 95% critical values
T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 9: 2.262, 10: 2.228}


def parse_eval_log(path: Path):
    """Try multiple regex patterns to find pos_mae_m and ori_deg in a log."""
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")

    # eval_model_single_task.py emits:
    #   Mean position error: 0.008594 m
    #   Mean orientation error: 1.108561 deg
    # Older scripts used pos_mae_m=... / ori_deg=...; both supported.
    pos_match = (
        re.search(r"Mean position error\s*:\s*([\d.eE+-]+)", text)
        or re.search(r"pos_mae_m\s*[:=]\s*([\d.eE+-]+)", text)
        or re.search(r"position[_ ]error[^:=]*[:=]\s*([\d.eE+-]+)", text, re.IGNORECASE)
    )
    ori_match = (
        re.search(r"Mean orientation error\s*:\s*([\d.eE+-]+)", text)
        or re.search(r"ori_deg\s*[:=]\s*([\d.eE+-]+)", text)
        or re.search(r"orientation[_ ]error[^:=]*[:=]\s*([\d.eE+-]+)", text, re.IGNORECASE)
    )

    if pos_match and ori_match:
        return {
            "pos_m": float(pos_match.group(1)),
            "ori_deg": float(ori_match.group(1)),
        }
    return None


def stats(values):
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(var)
    t = T95.get(n - 1, 1.96)
    half_ci = t * std / math.sqrt(n)
    return mean, std, half_ci


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <expG_outroot>")
        sys.exit(1)
    outroot = Path(sys.argv[1])

    print(f"=== Stage-1 single-task multi-seed aggregator ===")
    print(f"Output dir: {outroot}\n")

    seeds = [42, 1, 2]
    print(f"{'DoF':>3}  {'seed':>4}  {'pos_mae_m':>10}  {'ori_deg':>10}")
    print("-" * 50)
    by_dof = {}
    for dof in [5, 6, 7]:
        by_dof[dof] = []
        for seed in seeds:
            log = outroot / f"dof{dof}_seed{seed}" / "logs" / f"dof{dof}_seed{seed}_eval.log"
            # Try alternative log path
            if not log.exists():
                log = outroot / "logs" / f"dof{dof}_seed{seed}_eval.log"
            res = parse_eval_log(log)
            if res:
                print(f"{dof:>3}  {seed:>4}  {res['pos_m']:>10.4f}  {res['ori_deg']:>10.4f}")
                by_dof[dof].append(res)
            else:
                print(f"{dof:>3}  {seed:>4}  (eval log missing or unreadable: {log})")

    print()
    print("=== n=3 statistics (df=2, t-multiplier=12.706, 95% CI) ===")
    print(f"{'DoF':>3}  {'metric':>10}  {'mean':>10}  {'std':>10}  {'95% CI':>20}  {'headline':>10}  {'matches?':>10}")
    print("-" * 90)
    for dof in [5, 6, 7]:
        results = by_dof[dof]
        if len(results) < 2:
            print(f"{dof:>3}  (insufficient data)")
            continue
        pos_vals = [r["pos_m"] for r in results]
        ori_vals = [r["ori_deg"] for r in results]
        ps = stats(pos_vals); os = stats(ori_vals)
        if ps:
            m, s, h = ps
            head = HEADLINE[dof]["pos_m"]
            within = m - h <= h <= m + h
            print(f"{dof:>3}  {'pos_mae_m':>10}  {m:>10.4f}  {s:>10.4f}  [{m-h:.4f}, {m+h:.4f}]  {head:>10.4f}  {'yes' if abs(m-head) < h else 'no':>10}")
        if os:
            m, s, h = os
            head = HEADLINE[dof]["ori_deg"]
            print(f"{dof:>3}  {'ori_deg':>10}  {m:>10.4f}  {s:>10.4f}  [{m-h:.4f}, {m+h:.4f}]  {head:>10.4f}  {'yes' if abs(m-head) < h else 'no':>10}")

    print()
    print("=== For Table 5.1 update ===")
    print("Replace the Single-task row's point estimates with mean ± std (n=3, seeds 42, 1, 2):")
    for dof in [5, 6, 7]:
        results = by_dof[dof]
        if len(results) < 2: continue
        pos_vals = [r["pos_m"] for r in results]
        ori_vals = [r["ori_deg"] for r in results]
        ps = stats(pos_vals); os = stats(ori_vals)
        if ps and os:
            print(f"  {dof}-DoF: pos = {ps[0]:.4f} ± {ps[1]:.4f} m, ori = {os[0]:.3f} ± {os[1]:.3f}°")


if __name__ == "__main__":
    main()
