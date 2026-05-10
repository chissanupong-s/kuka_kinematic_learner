#!/usr/bin/env python3
"""
Aggregate Stage-3 7-DoF adaptation, n=3 (seeds 42, 1, 2) on part000.pt.

Usage:
    python aggregate_adapt_7dof_n3.py <expH_outroot>

Reports n=3 mean ± std and 95% Student's t CI (df=2, t=12.706), and compares
against:
  - the report's headline 7-DoF single-task baseline (0.0101 m, 2.0853°)
  - the report's headline 7-DoF adapted on part001 (~0.0135 m, ~2.59°)
"""
from __future__ import annotations
import math
import re
import sys
from pathlib import Path

# Headline values from the report (Table 5.1, 7-DoF row)
SINGLETASK_7 = {"pos_m": 0.0101, "ori_deg": 2.0853}
ADAPTED_7_PART001 = {"pos_m": 0.0135, "ori_deg": 2.59}

T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 9: 2.262}


def parse_best_line(path: Path):
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    # adapt_multitask_newest.py emits:
    #   [INFO] BEST step=99000 metrics={'pos_mae_m': 0.00990, 'pos_rmse_m': ..., 'ori_deg': 1.72} score=...
    matches = re.findall(
        r"\[INFO\] BEST.*?'pos_mae_m':\s*([\d.eE+-]+).*?'ori_deg':\s*([\d.eE+-]+)",
        text,
    )
    if not matches:
        return None
    pos_m, ori_deg = matches[-1]
    return {"pos_m": float(pos_m), "ori_deg": float(ori_deg)}


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
        print(f"Usage: {sys.argv[0]} <expH_outroot>")
        sys.exit(1)
    outroot = Path(sys.argv[1])
    print(f"=== 7-DoF adapted n=3 (part000.pt) aggregator ===")
    print(f"Output dir: {outroot}\n")

    seeds = [42, 1, 2]
    rows = []
    print(f"{'seed':>4}  {'pos_mae_m':>10}  {'ori_deg':>10}")
    print("-" * 30)
    for seed in seeds:
        log = outroot / "logs" / f"adapt_dof7_seed{seed}_part000.log"
        res = parse_best_line(log)
        if res:
            print(f"{seed:>4}  {res['pos_m']:>10.4f}  {res['ori_deg']:>10.4f}")
            rows.append(res)
        else:
            print(f"{seed:>4}  (BEST line missing in {log})")

    if len(rows) < 2:
        print("\n[!] Not enough completed seeds to compute statistics.")
        return

    pos_vals = [r["pos_m"] for r in rows]
    ori_vals = [r["ori_deg"] for r in rows]
    ps, os = stats(pos_vals), stats(ori_vals)

    print()
    print("=== n=3 statistics (df=2, t-multiplier=12.706, 95% CI) ===")
    if ps:
        m, s, h = ps
        print(f"  pos_mae_m = {m:.4f} ± {s:.4f} m   95% CI [{m-h:.4f}, {m+h:.4f}]")
    if os:
        m, s, h = os
        print(f"  ori_deg   = {m:.4f} ± {s:.4f}°  95% CI [{m-h:.4f}, {m+h:.4f}]")

    print()
    print("=== Comparison ===")
    if ps:
        m, s, h = ps
        st = SINGLETASK_7["pos_m"]
        ad = ADAPTED_7_PART001["pos_m"]
        print(f"  vs single-task headline (0.0101 m): mean {m:.4f} m -> "
              f"{'BEATS single-task' if m + h < st else 'overlaps' if m - h < st < m + h else 'WORSE'}")
        print(f"  vs adapted-part001    (0.0135 m): mean {m:.4f} m -> "
              f"{'improves' if m < ad - h else 'no improvement'}")

    print()
    print("=== For Table 5.1 update (7-DoF Adapted (best) row) ===")
    if ps and os:
        print(f"  pos = {ps[0]:.4f} ± {ps[1]:.4f} m, ori = {os[0]:.3f} ± {os[1]:.3f}°  (n=3, part000.pt)")


if __name__ == "__main__":
    main()
