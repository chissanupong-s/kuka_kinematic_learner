#!/usr/bin/env python3
"""
Aggregate the 10-seed 7-DoF adaptation results into a single summary.

Pulls BEST results from:
  - seed 42: existing canonical run (different protocol — 12k steps; flagged)
  - seed 1, 2: expB_multiseed_smart (100k steps)
  - seed 3: expC_seeds34_7dof (100k steps; seed 4 was killed there)
  - seeds 4-10: expD_seeds_4to10_7dof (100k steps)

Reports n=10 mean ± std, 95% CI (Student's t, df=9), and one-sample t-test
against the single-task baseline (0.0101 m, 2.0853°).
"""
import re
import glob
import os
from pathlib import Path
import math

BASE = Path("/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/tier4_runs")
SINGLE_TASK_POS = 0.0101
SINGLE_TASK_ORI = 2.0853

# Find log files for each seed
LOG_PATTERNS = [
    # seed 1, 2 (expB)
    ("seed1", str(BASE / "expB_multiseed_smart_*/logs/adapt_seed1_dof7.log"), "100k"),
    ("seed2", str(BASE / "expB_multiseed_smart_*/logs/adapt_seed2_dof7.log"), "100k"),
    # seed 3 (expC)
    ("seed3", str(BASE / "expC_seeds34_7dof_*/logs/adapt_dof7_seed3.log"), "100k"),
    # seeds 4-10 (expD)
    ("seed4", str(BASE / "expD_seeds_4to10_7dof_*/logs/adapt_dof7_seed4.log"), "100k"),
    ("seed5", str(BASE / "expD_seeds_4to10_7dof_*/logs/adapt_dof7_seed5.log"), "100k"),
    ("seed6", str(BASE / "expD_seeds_4to10_7dof_*/logs/adapt_dof7_seed6.log"), "100k"),
    ("seed7", str(BASE / "expD_seeds_4to10_7dof_*/logs/adapt_dof7_seed7.log"), "100k"),
    ("seed8", str(BASE / "expD_seeds_4to10_7dof_*/logs/adapt_dof7_seed8.log"), "100k"),
    ("seed9", str(BASE / "expD_seeds_4to10_7dof_*/logs/adapt_dof7_seed9.log"), "100k"),
    ("seed10", str(BASE / "expD_seeds_4to10_7dof_*/logs/adapt_dof7_seed10.log"), "100k"),
]

# Seed 42 canonical (different protocol — included for the multi-seed mean
# but flagged in the summary as having a shorter step budget)
SEED_42 = ("seed42", "0.0099 / 1.7104  (canonical, 12k step budget — different protocol)")

def parse_best(logfile):
    if not os.path.exists(logfile):
        return None
    with open(logfile, errors="ignore") as f:
        text = f.read()
    matches = list(re.finditer(r"\[INFO\]\s+BEST\s+step=(\d+)\s+metrics=(\{[^}]*\})", text))
    if not matches:
        return None
    last = matches[-1]
    step = int(last.group(1))
    d_str = last.group(2)
    # Parse pos_mae, ori_deg
    pos_m = re.search(r"'pos_mae_m':\s*([\d.eE+-]+)", d_str)
    ori_m = re.search(r"'ori_deg':\s*([\d.eE+-]+)", d_str)
    pos_rmse_m = re.search(r"'pos_rmse_m':\s*([\d.eE+-]+)", d_str)
    if not (pos_m and ori_m):
        return None
    return {
        "step": step,
        "pos_mae_m": float(pos_m.group(1)),
        "pos_rmse_m": float(pos_rmse_m.group(1)) if pos_rmse_m else None,
        "ori_deg": float(ori_m.group(1)),
    }

# Hard-coded seed 42 numbers (canonical)
seed42_vals = {"step": "(canonical)", "pos_mae_m": 0.0099, "pos_rmse_m": None, "ori_deg": 1.7104}

results = {"seed42": (seed42_vals, "12k (canonical)")}
print(f"{'seed':>6} {'step':>10} {'pos_mae_m':>12} {'ori_deg':>10}  protocol")
print("-" * 60)
print(f"{'seed42':>6} {'(canonical)':>10} {0.0099:>12.4f} {1.7104:>10.4f}  12k (canonical, different protocol)")

for label, pat, protocol in LOG_PATTERNS:
    matches = sorted(glob.glob(pat))
    if not matches:
        print(f"{label:>6} {'(missing)':>10}")
        results[label] = (None, protocol)
        continue
    log = matches[-1]  # latest
    parsed = parse_best(log)
    if parsed is None:
        print(f"{label:>6} {'(no BEST)':>10}")
        results[label] = (None, protocol)
        continue
    print(f"{label:>6} {parsed['step']:>10} {parsed['pos_mae_m']:>12.4f} {parsed['ori_deg']:>10.4f}  {protocol}")
    results[label] = (parsed, protocol)

# Compute statistics for valid results
valid = [(k, v[0]) for k, v in results.items() if v[0] is not None]
n = len(valid)
print(f"\nN = {n} seeds")

if n >= 2:
    pos_vals = [v["pos_mae_m"] for _, v in valid]
    ori_vals = [v["ori_deg"] for _, v in valid]
    pos_mean = sum(pos_vals) / n
    ori_mean = sum(ori_vals) / n
    pos_std = math.sqrt(sum((x - pos_mean) ** 2 for x in pos_vals) / (n - 1))
    ori_std = math.sqrt(sum((x - ori_mean) ** 2 for x in ori_vals) / (n - 1))
    # Student's t-distribution two-sided 95% critical values
    t_crit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
              8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.179}
    df = n - 1
    t = t_crit.get(df, 1.96)
    pos_ci_half = t * pos_std / math.sqrt(n)
    ori_ci_half = t * ori_std / math.sqrt(n)
    # One-sample t-statistic against single-task baseline
    pos_t = (pos_mean - SINGLE_TASK_POS) / (pos_std / math.sqrt(n)) if pos_std > 0 else float("inf")
    ori_t = (ori_mean - SINGLE_TASK_ORI) / (ori_std / math.sqrt(n)) if ori_std > 0 else float("inf")
    print(f"\n=== n = {n} statistics (df = {df}, 2-sided 95% CI) ===")
    print(f"  pos_mae_m: {pos_mean:.4f} +/- {pos_std:.4f}  CI [{pos_mean-pos_ci_half:.4f}, {pos_mean+pos_ci_half:.4f}] m")
    print(f"  ori_deg:   {ori_mean:.4f} +/- {ori_std:.4f}  CI [{ori_mean-ori_ci_half:.4f}, {ori_mean+ori_ci_half:.4f}] deg")
    print(f"\n=== One-sample t-test against single-task baseline ===")
    print(f"  Position (baseline {SINGLE_TASK_POS}): t = {pos_t:+.3f}  ({'rejects' if abs(pos_t) > t else 'fails to reject'} H0 at 5%)")
    print(f"  Orientation (baseline {SINGLE_TASK_ORI}): t = {ori_t:+.3f}  ({'rejects' if abs(ori_t) > t else 'fails to reject'} H0 at 5%)")
    if pos_mean - pos_ci_half <= SINGLE_TASK_POS <= pos_mean + pos_ci_half:
        print(f"  → Position 95% CI ENCOMPASSES baseline (statistical equivalence)")
    elif pos_mean + pos_ci_half < SINGLE_TASK_POS:
        print(f"  → Position 95% CI is ENTIRELY BELOW baseline (OUTPERFORMANCE)")
    else:
        print(f"  → Position 95% CI is ENTIRELY ABOVE baseline (statistical underperformance)")
