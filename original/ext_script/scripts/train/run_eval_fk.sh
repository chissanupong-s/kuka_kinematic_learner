#!/usr/bin/env bash
set -euo pipefail

# --------- EDIT THESE PATHS ----------
CKPT="/home/wish/isaaclab/kuka_14_kinematic_learner/original/ext_script/scripts/train/runs/multitask/5/fk_iiwa_5_6_7/multitask_fk_best.pt"

DATA_5="/home/wish/isaaclab/kuka_14_kinematic_learner/original/ext_script/data/narrowed/5DOF_8deg.pt_part000.pt"
DATA_6="/home/wish/isaaclab/kuka_14_kinematic_learner/original/ext_script/data/narrowed/6DOF_12deg.pt_part000.pt"
DATA_7="/home/wish/isaaclab/kuka_14_kinematic_learner/original/ext_script/data/narrowed/7DOF_15deg/7DOF_15deg_part001.pt"

EVAL_SCRIPT="eval_multitask.py"

# Use the real IsaacLab launcher path (not an alias)
ISAACLAB_SH="/home/wish/isaaclab/IsaacLab/isaaclab.sh"
# ------------------------------------

DEVICE="cuda"
BATCH_SIZE="8192"
NUM_WORKERS="8"

OUTDIR="./runs/eval_fk_multitask_$(date +%Y%m%d_%H%M%S)"
LOGDIR="$OUTDIR/logs"
mkdir -p "$LOGDIR"

SUMMARY_TXT="$OUTDIR/summary.txt"
echo "Multitask FK evaluation summary" | tee "$SUMMARY_TXT"
echo "ckpt: $CKPT" | tee -a "$SUMMARY_TXT"
echo "time: $(date)" | tee -a "$SUMMARY_TXT"
echo "" | tee -a "$SUMMARY_TXT"

run_one () {
  local dof="$1"
  local data="$2"

  local logfile="$LOGDIR/eval_dof${dof}.log"

  echo "============================================================" | tee -a "$SUMMARY_TXT"
  echo "[RUN] dof=${dof}  data=${data}" | tee -a "$SUMMARY_TXT"
  echo "log: ${logfile}" | tee -a "$SUMMARY_TXT"
  echo "============================================================" | tee -a "$SUMMARY_TXT"

  ( time "$ISAACLAB_SH" -p "$EVAL_SCRIPT" \
      --ckpt "$CKPT" \
      --dof "$dof" \
      --data "$data" \
      --device "$DEVICE" \
      --batch_size "$BATCH_SIZE" \
      --num_workers "$NUM_WORKERS" \
      --eval_pbar \
    ) 2>&1 | tee "$logfile"

  metrics_line="$(grep -E "^\[FK EVAL\]" "$logfile" | tail -n 1 || true)"
  echo "$metrics_line" | tee -a "$SUMMARY_TXT"
  echo "" | tee -a "$SUMMARY_TXT"
}

run_one 5 "$DATA_5"
run_one 6 "$DATA_6"
run_one 7 "$DATA_7"

echo "DONE. Outputs saved in: $OUTDIR"
echo "Summary: $SUMMARY_TXT"
