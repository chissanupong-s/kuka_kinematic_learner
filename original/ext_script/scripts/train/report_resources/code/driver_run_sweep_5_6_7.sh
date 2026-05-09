#!/usr/bin/env bash
set -euo pipefail

# ===================== EDIT THESE PATHS =====================
# FK checkpoint to adapt
CKPT="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/runs/multitask/seperate_weight/1/fk_iiwa_5_6_7/multitask_fk_best.pt"

DATA_5="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/data/narrowed/5DOF_8deg.pt_part000.pt"
DATA_6="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/data/narrowed/6DOF_12deg.pt_part000.pt"
DATA_7="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/data/narrowed/7DOF_15deg/7DOF_15deg_part001.pt"

# FK-weighted adaptation script (use absolute path)
# Option 1: if you copied it into your repo:
ADAPT_SCRIPT="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/adapt_multitask_newest.py"
# Option 2: run the sandbox copy directly (only works in this environment):
# ADAPT_SCRIPT="/mnt/data/adapt_multitask_fk_weighted.py"

# Real IsaacLab launcher path
ISAACLAB_SH="/home/ubuntu/IsaacLab/isaaclab.sh"
# ============================================================

# ===================== RUN CONFIG ===========================
DEVICE="cuda"
MODE="fk"  # force FK (recommended)

# Support / Query sizes (must satisfy support_size + query_size <= N)
SUPPORT_SIZE="50000"
QUERY_SIZE="2000000"

# Adaptation
ADAPT_STEPS="100000"
INNER_LR="1e-5"

# Batches
BATCH_SIZE="8192"
QUERY_BATCH_SIZE="8192"
NUM_WORKERS="8"

# Trainable params during adaptation
ADAPT_WHAT="all"   # or "head"

# Regularization / stability
L2_REG="1e-6"
GRAD_CLIP="1.0"
ADAM_EPS="1e-7"
STD_FLOOR_Q_DEG="1.0"
ENABLE_TF32="1"

# Logging / eval cadence
LOG_EVERY="2000"
EVAL_EVERY="2000"
EVAL_PBAR="1"
EVAL_STEPS_LIST=""   # optional: "0,500,1000,2000,5000,10000,50000,100000"

# ================= FK WEIGHTS (MAIN THING) ==================
# Support loss: pos_weight * pos_mse(m^2) + ori_weight * angle_rad^2
POS_WEIGHT="1.0"
ORI_WEIGHT="0.30"

# Best-step score: score_pos_w * pos_mae(m) + score_ori_w * ori_deg
SCORE_POS_W="1.0"
SCORE_ORI_W="0.01"
# ============================================================

SEED="42"
# ============================================================

OUTDIR="./runs/adapt_fk_weighted_7_$(date +%Y%m%d_%H%M%S)"
LOGDIR="$OUTDIR/logs"
TBROOT="$OUTDIR/tb"
SAVEROOT="$OUTDIR/ckpts"
mkdir -p "$LOGDIR" "$TBROOT" "$SAVEROOT"

SUMMARY_TXT="$OUTDIR/summary.txt"
echo "FK-weighted Adaptation summary" | tee "$SUMMARY_TXT"
echo "ckpt: $CKPT" | tee -a "$SUMMARY_TXT"
echo "script: $ADAPT_SCRIPT" | tee -a "$SUMMARY_TXT"
echo "mode: $MODE | device: $DEVICE" | tee -a "$SUMMARY_TXT"
echo "K=$SUPPORT_SIZE Q=$QUERY_SIZE steps=$ADAPT_STEPS lr=$INNER_LR bs=$BATCH_SIZE qbs=$QUERY_BATCH_SIZE adapt=$ADAPT_WHAT" | tee -a "$SUMMARY_TXT"
echo "FK weights: pos_w=$POS_WEIGHT ori_w=$ORI_WEIGHT | score_pos_w=$SCORE_POS_W score_ori_w=$SCORE_ORI_W" | tee -a "$SUMMARY_TXT"
echo "time: $(date)" | tee -a "$SUMMARY_TXT"
echo "" | tee -a "$SUMMARY_TXT"

common_args () {
  local dof="$1"
  local data="$2"
  local tb_name="$3"
  local save_dir="$4"

  ARGS=(
    --ckpt "$CKPT"
    --mode "$MODE"
    --dof "$dof"
    --data "$data"
    --device "$DEVICE"

    --support_size "$SUPPORT_SIZE"
    --query_size "$QUERY_SIZE"
    --adapt_steps "$ADAPT_STEPS"
    --inner_lr "$INNER_LR"

    --batch_size "$BATCH_SIZE"
    --query_batch_size "$QUERY_BATCH_SIZE"
    --num_workers "$NUM_WORKERS"

    --adapt "$ADAPT_WHAT"
    --l2_reg "$L2_REG"
    --grad_clip "$GRAD_CLIP"
    --adam_eps "$ADAM_EPS"
    --std_floor_q_deg "$STD_FLOOR_Q_DEG"

    --log_every "$LOG_EVERY"
    --eval_every "$EVAL_EVERY"
    --eval_steps_list "$EVAL_STEPS_LIST"

    --pos_weight "$POS_WEIGHT"
    --ori_weight "$ORI_WEIGHT"
    --score_pos_w "$SCORE_POS_W"
    --score_ori_w "$SCORE_ORI_W"

    --tb_logdir "$TBROOT"
    --tb_name "$tb_name"
    --save_dir "$save_dir"

    --seed "$SEED"
  )

  if [[ "$ENABLE_TF32" == "1" ]]; then
    ARGS+=( --enable_tf32 )
  fi
  if [[ "$EVAL_PBAR" == "1" ]]; then
    ARGS+=( --eval_pbar )
  fi
}

run_one () {
  local dof="$1"
  local data="$2"

  local tb_name="dof${dof}"
  local save_dir="$SAVEROOT/dof${dof}"
  mkdir -p "$save_dir"

  local logfile="$LOGDIR/adapt_dof${dof}.log"

  echo "============================================================" | tee -a "$SUMMARY_TXT"
  echo "[RUN] dof=${dof} data=${data}" | tee -a "$SUMMARY_TXT"
  echo "log: ${logfile}" | tee -a "$SUMMARY_TXT"
  echo "tb:  ${TBROOT}/${tb_name}" | tee -a "$SUMMARY_TXT"
  echo "save:${save_dir}" | tee -a "$SUMMARY_TXT"
  echo "============================================================" | tee -a "$SUMMARY_TXT"

  common_args "$dof" "$data" "$tb_name" "$save_dir"

  ( time "$ISAACLAB_SH" -p "$ADAPT_SCRIPT" "${ARGS[@]}" ) 2>&1 | tee "$logfile"

  best_line="$(grep -E "^\[INFO\] BEST" "$logfile" | tail -n 1 || true)"
  echo "$best_line" | tee -a "$SUMMARY_TXT"
  echo "" | tee -a "$SUMMARY_TXT"
}

# run_one 5 "$DATA_5"
# run_one 6 "$DATA_6"
run_one 7 "$DATA_7"

echo "DONE. Outputs saved in: $OUTDIR"
echo "Summary: $SUMMARY_TXT"
echo "TensorBoard: tensorboard --logdir $TBROOT"