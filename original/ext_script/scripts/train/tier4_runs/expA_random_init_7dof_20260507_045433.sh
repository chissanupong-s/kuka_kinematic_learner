#!/usr/bin/env bash
# Tier-4 Experiment A — Random-init 7-DoF adaptation (Ablation B 7-DoF row)
# Generated: 2026-05-07 04:54:33
# Runs the SAME adaptation protocol as the headline 7-DoF row of Table 5.1,
# but starting from a randomly-initialised ResidualMLP_Mask instead of the
# shared meta-kinematics checkpoint. Same wall-clock budget (~0.111 hr).
set -euo pipefail

TIMESTAMP="20260507_045433"
CKPT="/tmp/random_init_resmlp_${TIMESTAMP}.pt"
DATA_7="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/data/narrowed/7DOF_15deg/7DOF_15deg_part001.pt"

ADAPT_SCRIPT="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/adapt_multitask_newest.py"
ISAACLAB_SH="/home/ubuntu/IsaacLab/isaaclab.sh"

DEVICE="cuda"
MODE="fk"
SUPPORT_SIZE="50000"
QUERY_SIZE="2000000"
ADAPT_STEPS="100000"
INNER_LR="1e-5"
BATCH_SIZE="8192"
QUERY_BATCH_SIZE="8192"
NUM_WORKERS="8"
ADAPT_WHAT="all"
L2_REG="1e-6"
GRAD_CLIP="1.0"
ADAM_EPS="1e-7"
STD_FLOOR_Q_DEG="1.0"
ENABLE_TF32="1"
LOG_EVERY="2000"
EVAL_EVERY="2000"
EVAL_PBAR="1"
EVAL_STEPS_LIST=""
POS_WEIGHT="1.0"
ORI_WEIGHT="0.30"
SCORE_POS_W="1.0"
SCORE_ORI_W="0.01"
SEED="42"

OUTDIR="$(dirname "$ADAPT_SCRIPT")/tier4_runs/expA_random_init_7dof_${TIMESTAMP}"
LOGDIR="$OUTDIR/logs"
TBROOT="$OUTDIR/tb"
SAVEROOT="$OUTDIR/ckpts"
mkdir -p "$LOGDIR" "$TBROOT" "$SAVEROOT"

SUMMARY_TXT="$OUTDIR/summary.txt"
echo "EXPERIMENT A — Random-init 7-DoF adaptation"        | tee "$SUMMARY_TXT"
echo "timestamp: ${TIMESTAMP}"                             | tee -a "$SUMMARY_TXT"
echo "ckpt: $CKPT"                                         | tee -a "$SUMMARY_TXT"
echo "data: $DATA_7"                                        | tee -a "$SUMMARY_TXT"
echo "support_size=$SUPPORT_SIZE  query_size=$QUERY_SIZE  adapt_steps=$ADAPT_STEPS  inner_lr=$INNER_LR" | tee -a "$SUMMARY_TXT"
echo "pos_weight=$POS_WEIGHT  ori_weight=$ORI_WEIGHT  score_pos_w=$SCORE_POS_W  score_ori_w=$SCORE_ORI_W" | tee -a "$SUMMARY_TXT"
echo "started: $(date)"                                    | tee -a "$SUMMARY_TXT"

ARGS=(
  --ckpt "$CKPT"
  --mode "$MODE"
  --dof 7
  --data "$DATA_7"
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
  --tb_name "dof7_random_init"
  --save_dir "$SAVEROOT/dof7_random_init"
  --seed "$SEED"
)
[[ "$ENABLE_TF32" == "1" ]] && ARGS+=( --enable_tf32 )
[[ "$EVAL_PBAR" == "1" ]] && ARGS+=( --eval_pbar )

LOGFILE="$LOGDIR/adapt_dof7_random_init.log"
( time "$ISAACLAB_SH" -p "$ADAPT_SCRIPT" "${ARGS[@]}" ) 2>&1 | tee "$LOGFILE"

echo "finished: $(date)" | tee -a "$SUMMARY_TXT"
best_line="$(grep -E "^\[INFO\] BEST" "$LOGFILE" | tail -n 1 || true)"
echo "$best_line" | tee -a "$SUMMARY_TXT"
echo "DONE. Outputs in: $OUTDIR" | tee -a "$SUMMARY_TXT"
