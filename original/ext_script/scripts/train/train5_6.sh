#!/usr/bin/env bash
set -euo pipefail

# ===================== EDIT THESE PATHS =====================
SCRIPT="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/train_kinematics_nn_pol_pt_2.py"
EVAL_SCRIPT="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/eval_model_single_task.py"

DATA_5="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/data/narrowed/5DOF_8deg.pt_part000.pt"
DATA_6="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/data/narrowed/6DOF_12deg.pt_part000.pt"

# Use Isaac Lab python environment if needed
ISAACLAB_SH="/home/ubuntu/IsaacLab/isaaclab.sh"
# ===========================================================

DEVICE="cuda"
MODE="fk"

EPOCHS="200"
BATCH_SIZE="4096"
LR="5e-4"
HIDDEN_DIM="1024"
NUM_BLOCKS="8"
NUM_WORKERS="8"
WEIGHT_DECAY="1e-5"
SCHEDULER_PATIENCE="10"
GRAD_CLIP="1.0"

TRAIN_FRAC="0.5"
VAL_FRAC="0.2"

OUTROOT="./runs/single_task_fk_$(date +%Y%m%d_%H%M%S)"
LOGDIR="$OUTROOT/logs"
mkdir -p "$LOGDIR"

run_one () {
  local tag="$1"
  local data="$2"

  local run_root="$OUTROOT/$tag"
  local train_log="$LOGDIR/${tag}_train.log"
  local eval_log="$LOGDIR/${tag}_eval.log"

  local train_logdir="$run_root/tb"
  local out_dir="$run_root/models"

  mkdir -p "$run_root" "$train_logdir" "$out_dir"

  echo "============================================================"
  echo "[TRAIN] $tag"
  echo "data: $data"
  echo "============================================================"

  ( time "$ISAACLAB_SH" -p "$SCRIPT" \
      --csv "$data" \
      --mode "$MODE" \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --lr "$LR" \
      --hidden_dim "$HIDDEN_DIM" \
      --num_blocks "$NUM_BLOCKS" \
      --train_frac "$TRAIN_FRAC" \
      --val_frac "$VAL_FRAC" \
      --device "$DEVICE" \
      --out_dir "$out_dir" \
      --log_dir "$train_logdir" \
      --weight_decay "$WEIGHT_DECAY" \
      --num_workers "$NUM_WORKERS" \
      --scheduler_patience "$SCHEDULER_PATIENCE" \
      --grad_clip "$GRAD_CLIP" \
    ) 2>&1 | tee "$train_log"

  local ckpt="$train_logdir/$MODE/${MODE}_pose_best.pt"

  echo "============================================================"
  echo "[EVAL] $tag"
  echo "ckpt: $ckpt"
  echo "============================================================"

  ( time "$ISAACLAB_SH" -p "$EVAL_SCRIPT" \
      --csv "$data" \
      --checkpoint "$ckpt" \
      --batch_size "$BATCH_SIZE" \
      --device "$DEVICE" \
    ) 2>&1 | tee "$eval_log"
}

run_one "5dof" "$DATA_5"
run_one "6dof" "$DATA_6"

echo "DONE"
echo "Outputs: $OUTROOT"