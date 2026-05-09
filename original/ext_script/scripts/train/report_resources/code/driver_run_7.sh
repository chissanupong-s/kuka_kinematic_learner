#!/usr/bin/env bash
set -euo pipefail

# ===================== EDIT THESE PATHS =====================
CKPT="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/runs/multitask/seperate_weight/1/fk_iiwa_5_6_7/multitask_fk_best.pt"
DATA_7="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/data/narrowed/7DOF_15deg/7DOF_15deg_part000.pt"

ADAPT_SCRIPT="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/adapt_multitask_newest.py"
ISAACLAB_SH="/home/ubuntu/IsaacLab/isaaclab.sh"
# ============================================================

DEVICE="cuda"
MODE="fk"
DOF="7"

# Keep these stable for now
SUPPORT_SIZE="50000"
QUERY_SIZE="200000"
BATCH_SIZE="1024"
QUERY_BATCH_SIZE="8192"
NUM_WORKERS="8"

ADAPT_WHAT="all"
GRAD_CLIP="1.0"
ADAM_EPS="1e-7"
STD_FLOOR_Q_DEG="1.0"
ENABLE_TF32="1"

# Metrics selection weights
SCORE_POS_W="1.0"
SCORE_ORI_W="0.01"

# Evaluation steps. Dense early steps matter.
EVAL_STEPS_LIST="0,10,25,50,100,200,500,1000,2000"
LOG_EVERY="200"
EVAL_EVERY="200"
EVAL_PBAR="1"

SEED="42"

# ===================== SWEEP GRID =====================
# You said 1e-5 to 1e-6 is best. Include 3e-6 and 3e-5 too.
INNER_LRS=( "1e-6" "3e-6" "1e-5" "3e-5" )

# Drift control. This is usually the missing piece with adapt=all.
L2_REGS=( "0" "1e-6" "1e-5" "3e-5" "1e-4" )

# Total steps. Too many steps breaks generalization.
ADAPT_STEPS_LIST=( "10000" "20000" "50000" "100000" )

# Orientation balance. Keep pos_weight fixed at 1.0.
ORI_WEIGHTS=( "0.05" "0.10" "0.30" "0.50" )
POS_WEIGHT="1.0"
# ======================================================

OUTDIR="./runs/sweep_fk_dof7_$(date +%Y%m%d_%H%M%S)"
LOGDIR="$OUTDIR/logs"
TBROOT="$OUTDIR/tb"
SAVEROOT="$OUTDIR/ckpts"
mkdir -p "$LOGDIR" "$TBROOT" "$SAVEROOT"

SUMMARY_TXT="$OUTDIR/summary.csv"
echo "tag,inner_lr,l2_reg,adapt_steps,ori_weight,best_line" | tee "$SUMMARY_TXT"

run_one () {
  local inner_lr="$1"
  local l2_reg="$2"
  local adapt_steps="$3"
  local ori_weight="$4"

  local tag="lr${inner_lr}_l2${l2_reg}_S${adapt_steps}_ow${ori_weight}"
  local tb_name="$tag"
  local save_dir="$SAVEROOT/$tag"
  local logfile="$LOGDIR/$tag.log"
  mkdir -p "$save_dir"

  echo "============================================================"
  echo "[RUN] $tag"
  echo "============================================================"

  args=(
    --ckpt "$CKPT"
    --mode "$MODE"
    --dof "$DOF"
    --data "$DATA_7"
    --device "$DEVICE"

    --support_size "$SUPPORT_SIZE"
    --query_size "$QUERY_SIZE"
    --adapt_steps "$adapt_steps"
    --inner_lr "$inner_lr"

    --batch_size "$BATCH_SIZE"
    --query_batch_size "$QUERY_BATCH_SIZE"
    --num_workers "$NUM_WORKERS"

    --adapt "$ADAPT_WHAT"
    --l2_reg "$l2_reg"
    --grad_clip "$GRAD_CLIP"
    --adam_eps "$ADAM_EPS"
    --std_floor_q_deg "$STD_FLOOR_Q_DEG"

    --log_every "$LOG_EVERY"
    --eval_every "$EVAL_EVERY"
    --eval_steps_list "$EVAL_STEPS_LIST"

    --pos_weight "$POS_WEIGHT"
    --ori_weight "$ori_weight"
    --score_pos_w "$SCORE_POS_W"
    --score_ori_w "$SCORE_ORI_W"

    --tb_logdir "$TBROOT"
    --tb_name "$tb_name"
    --save_dir "$save_dir"

    --seed "$SEED"
  )

  if [[ "$ENABLE_TF32" == "1" ]]; then
    args+=( --enable_tf32 )
  fi
  if [[ "$EVAL_PBAR" == "1" ]]; then
    args+=( --eval_pbar )
  fi

  ( time "$ISAACLAB_SH" -p "$ADAPT_SCRIPT" "${args[@]}" ) 2>&1 | tee "$logfile"

  best_line="$(grep -E "^\[INFO\] BEST" "$logfile" | tail -n 1 || true)"
  echo "${tag},${inner_lr},${l2_reg},${adapt_steps},${ori_weight},\"${best_line}\"" | tee -a "$SUMMARY_TXT"
}

for inner_lr in "${INNER_LRS[@]}"; do
  for l2_reg in "${L2_REGS[@]}"; do
    for adapt_steps in "${ADAPT_STEPS_LIST[@]}"; do
      for ori_weight in "${ORI_WEIGHTS[@]}"; do
        run_one "$inner_lr" "$l2_reg" "$adapt_steps" "$ori_weight"
      done
    done
  done
done

echo "DONE"
echo "OUTDIR: $OUTDIR"
echo "SUMMARY: $SUMMARY_TXT"
echo "TensorBoard: tensorboard --logdir $TBROOT"