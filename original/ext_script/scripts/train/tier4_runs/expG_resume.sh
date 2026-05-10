#!/usr/bin/env bash
# One-time recovery script: resume expG_singletask_multiseed_15M after the
# grep-mismatch crash that killed the chain at dof5_seed42's summary step
# (the eval ran fine and metrics are in dof5_seed42_eval.log; only the
# summary-extraction step crashed). The original script and aggregator
# regex have since been fixed.
#
# This script reuses the existing OUTROOT (expG_singletask_multiseed_15M_20260509_165051)
# and runs only the 8 missing seed/DoF combinations:
#   5-DoF: seeds 1, 2  (skip 42 - already done)
#   6-DoF: seeds 42, 1, 2
#   7-DoF: seeds 42, 1, 2
#
# After completion, run aggregate_singletask_multiseed.py on the OUTROOT.
set -euo pipefail

ROOT=/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train
DATA_5="$ROOT/../../data/narrowed/5DOF_8deg.pt_part000.pt"
DATA_6="$ROOT/../../data/narrowed/6DOF_12deg.pt_part000.pt"
DATA_7="$ROOT/../../data/narrowed/7DOF_15deg/7DOF_15deg_part000.pt"
SCRIPT="$ROOT/train_kinematics_nn_pol_pt_2.py"
EVAL_SCRIPT="$ROOT/eval_model_single_task.py"
ISAACLAB_SH="/home/ubuntu/IsaacLab/isaaclab.sh"

# Reuse the existing OUTROOT from the partial run
OUTROOT="$ROOT/tier4_runs/expG_singletask_multiseed_15M_20260509_165051"
LOGDIR="$OUTROOT/logs"
mkdir -p "$LOGDIR"

SUMMARY="$OUTROOT/summary.txt"

# Hyperparameters identical to the original expG run
EPOCHS=200
BATCH_SIZE=4096
LR=5e-4
HIDDEN_DIM=1024
NUM_BLOCKS=8
WEIGHT_DECAY=1e-5
SCHEDULER_PATIENCE=10
GRAD_CLIP=1.0
TRAIN_FRAC=0.7
VAL_FRAC=0.1
MAX_SAMPLES=15000000

echo "[$(date)] === RESUME === Continuing expG into existing OUTROOT" | tee -a "$SUMMARY"
echo "  OUTROOT: $OUTROOT" | tee -a "$SUMMARY"
echo "  Skipping: dof5_seed42 (already complete: pos=0.008594m ori=1.109deg)" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

run_one () {
    local dof="$1"
    local data="$2"
    local seed="$3"
    local tag="dof${dof}_seed${seed}"
    local run_root="$OUTROOT/$tag"
    local train_log="$LOGDIR/${tag}_train.log"
    local eval_log="$LOGDIR/${tag}_eval.log"
    local train_logdir="$run_root/tb"
    local out_dir="$run_root/models"

    # Skip-if-already-complete guard
    if [ -f "$run_root/tb/fk/fk_pose_best.pt" ] \
       && [ -f "$eval_log" ] \
       && grep -q "Mean position error" "$eval_log" 2>/dev/null; then
        echo "[SKIP] $tag (best.pt + eval log already present)" | tee -a "$SUMMARY"
        return
    fi

    mkdir -p "$run_root" "$train_logdir" "$out_dir"

    echo "============================================================" | tee -a "$SUMMARY"
    echo "[TRAIN] $tag" | tee -a "$SUMMARY"
    echo "  start: $(date)" | tee -a "$SUMMARY"
    echo "============================================================" | tee -a "$SUMMARY"

    ( time "$ISAACLAB_SH" -p "$SCRIPT" \
        --csv "$data" --mode fk \
        --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --lr "$LR" \
        --hidden_dim "$HIDDEN_DIM" --num_blocks "$NUM_BLOCKS" \
        --train_frac "$TRAIN_FRAC" --val_frac "$VAL_FRAC" \
        --weight_decay "$WEIGHT_DECAY" --scheduler_patience "$SCHEDULER_PATIENCE" \
        --grad_clip "$GRAD_CLIP" --num_workers 8 --device cuda \
        --out_dir "$out_dir" --log_dir "$train_logdir" \
        --seed "$seed" --max_samples "$MAX_SAMPLES" \
    ) 2>&1 | tee "$train_log"

    local ckpt="$train_logdir/fk/fk_pose_best.pt"
    echo "[EVAL] $tag" | tee -a "$SUMMARY"
    ( time "$ISAACLAB_SH" -p "$EVAL_SCRIPT" \
        --csv "$data" --checkpoint "$ckpt" \
        --batch_size "$BATCH_SIZE" --device cuda \
    ) 2>&1 | tee "$eval_log"

    # Summary extract (with || true so empty grep doesn't kill the script)
    { grep -E "Mean position error|RMSE position error|Mean orientation error|RMSE orientation error|Total MSE" "$eval_log" 2>/dev/null || true; } | tail -10 | sed 's/^/  /' | tee -a "$SUMMARY"
    echo "  end:   $(date)" | tee -a "$SUMMARY"
    echo | tee -a "$SUMMARY"
}

# 5-DoF: skip seed 42 (already done)
run_one 5 "$DATA_5" 1
run_one 5 "$DATA_5" 2

# 6-DoF: full n=3
for SEED in 42 1 2; do run_one 6 "$DATA_6" "$SEED"; done

# 7-DoF: full n=3
for SEED in 42 1 2; do run_one 7 "$DATA_7" "$SEED"; done

echo "[$(date)] === RESUME complete ===" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"
echo "Aggregating now..." | tee -a "$SUMMARY"
python3 "$ROOT/tier4_runs/aggregate_singletask_multiseed.py" "$OUTROOT" 2>&1 | tee -a "$SUMMARY"
