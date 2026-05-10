#!/usr/bin/env bash
# expJ: 7-DoF single-task FK trained for GENERALISATION.
#
# Strategy:
#   1. Combine 4 angle ranges into one training pool (5deg + 10deg + 15deg + 20deg
#      part000 files). The model has to learn the FK transform across diverse
#      joint configurations, not memorise one range's specific samples.
#   2. Cap combined pool to 15M samples (~3.75M from each angle, balanced via
#      random sub-sample). Same total cap as the original expG so per-epoch
#      cost is comparable, but with 4x the joint-config diversity.
#   3. Stronger regularisation: dropout=0.2, weight_decay=1e-4 (10x baseline).
#   4. Early stopping with patience=15 (stops once val plateaus, prevents the
#      late-epoch overfit tail seen in expG).
#   5. OOD validation: hold out 7DOF_15deg_part001.pt as a separate distribution
#      to track each epoch — visibility into whether generalisation is still
#      improving when in-distribution val plateaus.
#
# Time estimate: ~1.5 h per seed (15M samples, expects early stop around epoch
# 60-100); ~5 h for n=3 total.
set -euo pipefail

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ROOT=/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train
DATA_DIR="$ROOT/../../data/narrowed"
SCRIPT="$ROOT/train_kinematics_nn_pol_pt_generalize.py"
EVAL_SCRIPT="$ROOT/eval_model_single_task.py"
ISAACLAB_SH="/home/ubuntu/IsaacLab/isaaclab.sh"

# Combine all four angle-range part000 files for training
DATA_TRAIN="$DATA_DIR/7DOF_5deg_part000.pt,$DATA_DIR/7DOF_10deg_part000.pt,$DATA_DIR/7DOF_15deg/7DOF_15deg_part000.pt,$DATA_DIR/7DOF_20deg_part000.pt"

# OOD val (different sub-sample of 15deg)
DATA_OOD="$DATA_DIR/7DOF_15deg/7DOF_15deg_part001.pt"

OUTROOT="$ROOT/tier4_runs/expJ_singletask_7dof_generalize_${TIMESTAMP}"
LOGDIR="$OUTROOT/logs"
mkdir -p "$LOGDIR"
SUMMARY="$OUTROOT/summary.txt"

# Hyperparameters
EPOCHS=200
BATCH_SIZE=4096
LR=5e-4
HIDDEN_DIM=1024
NUM_BLOCKS=8
WEIGHT_DECAY=1e-4         # 10x baseline (was 1e-5)
DROPOUT=0.2               # actually applied (baseline ResBlock had dropout disabled)
EARLY_STOP_PATIENCE=15
TRAIN_FRAC=0.8            # raised from 0.7: more training data, smaller in-dist val
VAL_FRAC=0.1
MAX_SAMPLES=15000000      # same cap as expG, but spread across 4 angle ranges
GRAD_CLIP=1.0
SCHED_PATIENCE=10

echo "[$(date)] === expJ === 7-DoF single-task generalisation training (n=3)" | tee "$SUMMARY"
echo "  train data: 5deg + 10deg + 15deg + 20deg part000 (combined, capped to ${MAX_SAMPLES})" | tee -a "$SUMMARY"
echo "  ood val:    7DOF_15deg_part001.pt" | tee -a "$SUMMARY"
echo "  reg:        dropout=$DROPOUT, weight_decay=$WEIGHT_DECAY, early_stop_patience=$EARLY_STOP_PATIENCE" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

run_one () {
    local seed="$1"
    local tag="dof7_seed${seed}_generalize"
    local run_root="$OUTROOT/$tag"
    local train_log="$LOGDIR/${tag}_train.log"
    local train_logdir="$run_root/tb"
    local out_dir="$run_root/models"

    if [ -f "$run_root/tb/fk/fk_pose_best.pt" ]; then
        echo "[SKIP] $tag (best.pt already exists)" | tee -a "$SUMMARY"
        return
    fi
    mkdir -p "$run_root" "$train_logdir" "$out_dir"

    echo "============================================================" | tee -a "$SUMMARY"
    echo "[TRAIN] $tag" | tee -a "$SUMMARY"
    echo "  start: $(date)" | tee -a "$SUMMARY"
    echo "============================================================" | tee -a "$SUMMARY"

    ( time "$ISAACLAB_SH" -p "$SCRIPT" \
        --csv "$DATA_TRAIN" --ood_csv "$DATA_OOD" --mode fk \
        --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --lr "$LR" \
        --hidden_dim "$HIDDEN_DIM" --num_blocks "$NUM_BLOCKS" \
        --train_frac "$TRAIN_FRAC" --val_frac "$VAL_FRAC" \
        --weight_decay "$WEIGHT_DECAY" --dropout "$DROPOUT" \
        --early_stopping_patience "$EARLY_STOP_PATIENCE" \
        --scheduler_patience "$SCHED_PATIENCE" --grad_clip "$GRAD_CLIP" \
        --num_workers 8 --device cuda \
        --out_dir "$out_dir" --log_dir "$train_logdir" \
        --seed "$seed" --max_samples "$MAX_SAMPLES" \
    ) 2>&1 | tee "$train_log"

    # Evaluate the resulting checkpoint against 4 distributions to characterise generalisation
    local ckpt="$train_logdir/fk/fk_pose_best.pt"
    echo "[EVAL] $tag — running 4-distribution generalisation grid" | tee -a "$SUMMARY"
    for eval_data in \
        "5deg:$DATA_DIR/7DOF_5deg_part000.pt" \
        "10deg:$DATA_DIR/7DOF_10deg_part000.pt" \
        "15deg-part000:$DATA_DIR/7DOF_15deg/7DOF_15deg_part000.pt" \
        "15deg-part001:$DATA_DIR/7DOF_15deg/7DOF_15deg_part001.pt" \
        "20deg:$DATA_DIR/7DOF_20deg_part000.pt"
    do
        local label="${eval_data%%:*}"
        local datapath="${eval_data##*:}"
        local elog="$LOGDIR/${tag}_eval_${label}.log"
        echo "  [eval on $label]" | tee -a "$SUMMARY"
        ( "$ISAACLAB_SH" -p "$EVAL_SCRIPT" \
            --csv "$datapath" --checkpoint "$ckpt" \
            --batch_size "$BATCH_SIZE" --device cuda \
        ) 2>&1 | tee "$elog" > /dev/null
        { grep -E "Mean position error|RMSE position error|Mean orientation error|RMSE orientation error|Total MSE" "$elog" 2>/dev/null || true; } | sed 's/^/    /' | tee -a "$SUMMARY"
    done
    echo "  end:   $(date)" | tee -a "$SUMMARY"
    echo | tee -a "$SUMMARY"
}

for SEED in 42 1 2; do run_one "$SEED"; done

echo "[$(date)] === expJ complete ===" | tee -a "$SUMMARY"
echo "Summary in: $SUMMARY" | tee -a "$SUMMARY"
