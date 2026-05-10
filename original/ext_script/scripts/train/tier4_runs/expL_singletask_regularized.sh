#!/usr/bin/env bash
# expL: Regularized single-task baselines for 5/6/7-DoF, n=3 each.
#
# Goal: produce Table 5.1 Single-task entries that don't suffer from the
# overfitting seen in the original expG runs. Each DoF is trained on its
# primary single-distribution dataset (matching Table 4.1 — no methodology
# change), but with the regularization techniques from expJ (dropout=0.2,
# weight_decay=1e-4, early stopping). This removes the late-epoch overfit
# tail without changing the data composition, so the resulting numbers
# are directly comparable to the Adapted (best) row.
#
# Order: 7-DoF first (highest priority), then 5-DoF, then 6-DoF. If the
# script is interrupted partway, 7-DoF will already be complete.
#
# Time estimate: ~2.5-3 hr/seed × 9 = ~22-27 hr if no early-stop fires; early-stop
# (patience=15) typically cuts each run to 60-80% of the cap once val plateaus.
# epochs=300 used (instead of 200) because expJ seed 42 was still saving best at
# epoch 199/200, indicating the model needs headroom to verify convergence.
set -euo pipefail

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ROOT=/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train
DATA_5="$ROOT/../../data/narrowed/5DOF_8deg.pt_part000.pt"
DATA_6="$ROOT/../../data/narrowed/6DOF_12deg.pt_part000.pt"
DATA_7="$ROOT/../../data/narrowed/7DOF_15deg/7DOF_15deg_part000.pt"
SCRIPT="$ROOT/train_kinematics_nn_pol_pt_generalize.py"
EVAL_SCRIPT="$ROOT/eval_model_single_task.py"
ISAACLAB_SH="/home/ubuntu/IsaacLab/isaaclab.sh"

OUTROOT="$ROOT/tier4_runs/expL_singletask_regularized_${TIMESTAMP}"
LOGDIR="$OUTROOT/logs"
mkdir -p "$LOGDIR"
SUMMARY="$OUTROOT/summary.txt"

# Hyperparameters
EPOCHS=300
BATCH_SIZE=4096
LR=5e-4
HIDDEN_DIM=1024
NUM_BLOCKS=8
WEIGHT_DECAY=1e-4
DROPOUT=0.2
EARLY_STOP_PATIENCE=15
SCHED_PATIENCE=10
GRAD_CLIP=1.0
TRAIN_FRAC=0.7
VAL_FRAC=0.1
MAX_SAMPLES=15000000

echo "[$(date)] === expL === Regularized single-task baselines (n=3 per DoF)" | tee "$SUMMARY"
echo "  reg: dropout=$DROPOUT, weight_decay=$WEIGHT_DECAY, early_stop=$EARLY_STOP_PATIENCE" | tee -a "$SUMMARY"
echo "  data: per-DoF primary single distribution, capped at ${MAX_SAMPLES} samples" | tee -a "$SUMMARY"
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

    if [ -f "$run_root/tb/fk/fk_pose_best.pt" ] && [ -f "$eval_log" ] \
       && grep -q "Mean position error" "$eval_log" 2>/dev/null; then
        echo "[SKIP] $tag (already complete)" | tee -a "$SUMMARY"
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
        --weight_decay "$WEIGHT_DECAY" --dropout "$DROPOUT" \
        --early_stopping_patience "$EARLY_STOP_PATIENCE" \
        --scheduler_patience "$SCHED_PATIENCE" --grad_clip "$GRAD_CLIP" \
        --num_workers 8 --device cuda \
        --out_dir "$out_dir" --log_dir "$train_logdir" \
        --seed "$seed" --max_samples "$MAX_SAMPLES" \
    ) 2>&1 | tee "$train_log"

    local ckpt="$train_logdir/fk/fk_pose_best.pt"
    echo "[EVAL] $tag" | tee -a "$SUMMARY"
    ( time "$ISAACLAB_SH" -p "$EVAL_SCRIPT" \
        --csv "$data" --checkpoint "$ckpt" \
        --batch_size "$BATCH_SIZE" --device cuda \
    ) 2>&1 | tee "$eval_log"

    { grep -E "Mean position error|RMSE position error|Mean orientation error|RMSE orientation error|Total MSE" "$eval_log" 2>/dev/null || true; } | tail -10 | sed 's/^/  /' | tee -a "$SUMMARY"
    echo "  end:   $(date)" | tee -a "$SUMMARY"
    echo | tee -a "$SUMMARY"
}

# THIS MACHINE: 7-DoF only (n=3). 5-DoF and 6-DoF are running on the second
# machine in parallel via expL_5_6dof.sh — see that script for the matching
# protocol. Combined output is aggregated by hand into Table 5.1 once both
# machines finish.
echo "[$(date)] --- 7-DoF block (this machine; 5/6-DoF on remote) ---" | tee -a "$SUMMARY"
for SEED in 42 1 2; do run_one 7 "$DATA_7" "$SEED"; done

echo "[$(date)] === expL complete ===" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"
echo "Aggregating..." | tee -a "$SUMMARY"
python3 "$ROOT/tier4_runs/aggregate_singletask_multiseed.py" "$OUTROOT" 2>&1 | tee -a "$SUMMARY"
