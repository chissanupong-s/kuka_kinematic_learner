#!/usr/bin/env bash
# Phase 2.1: Multi-seed single-task training (n=3 seeds × 3 DoFs) using the
# 15M-sample cap to match the report's stated dataset cap.
#
# Goal: give the Single-task row of Table 5.1 its own ± std, matching the
# Adapted (best) row's n=3 protocol (seeds 42, 1, 2).
#
# Time estimate (per seed):
#   5-DoF: ~1.7 hr  (15M of 16M, batch 4096, 200 epochs)
#   6-DoF: ~2.1 hr  (15M of 35M)
#   7-DoF: ~6.6 hr  (15M of 50M)
#   Total per seed: ~10.4 hr
#   Total n=3:      ~31 hr ≈ 1.3 days
#
# Auto-waits for current GPU jobs to finish before starting (so it can be
# launched in tmux at any time and will queue politely).
set -euo pipefail

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ROOT=/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train
DATA_5="$ROOT/../../data/narrowed/5DOF_8deg.pt_part000.pt"
DATA_6="$ROOT/../../data/narrowed/6DOF_12deg.pt_part000.pt"
DATA_7="$ROOT/../../data/narrowed/7DOF_15deg/7DOF_15deg_part000.pt"
SCRIPT="$ROOT/train_kinematics_nn_pol_pt_2.py"
EVAL_SCRIPT="$ROOT/eval_model_single_task.py"
ISAACLAB_SH="/home/ubuntu/IsaacLab/isaaclab.sh"

OUTROOT="$ROOT/tier4_runs/expG_singletask_multiseed_15M_${TIMESTAMP}"
LOGDIR="$OUTROOT/logs"
mkdir -p "$LOGDIR"

SUMMARY="$OUTROOT/summary.txt"

# Hyperparameters MATCHING the headline single-task run (per train5_6.sh and run_7_only_1.sh)
EPOCHS=200
BATCH_SIZE=4096
LR=5e-4
HIDDEN_DIM=1024
NUM_BLOCKS=8
WEIGHT_DECAY=1e-5
SCHEDULER_PATIENCE=10
GRAD_CLIP=1.0
TRAIN_FRAC=0.7
VAL_FRAC=0.1   # 0.7 train + 0.1 val + 0.2 test = matches Stage-1 baseline split
MAX_SAMPLES=15000000   # 15M cap

# Wait for GPU to free up. We only block on *other* GPU-heavy jobs we know
# about (adapt_multitask_newest.py). Don't include this script's own training
# binary (train_kinematics_nn_pol_pt_2.py) here — pgrep -f would match the
# child process we just forked and the loop would never exit.
# Note: pgrep uses ERE; OR is plain `|` (NOT `\|`, which matches a literal pipe).
echo "[$(date)] Waiting for any current GPU jobs to finish..." | tee "$SUMMARY"
while pgrep -f "adapt_multitask_newest.py" > /dev/null 2>&1; do
    sleep 60
done
echo "[$(date)] GPU is free. Starting Stage-1 multi-seed training (n=3 × 3 DoFs)" | tee -a "$SUMMARY"
echo "  Sub-sample cap: $MAX_SAMPLES" | tee -a "$SUMMARY"
echo "  Hyperparameters match the report's Stage-1 baseline (epochs=$EPOCHS, hidden=$HIDDEN_DIM, blocks=$NUM_BLOCKS)" | tee -a "$SUMMARY"
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

    # Extract final eval line from the log for the summary.
    # eval_model_single_task.py emits "Mean position error: X m" / "Mean orientation error: Y deg"
    # `|| true` prevents grep's exit-1-when-no-match from killing the script under set -e + pipefail.
    { grep -E "Mean position error|RMSE position error|Mean orientation error|RMSE orientation error|Total MSE|pos_mae_m|pos_rmse_m|ori_deg" "$eval_log" 2>/dev/null || true; } | tail -10 | sed 's/^/  /' | tee -a "$SUMMARY"
    echo "  end:   $(date)" | tee -a "$SUMMARY"
    echo | tee -a "$SUMMARY"
}

# n=3 seeds × 3 DoFs (matches the Adapted (best) row's protocol)
for SEED in 42 1 2; do
    run_one 5 "$DATA_5" "$SEED"
done
for SEED in 42 1 2; do
    run_one 6 "$DATA_6" "$SEED"
done
for SEED in 42 1 2; do
    run_one 7 "$DATA_7" "$SEED"
done

echo "[$(date)] === All Stage-1 multi-seed training complete ===" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"
echo "Output: $OUTROOT" | tee -a "$SUMMARY"
echo "Per-seed best ckpts: $OUTROOT/dof{5,6,7}_seed{42,1,2}/tb/fk/fk_pose_best.pt" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"
echo "Run aggregator after this finishes:" | tee -a "$SUMMARY"
echo "  python3 $ROOT/tier4_runs/aggregate_singletask_multiseed.py $OUTROOT" | tee -a "$SUMMARY"
