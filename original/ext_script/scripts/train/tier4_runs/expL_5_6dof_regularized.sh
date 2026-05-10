#!/usr/bin/env bash
# expL — REMOTE MACHINE half: 5-DoF and 6-DoF only (n=3 each = 6 runs total).
# Mirrors expL_singletask_regularized.sh exactly except for which DoFs run.
# 7-DoF runs on the primary machine in parallel; results are merged by hand.
#
# Hyperparameters MUST match the 7-DoF run on the primary machine for the
# Single-task row of Table 5.1 to be comparable across DoFs:
#   epochs=300, batch=4096, lr=5e-4, hidden=1024, blocks=8,
#   weight_decay=1e-4, dropout=0.2, early_stop_patience=15,
#   max_samples=15M, train_frac=0.7, val_frac=0.1, seeds 42/1/2.
#
# Time estimate: ~2.5-3 hr/seed × 6 seeds = ~15-18 hr; early-stop may cut to ~12-14 hr.
set -euo pipefail

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ROOT=/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train
DATA_5="$ROOT/../../data/narrowed/5DOF_8deg.pt_part000.pt"
DATA_6="$ROOT/../../data/narrowed/6DOF_12deg.pt_part000.pt"
SCRIPT="$ROOT/train_kinematics_nn_pol_pt_generalize.py"
EVAL_SCRIPT="$ROOT/eval_model_single_task.py"
ISAACLAB_SH="/home/ubuntu/IsaacLab/isaaclab.sh"

OUTROOT="$ROOT/tier4_runs/expL_5_6dof_regularized_${TIMESTAMP}"
LOGDIR="$OUTROOT/logs"
mkdir -p "$LOGDIR"
SUMMARY="$OUTROOT/summary.txt"

EPOCHS=200
BATCH_SIZE=4096
LR=2.5e-5                # FIXED LR — matches the primary machine's expL
                         # (no scheduler decay; mid-low rate held throughout).
                         # Lowered from 3e-5 after early-stop fired too soon
                         # at patience=15.
HIDDEN_DIM=1024
NUM_BLOCKS=8
WEIGHT_DECAY=1e-4
DROPOUT=0.2
EARLY_STOP_PATIENCE=50
SCHED_PATIENCE=999       # effectively disables scheduler at 300-epoch cap
GRAD_CLIP=1.0
TRAIN_FRAC=0.7
VAL_FRAC=0.1
MAX_SAMPLES=15000000

echo "[$(date)] === expL 5/6-DoF (remote machine) === Regularized single-task" | tee "$SUMMARY"
echo "  reg: dropout=$DROPOUT, weight_decay=$WEIGHT_DECAY, early_stop=$EARLY_STOP_PATIENCE" | tee -a "$SUMMARY"
echo "  data: per-DoF primary single distribution, capped at ${MAX_SAMPLES} samples" | tee -a "$SUMMARY"
echo "  this script runs in parallel with 7-DoF on the primary machine." | tee -a "$SUMMARY"
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

# 5-DoF first (slightly cheaper than 6-DoF — gets done sooner so the user can
# inspect partial results faster on the remote machine).
echo "[$(date)] --- 5-DoF block ---" | tee -a "$SUMMARY"
for SEED in 42 1 2; do run_one 5 "$DATA_5" "$SEED"; done

# 6-DoF
echo "[$(date)] --- 6-DoF block ---" | tee -a "$SUMMARY"
for SEED in 42 1 2; do run_one 6 "$DATA_6" "$SEED"; done

echo "[$(date)] === expL 5/6-DoF complete ===" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"
echo "Aggregating..." | tee -a "$SUMMARY"
python3 "$ROOT/tier4_runs/aggregate_singletask_multiseed.py" "$OUTROOT" 2>&1 | tee -a "$SUMMARY"
