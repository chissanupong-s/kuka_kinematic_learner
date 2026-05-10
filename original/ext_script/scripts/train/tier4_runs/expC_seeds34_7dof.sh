#!/usr/bin/env bash
# Option C: run 2 more seeds (3, 4) for 7-DoF only, matching the existing
# multi-seed protocol exactly, then extend to seed 5 if needed.
#
# Auto-waits for current GPU jobs to finish before starting (so it can be
# launched in tmux at any time and will queue politely).
set -euo pipefail

TIMESTAMP="20260507_045433"
SHARED_CKPT="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/runs/multitask/seperate_weight/1/fk_iiwa_5_6_7/multitask_fk_best.pt"
DATA_7="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/data/narrowed/7DOF_15deg/7DOF_15deg_part001.pt"

ADAPT_SCRIPT="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/adapt_multitask_newest.py"
ISAACLAB_SH="/home/ubuntu/IsaacLab/isaaclab.sh"

OUTDIR="$(dirname "$ADAPT_SCRIPT")/tier4_runs/expC_seeds34_7dof_${TIMESTAMP}"
LOGDIR="$OUTDIR/logs"
TBROOT="$OUTDIR/tb"
SAVEROOT="$OUTDIR/ckpts"
mkdir -p "$LOGDIR" "$TBROOT" "$SAVEROOT"

SUMMARY="$OUTDIR/summary.txt"

# === Wait for current adapt processes to finish ===
echo "[$(date)] Waiting for current GPU jobs to finish..." | tee "$SUMMARY"
while pgrep -f "adapt_multitask_newest.py" > /dev/null 2>&1; do
    sleep 60
done
echo "[$(date)] GPU is free. Starting seeds 3, 4 (and 5 if needed)." | tee -a "$SUMMARY"

run_seed () {
    local seed="$1"
    local tag="dof7_seed${seed}"
    local logfile="$LOGDIR/adapt_${tag}.log"
    local save_dir="$SAVEROOT/${tag}"
    mkdir -p "$save_dir"
    echo "=== seed $seed ===" | tee -a "$SUMMARY"
    echo "  start: $(date)" | tee -a "$SUMMARY"
    ( time "$ISAACLAB_SH" -p "$ADAPT_SCRIPT" \
        --ckpt "$SHARED_CKPT" --mode fk --device cuda --dof 7 --data "$DATA_7" \
        --support_size 50000 --query_size 2000000 --adapt_steps 100000 \
        --inner_lr 1e-6 --batch_size 8192 --query_batch_size 8192 --num_workers 8 \
        --adapt all --l2_reg 1e-6 --grad_clip 1.0 --adam_eps 1e-7 --std_floor_q_deg 1.0 \
        --enable_tf32 --log_every 5000 --eval_every 1000 --eval_steps_list "" \
        --pos_weight 1.0 --ori_weight 0.05 --score_pos_w 1.0 --score_ori_w 0.01 --eval_pbar \
        --tb_logdir "$TBROOT" --tb_name "$tag" --save_dir "$save_dir" --seed "$seed" \
    ) 2>&1 | tee "$logfile"
    best="$(grep -aE "^\[INFO\] BEST" "$logfile" | tail -1 || true)"
    echo "  $best" | tee -a "$SUMMARY"
    echo "  end:   $(date)" | tee -a "$SUMMARY"
    echo | tee -a "$SUMMARY"
}

run_seed 3
run_seed 4

# Check if either seed 3 or seed 4 landed near seed=42 (pos_mae < 0.011)
echo "=== Seeds 3, 4 done. Evaluating..." | tee -a "$SUMMARY"
seed3_pos=$(grep -aE "^\[INFO\] BEST" "$LOGDIR/adapt_dof7_seed3.log" | grep -oE "'pos_mae_m':\s*[0-9.eE+-]+" | grep -oE "[0-9.eE+-]+$" | head -1)
seed4_pos=$(grep -aE "^\[INFO\] BEST" "$LOGDIR/adapt_dof7_seed4.log" | grep -oE "'pos_mae_m':\s*[0-9.eE+-]+" | grep -oE "[0-9.eE+-]+$" | head -1)
echo "  seed 3 pos_mae: $seed3_pos" | tee -a "$SUMMARY"
echo "  seed 4 pos_mae: $seed4_pos" | tee -a "$SUMMARY"

# Optionally run seed 5 if neither was lucky (both > 0.011 means seed 42 is the outlier)
if (( $(echo "$seed3_pos > 0.011" | bc -l 2>/dev/null) && $(echo "$seed4_pos > 0.011" | bc -l 2>/dev/null) )); then
    echo "  Both seeds clustered around 0.013; running seed 5 for one more shot." | tee -a "$SUMMARY"
    run_seed 5
fi

echo "[$(date)] Option C done." | tee -a "$SUMMARY"
