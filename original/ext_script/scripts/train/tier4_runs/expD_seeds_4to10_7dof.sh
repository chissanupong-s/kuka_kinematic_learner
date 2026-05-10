#!/usr/bin/env bash
# Run seeds 4-10 of 7-DoF adaptation (re-doing seed 4 since it was killed earlier,
# plus seeds 5-10) to get n=10 multi-seed evaluation matching the seed 1 / seed 2 protocol.
#
# Each seed: K=50000 support, 100000 adaptation steps. Expected time: ~78 minutes per seed.
# Total: 7 seeds × ~78 min = ~9 hours.
#
# Auto-waits for current GPU jobs to finish before starting (tmux-friendly).
set -euo pipefail

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SHARED_CKPT="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/runs/multitask/seperate_weight/1/fk_iiwa_5_6_7/multitask_fk_best.pt"
DATA_7="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/data/narrowed/7DOF_15deg/7DOF_15deg_part001.pt"
ADAPT_SCRIPT="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/adapt_multitask_newest.py"
ISAACLAB_SH="/home/ubuntu/IsaacLab/isaaclab.sh"

OUTDIR="$(dirname "$ADAPT_SCRIPT")/tier4_runs/expD_seeds_4to10_7dof_${TIMESTAMP}"
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
echo "[$(date)] GPU is free. Starting seeds 4-10 for 7-DoF (n=10 multi-seed evaluation)." | tee -a "$SUMMARY"

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

# Run seeds 4 through 10
for seed in 4 5 6 7 8 9 10; do
    run_seed "$seed"
done

echo "[$(date)] === All seeds 4-10 done. ===" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

# Auto-aggregate n=10 statistics
echo "=== n=10 Aggregate (seeds 42, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10) ===" | tee -a "$SUMMARY"
echo "  Existing seeds 1, 2 logs are at expB_multiseed_smart/logs/" | tee -a "$SUMMARY"
echo "  Existing seeds 3, 4 logs are at expC_seeds34_7dof/logs/ (seed 3 only; seed 4 redone here)" | tee -a "$SUMMARY"
echo "  New seeds 4-10 logs are in this directory's logs/" | tee -a "$SUMMARY"
echo "  Run aggregate_n10_stats.py after this script finishes." | tee -a "$SUMMARY"

echo "[$(date)] Phase-1 (n=10 7-DoF multi-seed) batch complete." | tee -a "$SUMMARY"
