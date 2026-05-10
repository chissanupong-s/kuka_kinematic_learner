#!/usr/bin/env bash
# expH: 7-DoF Stage-3 adaptation n=3 on part000 (seeds 42 1 2, steps=40k)
#
# This script does NOT auto-wait for other GPU jobs. Launch it in tmux only
# after part000_test (seed 11) has finished. Verify with `nvidia-smi`.
set -euo pipefail

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SHARED_CKPT="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/runs/multitask/seperate_weight/1/fk_iiwa_5_6_7/multitask_fk_best.pt"
DATA_7_LARGE="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/data/narrowed/7DOF_15deg/7DOF_15deg_part000.pt"
ADAPT_SCRIPT="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/adapt_multitask_newest.py"
ISAACLAB_SH="/home/ubuntu/IsaacLab/isaaclab.sh"

OUTDIR="$(dirname "$ADAPT_SCRIPT")/tier4_runs/expH_adapt_7dof_n3_part000_${TIMESTAMP}"
LOGDIR="$OUTDIR/logs"
TBROOT="$OUTDIR/tb"
SAVEROOT="$OUTDIR/ckpts"
mkdir -p "$LOGDIR" "$TBROOT" "$SAVEROOT"

SUMMARY="$OUTDIR/summary.txt"
echo "Stage-3 7-DoF adaptation, n=3 on part000 (50M samples)" | tee "$SUMMARY"
echo "ckpt: $SHARED_CKPT" | tee -a "$SUMMARY"
echo "data: $DATA_7_LARGE" | tee -a "$SUMMARY"
echo "started: $(date)" | tee -a "$SUMMARY"
echo | tee -a "$SUMMARY"

run_one () {
    local seed="$1"
    local tag="dof7_seed${seed}_part000"
    local logfile="$LOGDIR/adapt_${tag}.log"
    local save_dir="$SAVEROOT/${tag}"
    mkdir -p "$save_dir"

    echo "=== RUN $tag ===" | tee -a "$SUMMARY"
    echo "  log: $logfile" | tee -a "$SUMMARY"
    echo "  start: $(date)" | tee -a "$SUMMARY"

    ( time "$ISAACLAB_SH" -p "$ADAPT_SCRIPT" \
        --ckpt "$SHARED_CKPT" --mode fk --device cuda --dof 7 --data "$DATA_7_LARGE" \
        --support_size 50000 --query_size 2000000 --adapt_steps 40000 \
        --inner_lr 1e-6 --batch_size 8192 --query_batch_size 8192 --num_workers 8 \
        --adapt all --l2_reg 1e-6 --grad_clip 1.0 --adam_eps 1e-7 --std_floor_q_deg 1.0 \
        --enable_tf32 --log_every 5000 --eval_every 1000 --eval_steps_list "" \
        --pos_weight 1.0 --ori_weight 0.05 --score_pos_w 1.0 --score_ori_w 0.01 --eval_pbar \
        --tb_logdir "$TBROOT" --tb_name "$tag" --save_dir "$save_dir" --seed "$seed" \
    ) 2>&1 | tee "$logfile"

    best="$(grep -aE '^\[INFO\] BEST' "$logfile" | tail -1 || true)"
    echo "  $best" | tee -a "$SUMMARY"
    echo "  end:   $(date)" | tee -a "$SUMMARY"
    echo | tee -a "$SUMMARY"
}

# Seeds match the report's adapted-row protocol (42, 1, 2).
# Run cheap-to-relaunch order: 42 first so the headline is available even
# if the script is interrupted.
for SEED in 42 1 2; do
    run_one "$SEED"
done

echo | tee -a "$SUMMARY"
echo "finished: $(date)" | tee -a "$SUMMARY"
echo "DONE. Outputs in: $OUTDIR" | tee -a "$SUMMARY"
echo | tee -a "$SUMMARY"
echo "Aggregate with:" | tee -a "$SUMMARY"
echo "  python3 $(dirname "$0")/aggregate_adapt_7dof_n3.py $OUTDIR" | tee -a "$SUMMARY"
