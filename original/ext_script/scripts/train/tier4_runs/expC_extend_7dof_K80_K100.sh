#!/usr/bin/env bash
# Extend 7-DoF data-size sweep with K=80000 and K=100000
# Same configuration as the main expC sweep so points are directly comparable.
set -euo pipefail

TIMESTAMP="20260507_045433"
SHARED_CKPT="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/runs/multitask/seperate_weight/1/fk_iiwa_5_6_7/multitask_fk_best.pt"
DATA_7="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/data/narrowed/7DOF_15deg/7DOF_15deg_part001.pt"

ADAPT_SCRIPT="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/adapt_multitask_newest.py"
ISAACLAB_SH="/home/ubuntu/IsaacLab/isaaclab.sh"

OUTDIR="$(dirname "$ADAPT_SCRIPT")/tier4_runs/expC_datasize_sweep_${TIMESTAMP}"
LOGDIR="$OUTDIR/logs"
TBROOT="$OUTDIR/tb"
SAVEROOT="$OUTDIR/ckpts"

SUMMARY="$OUTDIR/summary.txt"
echo "" | tee -a "$SUMMARY"
echo "=========== EXTEND 7-DoF $(date) — K=80k and K=100k @ 30k step budget ===========" | tee -a "$SUMMARY"

run_one_7 () {
    local k="$1"
    local tag="dof7_K${k}"
    local logfile="$LOGDIR/adapt_${tag}.log"
    if [ -f "$logfile" ] && grep -aE "^\[INFO\] BEST" "$logfile" > /dev/null 2>&1; then
        echo "[SKIP] $tag already has BEST line" | tee -a "$SUMMARY"
        return
    fi
    local save_dir="$SAVEROOT/${tag}"
    mkdir -p "$save_dir"

    echo "=== RUN $tag ===" | tee -a "$SUMMARY"
    echo "  start: $(date)" | tee -a "$SUMMARY"

    ARGS=(
        --ckpt "$SHARED_CKPT" --mode fk --device cuda --dof 7 --data "$DATA_7"
        --support_size "$k" --query_size 2000000
        --adapt_steps 30000 --inner_lr 1e-6 --batch_size 8192 --query_batch_size 8192
        --num_workers 8 --adapt all --l2_reg 1e-6 --grad_clip 1.0 --adam_eps 1e-7
        --std_floor_q_deg 1.0 --enable_tf32
        --log_every 5000 --eval_every 1000 --eval_steps_list ""
        --pos_weight 1.0 --ori_weight 0.05 --score_pos_w 1.0 --score_ori_w 0.01
        --eval_pbar --tb_logdir "$TBROOT" --tb_name "$tag" --save_dir "$save_dir"
        --seed 42
    )
    ( time "$ISAACLAB_SH" -p "$ADAPT_SCRIPT" "${ARGS[@]}" ) 2>&1 | tee "$logfile"

    best="$(grep -aE "^\[INFO\] BEST" "$logfile" | tail -1 || true)"
    echo "  $best" | tee -a "$SUMMARY"
    echo "  end:   $(date)" | tee -a "$SUMMARY"
    echo                    | tee -a "$SUMMARY"
}

run_one_7 80000
run_one_7 100000

echo "EXTEND PHASE finished: $(date)" | tee -a "$SUMMARY"
