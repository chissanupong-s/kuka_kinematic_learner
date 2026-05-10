#!/usr/bin/env bash
# expK: Re-run seed 42 of Stage-3 adaptation for 5-DoF and 6-DoF with the
# expB protocol, so Table 5.1's Adapted (best) row has consistent full-
# precision n=3 provenance for ALL three DoF configurations.
#
# Why: the report's existing 5/6-DoF Adapted means were computed from data
# that no longer exists locally (seed 42's raw checkpoints/logs were in
# runs/adapt_fk_weighted_*, deleted in the disk cleanup). Without seed 42's
# full-precision values we cannot show 5/6-DoF means with the same precision
# as the new 7-DoF expH numbers (9.901 ± 0.014 mm). This script re-creates
# seed 42 for 5/6-DoF using the same hyperparameters as expB seeds 1 and 2,
# so the new Table 5.1 cells have provenance:
#   "n=3, seeds 42, 1, 2 — same protocol throughout".
#
# Hyperparameters match expB exactly:
#   5-DoF: lr=1e-5, bs=2048, adapt_steps=30000, ori_w=0.30, l2=1e-6
#   6-DoF: lr=1e-6, bs=2048, adapt_steps=30000, ori_w=0.30, l2=1e-6
# Both: support=50000, query=2000000, query_batch=8192.
#
# Time: ~10-15 min/seed on a clean GPU; ~25 min total for both. Will share
# the GPU with the running expJ (slight slowdown, expected — both will
# still complete).
set -euo pipefail

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ROOT=/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train
SHARED_CKPT="$ROOT/runs/multitask/seperate_weight/1/fk_iiwa_5_6_7/multitask_fk_best.pt"
DATA_5="$ROOT/../../data/narrowed/5DOF_8deg.pt_part000.pt"
DATA_6="$ROOT/../../data/narrowed/6DOF_12deg.pt_part000.pt"
ADAPT_SCRIPT="$ROOT/adapt_multitask_newest.py"
ISAACLAB_SH="/home/ubuntu/IsaacLab/isaaclab.sh"

OUTROOT="$ROOT/tier4_runs/expK_adapt_5_6dof_seed42_${TIMESTAMP}"
LOGDIR="$OUTROOT/logs"
TBROOT="$OUTROOT/tb"
SAVEROOT="$OUTROOT/ckpts"
mkdir -p "$LOGDIR" "$TBROOT" "$SAVEROOT"

SUMMARY="$OUTROOT/summary.txt"

echo "[$(date)] === expK === seed 42 adaptation for 5/6-DoF (expB protocol)" | tee "$SUMMARY"
echo "  ckpt: $SHARED_CKPT" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

run_one () {
    local dof="$1"
    local data="$2"
    local lr="$3"
    local bs="$4"
    local steps="$5"
    local ori_w="$6"
    local l2="$7"
    local seed=42
    local tag="seed${seed}_dof${dof}"
    local logfile="$LOGDIR/adapt_${tag}.log"
    local save_dir="$SAVEROOT/${tag}"
    mkdir -p "$save_dir"

    echo "============================================================" | tee -a "$SUMMARY"
    echo "[RUN] $tag (lr=$lr bs=$bs steps=$steps ori_w=$ori_w l2=$l2)" | tee -a "$SUMMARY"
    echo "  start: $(date)" | tee -a "$SUMMARY"
    echo "============================================================" | tee -a "$SUMMARY"

    ( time "$ISAACLAB_SH" -p "$ADAPT_SCRIPT" \
        --ckpt "$SHARED_CKPT" --mode fk --device cuda --dof "$dof" --data "$data" \
        --support_size 50000 --query_size 2000000 --adapt_steps "$steps" \
        --inner_lr "$lr" --batch_size "$bs" --query_batch_size 8192 --num_workers 8 \
        --adapt all --l2_reg "$l2" --grad_clip 1.0 --adam_eps 1e-7 --std_floor_q_deg 1.0 \
        --enable_tf32 --log_every 5000 --eval_every 1000 --eval_steps_list "" \
        --pos_weight 1.0 --ori_weight "$ori_w" --score_pos_w 1.0 --score_ori_w 0.01 --eval_pbar \
        --tb_logdir "$TBROOT" --tb_name "$tag" --save_dir "$save_dir" --seed "$seed" \
    ) 2>&1 | tee "$logfile"

    best="$(grep -aE '^\[INFO\] BEST' "$logfile" | tail -1 || true)"
    echo "  $best" | tee -a "$SUMMARY"
    echo "  end:   $(date)" | tee -a "$SUMMARY"
    echo | tee -a "$SUMMARY"
}

# Match expB exactly
run_one 5 "$DATA_5" "1e-5" "2048" "30000" "0.30" "1e-6"
run_one 6 "$DATA_6" "1e-6" "2048" "30000" "0.30" "1e-6"

echo "[$(date)] === expK complete ===" | tee -a "$SUMMARY"
