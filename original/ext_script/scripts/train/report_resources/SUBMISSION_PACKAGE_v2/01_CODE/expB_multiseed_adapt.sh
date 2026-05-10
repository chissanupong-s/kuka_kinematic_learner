#!/usr/bin/env bash
# Multi-seed Stage-3 adaptation — SMART VARIANT
# Generated: 2026-05-07 ~05:15.
#
# Run two extra seeds (seed=1 and seed=2) of per-DoF adaptation, holding the
# Stage-2 shared meta-kinematics checkpoint fixed. Combined with the existing
# seed=42 headline runs, this gives n=3 measurements for EACH DoF row of
# Table 5.1, letting "Adapted (best)" report mean ± std.
#
# Hyperparameters per DoF MATCH each existing headline configuration. Step
# budgets are reduced for 5-DoF and 6-DoF based on the convergence evidence
# from Figure 5.6 (both reached within 5% of best at step 4 000, so 10 000
# is 2.5× past convergence — a defensible budget reduction):
#
#   5-DoF:  lr=1e-5, bs=2048, ori_w=0.30, l2=1e-6, steps=30 000  (~5 min/run; within 1.8% of best at this step)
#   6-DoF:  lr=1e-6, bs=2048, ori_w=0.30, l2=1e-6, steps=30 000  (~5 min/run; ~at best at this step)
#   7-DoF:  lr=1e-6, bs=8192, ori_w=0.05, l2=1e-6, steps=100 000 (~55 min/run, full headline)
#
# Order: by-DoF-then-by-seed. We collect a complete set of multi-seed data
# for the cheap DoFs first, so even if the long 7-DoF runs are interrupted
# we still have the 5-DoF and 6-DoF mean ± std values.
set -euo pipefail

TIMESTAMP="20260507_045433"
SHARED_CKPT="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/runs/multitask/seperate_weight/1/fk_iiwa_5_6_7/multitask_fk_best.pt"
DATA_5="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/data/narrowed/5DOF_8deg.pt_part000.pt"
DATA_6="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/data/narrowed/6DOF_12deg.pt_part000.pt"
DATA_7="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/data/narrowed/7DOF_15deg/7DOF_15deg_part001.pt"

ADAPT_SCRIPT="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/adapt_multitask_newest.py"
ISAACLAB_SH="/home/ubuntu/IsaacLab/isaaclab.sh"

OUTDIR="$(dirname "$ADAPT_SCRIPT")/tier4_runs/expB_multiseed_smart_${TIMESTAMP}"
LOGDIR="$OUTDIR/logs"
TBROOT="$OUTDIR/tb"
SAVEROOT="$OUTDIR/ckpts"
mkdir -p "$LOGDIR" "$TBROOT" "$SAVEROOT"

SUMMARY="$OUTDIR/summary.txt"
echo "Multi-seed Stage-3 adaptation — smart variant (all 3 DoFs)"  | tee "$SUMMARY"
echo "ckpt: $SHARED_CKPT"                                            | tee -a "$SUMMARY"
echo "started: $(date)"                                              | tee -a "$SUMMARY"
echo                                                                 | tee -a "$SUMMARY"

run_one () {
    local seed="$1"; local dof="$2"; local data="$3"
    local lr="$4"; local bs="$5"; local steps="$6"; local ori_w="$7"; local l2="$8"

    local tag="seed${seed}_dof${dof}"
    local logfile="$LOGDIR/adapt_${tag}.log"
    local save_dir="$SAVEROOT/${tag}"
    mkdir -p "$save_dir"

    echo "=== RUN $tag (lr=$lr bs=$bs steps=$steps ori_w=$ori_w l2=$l2) ===" | tee -a "$SUMMARY"
    echo "  log: $logfile"                                                  | tee -a "$SUMMARY"
    echo "  start: $(date)"                                                 | tee -a "$SUMMARY"

    ARGS=(
        --ckpt "$SHARED_CKPT"
        --mode fk
        --device cuda
        --dof "$dof"
        --data "$data"
        --support_size 50000
        --query_size 2000000
        --adapt_steps "$steps"
        --inner_lr "$lr"
        --batch_size "$bs"
        --query_batch_size 8192
        --num_workers 8
        --adapt all
        --l2_reg "$l2"
        --grad_clip 1.0
        --adam_eps 1e-7
        --std_floor_q_deg 1.0
        --enable_tf32
        --log_every 5000
        --eval_every 1000
        --eval_steps_list ""
        --pos_weight 1.0
        --ori_weight "$ori_w"
        --score_pos_w 1.0
        --score_ori_w 0.01
        --eval_pbar
        --tb_logdir "$TBROOT"
        --tb_name "$tag"
        --save_dir "$save_dir"
        --seed "$seed"
    )

    ( time "$ISAACLAB_SH" -p "$ADAPT_SCRIPT" "${ARGS[@]}" ) 2>&1 | tee "$logfile"

    best="$(grep -aE "^\[INFO\] BEST" "$logfile" | tail -1 || true)"
    echo "  $best"                                                          | tee -a "$SUMMARY"
    echo "  end: $(date)"                                                   | tee -a "$SUMMARY"
    echo                                                                    | tee -a "$SUMMARY"
}

# ===== 5-DoF (cheap, both seeds first) =====
run_one 1 5 "$DATA_5" "1e-5" "2048" "30000" "0.30" "1e-6"
run_one 2 5 "$DATA_5" "1e-5" "2048" "30000" "0.30" "1e-6"

# ===== 6-DoF (cheap, both seeds) =====
run_one 1 6 "$DATA_6" "1e-6" "2048" "30000" "0.30" "1e-6"
run_one 2 6 "$DATA_6" "1e-6" "2048" "30000" "0.30" "1e-6"

# ===== 7-DoF (heavy, both seeds; full 100k headline budget) =====
run_one 1 7 "$DATA_7" "1e-6" "8192" "100000" "0.05" "1e-6"
run_one 2 7 "$DATA_7" "1e-6" "8192" "100000" "0.05" "1e-6"

echo                                                                        | tee -a "$SUMMARY"
echo "finished: $(date)"                                                    | tee -a "$SUMMARY"
echo "DONE. Outputs in: $OUTDIR"                                            | tee -a "$SUMMARY"
