#!/usr/bin/env bash
# Tier-4 / Experiment C — data-efficiency sweep for adaptation
# Generated: 2026-05-07 ~09:48
#
# Sweep support_size K ∈ {1000, 5000, 10000, 15000, ..., 60000} (1k start, then
# +5k each step up to 60k) for 5-DoF, 6-DoF and 7-DoF adaptation, holding all
# other hyperparameters fixed at their respective headline configurations.
# Single seed (42). 13 K values × 3 DoFs = 39 runs.
#
# Per-DoF protocol — equal compute budget (30k steps) across all DoFs so the
# data-efficiency curves are directly comparable. 7-DoF was reduced from its
# 100k-step headline budget to keep total wall-clock under 8 hr.
#   5-DoF:  lr=1e-5, bs=2048, ori_w=0.30, l2=1e-6, steps=30000
#   6-DoF:  lr=1e-6, bs=2048, ori_w=0.30, l2=1e-6, steps=30000
#   7-DoF:  lr=1e-6, bs=8192, ori_w=0.05, l2=1e-6, steps=30000
#
# Run order: by-DoF-then-by-K (cheapest first), so 5/6-DoF complete in <90 min
# even if the long 7-DoF runs are interrupted.
set -euo pipefail

TIMESTAMP="20260507_045433"
SHARED_CKPT="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/runs/multitask/seperate_weight/1/fk_iiwa_5_6_7/multitask_fk_best.pt"
DATA_5="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/data/narrowed/5DOF_8deg.pt_part000.pt"
DATA_6="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/data/narrowed/6DOF_12deg.pt_part000.pt"
DATA_7="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/data/narrowed/7DOF_15deg/7DOF_15deg_part001.pt"

ADAPT_SCRIPT="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/adapt_multitask_newest.py"
ISAACLAB_SH="/home/ubuntu/IsaacLab/isaaclab.sh"

OUTDIR="$(dirname "$ADAPT_SCRIPT")/tier4_runs/expC_datasize_sweep_${TIMESTAMP}"
LOGDIR="$OUTDIR/logs"
TBROOT="$OUTDIR/tb"
SAVEROOT="$OUTDIR/ckpts"
mkdir -p "$LOGDIR" "$TBROOT" "$SAVEROOT"

SUMMARY="$OUTDIR/summary.txt"
echo "Data-size sweep — 5/6/7-DoF, K ∈ {2000, 5000, 10000, 20000, 40000, 60000}"  | tee "$SUMMARY"
echo "ckpt: $SHARED_CKPT"                                                          | tee -a "$SUMMARY"
echo "started: $(date)"                                                            | tee -a "$SUMMARY"
echo                                                                               | tee -a "$SUMMARY"

K_VALUES=(1000 5000 10000 15000 20000 25000 30000 35000 40000 45000 50000 55000 60000)

run_one () {
    local dof="$1"; local data="$2"
    local lr="$3"; local bs="$4"; local steps="$5"; local ori_w="$6"
    local k="$7"
    local tag="dof${dof}_K${k}"
    local logfile="$LOGDIR/adapt_${tag}.log"
    local save_dir="$SAVEROOT/${tag}"
    mkdir -p "$save_dir"

    echo "=== RUN $tag (lr=$lr bs=$bs steps=$steps ori_w=$ori_w K=$k) ===" | tee -a "$SUMMARY"
    echo "  start: $(date)"                                                | tee -a "$SUMMARY"

    ARGS=(
        --ckpt "$SHARED_CKPT"
        --mode fk
        --device cuda
        --dof "$dof"
        --data "$data"
        --support_size "$k"
        --query_size 2000000
        --adapt_steps "$steps"
        --inner_lr "$lr"
        --batch_size "$bs"
        --query_batch_size 8192
        --num_workers 8
        --adapt all
        --l2_reg 1e-6
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
        --seed 42
    )

    ( time "$ISAACLAB_SH" -p "$ADAPT_SCRIPT" "${ARGS[@]}" ) 2>&1 | tee "$logfile"

    best="$(grep -aE "^\[INFO\] BEST" "$logfile" | tail -1 || true)"
    echo "  $best"                                                         | tee -a "$SUMMARY"
    echo "  end:   $(date)"                                                | tee -a "$SUMMARY"
    echo                                                                   | tee -a "$SUMMARY"
}

# Cheapest first: 5-DoF and 6-DoF at 30k steps
for k in "${K_VALUES[@]}"; do
    run_one 5 "$DATA_5" "1e-5" "2048" "30000" "0.30" "$k"
done
for k in "${K_VALUES[@]}"; do
    run_one 6 "$DATA_6" "1e-6" "2048" "30000" "0.30" "$k"
done

# 7-DoF at 30k steps (equal-compute-budget data-efficiency curve)
for k in "${K_VALUES[@]}"; do
    run_one 7 "$DATA_7" "1e-6" "8192" "30000" "0.05" "$k"
done

echo                                                                       | tee -a "$SUMMARY"
echo "finished: $(date)"                                                   | tee -a "$SUMMARY"
echo "DONE. Outputs in: $OUTDIR"                                           | tee -a "$SUMMARY"
