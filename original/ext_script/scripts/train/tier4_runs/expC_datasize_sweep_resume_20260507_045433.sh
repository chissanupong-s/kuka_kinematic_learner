#!/usr/bin/env bash
# Tier-4 / Experiment C — RESUMED with smart per-DoF step budget
# Generated: 2026-05-07 ~10:35
#
# 5-DoF runs K=1000…20000 already complete at 30k steps (preserved).
# This script picks up from K=25000 for 5-DoF, runs 6-DoF and 7-DoF.
#
# Step budget per DoF (justified by multi-seed data — BEST always reached
# well before step budget for 5/6-DoF):
#   5-DoF (remaining K):  10 000 steps (BEST always at ~step 2 000)
#   6-DoF (all K):        10 000 steps (BEST at ~step 2 000–18 000 in multi-seed)
#   7-DoF (all K):        30 000 steps (model still descending at step 30k)
set -euo pipefail

TIMESTAMP="20260507_045433"
SHARED_CKPT="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/runs/multitask/seperate_weight/1/fk_iiwa_5_6_7/multitask_fk_best.pt"
DATA_5="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/data/narrowed/5DOF_8deg.pt_part000.pt"
DATA_6="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/data/narrowed/6DOF_12deg.pt_part000.pt"
DATA_7="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/data/narrowed/7DOF_15deg/7DOF_15deg_part001.pt"

ADAPT_SCRIPT="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/adapt_multitask_newest.py"
ISAACLAB_SH="/home/ubuntu/IsaacLab/isaaclab.sh"

# Use the SAME output dir so we keep the already-completed 5-DoF runs
OUTDIR="$(dirname "$ADAPT_SCRIPT")/tier4_runs/expC_datasize_sweep_${TIMESTAMP}"
LOGDIR="$OUTDIR/logs"
TBROOT="$OUTDIR/tb"
SAVEROOT="$OUTDIR/ckpts"
mkdir -p "$LOGDIR" "$TBROOT" "$SAVEROOT"

SUMMARY="$OUTDIR/summary.txt"
echo "" | tee -a "$SUMMARY"
echo "=========== RESUMED $(date) — smart per-DoF step budget ===========" | tee -a "$SUMMARY"

K_VALUES_REMAINING_5DOF=(25000 30000 35000 40000 45000 50000 55000 60000)
K_VALUES_FULL=(1000 5000 10000 15000 20000 25000 30000 35000 40000 45000 50000 55000 60000)

run_one () {
    local dof="$1"; local data="$2"
    local lr="$3"; local bs="$4"; local steps="$5"; local ori_w="$6"
    local k="$7"
    local tag="dof${dof}_K${k}"
    local logfile="$LOGDIR/adapt_${tag}.log"
    # If log already exists with a BEST line, skip
    if [ -f "$logfile" ]; then
        if grep -aE "^\[INFO\] BEST" "$logfile" > /dev/null 2>&1; then
            echo "[SKIP] $tag already has BEST line — preserving existing result" | tee -a "$SUMMARY"
            return
        fi
    fi
    local save_dir="$SAVEROOT/${tag}"
    mkdir -p "$save_dir"

    echo "=== RUN $tag (lr=$lr bs=$bs steps=$steps ori_w=$ori_w K=$k) ===" | tee -a "$SUMMARY"
    echo "  start: $(date)"                                                | tee -a "$SUMMARY"

    ARGS=(
        --ckpt "$SHARED_CKPT"
        --mode fk --device cuda --dof "$dof" --data "$data"
        --support_size "$k" --query_size 2000000
        --adapt_steps "$steps" --inner_lr "$lr"
        --batch_size "$bs" --query_batch_size 8192 --num_workers 8
        --adapt all --l2_reg 1e-6 --grad_clip 1.0 --adam_eps 1e-7 --std_floor_q_deg 1.0
        --enable_tf32 --log_every 5000 --eval_every 1000 --eval_steps_list ""
        --pos_weight 1.0 --ori_weight "$ori_w" --score_pos_w 1.0 --score_ori_w 0.01
        --eval_pbar --tb_logdir "$TBROOT" --tb_name "$tag" --save_dir "$save_dir"
        --seed 42
    )

    ( time "$ISAACLAB_SH" -p "$ADAPT_SCRIPT" "${ARGS[@]}" ) 2>&1 | tee "$logfile"

    best="$(grep -aE "^\[INFO\] BEST" "$logfile" | tail -1 || true)"
    echo "  $best"                                                         | tee -a "$SUMMARY"
    echo "  end:   $(date)"                                                | tee -a "$SUMMARY"
    echo                                                                   | tee -a "$SUMMARY"
}

# 5-DoF: only remaining K values at 10k steps (the already-completed 5/15k @ 30k steps stay)
for k in "${K_VALUES_REMAINING_5DOF[@]}"; do
    run_one 5 "$DATA_5" "1e-5" "2048" "10000" "0.30" "$k"
done

# 6-DoF: all K at 10k
for k in "${K_VALUES_FULL[@]}"; do
    run_one 6 "$DATA_6" "1e-6" "2048" "10000" "0.30" "$k"
done

# 7-DoF: all K at 30k (Option B — small-K runs converge well before 30k
# (data-limited regime); large-K runs end ~23% above the 100k-step asymptote
# but the data-efficiency curve SHAPE is preserved. Going to 50k+ does not
# change the headline finding "K=1000 is data-limited for 7-DoF" but costs
# significant extra compute. Curve interpretation in §5.5 emphasises shape.)
for k in "${K_VALUES_FULL[@]}"; do
    run_one 7 "$DATA_7" "1e-6" "8192" "30000" "0.05" "$k"
done

echo                                                                       | tee -a "$SUMMARY"
echo "RESUMED PHASE finished: $(date)"                                     | tee -a "$SUMMARY"
echo "DONE. Outputs in: $OUTDIR"                                           | tee -a "$SUMMARY"
