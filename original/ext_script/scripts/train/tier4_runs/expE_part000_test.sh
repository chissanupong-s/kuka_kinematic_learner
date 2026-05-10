#!/usr/bin/env bash
# Sanity test: run ONE 7-DoF adaptation seed on part000 (50M samples) instead of
# part001 (3.85M). Identical hyperparameters and seed protocol as expD; the ONLY
# change is the --data flag.
#
# Goal: rule out (or confirm) "dataset is the cause of high error" hypothesis.
#
# Expected runtime: ~78 minutes for one seed.
set -euo pipefail

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SHARED_CKPT="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/runs/multitask/seperate_weight/1/fk_iiwa_5_6_7/multitask_fk_best.pt"
DATA_7_LARGE="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/data/narrowed/7DOF_15deg/7DOF_15deg_part000.pt"  # 50M samples
ADAPT_SCRIPT="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/adapt_multitask_newest.py"
ISAACLAB_SH="/home/ubuntu/IsaacLab/isaaclab.sh"

OUTDIR="$(dirname "$ADAPT_SCRIPT")/tier4_runs/expE_part000_test_${TIMESTAMP}"
LOGDIR="$OUTDIR/logs"
TBROOT="$OUTDIR/tb"
SAVEROOT="$OUTDIR/ckpts"
mkdir -p "$LOGDIR" "$TBROOT" "$SAVEROOT"

SUMMARY="$OUTDIR/summary.txt"

# Use seed 11 (fresh seed, never run before)
SEED=11
TAG="dof7_seed${SEED}_part000"
LOGFILE="$LOGDIR/adapt_${TAG}.log"
SAVE_DIR="$SAVEROOT/${TAG}"
mkdir -p "$SAVE_DIR"

echo "[$(date)] Launching part000 sanity test (seed $SEED)" | tee "$SUMMARY"
echo "  --data: $DATA_7_LARGE  (50M samples, vs. seed 1-10's part001 at 3.85M)" | tee -a "$SUMMARY"
echo "  All other hyperparameters identical to expD." | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

echo "=== seed ${SEED} (part000) ===" | tee -a "$SUMMARY"
echo "  start: $(date)" | tee -a "$SUMMARY"
( time "$ISAACLAB_SH" -p "$ADAPT_SCRIPT" \
    --ckpt "$SHARED_CKPT" --mode fk --device cuda --dof 7 --data "$DATA_7_LARGE" \
    --support_size 50000 --query_size 2000000 --adapt_steps 100000 \
    --inner_lr 1e-6 --batch_size 8192 --query_batch_size 8192 --num_workers 8 \
    --adapt all --l2_reg 1e-6 --grad_clip 1.0 --adam_eps 1e-7 --std_floor_q_deg 1.0 \
    --enable_tf32 --log_every 5000 --eval_every 1000 --eval_steps_list "" \
    --pos_weight 1.0 --ori_weight 0.05 --score_pos_w 1.0 --score_ori_w 0.01 --eval_pbar \
    --tb_logdir "$TBROOT" --tb_name "$TAG" --save_dir "$SAVE_DIR" --seed "$SEED" \
) 2>&1 | tee "$LOGFILE"
best="$(grep -aE "^\[INFO\] BEST" "$LOGFILE" | tail -1 || true)"
echo "  $best" | tee -a "$SUMMARY"
echo "  end:   $(date)" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

echo "[$(date)] === Test complete ===" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"
echo "=== Comparison with expD seeds (part001) ===" | tee -a "$SUMMARY"
echo "  seeds 1-10 mean (part001): ~0.0135 m / ~2.59°" | tee -a "$SUMMARY"
echo "  seed 11 (part000):         (see BEST above)" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"
echo "  Interpretation:" | tee -a "$SUMMARY"
echo "    - if seed 11 ≈ 0.0135 m: dataset NOT the cause; optimization landscape is" | tee -a "$SUMMARY"
echo "    - if seed 11 < 0.0125 m: dataset matters; would justify re-running seeds with part000" | tee -a "$SUMMARY"
echo "    - if seed 11 < 0.0101 m: dataset IS the cause; major change to plan" | tee -a "$SUMMARY"
