#!/usr/bin/env bash
# End-to-end pipeline for the MetaKinFormer (joint-as-token Transformer) architecture.
# Runs Stage 1 (single-task per DoF) → Stage 2 (shared multitask) → Stage 3 (per-DoF
# adaptation), with multi-seed evaluation on Stage 3.
#
# Compute estimate (rough, based on 800k-param Transformer):
#   Stage 1 (5-DoF):  ~ 30-60 min
#   Stage 1 (6-DoF):  ~ 45-90 min
#   Stage 1 (7-DoF):  ~ 1.5-3 hr  (largest dataset)
#   Stage 2 (shared): ~ 30-60 min for 1M steps at batch 4096
#   Stage 3 (per-DoF, 3 seeds × 3 DoFs): ~ 7 × 1.3 hr = ~9 hr
#   TOTAL:           ~ 13-17 hr  (well within 5-day window)
#
# This is the "headline pipeline" for the conference paper. Multi-seed evaluation
# of the Adapted (best) numbers is across 3 seeds (42, 1, 2) per DoF.

set -euo pipefail

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ROOT=/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train_meta_kin_former
DATA_5="$ROOT/../../data/narrowed/5DOF_8deg.pt_part000.pt"
DATA_6="$ROOT/../../data/narrowed/6DOF_12deg.pt_part000.pt"
DATA_7="$ROOT/../../data/narrowed/7DOF_15deg/7DOF_15deg_part000.pt"

OUTDIR="$ROOT/runs/expF_transformer_pipeline_${TIMESTAMP}"
S1_OUT="$OUTDIR/stage1"
S2_OUT="$OUTDIR/stage2"
S3_OUT="$OUTDIR/stage3"
mkdir -p "$S1_OUT" "$S2_OUT" "$S3_OUT"

PY=/home/ubuntu/miniconda3/envs/env_isaaclab/bin/python
SUMMARY="$OUTDIR/summary.txt"
echo "[$(date)] Pipeline start" | tee "$SUMMARY"

# ===== Stage 1 — single-task per DoF =====
for DOF in 5 6 7; do
    case $DOF in
        5) DATA=$DATA_5 ;;
        6) DATA=$DATA_6 ;;
        7) DATA=$DATA_7 ;;
    esac
    OUT="$S1_OUT/dof${DOF}"
    LOG="$OUT/train.log"
    mkdir -p "$OUT"
    echo "[$(date)] === Stage 1: ${DOF}-DoF ===" | tee -a "$SUMMARY"
    ( time "$PY" "$ROOT/train_singletask_transformer.py" \
        --data "$DATA" --dof $DOF --mode fk \
        --d_model 128 --n_layers 4 --n_heads 4 --dim_feedforward 512 --dropout 0.1 \
        --batch_size 4096 --epochs 200 --lr 5e-4 --weight_decay 1e-5 \
        --device cuda --num_workers 4 --enable_tf32 --seed 42 \
        --tb_logdir "$OUTDIR/tb" --tb_name "stage1_dof${DOF}" \
        --save_dir "$OUT" \
    ) 2>&1 | tee "$LOG"
done

# ===== Stage 2 — shared multitask =====
echo "[$(date)] === Stage 2: shared multitask ===" | tee -a "$SUMMARY"
S2_LOG="$S2_OUT/train.log"
( time "$PY" "$ROOT/train_multitask_transformer.py" \
    --data_5 "$DATA_5" --data_6 "$DATA_6" --data_7 "$DATA_7" \
    --init_from_stage1 \
        "$S1_OUT/dof5/best.pt" \
        "$S1_OUT/dof6/best.pt" \
        "$S1_OUT/dof7/best.pt" \
    --d_model 128 --n_layers 4 --n_heads 4 --dim_feedforward 512 --dropout 0.1 \
    --steps 1000000 --batch_size 4096 --lr 3e-4 --warmup 2000 \
    --grad_clip 1.0 \
    --eval_every 2000 --log_every 500 --save_every 100000 \
    --device cuda --enable_tf32 --seed 42 \
    --tb_logdir "$OUTDIR/tb" --tb_name "stage2_shared" \
    --save_dir "$S2_OUT" \
) 2>&1 | tee "$S2_LOG"

S2_BEST="$S2_OUT/multitask_fk_best.pt"
echo "[$(date)] Stage 2 done. Best ckpt: $S2_BEST" | tee -a "$SUMMARY"

# ===== Stage 3 — per-DoF adaptation, multi-seed (3 seeds × 3 DoFs) =====
echo "[$(date)] === Stage 3: per-DoF adaptation (3 seeds × 3 DoFs) ===" | tee -a "$SUMMARY"
for DOF in 5 6 7; do
    case $DOF in
        5) DATA=$DATA_5 ;;
        6) DATA=$DATA_6 ;;
        7) DATA=$DATA_7 ;;
    esac
    for SEED in 42 1 2; do
        TAG="dof${DOF}_seed${SEED}"
        OUT="$S3_OUT/${TAG}"
        LOG="$OUT/adapt.log"
        mkdir -p "$OUT"
        echo "  [$(date)] adapt ${DOF}-DoF seed ${SEED}" | tee -a "$SUMMARY"
        ( time "$PY" "$ROOT/adapt_multitask_transformer.py" \
            --ckpt "$S2_BEST" --mode fk --device cuda --dof $DOF --data "$DATA" \
            --support_size 50000 --query_size 2000000 --adapt_steps 100000 \
            --inner_lr 1e-6 --batch_size 8192 --query_batch_size 8192 --num_workers 8 \
            --pos_weight 1.0 --ori_weight 0.05 \
            --score_pos_w 1.0 --score_ori_w 0.01 \
            --grad_clip 1.0 --adam_eps 1e-7 --l2_reg 1e-6 \
            --eval_every 1000 --log_every 5000 --eval_pbar --enable_tf32 \
            --seed $SEED \
            --tb_logdir "$OUTDIR/tb" --tb_name "$TAG" \
            --save_dir "$OUT" \
        ) 2>&1 | tee "$LOG"
        # Append the BEST line to summary
        BEST=$(grep -aE "^\[INFO\] BEST" "$LOG" | tail -1 || true)
        echo "    $BEST" | tee -a "$SUMMARY"
    done
done

echo "[$(date)] === Pipeline complete ===" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"
echo "Output dir: $OUTDIR" | tee -a "$SUMMARY"
echo "Stage 2 best: $S2_BEST" | tee -a "$SUMMARY"
echo "Per-DoF adapted ckpts: $S3_OUT/dof{5,6,7}_seed{42,1,2}/best.pt" | tee -a "$SUMMARY"
