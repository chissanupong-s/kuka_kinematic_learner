#!/usr/bin/env bash
# Run 3 extra experiments to address v28 review:
#  1. 5-DoF random-init Ablation B (Table 5.3 5-DoF row)
#  2. 6-DoF random-init Ablation B (Table 5.3 6-DoF row)
#  3. 7-DoF K=120000 data-size point (Fig 5.8 plateau verification)
set -euo pipefail

TIMESTAMP="20260507_045433"
ADAPT_SCRIPT="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/adapt_multitask_newest.py"
ISAACLAB_SH="/home/ubuntu/IsaacLab/isaaclab.sh"

RANDOM_CKPT="/tmp/random_init_resmlp_${TIMESTAMP}.pt"
SHARED_CKPT="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/runs/multitask/seperate_weight/1/fk_iiwa_5_6_7/multitask_fk_best.pt"

DATA_5="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/data/narrowed/5DOF_8deg.pt_part000.pt"
DATA_6="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/data/narrowed/6DOF_12deg.pt_part000.pt"
DATA_7="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/data/narrowed/7DOF_15deg/7DOF_15deg_part001.pt"

OUTDIR="$(dirname "$ADAPT_SCRIPT")/tier4_runs/expC_v28_extras_${TIMESTAMP}"
LOGDIR="$OUTDIR/logs"
TBROOT="$OUTDIR/tb"
SAVEROOT="$OUTDIR/ckpts"
mkdir -p "$LOGDIR" "$TBROOT" "$SAVEROOT"

SUMMARY="$OUTDIR/summary.txt"
echo "v28-extras started $(date)" | tee "$SUMMARY"

run () {
    local tag="$1"; shift
    local logfile="$LOGDIR/adapt_${tag}.log"
    local save_dir="$SAVEROOT/${tag}"
    mkdir -p "$save_dir"
    echo "=== $tag ===" | tee -a "$SUMMARY"
    echo "  start: $(date)" | tee -a "$SUMMARY"
    ( time "$ISAACLAB_SH" -p "$ADAPT_SCRIPT" "$@" --tb_logdir "$TBROOT" --tb_name "$tag" --save_dir "$save_dir" --seed 42 ) 2>&1 | tee "$logfile"
    best="$(grep -aE "^\[INFO\] BEST" "$logfile" | tail -1 || true)"
    echo "  $best" | tee -a "$SUMMARY"
    echo "  end:   $(date)" | tee -a "$SUMMARY"
    echo | tee -a "$SUMMARY"
}

# 1. 5-DoF Ablation B (random-init, matched 0.365 hr ≈ 30k step budget)
run "ablB_5dof_random_init" \
    --ckpt "$RANDOM_CKPT" --mode fk --device cuda --dof 5 --data "$DATA_5" \
    --support_size 50000 --query_size 2000000 --adapt_steps 30000 \
    --inner_lr 1e-5 --batch_size 2048 --query_batch_size 8192 --num_workers 8 \
    --adapt all --l2_reg 1e-6 --grad_clip 1.0 --adam_eps 1e-7 --std_floor_q_deg 1.0 \
    --enable_tf32 --log_every 5000 --eval_every 1000 --eval_steps_list "" \
    --pos_weight 1.0 --ori_weight 0.30 --score_pos_w 1.0 --score_ori_w 0.01 --eval_pbar

# 2. 6-DoF Ablation B
run "ablB_6dof_random_init" \
    --ckpt "$RANDOM_CKPT" --mode fk --device cuda --dof 6 --data "$DATA_6" \
    --support_size 50000 --query_size 2000000 --adapt_steps 30000 \
    --inner_lr 1e-6 --batch_size 2048 --query_batch_size 8192 --num_workers 8 \
    --adapt all --l2_reg 1e-6 --grad_clip 1.0 --adam_eps 1e-7 --std_floor_q_deg 1.0 \
    --enable_tf32 --log_every 5000 --eval_every 1000 --eval_steps_list "" \
    --pos_weight 1.0 --ori_weight 0.30 --score_pos_w 1.0 --score_ori_w 0.01 --eval_pbar

# 3. 7-DoF K=120000 data-size point (plateau verification)
run "dof7_K120000" \
    --ckpt "$SHARED_CKPT" --mode fk --device cuda --dof 7 --data "$DATA_7" \
    --support_size 120000 --query_size 2000000 --adapt_steps 30000 \
    --inner_lr 1e-6 --batch_size 8192 --query_batch_size 8192 --num_workers 8 \
    --adapt all --l2_reg 1e-6 --grad_clip 1.0 --adam_eps 1e-7 --std_floor_q_deg 1.0 \
    --enable_tf32 --log_every 5000 --eval_every 1000 --eval_steps_list "" \
    --pos_weight 1.0 --ori_weight 0.05 --score_pos_w 1.0 --score_ori_w 0.01 --eval_pbar

echo "v28-extras done $(date)" | tee -a "$SUMMARY"
