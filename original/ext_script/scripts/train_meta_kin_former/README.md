# MetaKinFormer — Joint-as-Token Transformer for Cross-DoF FK

This directory contains a **Transformer-based architecture** for cross-DoF
forward-kinematics learning, designed for the conference paper. It is a
drop-in replacement for the residual MLP backbone used in the FYP report.

## Why a Transformer?

The supervisor flagged that a generic ResMLP is not adapted to the kinematic
structure. The MetaKinFormer addresses this by treating **each joint as a token**
and using **self-attention** to model joint-to-joint dependencies — which is
exactly what makes 7-DoF hard for ResMLP (long-range coupling between proximal
and distal joints).

| Property | ResMLP_Mask (report baseline) | MetaKinFormer (this) |
|---|---|---|
| Parameters | ~17 M | **~800 k** (20× smaller) |
| Architecture | 8 residual MLP blocks | 4 transformer encoder layers |
| Mask handling | Learned additive `W_mask` projection | Attention key-padding mask |
| Joint coupling | Captured implicitly through depth | Explicit via self-attention |
| Inductive bias for FK | Generic regression | Joint-as-token + positional encoding |

## Files

| File | Role |
|---|---|
| `meta_kin_former.py` | Architecture (`MetaKinFormer` class) |
| `train_singletask_transformer.py` | **Stage 1** — per-DoF single-task training |
| `train_multitask_transformer.py` | **Stage 2** — shared multitask training |
| `adapt_multitask_transformer.py` | **Stage 3** — per-DoF adaptation |
| `tier4_runs/expF_transformer_pipeline.sh` | End-to-end pipeline (all 3 stages, 3 seeds × 3 DoFs) |

## Quick architecture details

```
Input:  q ∈ ℝ⁷  (joint angles, inactive joints clamped to 0)
        m ∈ {0,1}⁷  (active-joint mask)

  ↓ JointTokenEmbedding (Linear: 2 → d_model)
  ↓ JointPositionalEncoding (sinusoidal, by joint index)
  ↓ TransformerEncoder (4 layers, 4 heads, d_model=128, d_ff=512)
       — src_key_padding_mask = ¬m  (inactive joints excluded from attention)
       — pre-LN for stable training
  ↓ Concat-pool (B, 7×128)
  ↓ Linear (7×128 → 7)

Output: pose_vec = [x, y, z, q_w, q_x, q_y, q_z]  (standardised per task)
```

## Running the conference-paper pipeline

```bash
# Activate the env (Isaac Lab not needed; just PyTorch + tensorboard)
source /home/ubuntu/miniconda3/bin/activate env_isaaclab

# Launch the full pipeline (Stages 1 → 2 → 3, ~13-17 hours total)
cd /home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train
bash tier4_runs/expF_transformer_pipeline.sh
```

The script writes everything under `runs/expF_transformer_pipeline_<TIMESTAMP>/`:
- `stage1/dof{5,6,7}/best.pt` — Stage 1 single-task models
- `stage2/multitask_fk_best.pt` — Stage 2 shared meta-kinematics model
- `stage3/dof{5,6,7}_seed{42,1,2}/best.pt` — Stage 3 adapted models (3 seeds × 3 DoFs)
- `summary.txt` — BEST results per stage

## Running stages individually

```bash
PY=/home/ubuntu/miniconda3/envs/env_isaaclab/bin/python
DATA_DIR=/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/data/narrowed

# Stage 1 — single-task 7-DoF
$PY train_singletask_transformer.py \
    --data $DATA_DIR/7DOF_15deg/7DOF_15deg_part000.pt --dof 7 --mode fk \
    --batch_size 4096 --epochs 200 --lr 5e-4 \
    --device cuda --enable_tf32 --seed 42 \
    --tb_logdir runs/transformer_stage1 --tb_name dof7 \
    --save_dir runs/transformer_stage1/dof7

# Stage 2 — shared multitask (warm-start from Stage 1)
$PY train_multitask_transformer.py \
    --data_5 $DATA_DIR/5DOF_8deg.pt_part000.pt \
    --data_6 $DATA_DIR/6DOF_12deg.pt_part000.pt \
    --data_7 $DATA_DIR/7DOF_15deg/7DOF_15deg_part000.pt \
    --init_from_stage1 \
        runs/transformer_stage1/dof5/best.pt \
        runs/transformer_stage1/dof6/best.pt \
        runs/transformer_stage1/dof7/best.pt \
    --steps 1000000 --batch_size 4096 --lr 3e-4 --warmup 2000 \
    --device cuda --enable_tf32 --seed 42 \
    --tb_logdir runs/transformer_stage2 --tb_name shared \
    --save_dir runs/transformer_stage2

# Stage 3 — per-DoF adaptation
$PY adapt_multitask_transformer.py \
    --ckpt runs/transformer_stage2/multitask_fk_best.pt \
    --dof 7 --data $DATA_DIR/7DOF_15deg/7DOF_15deg_part000.pt \
    --support_size 50000 --query_size 2000000 --adapt_steps 100000 \
    --inner_lr 1e-6 --batch_size 8192 \
    --pos_weight 1.0 --ori_weight 0.05 \
    --device cuda --enable_tf32 --seed 42 \
    --tb_logdir runs/transformer_stage3 --tb_name dof7_seed42 \
    --save_dir runs/transformer_stage3/dof7_seed42
```

## Multi-seed evaluation

The Stage 3 adaptation supports `--seed` for multi-seed runs (matches the
n=3 protocol used in the report). Run with seeds 42, 1, 2 for headline n=3,
or extend to 5–10 seeds for tighter CIs in the conference paper.

## Architecture variations to try (for ablations)

- **Smaller**: `--d_model 64 --n_layers 2 --n_heads 2` (~150k params)
- **Larger**: `--d_model 256 --n_layers 6 --n_heads 8` (~3.5M params)
- **Shallower**: `--n_layers 2`
- **Heads only**: `--n_heads 1` (no multi-head)

For the conference paper, an **ablation table** comparing these to the
ResMLP baseline at matched compute budgets would strengthen the architecture
analysis section.

## Compute estimates (single RTX 4080 Laptop, 12 GB VRAM)

| Stage | Time |
|---|---|
| Stage 1 (5-DoF) | ~30 min |
| Stage 1 (6-DoF) | ~45 min |
| Stage 1 (7-DoF) | ~1.5–3 hr |
| Stage 2 (1M steps) | ~30–60 min |
| Stage 3 (per seed × DoF) | ~1.3 hr |
| **Full pipeline (3 seeds × 3 DoFs)** | **~13–17 hr** |

## Headline numbers to compare against (ResMLP baseline from report)

| 7-DoF Adapted (best, seed 42) | ResMLP_Mask | MetaKinFormer (target) |
|---|---|---|
| Position error | 0.0099 m | < 0.0099 m ✓ to surpass |
| Orientation error | 1.71° | < 1.71° ✓ to surpass |
| n=10 mean | 0.0132 m | < 0.0101 m ✓ to beat single-task |

If MetaKinFormer beats single-task on 7-DoF n=10, that's the publishable result
the conference paper needs.
