# Tier 4 FYP Experiments — Claude Code Briefing

## CONTEXT (for you, Claude Code)

You are running on the user's remote training machine. The user is Chissanupong Saengsint (BEng FYP, University of Birmingham, supervisor Dr Yongjing Wang). The project is "Learning and Transfer of Robot Forward Kinematics Across Varying Degrees of Freedom" — a meta-kinematics framework on the KUKA iiwa 14 in 5/6/7 DoF settings, using a ResMLP backbone trained in Isaac Lab. The FYP report is being submitted on Friday 8 May 2026.

Headline results already obtained from the existing single-seed=42 runs:

| DoF | Single-task pos (m) | Single-task ori (°) | Shared meta pos (m) | Shared meta ori (°) | Adapted pos (m) | Adapted ori (°) | Single-task hr | Adapted hr |
|---|---|---|---|---|---|---|---|---|
| 5  | 0.0093 | 1.2039 | 0.0068 | 0.9096 | 0.0062 | 0.8036 | 1.873  | 0.365 |
| 6  | 0.0110 | 1.7136 | 0.0092 | 1.3689 | 0.0090 | 1.3385 | 4.973  | 0.363 |
| 7  | 0.0101 | 2.0853 | 0.0109 | 1.9953 | 0.0099 | 1.7104 | 22.12  | 0.111 |

The user is now executing **Tier 4** of a four-experiment plan to push the FYP mark from ~85% into the 93–96% band:

- **Experiment A** — Ablation B for 7-DoF only: random-init adaptation in the same wall-clock budget. Tests whether the shared representation does real work or just provides more compute. ~7 min.
- **Experiment B** — Multi-seed (seeds 1, 2) for the shared multitask model + 7-DoF adapt. Adds mean ± std to the headline 7-DoF row. ~2.5 hr.
- **Experiment C** — Ablation A for 5-DoF: shared multitask trained from random init (no single-task warm-start). Tests whether the warm-start checkpoints actually help. ~1.2 hr.
- **Experiment D** — t-SNE feature projection of the shared model's penultimate-layer activations on held-out samples from each DoF. Produces Figure 6.1 (representation analysis). ~30 min, no GPU.

This briefing covers all four experiments. The outputs you produce will be sent back for the user's FYP report. **Be careful, deterministic, and faithful to the existing scripts** — do not invent hyperparameters or change architecture.

## REPO LAYOUT (reference paths)

```
TRAIN_DIR=/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train
DATA_5DOF=/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/data/narrowed/5DOF_8deg.pt_part000.pt
DATA_6DOF=/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/data/narrowed/6DOF_12deg.pt_part000.pt
DATA_7DOF=/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/data/narrowed/7DOF_15deg/7DOF_15deg_part001.pt
SHARED_CKPT=$TRAIN_DIR/runs/multitask/seperate_weight/1/fk_iiwa_5_6_7/multitask_fk_best.pt
ISAACLAB=/home/ubuntu/IsaacLab/isaaclab.sh
```

The relevant scripts in `$TRAIN_DIR` are:
- `train_multitask_separate_weight.py` — Stage 2 multitask training (FK or IK). Defines `ResidualMLP_Mask(in_dim=7, out_dim=7, hidden_dim=1024, num_blocks=8)` inline. FK model has `fc_in`, `mask_proj`, 8× `blocks`, `fc_out`.
- `adapt_multitask_newest.py` — Stage 3 per-DoF adaptation. Loads checkpoints with `model_state_dict` key, infers architecture from state dict, supports `--mode fk`.
- `run_sweep_5_6_7 (1).sh` — driver that launches adapt with all the right flags. Note the space in the filename — quote it or rename.

## FIRST: PREP STEP

Verify the environment is ready before launching anything:

```bash
# Verify paths exist
TRAIN_DIR=/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train
ls -la $TRAIN_DIR/train_multitask_separate_weight.py
ls -la $TRAIN_DIR/adapt_multitask_newest.py
ls -la $TRAIN_DIR/runs/multitask/seperate_weight/1/fk_iiwa_5_6_7/multitask_fk_best.pt
ls -la /home/ubuntu/IsaacLab/isaaclab.sh

# Verify GPU
nvidia-smi | head -20

# Make a workspace directory under the train dir
mkdir -p $TRAIN_DIR/tier4_runs
cd $TRAIN_DIR
```

If any path is wrong, ASK the user before proceeding. Do not guess paths.

## EXPERIMENT A — Random-init 7-DoF adaptation (~7 min)

Goal: produce a baseline showing what 0.111 hr of adaptation achieves when starting from a randomly-initialised model rather than the trained shared meta-kinematics checkpoint. Expected outcome: errors substantially worse than 0.0099 m / 1.71° (the shared-init result).

### Step A.1 — Build a random-init FK checkpoint that the adapt script can load

Create `/tmp/make_random_init.py`:

```python
import sys, importlib.util, torch

TRAIN_DIR = "/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train"
TRAIN_PY  = f"{TRAIN_DIR}/train_multitask_separate_weight.py"

spec = importlib.util.spec_from_file_location("train_mod", TRAIN_PY)
mod  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

torch.manual_seed(42)
model = mod.ResidualMLP_Mask(in_dim=7, out_dim=7, hidden_dim=1024, num_blocks=8)

out_path = "/tmp/random_init_resmlp.pt"
torch.save({"model_state_dict": model.state_dict()}, out_path)

n_params = sum(p.numel() for p in model.parameters())
print(f"Saved random-init FK ResMLP_Mask to {out_path}")
print(f"Total parameters: {n_params:,}")
print(f"Has fc_in: {'fc_in.weight' in model.state_dict()}")
print(f"Has fc_out (FK head): {'fc_out.weight' in model.state_dict()}")
print(f"Number of residual blocks: {sum(1 for k in model.state_dict() if k.startswith('blocks.') and k.endswith('.fc1.weight'))}")
```

Run it:

```bash
python /tmp/make_random_init.py
```

Verify output contains:
- "Total parameters: 16,816,135" (or similar — this is exactly what training uses)
- "Has fc_out (FK head): True"
- "Number of residual blocks: 8"

If parameter count differs by a lot, STOP and ask the user — the architecture might have drifted.

### Step A.2 — Build the adapt sweep script for random-init 7-DoF

Save as `$TRAIN_DIR/run_expA_random_init_7dof.sh`. This is a copy of the existing `run_sweep_5_6_7 (1).sh` with three things changed: `CKPT` points at the random checkpoint, `OUTDIR` is namespaced for this experiment, and only `run_one 7` is active.

```bash
cat > $TRAIN_DIR/run_expA_random_init_7dof.sh << 'BASH_EOF'
#!/usr/bin/env bash
set -euo pipefail

# Experiment A — random-init 7-DoF adaptation (Tier 4 ablation B)
CKPT="/tmp/random_init_resmlp.pt"
DATA_7="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/data/narrowed/7DOF_15deg/7DOF_15deg_part001.pt"

ADAPT_SCRIPT="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/adapt_multitask_newest.py"
ISAACLAB_SH="/home/ubuntu/IsaacLab/isaaclab.sh"

DEVICE="cuda"
MODE="fk"
SUPPORT_SIZE="50000"
QUERY_SIZE="2000000"
ADAPT_STEPS="100000"
INNER_LR="1e-5"
BATCH_SIZE="8192"
QUERY_BATCH_SIZE="8192"
NUM_WORKERS="8"
ADAPT_WHAT="all"
L2_REG="1e-6"
GRAD_CLIP="1.0"
ADAM_EPS="1e-7"
STD_FLOOR_Q_DEG="1.0"
ENABLE_TF32="1"
LOG_EVERY="2000"
EVAL_EVERY="2000"
EVAL_PBAR="1"
EVAL_STEPS_LIST=""
POS_WEIGHT="1.0"
ORI_WEIGHT="0.30"
SCORE_POS_W="1.0"
SCORE_ORI_W="0.01"
SEED="42"

OUTDIR="$(dirname "$ADAPT_SCRIPT")/tier4_runs/expA_random_init_7dof_$(date +%Y%m%d_%H%M%S)"
LOGDIR="$OUTDIR/logs"
TBROOT="$OUTDIR/tb"
SAVEROOT="$OUTDIR/ckpts"
mkdir -p "$LOGDIR" "$TBROOT" "$SAVEROOT"

SUMMARY_TXT="$OUTDIR/summary.txt"
echo "EXPERIMENT A — Random-init 7-DoF adaptation" | tee "$SUMMARY_TXT"
echo "ckpt: $CKPT" | tee -a "$SUMMARY_TXT"
echo "time: $(date)" | tee -a "$SUMMARY_TXT"

run_one_7 () {
  local dof=7
  local data="$DATA_7"
  local tb_name="dof7_random_init"
  local save_dir="$SAVEROOT/dof7_random_init"
  mkdir -p "$save_dir"
  local logfile="$LOGDIR/adapt_dof7_random_init.log"

  echo "=========== EXP A: random-init dof=7 ===========" | tee -a "$SUMMARY_TXT"
  echo "log: $logfile" | tee -a "$SUMMARY_TXT"

  ARGS=(
    --ckpt "$CKPT"
    --mode "$MODE"
    --dof "$dof"
    --data "$data"
    --device "$DEVICE"
    --support_size "$SUPPORT_SIZE"
    --query_size "$QUERY_SIZE"
    --adapt_steps "$ADAPT_STEPS"
    --inner_lr "$INNER_LR"
    --batch_size "$BATCH_SIZE"
    --query_batch_size "$QUERY_BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --adapt "$ADAPT_WHAT"
    --l2_reg "$L2_REG"
    --grad_clip "$GRAD_CLIP"
    --adam_eps "$ADAM_EPS"
    --std_floor_q_deg "$STD_FLOOR_Q_DEG"
    --log_every "$LOG_EVERY"
    --eval_every "$EVAL_EVERY"
    --eval_steps_list "$EVAL_STEPS_LIST"
    --pos_weight "$POS_WEIGHT"
    --ori_weight "$ORI_WEIGHT"
    --score_pos_w "$SCORE_POS_W"
    --score_ori_w "$SCORE_ORI_W"
    --tb_logdir "$TBROOT"
    --tb_name "$tb_name"
    --save_dir "$save_dir"
    --seed "$SEED"
  )
  [[ "$ENABLE_TF32" == "1" ]] && ARGS+=( --enable_tf32 )
  [[ "$EVAL_PBAR" == "1" ]] && ARGS+=( --eval_pbar )

  ( time "$ISAACLAB_SH" -p "$ADAPT_SCRIPT" "${ARGS[@]}" ) 2>&1 | tee "$logfile"

  best_line="$(grep -E "^\[INFO\] BEST" "$logfile" | tail -n 1 || true)"
  echo "$best_line" | tee -a "$SUMMARY_TXT"
}

run_one_7

echo "DONE. Outputs in: $OUTDIR"
echo "Summary: $SUMMARY_TXT"
BASH_EOF

chmod +x $TRAIN_DIR/run_expA_random_init_7dof.sh
```

### Step A.3 — Launch and capture the result

```bash
bash $TRAIN_DIR/run_expA_random_init_7dof.sh
```

When done, extract the result:

```bash
# Find the most recent expA output directory and print its summary
LATEST_A=$(ls -dt $TRAIN_DIR/tier4_runs/expA_random_init_7dof_* | head -1)
echo "=== Experiment A summary ==="
cat $LATEST_A/summary.txt
echo
echo "=== BEST line ==="
grep -E "^\[INFO\] BEST" $LATEST_A/logs/adapt_dof7_random_init.log | tail -1
```

REPORT BACK: the BEST line from this log. The user needs `pos_mae=` (in metres) and `ori_deg=` (in degrees). Print them clearly:

```
EXPERIMENT A RESULT (7-DoF, random-init adapt, 0.111 hr budget):
  Position error: X.XXXX m
  Orientation error: Y.YYYY °
```

## EXPERIMENT B — Multi-seed for shared + 7-DoF adapt (~2.5 hr)

Goal: re-train the shared multitask model with seeds 1 and 2, then adapt 7-DoF from each. Combined with the existing seed=42 result, this gives mean ± std for the headline 7-DoF row of Table 5.1.

**Critical wall-clock detail:** the existing seed=42 shared run produced a `multitask_fk_best.pt` with reported wall-clock 1.159 hr. Match this. Two ways:

**Option 1 (preferred):** read the step number from the existing seed=42 best checkpoint, then set `--steps` for the new seeds to that same step count.

```bash
python -c "
import torch
ckpt = torch.load('$TRAIN_DIR/runs/multitask/seperate_weight/1/fk_iiwa_5_6_7/multitask_fk_best.pt',
                  map_location='cpu', weights_only=False)
if isinstance(ckpt, dict):
    print('Keys:', list(ckpt.keys())[:10])
    print('Step:', ckpt.get('step', 'NOT FOUND'))
    print('Best metric:', ckpt.get('best_metric', 'NOT FOUND'))
"
```

If `step` is in the metadata, use that as `--steps` for the seed=1 and seed=2 retrainings.

**Option 2 (fallback):** run with `--steps 1000000` (the maximum) but kill the process at 1.16 hr wall-clock, then use the `multitask_fk_best.pt` file at that point.

### Step B.1 — Retrain shared multitask model with seed=1

The existing multitask training script accepts a long argument list. Use `argparse` flags found in `train_multitask_separate_weight.py` (search for `add_argument` lines if you need to verify any of these). The non-default flags are:

```bash
mkdir -p $TRAIN_DIR/tier4_runs/multitask_seed1

cd $TRAIN_DIR
$ISAACLAB -p train_multitask_separate_weight.py \
    --mode fk \
    --task_5dof "/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/data/narrowed/5DOF_8deg.pt_part000.pt" \
    --task_6dof "/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/data/narrowed/6DOF_12deg.pt_part000.pt" \
    --task_7dof "/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/data/narrowed/7DOF_15deg/7DOF_15deg_part001.pt" \
    --hidden_dim 1024 \
    --num_blocks 8 \
    --batch_size 4096 \
    --lr 3e-4 \
    --steps 1000000 \
    --lr_schedule cosine \
    --warmup_steps 2000 \
    --lr_min 1e-5 \
    --grad_clip 1.0 \
    --aux_loss_weight 0.03 \
    --seed 1 \
    --log_dir $TRAIN_DIR/tier4_runs/multitask_seed1 \
    --out_dir $TRAIN_DIR/tier4_runs/multitask_seed1 \
    2>&1 | tee $TRAIN_DIR/tier4_runs/multitask_seed1/train.log
```

**Important:** before launching, run a dry sanity check by viewing the argparse section of the script and confirm the flag names match. If any flag name in the command above is wrong, fix it from the script's actual argparse defaults rather than guessing.

```bash
grep -n "add_argument" $TRAIN_DIR/train_multitask_separate_weight.py
```

After the run completes (or after killing at ~1.16 hr if using Option 2), confirm `multitask_fk_best.pt` exists:

```bash
ls -la $TRAIN_DIR/tier4_runs/multitask_seed1/multitask_fk_best.pt
```

### Step B.2 — Adapt 7-DoF from seed=1 shared model

Make a copy of the Experiment A sweep, change the CKPT to point at the seed=1 shared model, and namespace the OUTDIR:

```bash
cat > $TRAIN_DIR/run_expB_adapt_seed1_7dof.sh << 'BASH_EOF'
#!/usr/bin/env bash
set -euo pipefail
CKPT="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/tier4_runs/multitask_seed1/multitask_fk_best.pt"
DATA_7="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/data/narrowed/7DOF_15deg/7DOF_15deg_part001.pt"
ADAPT_SCRIPT="/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/adapt_multitask_newest.py"
ISAACLAB_SH="/home/ubuntu/IsaacLab/isaaclab.sh"

DEVICE="cuda"; MODE="fk"
SUPPORT_SIZE="50000"; QUERY_SIZE="2000000"
ADAPT_STEPS="100000"; INNER_LR="1e-5"
BATCH_SIZE="8192"; QUERY_BATCH_SIZE="8192"; NUM_WORKERS="8"
ADAPT_WHAT="all"
L2_REG="1e-6"; GRAD_CLIP="1.0"; ADAM_EPS="1e-7"; STD_FLOOR_Q_DEG="1.0"
ENABLE_TF32="1"
LOG_EVERY="2000"; EVAL_EVERY="2000"; EVAL_PBAR="1"; EVAL_STEPS_LIST=""
POS_WEIGHT="1.0"; ORI_WEIGHT="0.30"; SCORE_POS_W="1.0"; SCORE_ORI_W="0.01"
SEED="1"

OUTDIR="$(dirname "$ADAPT_SCRIPT")/tier4_runs/expB_adapt_seed1_7dof_$(date +%Y%m%d_%H%M%S)"
LOGDIR="$OUTDIR/logs"; TBROOT="$OUTDIR/tb"; SAVEROOT="$OUTDIR/ckpts"
mkdir -p "$LOGDIR" "$TBROOT" "$SAVEROOT"

SUMMARY_TXT="$OUTDIR/summary.txt"
echo "EXPERIMENT B — adapt 7-DoF from seed=1 shared model" | tee "$SUMMARY_TXT"

ARGS=(
    --ckpt "$CKPT" --mode "$MODE" --dof 7 --data "$DATA_7" --device "$DEVICE"
    --support_size "$SUPPORT_SIZE" --query_size "$QUERY_SIZE"
    --adapt_steps "$ADAPT_STEPS" --inner_lr "$INNER_LR"
    --batch_size "$BATCH_SIZE" --query_batch_size "$QUERY_BATCH_SIZE" --num_workers "$NUM_WORKERS"
    --adapt "$ADAPT_WHAT" --l2_reg "$L2_REG" --grad_clip "$GRAD_CLIP"
    --adam_eps "$ADAM_EPS" --std_floor_q_deg "$STD_FLOOR_Q_DEG"
    --log_every "$LOG_EVERY" --eval_every "$EVAL_EVERY" --eval_steps_list "$EVAL_STEPS_LIST"
    --pos_weight "$POS_WEIGHT" --ori_weight "$ORI_WEIGHT"
    --score_pos_w "$SCORE_POS_W" --score_ori_w "$SCORE_ORI_W"
    --tb_logdir "$TBROOT" --tb_name "dof7_seed1" --save_dir "$SAVEROOT/dof7_seed1"
    --seed "$SEED"
)
[[ "$ENABLE_TF32" == "1" ]] && ARGS+=( --enable_tf32 )
[[ "$EVAL_PBAR" == "1" ]] && ARGS+=( --eval_pbar )

LOGFILE="$LOGDIR/adapt_dof7_seed1.log"
( time "$ISAACLAB_SH" -p "$ADAPT_SCRIPT" "${ARGS[@]}" ) 2>&1 | tee "$LOGFILE"

best_line="$(grep -E "^\[INFO\] BEST" "$LOGFILE" | tail -n 1 || true)"
echo "$best_line" | tee -a "$SUMMARY_TXT"
echo "DONE: $OUTDIR"
BASH_EOF
chmod +x $TRAIN_DIR/run_expB_adapt_seed1_7dof.sh

bash $TRAIN_DIR/run_expB_adapt_seed1_7dof.sh
```

Also evaluate the seed=1 shared model directly on 7-DoF held-out, *without* any adaptation, by reading the very first eval (step 0) from the adapt log:

```bash
LATEST_B1=$(ls -dt $TRAIN_DIR/tier4_runs/expB_adapt_seed1_7dof_* | head -1)
grep -E "step=0[^0-9]|step=     0" $LATEST_B1/logs/adapt_dof7_seed1.log | head -5
```

The "step=0" eval is the shared model's pre-adaptation performance — that's the seed=1 shared 7-DoF number for Table 5.1.

### Step B.3 — Repeat for seed=2

Same as B.1 and B.2 but replace `seed1` with `seed2` everywhere and `--seed 1` with `--seed 2`. Use sed or just duplicate the two scripts:

```bash
sed 's/seed1/seed2/g; s/--seed 1$/--seed 2/' \
    $TRAIN_DIR/run_expB_adapt_seed1_7dof.sh > $TRAIN_DIR/run_expB_adapt_seed2_7dof.sh
chmod +x $TRAIN_DIR/run_expB_adapt_seed2_7dof.sh

# Same for the multitask training command — re-run the Step B.1 block but with --seed 2
# and out_dir tier4_runs/multitask_seed2
```

### Step B.4 — Report Experiment B numbers

Collect six numbers:

```
For each seed in {1, 2}:
  Shared model on 7-DoF held-out: pos err (m), ori err (°)   [from step=0 eval in adapt log]
  Adapted 7-DoF: pos err (m), ori err (°)                    [from BEST line in adapt log]
```

Print clearly as:

```
EXPERIMENT B RESULTS (7-DoF, multi-seed):
  Seed 1 — shared at step 0:    pos = X.XXXX m,  ori = Y.YYYY °
  Seed 1 — adapted (BEST):       pos = X.XXXX m,  ori = Y.YYYY °
  Seed 2 — shared at step 0:    pos = X.XXXX m,  ori = Y.YYYY °
  Seed 2 — adapted (BEST):       pos = X.XXXX m,  ori = Y.YYYY °
```

## EXPERIMENT C — Random-init shared, 5-DoF only (~1.2 hr)

Goal: train the shared multitask model from random init (no warm-start from single-task checkpoints), evaluate on 5-DoF held-out, then short-adapt 5-DoF from it. This populates the 5-DoF row of Ablation A (Table 5.3 in the report).

**Note on warm-start:** check whether the existing multitask training script implements warm-start at all by searching for `init_ckpt` in the argparse:

```bash
grep -n "init_ckpt\|init_5dof\|init_6dof\|init_7dof\|warm_start\|init_paths" $TRAIN_DIR/train_multitask_separate_weight.py | head -20
```

If the script DOES implement warm-start (e.g., via `--init_ckpt_5dof` or similar), then the existing seed=42 run used those flags and the random-init run is "same training, no init flags". If the script does NOT implement warm-start, then the existing seed=42 run is already random-init, and the user needs to clarify what "Stage 1 → Stage 2 warm start" actually means in the methodology — DO NOT proceed with Experiment C; instead report this back and ask.

Assuming warm-start IS used (most likely case):

### Step C.1 — Train random-init shared multitask

```bash
mkdir -p $TRAIN_DIR/tier4_runs/multitask_random_init

cd $TRAIN_DIR
$ISAACLAB -p train_multitask_separate_weight.py \
    --mode fk \
    --task_5dof "/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/data/narrowed/5DOF_8deg.pt_part000.pt" \
    --task_6dof "/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/data/narrowed/6DOF_12deg.pt_part000.pt" \
    --task_7dof "/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/data/narrowed/7DOF_15deg/7DOF_15deg_part001.pt" \
    --hidden_dim 1024 \
    --num_blocks 8 \
    --batch_size 4096 \
    --lr 3e-4 \
    --steps 1000000 \
    --lr_schedule cosine \
    --warmup_steps 2000 \
    --lr_min 1e-5 \
    --grad_clip 1.0 \
    --aux_loss_weight 0.03 \
    --seed 42 \
    --log_dir $TRAIN_DIR/tier4_runs/multitask_random_init \
    --out_dir $TRAIN_DIR/tier4_runs/multitask_random_init \
    2>&1 | tee $TRAIN_DIR/tier4_runs/multitask_random_init/train.log
```

**Match wall-clock or step count** to the original seed=42 run. Same approach as Step B.1.

### Step C.2 — Adapt 5-DoF from random-init shared

Use the same pattern as Step B.2 but: CKPT points at `tier4_runs/multitask_random_init/multitask_fk_best.pt`, dof=5, data=DATA_5DOF, score_ori_w stays the same. Save as `run_expC_adapt_5dof_from_random.sh`.

### Step C.3 — Report Experiment C numbers

```
EXPERIMENT C RESULTS (Ablation A, 5-DoF):
  Random-init shared on 5-DoF held-out (step 0):  pos = X.XXXX m,  ori = Y.YYYY °
  Adapted 5-DoF from random-init shared (BEST):    pos = X.XXXX m,  ori = Y.YYYY °

For comparison (already known from seed=42 with warm-start):
  Warm-start shared on 5-DoF held-out:  pos = 0.0068 m,  ori = 0.9096 °
  Adapted 5-DoF from warm-start shared: pos = 0.0062 m,  ori = 0.8036 °
```

## EXPERIMENT D — t-SNE feature projection (~30 min, no GPU needed)

Goal: produce `fig_features.png` showing the 2D t-SNE projection of the shared model's penultimate-layer activations on held-out samples from each DoF, coloured by DoF. Used as Figure 6.1.

### Step D.1 — Install sklearn if not present

```bash
python -c "import sklearn; print(sklearn.__version__)" 2>/dev/null || pip install scikit-learn matplotlib --break-system-packages
```

### Step D.2 — Run the feature projection script

Save as `$TRAIN_DIR/tier4_runs/feature_projection.py`:

```python
import sys, importlib.util, numpy as np, torch, matplotlib.pyplot as plt
from sklearn.manifold import TSNE

TRAIN_DIR = "/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train"
DATA_DIR  = "/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/data/narrowed"

# Load model class
spec = importlib.util.spec_from_file_location(
    "train_mod", f"{TRAIN_DIR}/train_multitask_separate_weight.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

# Load shared seed=42 checkpoint
ckpt_path = f"{TRAIN_DIR}/runs/multitask/seperate_weight/1/fk_iiwa_5_6_7/multitask_fk_best.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt["model_state_dict"]
hidden_dim = sd["fc_in.weight"].shape[0]
num_blocks = sum(1 for k in sd if k.startswith("blocks.") and k.endswith(".fc1.weight"))
print(f"Loaded shared model: hidden={hidden_dim}, blocks={num_blocks}")

model = mod.ResidualMLP_Mask(in_dim=7, out_dim=7, hidden_dim=hidden_dim, num_blocks=num_blocks)
model.load_state_dict(sd, strict=True); model.eval()

# Hook the output of the last residual block (penultimate features, before fc_out)
penult = []
def hook(module, inp, out):
    penult.append(out.detach().cpu().numpy())
model.blocks[-1].register_forward_hook(hook)

# Datasets per DoF — paths must match what training used
data_paths = {
    5: f"{DATA_DIR}/5DOF_8deg.pt_part000.pt",
    6: f"{DATA_DIR}/6DOF_12deg.pt_part000.pt",
    7: f"{DATA_DIR}/7DOF_15deg/7DOF_15deg_part001.pt",
}

# Sample per DoF and run forward pass
np.random.seed(0)
labels = []
N_PER_DOF = 1000
for dof, path in data_paths.items():
    full = torch.load(path, map_location="cpu", weights_only=False)
    # Handle tensor or dict-wrapped tensor
    if isinstance(full, dict):
        for key in ("data", "tensor", "samples"):
            if key in full:
                full = full[key]; break
    if not torch.is_tensor(full):
        raise RuntimeError(f"Could not extract tensor from {path}; type={type(full)}")
    print(f"DoF {dof}: dataset shape {tuple(full.shape)}")

    idx = np.random.permutation(full.shape[0])[:N_PER_DOF]
    sample = full[idx].float()

    # FK convention: input is q1..q7. Verify column count >= 7.
    assert sample.shape[1] >= 7, f"Expected >=7 cols, got {sample.shape[1]}"
    q = sample[:, :7]

    # Build mask: 1 for active joints (1..dof), 0 for inactive (dof..7)
    mask = torch.zeros_like(q)
    mask[:, :dof] = 1.0
    q_clamped = q * mask

    with torch.no_grad():
        _ = model(q_clamped, mask)
    labels.extend([dof] * N_PER_DOF)

X = np.vstack(penult)
print(f"Penultimate features collected: {X.shape}")

# t-SNE projection
print("Running t-SNE...")
Z = TSNE(n_components=2, random_state=0, perplexity=30, init="pca",
         learning_rate="auto").fit_transform(X)

# Plot
plt.figure(figsize=(6, 5))
colors = {5: "#1f77b4", 6: "#ff7f0e", 7: "#2ca02c"}
for dof in (5, 6, 7):
    m = np.array(labels) == dof
    plt.scatter(Z[m, 0], Z[m, 1], s=8, alpha=0.6, c=colors[dof], label=f"{dof} DoF")
plt.legend(loc="best", frameon=True)
plt.xlabel("t-SNE dim 1")
plt.ylabel("t-SNE dim 2")
plt.title("Penultimate-layer features of the shared meta-kinematics model")
plt.tight_layout()

OUT_PNG = f"{TRAIN_DIR}/tier4_runs/fig_features.png"
plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
print(f"Saved: {OUT_PNG}")
```

Run it:

```bash
python $TRAIN_DIR/tier4_runs/feature_projection.py
```

If the dataset file is wrapped in a dict structure that the script doesn't handle, the assertion will fire — inspect the file with `python -c "import torch; d = torch.load('<path>', weights_only=False); print(type(d), d if not torch.is_tensor(d) else d.shape)"` and adjust the unwrap logic in the script.

### Step D.3 — Verify the output

```bash
ls -la $TRAIN_DIR/tier4_runs/fig_features.png
file $TRAIN_DIR/tier4_runs/fig_features.png   # should report PNG, ~6x5 inches at 200 dpi
```

REPORT BACK: confirm the file exists and its size. The user will scp it back to their machine.

## FINAL SUMMARY — what to send back

After all four experiments complete, produce a single summary file:

```bash
cat > $TRAIN_DIR/tier4_runs/RESULTS.md << 'MD_EOF'
# Tier 4 Results Summary

## Experiment A — Random-init 7-DoF adapt (Ablation B)
[Fill from $LATEST_A summary]
- Position error: ____ m
- Orientation error: ____ °

## Experiment B — Multi-seed 7-DoF
- Seed 1, shared at step 0 on 7-DoF: pos = ____ m, ori = ____ °
- Seed 1, adapted 7-DoF (BEST):       pos = ____ m, ori = ____ °
- Seed 2, shared at step 0 on 7-DoF: pos = ____ m, ori = ____ °
- Seed 2, adapted 7-DoF (BEST):       pos = ____ m, ori = ____ °

## Experiment C — Random-init shared, 5-DoF (Ablation A)
- Random-init shared on 5-DoF held-out: pos = ____ m, ori = ____ °
- Adapted 5-DoF from random-init shared: pos = ____ m, ori = ____ °

## Experiment D — t-SNE figure
- Saved to: tier4_runs/fig_features.png
- File size: ____ KB
MD_EOF
```

The user will collect this RESULTS.md plus `fig_features.png` and feed both to the assistant that builds the FYP report.

## RULES OF THUMB FOR YOU, CLAUDE CODE

1. Do not invent paths or hyperparameters. If something is missing, STOP and ASK the user before proceeding.
2. After each experiment, IMMEDIATELY print the BEST line and any key numbers — don't bury them in logs.
3. If a multitask training run hits OOM, reduce `--batch_size` to 2048 and ASK before continuing.
4. If `--steps 1000000` is taking longer than expected to converge, kill at the wall-clock budget that matches the original seed=42 run (~1.16 hr) and use whatever `multitask_fk_best.pt` exists at that point — it's the early-best checkpoint anyway.
5. The experiments are independent. If one fails, the others can still proceed. Report failures clearly.
6. After every experiment, run `ls -la <out_dir>/multitask_fk_best.pt` (or the relevant output) to confirm the checkpoint actually got saved.
7. **Do not modify the user's training scripts** — only create new sweep scripts in `tier4_runs/`. The training scripts are the ground truth.
