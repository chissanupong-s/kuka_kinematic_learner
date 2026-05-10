# Setup guide for a fresh / second computer

This repo lets you reproduce all FYP-report and conference-paper experiments on a different machine. The setup is split into 5 short sections — each is a copy-paste block.

**Hardware requirement:** any NVIDIA GPU with CUDA support (RTX, GTX, A-series — all fine). **Isaac Lab is NOT required** — datasets are pre-generated and stored on Hugging Face. You only need PyTorch.

---

## 1. Clone the repository

```bash
# Choose where to put it (replace /your/path with whatever you like)
git clone git@github-wish:chissanupong-s/kuka_kinematic_learner.git /your/path/kuka_kinematic_learner
cd /your/path/kuka_kinematic_learner
```

Repo size: ~250 MB tracked content + 64 MB Stage-2 checkpoint.

---

## 2. Symlink the Stage-2 shared checkpoint into the path scripts expect

```bash
bash original/ext_script/scripts/train/model_tracking/setup_checkpoints.sh
```

This creates `runs/multitask/seperate_weight/1/fk_iiwa_5_6_7/multitask_fk_best.pt` as a symlink to the tracked `model_tracking/multitask_fk_best.pt`. The Stage-3 adaptation scripts hardcode that path.

---

## 3. Set up the Python environment

```bash
# Create a fresh conda env (Isaac Lab NOT needed — just PyTorch)
conda create -n meta_kin python=3.10 -y
conda activate meta_kin

# Install PyTorch (adjust cu121 to match your CUDA version: cu118, cu121, cu124)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Other dependencies
pip install numpy tqdm tensorboard huggingface_hub python-docx
```

Verify CUDA works:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
```

---

## 4. Download datasets from Hugging Face (required — too large for git)

Total ~5.7 GB. Takes 10–30 min depending on bandwidth.

```bash
# Login with your HF token (paste when prompted; choose 'n' for git-credential)
hf auth login

# Download all four headline datasets into the path scripts expect
mkdir -p original/ext_script/data/narrowed
hf download Chissanupong/kuka-iiwa-meta-kinematics-data \
    --repo-type dataset \
    --local-dir original/ext_script/data/narrowed

# Verify (should show 4 .pt files + README.md)
ls -lh original/ext_script/data/narrowed/
ls -lh original/ext_script/data/narrowed/7DOF_15deg/
```

The dataset is **private** — you need to be logged in with a token from the same HF account that owns the dataset (`Chissanupong`).

---

## 5. Smoke test (1 minute)

Run a tiny adaptation to confirm everything is wired up:

```bash
cd original/ext_script/scripts/train

python adapt_multitask_newest.py \
    --ckpt model_tracking/multitask_fk_best.pt \
    --mode fk --device cuda --dof 7 \
    --data ../../data/narrowed/7DOF_15deg/7DOF_15deg_part001.pt \
    --support_size 50000 --query_size 100000 --adapt_steps 1000 \
    --inner_lr 1e-6 --batch_size 8192 --query_batch_size 8192 --num_workers 4 \
    --enable_tf32 --eval_every 500 --eval_pbar --seed 999 \
    --tb_logdir /tmp/tb --tb_name smoke --save_dir /tmp/smoke_ckpts
```

Expected output (after ~1 min):
```
[INFO] BEST step=1000 metrics={'pos_mae_m': ~0.X, 'ori_deg': ~Y} score=~Z
```

If you see a `[INFO] BEST` line at the end, the full pipeline (model, data, GPU, loss) is working.

---

## What to run next on this machine

The whole point of having a second machine is to run experiments in parallel. Some good options:

### Option A — extend multi-seed coverage on the report's protocol

```bash
# Run additional seeds (e.g., 11-20) using the same protocol as seeds 1-10
# Edit tier4_runs/expD_seeds_4to10_7dof.sh to use seeds 11-20, then:
cd original/ext_script/scripts/train
bash tier4_runs/expD_seeds_4to10_7dof.sh
```

### Option B — run the conference-paper Transformer pipeline

```bash
cd original/ext_script/scripts/train_meta_kin_former
bash expF_transformer_pipeline.sh
```
~13–17 hours. Trains all 3 stages of the MetaKinFormer (Transformer architecture for the conference paper) end-to-end.

### Option C — run hyperparameter sweeps for the conference paper

(See `train_meta_kin_former/README.md` for variation ideas.)

---

## Troubleshooting

**"No module named torch"** — your conda env isn't activated. Run `conda activate meta_kin`.

**"CUDA out of memory"** — reduce `--batch_size` (e.g., 4096 → 2048) or `--query_batch_size`.

**HF download fails with 401** — your token isn't logged in. Run `hf auth login` again.

**Stage-2 checkpoint not found at `runs/multitask/...`** — you skipped step 2; run the setup script.

**`tier4_runs/` files appear missing on this machine** — that's intentional, the experiment outputs are gitignored. The scripts auto-create them on first run.
