# Machine B — instructions for Claude / the user

This file tells Claude (or the user) running on the **second computer** what
to do. Machine A (the primary) is concurrently running the 7-DoF half of the
same experiment — see [MULTI_MACHINE_SYNC.md](MULTI_MACHINE_SYNC.md) for the
overall sync protocol.

---

## Goal

Run **expL** (regularized single-task FK training) on **5-DoF and 6-DoF
configurations**, **3 seeds each = 6 training runs total**. Machine A is
running the 7-DoF half. Together they produce the n=3 mean ± std for the
Single-task row of Table 5.1 in the FYP report.

## Hyperparameters (must match Machine A exactly)

```
epochs              = 200
batch_size          = 4096
lr                  = 3e-5      (FIXED — no scheduler decay)
hidden_dim          = 1024
num_blocks          = 8
weight_decay        = 1e-4
dropout             = 0.2
early_stop_patience = 15
scheduler_patience  = 999       (effectively disables ReduceLROnPlateau)
max_samples         = 15_000_000
train_frac          = 0.7
val_frac            = 0.1
seeds               = 42, 1, 2  (three seeds per DoF)
```

These are pinned in `expL_5_6dof_regularized.sh`. **Do not edit them**
unless explicitly asked — keeping them identical to Machine A's 7-DoF run
is what makes the n=3 results across DoFs comparable in Table 5.1.

---

## Pre-flight checklist

Run these and confirm each one passes before starting:

```bash
cd ~/wish/kuka_kinematic_learner    # adjust if your clone is elsewhere

# 0. Make sure you're on the latest commit (Machine A may have pushed updates)
git pull origin main

# 1. Conda env
conda activate env_isaaclab
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
# expected: CUDA: True | <some NVIDIA GPU name>

# 2. Required Python packages
python -c "import tensorboard, huggingface_hub" 2>&1
# if either ImportErrors: pip install tensorboard huggingface_hub python-docx

# 3. Datasets present
ls -lh original/ext_script/data/narrowed/5DOF_8deg.pt_part000.pt
ls -lh original/ext_script/data/narrowed/6DOF_12deg.pt_part000.pt
# expected: ~867 MB and ~1.96 GB respectively

# 4. Stage-2 shared checkpoint (only needed for adaptation, not for expL — but
#    set it up while you're here since later experiments may need it)
ls -lh original/ext_script/scripts/train/model_tracking/multitask_fk_best.pt
# expected: 65 MB
bash original/ext_script/scripts/train/model_tracking/setup_checkpoints.sh
# creates the symlink at runs/multitask/seperate_weight/1/fk_iiwa_5_6_7/multitask_fk_best.pt

# 5. Launcher script + trainer
ls -l original/ext_script/scripts/train/tier4_runs/expL_5_6dof_regularized.sh
ls -l original/ext_script/scripts/train/train_kinematics_nn_pol_pt_generalize.py

# 6. GPU is free (no other training already running)
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
# should show only minor things like gnome-remote-desktop-daemon, no python procs >500 MB
```

If any check fails, **stop and report** — do not launch training until all six pass.

---

## Launch

```bash
chmod +x original/ext_script/scripts/train/tier4_runs/expL_5_6dof_regularized.sh

tmux new -d -s expL_machineB "source ~/miniconda3/bin/activate env_isaaclab && \
  cd ~/wish/kuka_kinematic_learner/original/ext_script/scripts/train/tier4_runs && \
  ./expL_5_6dof_regularized.sh; exec bash"

# Verify it started:
tmux ls
sleep 5
tmux capture-pane -t expL_machineB -p | tail -10 | tr -d '\r'
```

The script's order is **5-DoF first** (3 seeds), then **6-DoF** (3 seeds).

---

## Expected runtime

- 5-DoF: ~1.5–2 hr per seed × 3 seeds ≈ 4.5–6 hr
- 6-DoF: ~2–2.5 hr per seed × 3 seeds ≈ 6–7.5 hr
- **Total: ~11–14 hr**

Early stopping (patience=15) will cut runs short once val plateaus.

---

## How to monitor without interrupting

```bash
# Latest training-loop progress:
tmux capture-pane -t expL_machineB -p | tail -3 | tr -d '\r'

# Saved-best events for the currently-running seed:
LATEST_LOG=$(ls -t original/ext_script/scripts/train/tier4_runs/expL_5_6dof_regularized_*/logs/*train.log | head -1)
grep -aoE "Saved best at epoch=[0-9]+, val_mse=[0-9.eE+-]+" "$LATEST_LOG" | tail -5

# Which seeds have completed eval (green-light for that seed being done):
grep -l "Mean position error" \
  original/ext_script/scripts/train/tier4_runs/expL_5_6dof_regularized_*/logs/*_eval.log 2>/dev/null

# Summary file (this is the file Machine A reads to know your progress):
tail -50 original/ext_script/scripts/train/tier4_runs/expL_5_6dof_regularized_*/summary.txt
```

---

## When each seed completes — push results so Machine A can see them

Per [MULTI_MACHINE_SYNC.md](MULTI_MACHINE_SYNC.md), commit + push as soon as
new eval-log files exist. This makes Machine A aware of the progress without
needing to share GPU resources.

```bash
cd ~/wish/kuka_kinematic_learner
git add original/ext_script/scripts/train/tier4_runs/expL_5_6dof_regularized_*/
git commit -m "machine-B: expL 5/6-DoF — <which seeds done>"
git push origin main
```

Do this whenever convenient. A reasonable rhythm: push after each completed
seed (so 6 commits over ~12 hr).

### If `git push` is rejected (remote ahead)

This is **expected** and harmless — Machine A has pushed something while
you were training. Do NOT ask the user; do NOT use `--force`. The fix is
always the same two-command sequence:

```bash
git pull origin main      # safely merges Machine A's changes into yours
git push origin main      # now succeeds
```

The pull never deletes your run directories — they have a unique timestamp
that doesn't overlap with Machine A's. Only the docx (`*.docx`) could
genuinely conflict, and Machine B is told elsewhere in this file not to
edit the docx, so that conflict can't arise here.

If the pull does surface an unexpected conflict, **stop and report** —
don't try to auto-resolve.

Full handling rule documented in
[MULTI_MACHINE_SYNC.md → "Push-rejected-because-remote-is-ahead"](MULTI_MACHINE_SYNC.md).

---

## When ALL 6 runs complete — final tasks

1. **Final commit** with the full summary:
   ```bash
   cd ~/wish/kuka_kinematic_learner
   git add original/ext_script/scripts/train/tier4_runs/expL_5_6dof_regularized_*/
   git commit -m "machine-B: expL 5/6-DoF complete (n=3 each)"
   git push origin main
   ```

2. **Optional: upload trained checkpoints to HF model repo** so Machine A can
   pull them if needed for further work:
   ```bash
   ./original/ext_script/scripts/train/tier4_runs/sync_checkpoints_to_hf.sh \
       expL_5_6dof_regularized_<TIMESTAMP> machine-B
   ```
   (Replace `<TIMESTAMP>` with the actual run-dir suffix shown by `ls -d expL_5_6dof_regularized_*`.)

3. **Tell the user** — they will then ask Claude on Machine A to merge the
   results from both machines into Table 5.1 of the docx.

---

## What NOT to do on Machine B

- **Do not edit the .docx**. Only Machine A edits the report. (Avoids merge
  conflicts on the binary file.)
- **Do not run 7-DoF training**. Machine A is doing that. Running it here
  duplicates work and wastes ~6 hr of GPU time.
- **Do not modify `expL_5_6dof_regularized.sh`** unless the user explicitly
  asks. The hyperparameters need to stay identical to Machine A's run.
- **Do not push WIP commits with broken state**. Wait until at least one seed
  has completed before pushing.

---

## If something fails

Common failure modes and fixes:

| Symptom | Fix |
|---|---|
| `CUDA out of memory` | Reduce `--batch_size` to 2048 in expL_5_6dof_regularized.sh (one-line edit), restart |
| `No such file: 5DOF_8deg.pt_part000.pt` | Re-run the `hf download` for the missing file |
| `Cannot find isaaclab.sh` | Edit the `ISAACLAB_SH=` line at the top of expL_5_6dof_regularized.sh to your actual path |
| Tmux session dies silently | Re-launch with `; exec bash` appended to the command, then re-attach to inspect |
| Grep / aggregation step fails at the end | The training itself succeeded — re-run the aggregator manually: `python3 ../tier4_runs/aggregate_singletask_multiseed.py /path/to/expL_5_6dof_regularized_<TS>` |

If unsure, **stop and report** — better to ask the user than to silently re-run something.

---

## Quick task summary (TL;DR for Claude)

1. `git pull` first
2. Run pre-flight checklist (six checks above)
3. Launch in tmux: `tmux new -d -s expL_machineB ... ./expL_5_6dof_regularized.sh ...`
4. Monitor progress, push commits as seeds complete
5. After all 6 runs done, final commit + (optional) HF model upload
6. Tell user it's done so they can ask Machine A to merge into Table 5.1
