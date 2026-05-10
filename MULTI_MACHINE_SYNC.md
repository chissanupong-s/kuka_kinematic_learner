# Multi-machine sync protocol

Two (or more) machines coordinating on the same project, using **GitHub** as the
shared source of truth for code + small text outputs and **Hugging Face** for
large data + (optionally) trained checkpoints.

## What's tracked where

| Artefact | Location | Reason |
|---|---|---|
| Source code (`.py`, `.sh`) | git | small, central reference |
| `tier4_runs/*.sh` launchers | git | both machines run the same protocol |
| `tier4_runs/*.py` aggregators | git | both machines aggregate the same way |
| `tier4_runs/<run>/summary.txt` | git | the result line each machine produces |
| `tier4_runs/<run>/logs/*.log` | git | full eval/train logs, small text |
| `*.docx` (latest report drafts) | git | report content |
| `narrowed/*.pt` (training data) | **HF dataset** `Chissanupong/kuka-iiwa-meta-kinematics-data` | too large for git (~5.7 GB) |
| `multitask_fk_best.pt` (Stage-2 ckpt) | git (66 MB, tracked specifically) | needed by adaptation pipeline |
| `tier4_runs/<run>/ckpts/*.pt` | NEITHER (regenerable from logs+code) | per-run checkpoints; only the eval numbers matter cross-machine |
| `tier4_runs/<run>/tb/` (TensorBoard) | NEITHER (large binary, regenerable) | local-only |
| `originals/` (pre-edit snapshots) | NEITHER | each machine keeps its own backups |

## Daily workflow

### Before starting work on either machine

```bash
cd ~/kuka_kinematic_learner
git pull origin main
```

This brings down whatever the other machine pushed, including any new
launcher scripts, completed `summary.txt` files, or `.docx` updates.

### After a meaningful piece of work

Whenever a machine finishes an experiment block (or a docx edit, or a new
script), commit + push so the other machine can see it:

```bash
git add original/ext_script/scripts/train/tier4_runs/*.sh \
        original/ext_script/scripts/train/tier4_runs/*.py \
        original/ext_script/scripts/train/tier4_runs/<your_new_run_dir>/summary.txt \
        original/ext_script/scripts/train/tier4_runs/<your_new_run_dir>/logs/*.log
git add original/ext_script/scripts/train/report_resources/report_drafts/*.docx
git commit -m "machine-A: expL 7-DoF n=3 done; pos = 9.x ± 0.x mm"
git push origin main
```

(If you also added a new training script (`train_kinematics_*.py`), include it.)

### Resolving conflicts

The two machines should write to **different timestamped run directories**
(`expL_singletask_regularized_<TS>` vs `expL_5_6dof_regularized_<TS>`), so
file conflicts are unlikely. The most likely conflict source is the docx —
only edit the docx on **one machine at a time**.

If git pull says there's a conflict in the docx:
1. Pick the version the other machine pushed (their edits are usually later)
2. `git checkout --theirs <file>`
3. Apply your local-only edits on top
4. Commit + push

### Push-rejected-because-remote-is-ahead — handling rule (for Claude / automation)

This **WILL** happen routinely when both machines push close together. The
expected pattern is:

```
$ git push origin main
! [rejected]        main -> main (fetch first)
error: failed to push some refs to '...'
hint: Updates were rejected because the remote contains work that you do
      not have locally...
```

**Standing rule for any Claude instance encountering this:**

1. Do **not** ask the user. This is the normal cross-machine race condition;
   the user has already authorised the push.
2. Do **not** use `--force` or `--force-with-lease`. Ever.
3. **Pull first, then re-push:**
   ```bash
   git pull origin main
   git push origin main
   ```
4. If the pull merges cleanly (the typical case for timestamped run dirs),
   the second push succeeds. Done — report the final commit hash.
5. If the pull surfaces a real merge conflict (almost always means the
   .docx — see "Resolving conflicts" above), **stop and report** the
   conflicting file to the user. Do not attempt to resolve a binary docx
   merge automatically.

For non-Claude users: same flow, manual.

A convenience alias to make this one command everywhere:
```bash
git config --global alias.sync '!git pull origin main && git push origin main'
# then use: git sync   instead of separate pull + push
```

For tier4_runs/, conflicts only happen if both machines wrote to the same
run-dir (which they shouldn't — timestamps make this nearly impossible).

## Hugging Face: when to use it

For the FYP timeline, **HF is only needed for the dataset** (already set up at
`Chissanupong/kuka-iiwa-meta-kinematics-data`). Both machines pull data from
there during initial setup.

For the conference-paper extension (later), HF can host:
- A new model repo for sharing trained checkpoints (so machine A's expJ ckpt
  can be downloaded by machine B for cross-evaluation)
- New datasets (e.g., the multi-distribution 5/6-DoF data the user discussed
  generating)

Set up an HF model repo only when needed:
```bash
hf repo create --type model Chissanupong/kuka-iiwa-meta-kin-checkpoints
hf upload Chissanupong/kuka-iiwa-meta-kin-checkpoints \
    /path/to/ckpt.pt fk_pose_best_expL_dof7_seed42.pt
```

## Quick "where am I" check (run on either machine)

```bash
cd ~/kuka_kinematic_learner
git log --oneline -10                          # last 10 commits across both machines
ls -d original/ext_script/scripts/train/tier4_runs/expL_*  # which expL runs exist locally
grep -l "Mean position error" \
   original/ext_script/scripts/train/tier4_runs/expL_*/logs/*.log  # which seeds are done
```

## When asking Claude to update the report

If you ask Claude on either machine to write/update the docx, just make sure
that machine has **just done** `git pull` so it sees the latest state from the
other. Then the docx edits Claude makes will be based on the most up-to-date
results.

After Claude commits its docx changes and pushes, the *other* machine should
`git pull` before doing any further docx edits — otherwise you risk merge
conflicts on the binary docx file.

**Rule of thumb:** docx editing happens on **one** machine at a time, and that
machine pushes before the other touches it.
