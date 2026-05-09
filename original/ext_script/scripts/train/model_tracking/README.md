# Tracked checkpoints

This directory previously held many model checkpoints (~900 MB total). To keep the repo lightweight,
**only one checkpoint is now tracked**:

## `multitask_fk_best.pt`

- **Stage**: 2 (shared meta-kinematics model)
- **Trained from**: Stage 1 single-task checkpoints averaged
- **Used as `--ckpt`** for all Stage-3 adaptation experiments (`expC_*.sh`, `expD_*.sh`)
- **Size**: ~67 MB
- **Output dim**: 7 (3-D position + 4-D quaternion)
- **Architecture**: ResMLP_Mask, hidden 1024, 8 residual blocks (see `train_kinematics_nn_pol_pt_2.py`)

## Reproducing Stage 1 / Stage 2 from scratch

If you need to retrain Stage 1 (single-task per-DoF) or Stage 2 (shared) from scratch:

1. Generate datasets: `bash data_generation/iiwa/gen_iiwa_data.sh`
2. Train Stage 1: `bash train5_6.sh; bash run_7_only_1.sh`
3. Train Stage 2: outputs `runs/multitask/seperate_weight/1/fk_iiwa_5_6_7/multitask_fk_best.pt`

After that, the Stage 2 output is identical to the tracked `multitask_fk_best.pt` (modulo seed effects).

## Setup after cloning

The Stage-3 adaptation scripts (`tier4_runs/expC_*.sh`, `expD_*.sh`) hardcode a path of:

```
runs/multitask/seperate_weight/1/fk_iiwa_5_6_7/multitask_fk_best.pt
```

After `git clone`, run `setup_checkpoints.sh` (in this directory) to symlink the tracked file to the
expected runs/ location, OR edit the scripts' `SHARED_CKPT=` to point at this directory.
