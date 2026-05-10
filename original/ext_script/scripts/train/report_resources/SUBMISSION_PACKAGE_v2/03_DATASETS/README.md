# Datasets

The four headline training datasets used in the report are released
publicly through a Hugging Face dataset repository:

  **`Chissanupong/kuka-iiwa-meta-kinematics-data`** (Hugging Face Hub)

The repository contains:

| File | DoF | Sampling step | Approx. samples | Approx. size |
|---|---|---|---|---|
| `5DOF_8deg.pt_part000.pt`   | 5  | 8°  | 16 M | 867 MB |
| `6DOF_12deg.pt_part000.pt`  | 6  | 12° | 35 M | 1.96 GB |
| `7DOF_15deg/7DOF_15deg_part000.pt` | 7 | 15° | 50 M | 2.7 GB |
| `7DOF_15deg/7DOF_15deg_part001.pt` | 7 | 15° | 3.85 M | 206 MB |

Each file is a single `[N, 14]` `torch.Tensor` where the first 7 columns
are the joint angles `q1…q7` (inactive joints clamped to 0 for the
lower-DoF configurations) and the last 7 columns are the end-effector
pose `[x, y, z, qw, qx, qy, qz]`.

Datasets are too large to include in this submission package directly.
Download with:
```bash
hf download Chissanupong/kuka-iiwa-meta-kinematics-data \
    --repo-type dataset --local-dir ./narrowed
```

The Isaac Lab generation script used to produce them is in
[01_CODE/generate_iiwa14_grid_dataset_7DOF.py](../01_CODE/generate_iiwa14_grid_dataset_7DOF.py).
