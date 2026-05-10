# Datasets

Hosted on Hugging Face — too large to include in this package.

`Chissanupong/kuka-iiwa-meta-kinematics-data` (dataset repo)

Files inside:
- `5DOF_8deg.pt_part000.pt`         — 5 DoF, 8° step, ~16 M samples
- `6DOF_12deg.pt_part000.pt`        — 6 DoF, 12° step, ~35 M samples
- `7DOF_15deg/7DOF_15deg_part000.pt` — 7 DoF, 15° step, ~50 M samples
- `7DOF_15deg/7DOF_15deg_part001.pt` — 7 DoF, 15° step, ~3.85 M samples

Each is a `[N, 14]` tensor: cols 0..6 are joint angles, cols 7..13 are pose (x,y,z,qw,qx,qy,qz).

```bash
hf download Chissanupong/kuka-iiwa-meta-kinematics-data \
    --repo-type dataset --local-dir ./narrowed
```

Generated with `01_CODE/generate_iiwa14_grid_dataset_7DOF.py` (Isaac Lab).
