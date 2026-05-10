# Submission package — FYP supporting files

Chissanupong Saengsint (2881058) — BEng FYP, 2025–26 academic year
Supervisor: Dr Yongjing Wang. School of Engineering, University of Birmingham.

This folder contains the **supporting files** for the FYP final report
*Meta-kinematics framework for the KUKA iiwa 14 across multiple DoF
configurations*. The full graded report is the PDF compiled from
[00_REPORT/](00_REPORT/) and uploaded separately to Canvas; the items in
this package are the source code, trained model, figures, reproducibility
instructions and declarations referenced by the report. Per the marking
scheme, "supporting files will not form part of the project assessment"
— this package is provided for transparency and to support independent
reproduction.

## What's in each subfolder

| Folder | Contents |
|---|---|
| [00_REPORT/](00_REPORT/) | The Word source of the final report (`Final_checkpoint_v2.docx`). The PDF is submitted separately to Canvas. |
| [01_CODE/](01_CODE/) | All training, adaptation, evaluation, dataset-generation, aggregator and figure-rendering scripts in Python and bash. |
| [02_TRAINED_MODELS/](02_TRAINED_MODELS/) | The Stage-2 shared meta-kinematics checkpoint (`multitask_fk_best.pt`, 65 MB). This is the single live artefact required by the Stage-3 adaptation pipeline. |
| [03_DATASETS/](03_DATASETS/) | Pointer to the publicly released dataset repository (data files are too large for this package). |
| [04_FIGURES/](04_FIGURES/) | PNG + PDF source for the bar-chart figures used in the report (5.3 position, 5.4 orientation, 5.5 wall-clock time). |
| [05_REPRODUCIBILITY/](05_REPRODUCIBILITY/) | Step-by-step setup guide for a fresh machine, and the multi-machine sync protocol used during the project. |
| [06_DECLARATIONS/](06_DECLARATIONS/) | Ethics questionnaire and risk assessment as separate PDF copies (also bound inside the report PDF as appendices). |

## Quick-start

If you want to reproduce a single result:

1. Read `05_REPRODUCIBILITY/SETUP_NEW_COMPUTER.md` to set up a fresh machine
   (clone repo, install conda environment, download datasets from Hugging Face).
2. Run any launcher in `01_CODE/` (e.g. `expH_adapt_7dof_n3_part000.sh` for the
   7-DoF Adapted (best) row of Table 5.1).
3. Every script is deterministic given its `--seed` argument, so the BEST
   pos_mae values produced match the report's headline numbers byte-for-byte.

## Repository, dataset and model URLs

- Project Git repository: `git clone git@github-wish:chissanupong-s/kuka_kinematic_learner.git`
- Datasets (5/6/7-DoF KUKA iiwa 14 joint–pose pairs): `Chissanupong/kuka-iiwa-meta-kinematics-data` on Hugging Face Hub (dataset repository)
- Trained model checkpoints: `Chissanupong/kuka-iiwa-meta-kin-checkpoints` on Hugging Face Hub (model repository)

Access to private repositories is granted on request to the author via
University email.

## Package metadata

- Total size: ~69 MB (dominated by the 65 MB Stage-2 checkpoint)
- Number of source files: ~30
- Compatible with: Python 3.10, PyTorch ≥ 2.0, NVIDIA Isaac Lab 2024.x
- License: project-internal (academic submission only)
