# Submission package — FYP supporting files

Chissanupong Saengsint (2881058) — BEng FYP, 2025–26 academic year
Supervisor: Dr Yongjing Wang. School of Engineering, University of Birmingham.

This folder contains the **supporting files** for the FYP final report
*Meta-kinematics framework for the KUKA iiwa 14 across multiple DoF
configurations*. The full graded report is the PDF compiled from
[00_REPORT/](00_REPORT/) and uploaded separately to Canvas; the items in
this package are the source code, trained models, headline experiment
summaries, figures and reproducibility instructions referenced by the
report. Per the marking scheme, "supporting files will not form part of
the project assessment" — this package is provided for transparency and
to support independent reproduction.

## What's in each subfolder

| Folder | Contents |
|---|---|
| [00_REPORT/](00_REPORT/) | The Word source of the final report (`Final_checkpoint_v2.docx`). The PDF is submitted separately to Canvas. |
| [01_CODE/](01_CODE/) | All training, adaptation, evaluation, dataset-generation, aggregator and figure-rendering scripts in Python and bash. |
| [02_TRAINED_MODELS/](02_TRAINED_MODELS/) | The Stage-2 shared meta-kinematics checkpoint (`multitask_fk_best.pt`, 65 MB). This is the single live artefact required by the Stage-3 adaptation pipeline. |
| [03_DATASETS/](03_DATASETS/) | Pointer to the publicly released dataset repository (data files are too large for this package). |
| [04_EXPERIMENT_LOGS/](04_EXPERIMENT_LOGS/) | Per-experiment `summary.txt` files for the five headline experiments cited in Table 5.1, Table 5.3 and Section 5.5 of the report. |
| [05_FIGURES/](05_FIGURES/) | PNG + PDF source for the three bar-chart figures (5.3 position, 5.4 orientation, 5.5 wall-clock time). |
| [06_REPRODUCIBILITY/](06_REPRODUCIBILITY/) | Step-by-step setup guide for a fresh machine and the multi-machine sync protocol used during the project. |
| [07_DECLARATIONS/](07_DECLARATIONS/) | Note on where the ethics questionnaire and risk assessment can be found (they live in the report's appendices, per submission requirements). |

## Quick-start

If you want to reproduce a single result:

1. Read `06_REPRODUCIBILITY/SETUP_NEW_COMPUTER.md` to set up a fresh machine
   (clone repo, install conda environment, download datasets from Hugging Face).
2. Run any launcher in `01_CODE/` (e.g. `expH_adapt_7dof_n3_part000.sh` for the
   7-DoF Adapted (best) row of Table 5.1).
3. Compare the produced summary.txt with the matching one in
   `04_EXPERIMENT_LOGS/` to verify byte-equivalent results (deterministic seed).

## Headline numbers traceability

Every number in Table 5.1 of the report can be traced to a file in this package:

| Table 5.1 cell | Where it comes from |
|---|---|
| 5-DoF Adapted (best) — 6.203 ± 0.041 mm | `04_EXPERIMENT_LOGS/expK_adapt_5_6dof_seed42_*/summary.txt` (seed 42) + `04_EXPERIMENT_LOGS/expB_multiseed_smart_*/summary.txt` (seeds 1, 2) |
| 6-DoF Adapted (best) — 8.900 ± 0.023 mm | same expK + expB |
| 7-DoF Adapted (best) — 9.901 ± 0.014 mm | `04_EXPERIMENT_LOGS/expH_adapt_7dof_n3_part000_*/summary.txt` (3 seeds, all on part000 data) |

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
