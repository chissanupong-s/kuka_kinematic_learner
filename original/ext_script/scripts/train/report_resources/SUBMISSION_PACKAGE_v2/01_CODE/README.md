# Source code

All Python and bash scripts referenced by the report. The launcher scripts
(`exp*.sh`) reproduce the headline experiments end-to-end; the Python
files contain the trainer, evaluator, model architecture and aggregators.

## Files

| File | Purpose | Cited in |
|---|---|---|
| `train_kinematics_nn_pol_pt_2.py` | Stage-1 single-task FK trainer (ResMLP). Used to produce the baseline rows of Table 5.1. | Section 4.4, Appendix E.2 |
| `train_kinematics_nn_pol_pt_generalize.py` | Variant of the Stage-1 trainer with explicit dropout, weight-decay, multi-distribution support and early stopping. Used for the exploratory generalisation runs. | Section 4.4 (supplementary) |
| `adapt_multitask_newest.py` | Stage-3 per-DoF adaptation script that takes the Stage-2 shared checkpoint and fine-tunes it. Implements mask conditioning, geodesic-angle loss and the support/query split. | Section 4.5, Equation 3.3b |
| `eval_model_single_task.py` | Held-out test evaluator for any Stage-1, Stage-2 or Stage-3 checkpoint. Produces the position / orientation error numbers reported in every results section. | Section 4.5 |
| `generate_iiwa14_grid_dataset_7DOF.py` | Isaac Lab dataset-generation script for the 7-DoF KUKA iiwa 14 configuration; 5- and 6-DoF variants are produced by changing the active-joint mask in the same script. | Section 4.3 |
| `expB_multiseed_adapt.sh` | Multi-seed Stage-3 adaptation for 5/6/7-DoF (seeds 1 and 2). The earliest multi-seed sweep. | Section 5.4 |
| `expH_adapt_7dof_n3_part000.sh` | The decisive 7-DoF n=3 Adapted run on the larger `part000.pt` dataset that produced the 9.901 ± 0.014 mm headline. | Section 5.4, Section 5.5 |
| `expK_adapt_5_6dof_seed42.sh` | Companion seed-42 reruns for 5/6-DoF with matched `expB` hyperparameters, completing the n=3 Adapted column of Table 5.1. | Section 5.4 |
| `aggregate_adapt_7dof_n3.py` | Aggregator that reads the per-seed `expH` logs and produces the n=3 mean ± std, 95 % CI and one-sample t-test verdict cited in para 187. | Section 5.4, Appendix E.7 |
| `render_figures_5_3_5_4_5_5.py` | Matplotlib renderer that produces Figures 5.3, 5.4 and 5.5 of the report from the final Table 5.1 values. | Figures 5.3–5.5 |

## To re-run a specific result

E.g. to reproduce the 7-DoF Adapted headline:
```bash
conda activate env_isaaclab
bash expH_adapt_7dof_n3_part000.sh
python aggregate_adapt_7dof_n3.py <path-to-this-run's-output-dir>
```

All scripts pin `--seed`, `numpy.random.seed` and `torch.manual_seed` at
the start, so results are bit-deterministic given the same data files.
