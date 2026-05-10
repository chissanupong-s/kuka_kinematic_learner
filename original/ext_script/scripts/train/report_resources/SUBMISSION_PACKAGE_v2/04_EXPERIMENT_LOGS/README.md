# Experiment logs

`summary.txt` files for the five headline experiments cited in the report.
Each summary contains the BEST line from the per-seed adaptation run plus
the per-DoF means and 95% CIs that feed Table 5.1.

| Experiment dir | What it produced | Table/section |
|---|---|---|
| `expB_multiseed_smart_*/` | Adapted (best) seeds 1 and 2 for 5/6/7-DoF on part001 (the original 7-DoF data) | Section 5.4 |
| `expD_seeds_4to10_7dof_*/` | Multi-seed scaling study at 7-DoF (seeds 4–10) | Section 5.4 / Appendix E.7 |
| `expE_part000_test_*/` | Single-seed sanity test of 7-DoF Adapted on the larger part000 data | Section 5.2 footnote, Section 6.4 |
| `expH_adapt_7dof_n3_part000_*/` | THE headline 7-DoF Adapted n=3 run on part000 (seeds 42, 1, 2) | Table 5.1, Section 5.4 |
| `expK_adapt_5_6dof_seed42_*/` | Seed-42 reruns for 5/6-DoF adapted to complete the n=3 Adapted column | Table 5.1, Section 5.4 |

Per-step training logs and per-seed `.pt` checkpoints are not included in
this submission to keep the package size manageable (≈600 MB of model
checkpoints would otherwise be required). The full artefacts are
recoverable by re-running the launcher scripts in [../01_CODE/](../01_CODE/)
on the public dataset — every script is deterministic given the same seed.
