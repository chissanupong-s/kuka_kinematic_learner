# `report_drafts/` — every saved version of the FYP report

All `.docx` versions of the report live here, ordered chronologically.

## ⭐ Latest submission-ready version

**`FYP_Report_2_Chissanupong_2881058__post_audit_v15_final_20260507_045433.docx`**

This is the file to open in Word and submit. It contains every audit fix, multi-seed mean ± std with 95% CIs and t-tests, supervisor's §6.3 update, all rendered figures (3.1, 3.2, 5.1–5.8, 6.1), self-assessment, Appendix D, and the completed data-efficiency curve.

## Version timeline

| Version | When | What was added on top of the previous |
|---|---|---|
| `…__pre_audit_snapshot.docx` | 04:38 | Frozen pre-audit copy. Read-only baseline. |
| `…__post_audit.docx` | 04:49 | All 16 audit findings applied. |
| `…__post_audit_v2_…` | 05:19 | + Figure 5.6 (adaptation curves) and Figure 6.1 (t-SNE). |
| `…__post_audit_v3_…` | 06:24 | + Table 5.4 7-DoF Ablation B row from Experiment A. |
| `…__post_audit_v4_multiseed_…` | 09:37 | + Table 5.1 mean ± std (n=3) from multi-seed sweep. |
| `…__post_audit_v5_variance_…` | 09:39 | + Honest 7-DoF variance discussion. |
| `…__post_audit_v6_final_…` | 09:41 | + Figure 5.7 (multi-seed reproducibility). |
| `…__post_audit_v7_polish_…` | 10:22 | + Figs 5.1–5.5 inserted; Cosmetic Unicode pass; List of Figures updated. |
| `…__post_audit_v8_discussion_…` | 10:38 | + §6.3 supervisor update (3-point novelty + SE(3) baseline comparison). |
| `…__post_audit_v9_stats_…` | 10:50 | + 95% CIs and t-test verdicts in §5.4. |
| `…__post_audit_v10_diagrams_…` | 11:05 | + Figs 3.1 (pipeline) and 3.2 (architecture). |
| `…__post_audit_v11_dataeff_…` | 11:34 | + Partial Fig 5.8 (5-DoF complete, 6-DoF partial). |
| `…__post_audit_v12_optionc_…` | 12:17 | + Option-C Fig 5.1 with best-val markers and ReduceLROnPlateau explainer. |
| `…__post_audit_v13_personal_…` | 12:55 | + Self-assessment Q1, Q2 and Appendix D Generative AI declaration. |
| `…__post_audit_v14_full_…` | 18:42 | + Fig 5.8 with all 39 data-efficiency points. |
| **`…__post_audit_v15_final_…`** | **18:43** | **+ §5.5 narrative rewritten with actual sweep numbers; Fig 5.8 caption updated. ← SUBMIT THIS ONE.** |

## What's still missing from v15

Two figures only the user can produce:

1. **Figure 4.1** — Isaac Lab simulation environment screenshot (5 min in Isaac Lab)
2. **Figure A.1** — Project Gantt chart from milestone Table A.1 (10 min in Excel/Sheets)

Both are figures-only; the prose around them is already in place.

## How to use this folder

- **For Word editing**: open the latest `__v{N}_…docx` file, edit, save under your own name (e.g. `…final_submission.docx`) — keep the `__v{N}_…` files read-only as the AI-edit history.
- **For roll-back**: any earlier version is intact; you can copy any of these out and continue from there.
- **For diff**: `FYP_Report_extracted_text.md` is a plain-text dump of the pre-audit content; if you want to diff a current version, regenerate via the python-docx extraction snippet in `methodology_audit/scripts/`.

## Where to find the supporting figures

- All TB-derived figures and bar charts: `report_resources/figures/appendix_tb_plots/20260507_045433/`
- Pipeline + architecture: `fig_3_1_pipeline.png` and `fig_3_2_architecture.png`
- Multi-seed reproducibility: `fig_multiseed_reproducibility_*.png` (and `_with_CI_*` variant)
- Data-efficiency curve: `fig_datasize_efficiency_*.png`
- Stage-1 Option C: `fig_5_1_singletask_option_c.png`
- t-SNE: under `tier4_runs/expD_tsne_20260507_045433/`
