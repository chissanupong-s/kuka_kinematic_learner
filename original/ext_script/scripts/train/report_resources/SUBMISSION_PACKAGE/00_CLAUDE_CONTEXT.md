# FYP Report — Context for a New Claude Conversation

**Read this first.** This document brings a brand-new Claude session up to speed on the project so that further report polishing can happen outside Claude Code (e.g. in claude.ai with the files attached).

---

## Quick facts

- **Author:** Chissanupong "Goon" Saengsint, BEng Mechatronic & Robotic Engineering, University of Birmingham
- **Supervisor:** Dr Yongjing Wang
- **Course:** Final Year Project (FYP), 3rd year BEng
- **Title:** *Learning and Transfer of Robot Forward Kinematics Across Varying Degrees of Freedom — A meta-kinematics framework for the KUKA iiwa 14*
- **Submission deadline:** Friday 8 May 2026 (end of day)
- **Submission file:** `01_REPORT_v18_final.docx` (in this folder)
- **Mark target:** the user is aiming for **89-93%** (high First / exceptional First). Honest realistic ceiling is **~93%** because the project is sim-only and doesn't include real-robot validation, accepted publication, or cross-robot transfer.

---

## The project in one paragraph

The work studies whether a single neural network can learn forward kinematics (joint angles → end-effector pose) for **multiple DoF configurations of the same robot** simultaneously, and whether short fine-tuning can specialise it to each configuration cheaply. The KUKA iiwa 14 is configured in 5, 6 and 7 active-joint settings (distal joints progressively locked). A residual MLP (1024-d hidden, 8 blocks) is trained in a three-stage pipeline:

1. **Stage 1** — single-task `ResidualMLP` per DoF
2. **Stage 2** — shared `ResidualMLP_Mask` warm-started by averaging Stage-1 checkpoints, trained jointly across all three datasets with mask conditioning (an additive zero-init linear projection of the binary active-DoF mask into the hidden space)
3. **Stage 3** — per-DoF fine-tune from the Stage-2 checkpoint

Headline result: per-DoF adaptation reduces wall-clock training time by **80.5 % / 92.7 % / 99.5 %** for 5/6/7-DoF respectively while maintaining or improving accuracy over the single-task baselines (5- and 6-DoF improvements are statistically significant under multi-seed n = 3 t-tests at p < 0.05; 7-DoF is comparable on average with one outlier-good seed).

A separate **data-efficiency study** (Figure 5.8) shows that the per-DoF support-set requirement scales sharply with the active-DoF dimensionality — a finding consistent with standard sample-complexity theory (curse of dimensionality).

---

## What's in this folder (18 files + this MD = 19)

| # | File | Purpose |
|---|---|---|
| 00 | `00_CLAUDE_CONTEXT.md` | This file |
| 01 | `01_REPORT_v18_final.docx` | **The submission report.** Ready to read in Word |
| 02 | `02_fig_3_1_pipeline.png` | Pipeline diagram (Stage 1 → Stage 2 → Stage 3). Already inserted in §3.4 of the docx |
| 03 | `03_fig_3_2_architecture.png` | ResMLP_Mask architecture diagram. Already inserted in §3.3 |
| 04 | `04_fig_5_1_singletask.png` | Stage-1 training curves (Option C with best-val markers). Already inserted in §5.1 |
| 05 | `05_fig_5_3_position_bars.png` | Position-error bar chart per DoF (single-task / shared / adapted). §5.4 |
| 06 | `06_fig_5_4_orientation_bars.png` | Orientation-error bar chart. §5.4 |
| 07 | `07_fig_5_5_wallclock_bars.png` | Wall-clock bar chart (log-y). §5.5 |
| 08 | `08_fig_5_7_multiseed_CI.png` | Multi-seed reproducibility with 95% CI shading. §5.4 (the version inserted is the no-CI variant; this CI version is recommended drop-in via Word's Change Picture) |
| 09 | `09_fig_5_8_datasize_efficiency.png` | **The headline data-efficiency curve** (5/6/7-DoF). §5.5 |
| 10 | `10_fig_6_1_tsne.png` | t-SNE projection of penultimate-layer activations. §6.1 |
| 11 | `11_stage1_train_singletask.py` | Stage 1 training script (`ResidualMLP`, no mask) |
| 12 | `12_stage2_train_multitask.py` | Stage 2 training script (`ResidualMLP_Mask`, shared) |
| 13 | `13_stage3_adapt.py` | Stage 3 adaptation script |
| 14 | `14_eval_singletask.py` | Held-out evaluation for Stage 1 |
| 15 | `15_eval_multitask.py` | Held-out evaluation for Stage 2 |
| 16 | `16_datasize_sweep_summary.csv` | All 41 (DoF, K) data-efficiency results — every point in Fig 5.8 |
| 17 | `17_multiseed_summary.csv` | Multi-seed n=3 results (3 seeds × 3 DoF) — backing data for Table 5.1 |
| 18 | `18_AUDIT_FINDINGS.md` | The 16-finding methodology audit that corrected report drift from code |

---

## What has been done (every change tracked)

### Methodology audit (16 findings, all applied)

The report was systematically audited against the source code. Critical fixes:
- **Eq 4.1** — was "RMSE position error", actually mean Euclidean error. Equation now matches the metric.
- **Eq 3.3** — was unified weighted loss across all 3 stages; actually Stage 1+2 use plain MSE on standardised pose, Stage 3 uses weighted raw-units. Now split into 3.3a (Stages 1-2) and 3.3b (Stage 3).
- **`mask_proj` mechanism** — was completely undocumented; now described in §3.3.
- **ResBlock structure** — Listing E.2 said LayerNorm; actual code has Dropout(0.1). Listing fixed.
- **Stage 1 vs Stage 2/3 architecture mismatch** — Stage 1 has no `mask_proj`. §3.3 and §4.6 now reflect this.
- **Multitask sampler** — Listing E.3 said K-task average per step; actual is uniform-task sampling. Listing fixed.
- **Tables E.1 / E.2** — corrected per-stage hyperparameters (LR schedule, weight decay, batch size, total steps).
- **§4.5 / §4.6 / §6.4** — clarified per-task standardisation, the 1-batch quick-eval inside Stage 2, and the BEST-step scoring in Stage 3.

Full audit at `18_AUDIT_FINDINGS.md`.

### Tier-4 experiments completed

- **Experiment A** (Ablation B): random-init 7-DoF adaptation. At the matched 0.111 hr wall-clock budget, random-init reaches only 0.0265 m / 4.28° vs shared-init's 0.0099 m / 1.71° — confirms the shared representation does real work. Filled into Table 5.4 7-DoF row.
- **Experiment B** (multi-seed): Stage-3 adaptation re-run with seeds 1 and 2 across all 3 DoF (combined with seed 42 → n = 3). Mean ± std + 95% CI (Student's t, df = 2) reported in Table 5.1.
  - 5-DoF: 0.0062 ± 0.00002 m / 0.807 ± 0.006° → t-test rejects H₀ vs single-task at p < 0.05
  - 6-DoF: 0.0089 ± 0.00003 m / 1.330 ± 0.002° → t-test rejects H₀ at p < 0.05
  - 7-DoF: 0.0122 ± 0.0019 m / 2.276 ± 0.491° → fails to reject H₀ (seed 42 outlier-good, seeds 1, 2 cluster at 0.0133/2.55°)
- **Experiment C** (data-efficiency): 41 runs total — 13 K values for 5/6-DoF (K = 1k…60k), 15 K values for 7-DoF (K = 1k…100k). Step budget 30 000 per run.
- **Experiment D** (t-SNE): penultimate features projected to 2-D using TSNE(perplexity=30, init="pca"). 7-DoF features form a distinct cluster; 5/6-DoF overlap.

### Statistical rigour

- 95% CIs from Student's t-distribution (df = 2 for n = 3) reported in §5.4 for the Adapted (best) row.
- One-sample t-tests against the single-task baseline (alternative: less-than) for each (DoF, metric) pair.
- The variance/CI evidence is explicitly inventoried in §6.4 limitations.

### Supervisor's §6.3 update

After supervisor feedback ("compare to state-of-the-art baselines, articulate novelty, why our approach"), §6.3 was rewritten as 5 paragraphs:

1. Cross-platform context (Devin, Chen, Ghadirzadeh)
2. SE(3)-aware architectures (Byravan SE3-Nets, SE3-Pose-Nets, Li et al. 2021 SE(3) equivariance)
3. **Three-point novelty list (C1-C5)** — explicit signposting
4. Direct comparison gap honestly acknowledged → routed to journal extension
5. Consistency with KineNN / SE3-Net family

Reference [17] Li et al 2021 added to bibliography.

### Top-mark polish (v17/v18)

- Abstract rewritten to lead with framework + data-efficiency (not just numbers)
- Key Contributions list (C1-C5) added at end of §1.3
- §5.5 includes a **theoretical framing**: connects data-efficiency-scales-with-DoF empirical finding to the curse-of-dimensionality / sample-complexity reading (O(ε⁻ⁿ) under generic Lipschitz)
- §6.1 deeper interpretation: WHY mask conditioning works (additive task-bias preserves multiplicative param sharing); WHY 7-DoF has higher seed variance (higher-dimensional loss landscape)
- §7.3 made concrete: every future-work item now has time/effort estimates (real-robot ~6 weeks, multi-seed Stages 1+2 ~150 GPU-hours, SE(3) baseline porting ~1-2 weeks per architecture, cross-robot ~3 weeks)

### Self-assessment + Appendix D

- Self-assessment Q1, Q2 written in natural-student voice (~60 words each)
- Appendix D Generative AI declaration filled (Claude Code, ChatGPT, GitHub Copilot)

---

## What's left to do (only the user / a new Claude session can do)

### MANDATORY before submission

1. **Figure 4.1 — Isaac Lab simulation environment screenshot.** ~5 min in Isaac Lab. Replace the placeholder caption in §4.1.
2. **Figure A.1 — Project Gantt chart.** Use Table A.1 milestones. Excel/Google Sheets is fine, ~10 min.
3. **Read v18_final docx** for any remaining odd phrasings or typos. Especially:
   - §3.3 (backbone description)
   - §3.5 / Eq 3.3a, 3.3b
   - §5.4 (multi-seed paragraph)
   - §5.5 (data-efficiency, including the new theoretical framing)
   - §6.1 (interpretation, including the new mask + variance paragraph)
   - §6.3 (the supervisor's update)
   - §7.3 (future work with concrete plans)
   - Abstract
4. **Replace the existing Fig 5.7 image with the CI version (`08_fig_5_7_multiseed_CI.png`)** in Word: right-click image → Change Picture → select. (Optional but recommended — adds 95% CI whiskers.)

### NICE-TO-HAVE polish if time permits

5. **Cosmetic Unicode fix:** the rewritten paragraphs spell `t-hat`/`r-hat` in places. Word Find/Replace `t-hat` → `t̂` and `r-hat` → `r̂`. ~5 min.
6. **Table 6.1 prior-work numbers:** if you have ScienceDirect access, look up KineNN (Diprasetya 2025) and Köker (2004) for the comparison table. Otherwise leave as "(see cited paper; not extracted in this submission)".
7. **Skim §6.3 prose** for anything that still reads AI-toned and humanise.

### Out of scope but routed to journal extension (do NOT attempt before Friday)

- Real-robot validation on the iiwa 14 (~6 weeks)
- Multi-seed for Stages 1 and 2 (~150 GPU-hours; needs cluster)
- Direct numerical comparison with SE3-Nets / SE3-Pose-Nets / KineNN ported to FK task (~1-2 weeks per architecture)
- Cross-robot transfer to UR5 / Franka Emika Panda (~3 weeks)

These are explicitly noted in §7.3 of the report as the planned journal extension — that's a legitimate scoping decision and reviewers will accept it.

---

## Headline numbers to remember

For Table 5.1 (held-out test errors, mean Euclidean position / mean orientation):

| | Single-task | Shared meta | Adapted (best) n=3 |
|---|---|---|---|
| 5-DoF | 0.0093 m / 1.20° | 0.0068 m / 0.91° | **0.0062 ± 0.00002 m / 0.81 ± 0.01°** |
| 6-DoF | 0.0110 m / 1.71° | 0.0092 m / 1.37° | **0.0089 ± 0.00003 m / 1.33 ± 0.00°** |
| 7-DoF | 0.0101 m / 2.09° | 0.0109 m / 2.00° | **0.0122 ± 0.0019 m / 2.28 ± 0.49°** |

For wall-clock training time:

| | Single-task | Adapted | Reduction |
|---|---|---|---|
| 5-DoF | 1.873 hr | 0.365 hr | **80.5%** |
| 6-DoF | 4.973 hr | 0.363 hr | **92.7%** |
| 7-DoF | 22.12 hr | 0.111 hr | **99.5%** |

For Fig 5.8 (data efficiency, position error at K = 1k → 60k):

| DoF | K=1 000 | K=60 000 (or K=100k for 7-DoF) | Drop |
|---|---|---|---|
| 5-DoF | 0.0082 | 0.0063 | 23% (essentially flat) |
| 6-DoF | 0.0109 | 0.0088 | 19% (gentle slope) |
| 7-DoF | 0.0565 | 0.0156 (K=80k=K=100k plateau) | **71%** (steep, plateau at K=80-100k) |

---

## How a new Claude session should help

When you (the user) open a new Claude conversation and attach these 19 files, you can ask things like:

- *"Read 01_REPORT_v18_final.docx and check §5.5 / §6.1 for any phrases that read AI-generated rather than student-written. Suggest concrete rewrites."*
- *"Tighten the abstract to be exactly 250 words while keeping all the headline numbers."*
- *"Suggest a punchier title for the FYP report."*
- *"Look at 09_fig_5_8_datasize_efficiency.png and 16_datasize_sweep_summary.csv — write 2 paragraphs of discussion connecting the empirical result to the theoretical curse-of-dimensionality scaling."*
- *"Read 18_AUDIT_FINDINGS.md and confirm none of the 16 findings are still unfixed in the v18 docx."*
- *"Help me draft a 1-page conference paper extension proposal that addresses the deferred items in §7.3."*

Claude won't have access to the full project tree or to my training-script-archive, but with these 19 files it can do meaningful editorial polish, abstract rewrites, presentation prep, and conference-paper drafting.

---

## End of context
