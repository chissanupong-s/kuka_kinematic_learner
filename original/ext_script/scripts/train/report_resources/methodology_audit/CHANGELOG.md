# Post-audit changelog — FYP Report

Generated: 2026-05-07.

This file lists every change applied to the FYP .docx by `apply_audit_fixes.py`. Use this as the diff record between:

- `report_drafts/FYP_Report_2_Chissanupong_2881058__pre_audit_snapshot.docx` (frozen original)
- `report_drafts/FYP_Report_2_Chissanupong_2881058__post_audit.docx` (audit-applied)

The original file at `train/FYP_Report_2_Chissanupong_2881058.docx` was **not** modified — the post-audit document is a sibling, named with the `__post_audit` suffix.

---

## Changes applied (mapped to AUDIT_FINDINGS.md)

### Body prose

| # | Section | Change |
|---|---|---|
| 4, 5, 7, 8 | §3.3 (paragraph 176) — backbone description | Replaced ResBlock description with the actual implementation (Linear → ReLU → Dropout(0.1) → Linear → skip → ReLU). Added paragraph on `mask_proj` (learned mask conditioning). Reworded the unit-norm quaternion claim to clarify it happens at loss/eval time, not at the model output. Added the per-task standardisation note. Distinguished Stage-1 backbone (no mask_proj) from Stages 2–3 (with mask_proj), and described how Stage-1 ckpts are loaded with `strict=False`. |
| 2 | §3.5 (paragraphs 191–193) — loss function | Eq 3.3 split into Eq 3.3a (Stages 1–2 unified MSE on standardised pose) and Eq 3.3b (Stage 3 weighted raw-units loss with squared geodesic angle). New explanatory paragraph distinguishes the two regimes. |
| 9, 8 | §4.4 (paragraph 207) — software stack | Per-task standardisation made explicit. Held-out-evaluation claim restated to acknowledge that Stage 2's in-script eval is a quick-eval over the training distribution, while held-out test metrics are computed by a separate eval script on a deterministic permutation subset disjoint from the support set. |
| 1 | §4.5 (paragraphs 209–211) — metrics | Position metric redefined from RMSE to mean Euclidean error (Eq 4.1 updated). Orientation-error formula added explicitly. The reported numbers in Chapter 5 are unchanged because they were already mean (not RMSE) values; only the description in §4.5 changes. |
| 5 | §4.6 (paragraph ~214) — statistical methodology | "Same ResMLP backbone defined in Chapter 3" replaced with "share the same residual-block stack and the same FK output head; Stages 2 and 3 additionally include the mask-conditioning projection". Loss-form difference between Stages 1–2 and Stage 3 added. |
| 10 | §5.3 (paragraph 228) — per-DoF adaptation | Added one sentence disclosing the scalarised score `s = 1.0 · E_pos + 0.01 · E_ori` used to select the BEST adapted step for the table. |

### Listings

| # | Listing | Change |
|---|---|---|
| 2 | E.1 (was `quat_distance` returning `1 - |dot|`) | Replaced with `quat_geodesic_loss_rad2`, the actual geodesic-squared-angle term that the Stage-3 adapt code uses. Caption relabelled to point at Eq 3.3b. |
| 4 | E.2 (was a LayerNorm-based block) | Replaced with the actual `ResBlock` from `train_multitask_separate_weight.py`/`adapt_multitask_newest.py`/`train_kinematics_nn_pol_pt_2.py`: Dropout(0.1) inside the block, ReLU after the additive skip, no LayerNorm. |
| 6 | E.3 (was a K-task average) | Replaced with the uniform-task sampler that the script actually uses — one DoF configuration sampled at random per Stage-2 step. |

### Tables

| # | Table | Change |
|---|---|---|
| 4, 12, 13, 15 | E.1 (Architecture & shared training parameters) | Row 5 (Activation): kept ReLU. Row 6 (Normalisation/Regularisation): "LayerNorm (within each residual block)" → "Dropout (p = 0.1) between the two linear layers; no LayerNorm or BatchNorm". Row 11 (Weight decay) → "1×10⁻⁵ (Adam L2, Stage 1); 0 (Stage 2); 1×10⁻⁶ L2-to-init (Stage 3 adaptation)". Row 12 (Mask conditioning): added the mask-projection description. Row 14 (Joint-noise floor) qualified as IK-only. |
| 11, 13, 16 | E.2 (Per-stage training configuration) | LR schedule split per stage: Stage 1 = ReduceLROnPlateau, Stage 2 = Cosine + warmup, Stage 3 = Constant. LR row: 5e-4 / 3e-4 / 1e-5. Batch size: 8192 / 4096 / 8192. Total steps: "Up to 200 epochs (early-stopped on val loss plateau)" / "300 000 (best at step 289 500)" / "100 000". Support/query column: "n/a (50/20/30 train/val/test split)" / "n/a" / "50 k support / 2 M query". Weight decay/L2 split: "1×10⁻⁵ (Adam L2)" / "0" / "1×10⁻⁶ (L2-to-init regulariser)". |

### What was *not* automatically edited

These items remain on the user's checklist and are noted here so they don't get missed:

1. **Figure 3.2 caption**: still describes the architecture without the mask projection. Author should update the figure (or its caption) to show the mask input merging into the input projection.
2. **Listing E.3 prose lead-in** in §3.4 / §E.4: still says "the multi-task average" in the surrounding sentence in some places. The listing has been corrected; the prose should be skim-checked for consistency.
3. **Table 5.3 / Table 5.4 `[fill]` cells**: ablation A & B numbers. Only producible by running Tier-4 Experiment A (7-DoF random-init adapt, ~7 min) and Experiment C (5-DoF random-init shared, ~1.2 hr), per the briefing.
4. **Figure 6.1 (t-SNE)**: still the `[Student to complete]` placeholder in §6.1. Tier-4 Experiment D produces this in ~30 min.
5. **Table 6.1 `[fill from source]`**: KineNN (Diprasetya 2025) and Köker 2004 numbers must be looked up in the source papers, or those rows should be dropped.
6. **Self-assessment box "What aspects… did you enjoy"**: still has the bracketed prompt. Author should write the 50-word reflections.
7. **Appendix D (Generative AI)**: tool table still has placeholder `[Tool name 1]` rows. Author should fill in the AI tools used (Claude Code for the audit, etc.).

### Variables rendered as plain text

Where the original docx used Unicode italic / superscript variables (e.g. `t̂`, `f_θ`, `ℝⁿₖ`), the post-audit script wrote them out as ASCII (`t-hat`, `f_θ`, `R^n_k`) inside the **rewritten** paragraphs only. The unmodified equations (Eq 3.1, Eq 3.2, etc.) keep their original Unicode rendering. This is a cosmetic inconsistency the author can prettify in Word in five minutes if desired — none of it changes the meaning.

---

## How to roll back to the original

```bash
# Discard the post-audit document; the original is untouched
rm /home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/FYP_Report_2_Chissanupong_2881058__post_audit.docx
```

The `report_drafts/FYP_Report_2_Chissanupong_2881058__pre_audit_snapshot.docx` and the file at the original path are byte-identical and were never touched by the script.
