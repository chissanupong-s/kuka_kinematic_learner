# `report_resources/` — Frozen archive for the FYP & future conference paper

**Last updated:** 2026-05-07 09:45 (after Tier-4 experiments + multi-seed sweep complete)
**Final report version:** `report_drafts/FYP_Report_2_Chissanupong_2881058__post_audit_v6_final_20260507_045433.docx`

This is the **cold archive** of every artefact used by the FYP report and intended to be the starting point for the conference paper. Nothing inside should be modified after creation; if you need to re-run, copy the relevant subfolder out and work in the copy.

---

## Final docx versions (in chronological order)

| File | Description |
|---|---|
| `report_drafts/FYP_Report_2_Chissanupong_2881058__pre_audit_snapshot.docx` | Original docx, pre-audit. Frozen. |
| `report_drafts/FYP_Report_2_Chissanupong_2881058__post_audit.docx` | After audit fixes (Findings 1–16) applied. |
| `…__post_audit_v2_20260507_045433.docx` | + Figure 6.1 (t-SNE) and Figure 5.6 (adaptation curves). |
| `…__post_audit_v3_20260507_045433.docx` | + Table 5.4 (Ablation B) 7-DoF row from Experiment A. |
| `…__post_audit_v4_multiseed_20260507_045433.docx` | + Table 5.1 mean ± std (n=3) for adapted rows. |
| `…__post_audit_v5_variance_20260507_045433.docx` | + Honest 7-DoF variance discussion in §5.4, §6.1, abstract. |
| **`…__post_audit_v6_final_20260507_045433.docx`** | **+ Figure 5.7 (multi-seed reproducibility). FINAL.** |

**Use the v6 file for submission.**

---

## Folder layout

```
report_resources/
├─ MANIFEST.md                          (this file)
├─ originals/                           — read-only backups before any AI edit (rollback path)
├─ code/                                — the 3 training scripts + eval scripts + sweep drivers
├─ figures/
│  └─ appendix_tb_plots/<TS>/           — every TB-derived figure rendered for appendix use
├─ report_drafts/                       — every saved docx version
├─ results/
│  ├─ single_task/                      — Stage 1 eval logs that produced Table 5.1 numbers
│  ├─ multitask/
│  ├─ adapt/                            — Stage 3 adapt logs (headline + matched runs)
│  └─ tb_extracted/<TS>/                — every TB scalar of every run, as CSVs + master MD index
├─ checkpoints_metadata/                — seed=42 best.pt metadata
├─ tier4_runs_archive/                  — outputs of Tier-4 experiments (A, D, multi-seed)
├─ tier4_scripts/                       — Tier-4 runner scripts (re-runnable)
├─ methodology_audit/
│  ├─ AUDIT_FINDINGS.md                 — 16-finding audit report
│  ├─ CHANGELOG.md                      — what each script changed
│  └─ scripts/                          — every "apply_*.py" docx-mutation script
└─ prompts_and_briefings/
   └─ CLAUDE_CODE_BRIEFING.md
```

---

## Headline numbers (post-Tier-4)

### Table 5.1 — Adapted (best) row, mean ± std (n = 3 seeds)

| DoF | pos error (m) | ori error (°) | Single-task baseline (m / °) |
|---|---|---|---|
| 5 | 0.0062 ± 0.00002 | 0.807 ± 0.006 | 0.0093 / 1.2039 |
| 6 | 0.0089 ± 0.00003 | 1.330 ± 0.002 | 0.0110 / 1.7136 |
| 7 | 0.0122 ± 0.0019 | 2.276 ± 0.491 | 0.0101 / 2.0853 |

**Key observation:** 5-DoF and 6-DoF are reproducible to <1% of mean across seeds. 7-DoF has high seed sensitivity — seed 42 attains 0.0099 / 1.71° but seeds 1, 2 cluster at 0.0133 / 2.55°. The §5.4 "fourth observation" paragraph and the §6.1 7-DoF interpretation paragraph have both been updated in v5/v6 to acknowledge this honestly.

### Table 5.4 — Ablation B 7-DoF row (Experiment A)

At matched 0.111 hr wall-clock budget:
- From shared init: 0.0099 m / 1.71° (Table 5.1 headline)
- From random init: 0.0265 m / 4.28° (~3× worse on position, ~2.5× worse on orientation)
- Random-init given full ~74 min: 0.00664 / 0.83° (eventually competitive but at 12× the wall-clock cost)

### Tier-4 Experiment D — t-SNE figure (Fig 6.1 in v6)

Penultimate-layer features projected into 2-D. 7-DoF features form a fairly distinct cluster; 5-DoF and 6-DoF overlap substantially (consistent with their shared kinematic chain). Implementation: `tier4_scripts/expD_tsne_20260507_045433.py`.

### Multi-seed reproducibility figure (Fig 5.7 in v6)

Per-seed scatter + n=3 mean±std error bars + single-task baseline reference, both axes. Implementation: `tier4_scripts/render_multiseed_comparison.py`.

---

## TB-extracted evidence pack

`results/tb_extracted/20260507_045433/TB_KEY_NUMBERS.md` is the master index. For every TB event file in the project, every scalar is saved as a CSV with `(step, value, wall_time)`. Use this to back any number cited in Tables 5.1, 5.2, 5.3, 5.4, E.2.

Relevant runs included (14 total):
- Stage 1 single-task FK: 5/6/7-DoF (3 runs)
- Stage 2 shared multitask seed=42 (1 run)
- Stage 3 adaptation seed=42: 5/6/7-DoF (3 runs, headline)
- Tier-4 Experiment A: random-init 7-DoF (1 run)
- Tier-4 multi-seed sweep: seed=1, seed=2 × 5/6/7-DoF (6 runs)

---

## Re-use protocol for the conference paper

1. `cp -r report_resources /path/to/conference_workspace/`
2. Read `methodology_audit/AUDIT_FINDINGS.md` first — apply the corrected methodology.
3. Use scripts under `tier4_scripts/` to re-run experiments (multi-seed, ablations, t-SNE) on a different machine or with extended budgets.
4. The `results/tb_extracted/` directory is the primary numerical reference — every row of every table can be traced back to a CSV here.
5. The §6.4 limitations paragraph in v6 still flags **multi-seed evaluation of Stages 1–2** (which were not re-seeded in this submission) as the first item of the journal extension. Plan: re-run Stage 1 single-task and Stage 2 multitask with seeds 1 and 2 on a faster machine before submission to RA-L / CoRL.

---

## How to roll back

The `originals/` folder contains byte-identical copies of the .docx and the 3 training scripts at the moment audit work began. To revert any single file, see `originals/README.md` for `cp` commands. The original training scripts have not been touched — only docx files and new tier4 scripts have been added.
