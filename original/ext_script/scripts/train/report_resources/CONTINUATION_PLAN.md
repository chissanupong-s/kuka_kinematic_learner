# Continuation Plan — FYP Report Session

**Last updated:** 2026-05-07 11:15
**Author/owner:** Chissanupong Saengsint (FYP, U Birmingham, supervisor Dr Yongjing Wang)
**Submission deadline:** Friday 8 May 2026

---

## ⚡ TL;DR for a new Claude session that gets "continue the project"

1. **Read this file first** — `report_resources/CONTINUATION_PLAN.md` (you're here).
2. **Activate the conda env before any training/eval/adapt:** `source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab`
3. **Latest submission-ready report:** [`report_resources/report_drafts/FYP_Report_2_Chissanupong_2881058__post_audit_v10_diagrams_20260507_045433.docx`](report_drafts/FYP_Report_2_Chissanupong_2881058__post_audit_v10_diagrams_20260507_045433.docx)
4. **Backup-before-edit policy is mandatory.** Snapshot any script or docx into `report_resources/originals/` with a timestamped suffix before modifying.
5. **Active sweep:** check whether `expC_datasize_sweep_20260507_045433.sh` is running with the grep one-liner in §"How to verify state" below; if not, decide whether to re-launch or skip.
6. **Memory:** `/home/ubuntu/.claude/projects/-home-ubuntu/memory/MEMORY.md` already has user profile, env policy, backup policy, project context.

---

## Project context (one paragraph)

The FYP studies a three-stage meta-kinematics pipeline on the KUKA iiwa 14 across 5/6/7-DoF configurations. **Stage 1** trains a per-DoF `ResidualMLP` (no mask) on each dataset. **Stage 2** averages those checkpoints to warm-start a `ResidualMLP_Mask` (mask projection added) trained jointly on the union of all three datasets. **Stage 3** fine-tunes per-DoF copies of the Stage-2 model. The headline finding is a **80–99% wall-clock reduction** with comparable or better accuracy. The submission report at v10 reports multi-seed mean ± std with 95% CIs and t-test verdicts; the data-size sweep (Fig 5.8, pending) adds the data-efficiency curve.

---

## 📂 File map (treat as ground truth)

```
TRAIN_DIR=/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train

TRAIN_DIR/
├─ adapt_multitask_newest.py          # Stage 3. EDITED: drop_last=False (line 456)
├─ train_multitask_separate_weight.py # Stage 2. UNTOUCHED.
├─ train_kinematics_nn_pol_pt_2.py    # Stage 1. UNTOUCHED.
├─ eval_*.py                          # Eval scripts. UNTOUCHED.
├─ runs/multitask/seperate_weight/1/fk_iiwa_5_6_7/multitask_fk_best.pt
│                                     # Stage-2 seed=42 best.pt (~280 MB).
│                                     # step=289500/300000, cosine LR, base_lr=3e-4.
├─ tier4_runs/                        # All Tier-4 experiments
│  ├─ expA_random_init_7dof_<TS>/     # Experiment A (random-init 7-DoF adapt)
│  ├─ expB_multiseed_smart_<TS>/      # Multi-seed Stage 3 (seeds 1, 2 × 5/6/7 DoF)
│  ├─ expC_datasize_sweep_<TS>/       # Data-size sweep (active OR done)
│  ├─ expD_tsne_<TS>/                 # t-SNE figure
│  ├─ option1_adaptation_curves_<TS>/ # Free convergence curves from existing logs
│  ├─ apply_*.py                      # Docx-mutation scripts (all idempotent)
│  ├─ render_*.py                     # Figure renderers
│  ├─ extract_tb_evidence.py          # TB → CSV evidence extractor
│  └─ *.sh                            # Sweep driver scripts
└─ report_resources/                  # FROZEN ARCHIVE for FYP + conference reuse
   ├─ MANIFEST.md                     # Top-level guide
   ├─ CONTINUATION_PLAN.md            # ← this file
   ├─ originals/                      # Read-only backups before any AI edit
   ├─ code/                           # Snapshot of the 3 training scripts
   ├─ figures/
   │  ├─ figures_3_1_and_3_2_ascii.txt           # Plain-text pipeline + arch diagrams
   │  └─ appendix_tb_plots/<TS>/
   │     ├─ stage1_singletask_*.{png,pdf}        # Stage 1 training curves
   │     ├─ stage2_multitask_*.{png,pdf}         # Stage 2 training curves
   │     ├─ stage3_adapt_{5,6,7}dof_curves.*     # Stage 3 support+query curves
   │     ├─ fig_3_1_pipeline.{png,pdf}           # Fig 3.1
   │     ├─ fig_3_2_architecture.{png,pdf}       # Fig 3.2
   │     ├─ fig_5_3_position_error_bars.*        # Fig 5.3
   │     ├─ fig_5_4_orientation_error_bars.*     # Fig 5.4
   │     ├─ fig_5_5_wallclock_bars.*             # Fig 5.5
   │     ├─ fig_multiseed_reproducibility_*      # Fig 5.7 (no CI)
   │     ├─ fig_multiseed_with_CI_*              # Fig 5.7 alt (with 95% CI shading)
   │     └─ fig_datasize_efficiency_*            # Fig 5.8 (renders when sweep done)
   ├─ report_drafts/                  # ALL .docx versions live here
   │  ├─ README.md                    # version timeline
   │  ├─ FYP_Report_…__pre_audit_snapshot.docx
   │  ├─ FYP_Report_…__post_audit.docx
   │  ├─ FYP_Report_…__post_audit_v2_…docx
   │  ├─ … (v3, v4, v5, v6, v7, v8, v9) …
   │  └─ FYP_Report_…__post_audit_v10_diagrams_…docx   ← LATEST
   ├─ results/
   │  ├─ single_task/                 # Stage 1 eval logs (5/6 DoF)
   │  ├─ adapt/                       # Stage 3 adapt logs (5/6 DoF + 7-DoF sweep csv)
   │  └─ tb_extracted/<TS>/           # Per-tag CSVs + TB_KEY_NUMBERS.md
   ├─ checkpoints_metadata/           # Stage-2 best.pt metadata
   ├─ tier4_scripts/                  # Re-runnable Tier-4 launchers
   ├─ methodology_audit/
   │  ├─ AUDIT_FINDINGS.md            # 16-finding audit
   │  ├─ CHANGELOG.md                 # what each script changed
   │  └─ scripts/                     # every apply_*.py copied here
   └─ prompts_and_briefings/CLAUDE_CODE_BRIEFING.md
```

`<TS>` = `20260507_045433` (the canonical run timestamp for this session).

---

## ✅ What's already done (so a new session doesn't re-do it)

### Methodology audit & report polish (all in v10)
- All 16 audit findings applied (Eq 4.1, Eq 3.3, ResBlock, mask_proj, Tables E.1/E.2, Listings E.1/E.2/E.3)
- Multi-seed mean ± std for Adapted (best) row of Table 5.1
- 95% CIs (Student's t, df=2) reported in §5.4
- One-sample t-test verdicts: 5/6-DoF reject H₀ at p < 0.05; 7-DoF fails (seed 42 outlier)
- Supervisor's §6.3 update with 3-point novelty + SE(3) baseline comparison
- Reference [17] Li et al 2021 added
- Figs 3.1, 3.2, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 6.1 inserted into the docx
- Cosmetic Unicode pass (t-hat → t̂) for the rewritten paragraphs
- List of Figures updated to include 5.6, 5.7

### Tier-4 experiments
- ✅ Experiment A: random-init 7-DoF adapt — 7-DoF Ablation B row of Table 5.4 filled
- ✅ Experiment D: t-SNE of penultimate features — Fig 6.1
- ✅ Experiment B: multi-seed Stage 3 (seeds 1, 2 × 5/6/7 DoF) — Table 5.1 mean ± std + Fig 5.7
- 🟡 Experiment C: data-size sweep (running, see "Active processes")

### Statistical figures
- Fig 5.7 alt with 95% CI shading is rendered; can swap into docx via Word's Change Picture
- TB evidence pack: `report_resources/results/tb_extracted/<TS>/TB_KEY_NUMBERS.md`

### Memory persisted across sessions
`/home/ubuntu/.claude/projects/-home-ubuntu/memory/MEMORY.md` indexes:
- `user_profile.md`
- `feedback_isaaclab_env.md` — must conda activate env_isaaclab
- `feedback_backup_policy.md` — snapshot before any edit
- `project_fyp_kuka_meta_kinematics.md` — project context

---

## 🔄 Active processes

Currently the data-size sweep is running. Verify with:

```bash
pgrep -af "adapt_multitask_newest.py" | head -1
```

The script is `tier4_runs/expC_datasize_sweep_resume_20260507_045433.sh`. Order:
1. 5-DoF: K=25k, 30k, …, 60k @ 10k steps (K=1k–20k already complete at 30k steps)
2. 6-DoF: K=1k, 5k, 10k, …, 60k @ 10k steps
3. 7-DoF: K=1k, 5k, 10k, …, 60k @ 20k steps (reduced from 30k for time)

If the process has died (e.g. machine restart), check `summary.txt` for completed BEST lines and re-launch the resume script — it skips already-completed K values automatically (line 56: `if grep -aE "^\[INFO\] BEST" "$logfile" > /dev/null; then return; fi`).

---

## ⏳ Time table for remaining tasks

| Task | Owner | Est | Depends on |
|---|---|---|---|
| **Sweep finishes** | (auto) | ~3.7 h from 11:15 → **~15:00** | — |
| **Apply data-efficiency results to docx → v11_dataeff** | Claude (auto) | 5 min | sweep done |
| **7-DoF data-efficiency assessment** | Claude (auto) | 5 min | sweep done |
| **Optional 7-DoF rerun at 100k steps** | user decides | 0 / 1.3 h / 17 h | depends on assessment |
| **Self-assessment box** (Tables 2, 3) | **user only** | 10 min | — |
| **Appendix D Generative AI declaration** (Table 14) | **user only** | 5 min | — |
| **Figure 4.1** Isaac Lab screenshot | **user only** | 5 min | — |
| **Figure A.1** Gantt chart | **user only** | 10 min | — |
| **Replace Fig 5.7 with CI version** in Word | **user only** | 1 min | — |
| **Final read-through** | user | 30–60 min | v11 |

### Two scenarios

**Scenario A — sweep finishes cleanly (~15:00)**
- 15:00–15:10: Claude applies v10 → v11_dataeff with Fig 5.8
- 15:10–15:15: Claude reports 7-DoF verdict
- 15:15+: user does final read-through and human-only items
- Submit Friday 8 May at any time

**Scenario B — sweep dies / machine restarts**
- New session reads this file, checks `expC_datasize_sweep_*/summary.txt` for completed BEST lines
- If most K values done: skip the rerun; just apply what we have
- If too few K values done: drop Fig 5.8 from v11 entirely; submit v10 (already submission-ready)

---

## 🔍 How to verify state (in any new session)

```bash
# 1. Where am I?
date && pwd && hostname

# 2. Latest report version
ls -t /home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/report_resources/report_drafts/FYP_Report_*_v*.docx | head -1

# 3. Is the sweep running?
pgrep -af "adapt_multitask_newest.py" | head -3 || echo "no sweep running"

# 4. How many BEST lines do we have for the sweep?
grep -c "^  \[INFO\] BEST" /home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/tier4_runs/expC_datasize_sweep_20260507_045433/summary.txt

# 5. Active run (if sweeping)
grep -ao "\[EVAL\] step=[0-9]\+ | {[^}]\+}" "$(ls -t /home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/tier4_runs/expC_datasize_sweep_20260507_045433/logs/*.log | head -1)" | tail -3

# 6. GPU
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free --format=csv | head -3
```

---

## 🛠 How to continue (concrete recipes)

### "Sweep is done; apply Fig 5.8 to the report"

```bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
python /home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/tier4_runs/apply_datasize_sweep_to_docx.py
# Output: report_drafts/FYP_Report_…__post_audit_v11_dataeff_<TS>.docx
```

The script also prints the 7-DoF verdict (GOOD / BORDERLINE / INSUFFICIENT) at the end.

### "Sweep died, re-launch from where we left off"

```bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
bash /home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/tier4_runs/expC_datasize_sweep_resume_20260507_045433.sh &
```

The resume script auto-skips already-completed K values.

### "Run the 7-DoF rerun at 100k steps for K = 30k, 50k, 60k"

```bash
# Quick patch script: take the resume script and edit only the 7-DoF block
# to use steps=100000 and K_VALUES_FULL=(30000 50000 60000).
# Backup script first per policy.
```

(The user can ask Claude to do this in a one-shot command.)

### "Refresh figures from latest data"

```bash
python /home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/tier4_runs/render_tb_appendix_figures.py
python /home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/tier4_runs/render_multiseed_comparison.py
python /home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/tier4_runs/render_figures_3_1_and_3_2.py
```

### "Re-extract TB evidence pack"

```bash
python /home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/tier4_runs/extract_tb_evidence.py
# Output: report_resources/results/tb_extracted/<TS>/TB_KEY_NUMBERS.md
```

---

## 🚨 Things to NOT do (gotchas)

1. **Do NOT modify the three training scripts directly** without backing up first. The only edited file is `adapt_multitask_newest.py` (drop_last=False); backup is at `report_resources/originals/scripts/adapt_multitask_newest.py.original_20260507_100115`.
2. **Do NOT run training/eval scripts outside `env_isaaclab`** — torch/CUDA/Isaac Lab won't be importable.
3. **Do NOT pkill bare `python`** — it'll kill unrelated processes. Use `pkill -9 -f "<specific script name>"`.
4. **Do NOT touch the v10 docx as the working copy** — open it, save under a different name. Word can corrupt the file if multiple processes edit it concurrently.
5. **Do NOT re-evaluate seed=42 multi-seed best.pt for "fresh" mean ± std** — the existing per-DoF eval logs already have the BEST values; re-running just produces the same numbers.

---

## 📞 Memory pointers (auto-loaded each session)

The /home/ubuntu/.claude/projects/-home-ubuntu/memory/ directory has indexed memories:
- **user_profile**: who the user is (FYP, U Birmingham, supervisor)
- **feedback_isaaclab_env**: must conda activate env_isaaclab
- **feedback_backup_policy**: snapshot before edit
- **project_fyp_kuka_meta_kinematics**: project context, three stages, deadline

A new Claude session will load these automatically and won't need to be told this stuff again. This `CONTINUATION_PLAN.md` is the project-specific complement that goes deeper.

---

## End of plan
