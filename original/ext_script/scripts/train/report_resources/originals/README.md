# `originals/` — read-only backups before any AI-driven edit

These are byte-identical copies of the project files made just before any agent modified them, kept here as a guaranteed rollback path. Each filename has the form `<original_name>.original_<YYYYMMDD_HHMMSS>`.

## How to roll back a single script

```bash
cd /home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train
cp report_resources/originals/scripts/train_multitask_separate_weight.py.original_YYYYMMDD_HHMMSS \
   train_multitask_separate_weight.py
```

## How to roll back the report

```bash
cp report_resources/originals/report/FYP_Report_2_Chissanupong_2881058.docx.original_YYYYMMDD_HHMMSS \
   /home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train/FYP_Report_2_Chissanupong_2881058.docx
```

## Policy

- Files in `originals/` should NEVER be modified after creation.
- Before any new edit, drop a fresh timestamped snapshot here so we have a recoverable point at every state.
- The earlier `code/` folder also contains copies of the same scripts, but those are working copies for the conference re-use; only `originals/` is the rollback point.
