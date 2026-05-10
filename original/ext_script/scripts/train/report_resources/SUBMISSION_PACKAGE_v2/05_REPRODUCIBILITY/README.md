# Reproducibility

Two documents describe how to reproduce results in this project:

- **`SETUP_NEW_COMPUTER.md`** — Step-by-step setup for a fresh GPU
  machine: clone the project repository, install the conda environment,
  download datasets from Hugging Face, run a smoke test. ~10 minutes
  from start to first training run.

- **`MULTI_MACHINE_SYNC.md`** — Protocol used during the project for
  coordinating two GPU machines through a single Git repository
  (each machine pushed its experiment summaries; either machine could
  pull and pick up the other's progress). Documents the conflict
  protocol and the rule that Machine A is the sole writer for `.docx`
  and `.tex` files to avoid binary merge conflicts.

Together with the deterministic seed protocol implemented in every
training script (`torch.manual_seed`, `numpy.random.seed`, `random.seed`
all set at the head of every script), these documents make every
reported result reproducible from the publicly available datasets and
code.
