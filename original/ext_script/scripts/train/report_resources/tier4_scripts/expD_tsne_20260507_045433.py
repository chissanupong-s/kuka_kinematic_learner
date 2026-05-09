#!/usr/bin/env python3
"""
Tier-4 Experiment D — t-SNE feature projection (Figure 6.1)
===========================================================
Project the penultimate-layer activations of the shared seed=42 ResidualMLP_Mask
onto a 2D plane using t-SNE, on held-out samples drawn from each DoF dataset.

Goal: produce a figure that visually argues the shared backbone has learned a
unified, smoothly-varying representation across DoF settings rather than three
disjoint clusters. Saved as a PNG with the run timestamp in the filename so it
is unambiguous which run produced it.
"""
import importlib.util, sys, time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


TIMESTAMP = "20260507_045433"
TRAIN_DIR = Path("/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train")
DATA_DIR  = Path("/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/data/narrowed")
OUT_DIR   = TRAIN_DIR / "tier4_runs" / f"expD_tsne_{TIMESTAMP}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_LINES = []
def log(s):
    print(s)
    LOG_LINES.append(s)

t0 = time.time()
log(f"=== EXPERIMENT D — t-SNE feature projection (timestamp {TIMESTAMP}) ===")

# Load the model class from the live training script
spec = importlib.util.spec_from_file_location("train_mod", str(TRAIN_DIR / "train_multitask_separate_weight.py"))
mod  = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

ckpt_path = TRAIN_DIR / "runs/multitask/seperate_weight/1/fk_iiwa_5_6_7/multitask_fk_best.pt"
log(f"Loading shared seed=42 checkpoint: {ckpt_path}")
ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
sd = ckpt["model_state_dict"]
hidden_dim = sd["fc_in.weight"].shape[0]
num_blocks = sum(1 for k in sd if k.startswith("blocks.") and k.endswith(".fc1.weight"))
log(f"Shared model architecture: hidden_dim={hidden_dim}, num_blocks={num_blocks}")
log(f"  step={ckpt.get('step')}, total_steps={ckpt.get('total_steps')}, base_lr={ckpt.get('base_lr')}")

model = mod.ResidualMLP_Mask(in_dim=7, out_dim=7, hidden_dim=hidden_dim, num_blocks=num_blocks)
model.load_state_dict(sd, strict=True)
model.eval()

# Hook the output of the LAST residual block (the penultimate-layer features just before fc_out)
penult = []
def hook(module, inp, out):
    penult.append(out.detach().cpu().numpy())
model.blocks[-1].register_forward_hook(hook)

# Pull held-out samples from each DoF dataset.
# We use the per-task standardisation stored in ckpt['task_norm'] so the input
# distribution the model sees here exactly matches what it was trained on.
data_paths = {
    5: DATA_DIR / "5DOF_8deg.pt_part000.pt",
    6: DATA_DIR / "6DOF_12deg.pt_part000.pt",
    7: DATA_DIR / "7DOF_15deg/7DOF_15deg_part001.pt",
}
task_norm = ckpt["task_norm"]   # keys: '5dof', '6dof', '7dof'

N_PER_DOF = 1000
labels = []

# Use a permutation seed disjoint from the support seed (42) so we project
# samples that the adapt scripts have NOT seen as their support set.
rng = np.random.RandomState(20260507)

for dof, path in data_paths.items():
    raw = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        for key in ("data", "tensor", "samples"):
            if key in raw:
                raw = raw[key]; break
    if not torch.is_tensor(raw):
        raise RuntimeError(f"Could not extract tensor from {path}; type={type(raw)}")
    log(f"DoF {dof}: dataset shape {tuple(raw.shape)}")

    idx = rng.permutation(raw.shape[0])[:N_PER_DOF]
    sample = raw[idx].float()
    q = sample[:, :7]   # joint columns

    # Per-task standardisation — same transform the model was trained with
    tn = task_norm[f"{dof}dof"]
    x_mean = torch.tensor(tn["x_mean"], dtype=torch.float32).view(1, 7)
    x_std  = torch.tensor(tn["x_std"],  dtype=torch.float32).view(1, 7) + 1e-8
    q_norm = (q - x_mean) / x_std

    # Active-DoF mask (same convention as training)
    mask = torch.zeros_like(q_norm)
    mask[:, :dof] = 1.0

    with torch.no_grad():
        _ = model(q_norm, mask)
    labels.extend([dof] * N_PER_DOF)

X = np.vstack(penult)   # (N_total, hidden_dim)
log(f"Penultimate features collected: shape={X.shape}")

log("Running t-SNE (this may take 1–3 min)...")
Z = TSNE(n_components=2, random_state=0, perplexity=30, init="pca",
         learning_rate="auto").fit_transform(X)

# Plot
fig, ax = plt.subplots(figsize=(6.5, 5.5))
colors = {5: "#1f77b4", 6: "#ff7f0e", 7: "#2ca02c"}
markers = {5: "o", 6: "s", 7: "^"}
labels_arr = np.array(labels)
for dof in (5, 6, 7):
    m = labels_arr == dof
    ax.scatter(Z[m, 0], Z[m, 1], s=10, alpha=0.55,
               c=colors[dof], marker=markers[dof],
               label=f"{dof} DoF (n={int(m.sum())})", edgecolors="none")
ax.legend(loc="best", frameon=True, fontsize=10)
ax.set_xlabel("t-SNE dim 1")
ax.set_ylabel("t-SNE dim 2")
ax.set_title("Penultimate-layer features of the shared meta-kinematics model")
fig.tight_layout()

out_png = OUT_DIR / f"fig_features_{TIMESTAMP}.png"
out_pdf = OUT_DIR / f"fig_features_{TIMESTAMP}.pdf"
fig.savefig(str(out_png), dpi=200, bbox_inches="tight")
fig.savefig(str(out_pdf), bbox_inches="tight")
log(f"Saved figure: {out_png}")
log(f"Saved figure: {out_pdf}")

# Also save the raw features and projection so a future paper can re-render
np.savez(str(OUT_DIR / f"features_and_projection_{TIMESTAMP}.npz"),
         X=X, Z=Z, labels=np.array(labels))
log(f"Saved features+projection: {OUT_DIR / f'features_and_projection_{TIMESTAMP}.npz'}")

dt = time.time() - t0
log(f"Total time: {dt:.1f} s")

with open(OUT_DIR / f"run_log_{TIMESTAMP}.txt", "w") as f:
    f.write("\n".join(LOG_LINES) + "\n")
log("DONE")
