#!/usr/bin/env python3
"""
Tier-4 Experiment A — Step 1
============================
Build a random-init FK ResMLP_Mask checkpoint that the adapt script can load
with strict=True. This is the baseline initialisation for Ablation B (7-DoF):
"What does 0.111 hr of adaptation achieve when starting from a randomly-initialised
model rather than the trained shared meta-kinematics checkpoint?"

Output: /tmp/random_init_resmlp_<TIMESTAMP>.pt
"""

import importlib.util, sys, torch

TIMESTAMP = "20260507_045433"
TRAIN_DIR = "/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train"
TRAIN_PY  = f"{TRAIN_DIR}/train_multitask_separate_weight.py"

# Import the model definition from the actual training script (so the architecture matches exactly)
spec = importlib.util.spec_from_file_location("train_mod", TRAIN_PY)
mod  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Match the seed=42 architecture (hidden=1024, num_blocks=8) but fully randomly initialised
torch.manual_seed(42)
model = mod.ResidualMLP_Mask(in_dim=7, out_dim=7, hidden_dim=1024, num_blocks=8)

out_path = f"/tmp/random_init_resmlp_{TIMESTAMP}.pt"
torch.save({"model_state_dict": model.state_dict()}, out_path)

n_params = sum(p.numel() for p in model.parameters())
print(f"Saved random-init FK ResMLP_Mask to {out_path}")
print(f"Total parameters: {n_params:,}")
print(f"Has fc_in: {'fc_in.weight' in model.state_dict()}")
print(f"Has mask_proj: {'mask_proj.weight' in model.state_dict()}")
print(f"Has fc_out (FK head): {'fc_out.weight' in model.state_dict()}")
print(f"Number of residual blocks: {sum(1 for k in model.state_dict() if k.startswith('blocks.') and k.endswith('.fc1.weight'))}")
