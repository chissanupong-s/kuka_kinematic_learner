# Trained models

`multitask_fk_best.pt` — the Stage-2 shared meta-kinematics checkpoint
(ResMLP with mask conditioning, hidden_dim = 1024, num_blocks = 8). 65 MB.

This is the single trained artefact required by the Stage-3 adaptation
pipeline. All Adapted (best) rows of Table 5.1 are produced by fine-tuning
from this checkpoint with the seed-specific `adapt_multitask_newest.py` runs.

The full set of Stage-1 single-task, Stage-2 shared, and Stage-3 adapted
checkpoints (≈600 MB total) is mirrored to a Hugging Face model repository,
`Chissanupong/kuka-iiwa-meta-kin-checkpoints` (private; access on request).

To verify the checkpoint loads:
```python
import torch
ckpt = torch.load("multitask_fk_best.pt", map_location="cpu", weights_only=False)
print(list(ckpt.keys()))
# expected keys include 'model_state_dict', 'input_mean', 'input_std',
#   'target_mean', 'target_std', 'mode', 'hidden_dim', 'num_blocks'
```
