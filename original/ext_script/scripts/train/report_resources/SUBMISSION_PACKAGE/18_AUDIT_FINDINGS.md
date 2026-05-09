# FYP Report — Methodology Audit Against Source Code
**Auditor pass:** 2026-05-07 (24 h before submission)
**Report:** `FYP_Report_2_Chissanupong_2881058.docx`
**Scripts audited:**
- Stage 1 (single-task): `train_kinematics_nn_pol_pt_2.py`
- Stage 2 (shared multitask): `train_multitask_separate_weight.py`
- Stage 3 (per-DoF adapt): `adapt_multitask_newest.py`

This document is a **claim-by-claim audit** of the report's methodology and appendix against what the three training scripts actually do. The aim is to flag every place where an examiner with access to the code (e.g. the supervisor) could spot a discrepancy and dock marks. Each finding is paired with a **concrete edit** that fixes the report without re-running any experiments.

---

## SEVERITY RANKING

| # | Finding | Severity | Risk if unaddressed | Fix effort |
|---|---|---|---|---|
| 1 | §4.5 / Eq 4.1 calls position metric **RMSE**, but the headline numbers are **Mean Euclidean error** | **CRITICAL** | Examiner spots the math mismatch; whole results table looks wrong | 1 sentence + 1 equation |
| 2 | Eq 3.3 / Listing E.1: unified pos+ori weighted loss is described, but **Stages 1 & 2 actually use plain MSE on the normalised 7-D vector** | **CRITICAL** | Methodology chapter doesn't describe what was trained | Rewrite Eq 3.3 + 1 paragraph |
| 3 | `mask_proj` (learned mask conditioning) is **completely missing** from §3.3 / Fig 3.2 / Table E.1 | **CRITICAL** | Architecture in report is not the architecture in code | 1 sentence + Fig caption + 1 row in E.1 |
| 4 | Listing E.2 ResBlock shows **LayerNorm**; the code has **Dropout** (no LayerNorm) | **CRITICAL** | Pseudocode contradicts source | Replace LayerNorm with Dropout in listing + Table E.1 row |
| 5 | §4.6 / §3.3 claim Stage 1, Stage 2, Stage 3 share the same backbone; **Stage 1 single-task uses `ResidualMLP` (no `mask_proj`)** while Stages 2/3 use `ResidualMLP_Mask` | HIGH | "Same backbone" claim is technically false | One-paragraph clarification |
| 6 | Listing E.3 shows multi-task average loss per step; **Stage 2 actually picks one task uniformly per step** | HIGH | Pseudocode mis-describes the training loop | Replace pseudocode |
| 7 | §3.3 says "the quaternion part is normalised to unit norm" at the **model output**; in reality the model returns un-normalised 4-D and unit-norm projection happens **only inside the loss/eval functions** | MEDIUM | Architecture claim is overstated | Reword 1 sentence in §3.3 |
| 8 | Stage 2 uses **per-task standardisation** (each DoF has its own x_mean/x_std); §4.4 just says "inputs and targets are standardised" | MEDIUM | Subtle but a sharp examiner could ask | 1 sentence |
| 9 | §4.4 says "validation loss curves are used to monitor training, and the best checkpoint on a held-out split is retained"; the **Stage 2 script has no held-out split** — best is selected on a 1-batch quick-eval that samples from the training distribution | MEDIUM | Held-out-leak concern | 2-sentence honest restatement |
| 10 | Stage 3 best-step is selected by `score = pos_mae_m + 0.01·ori_deg`; report calls these "BEST" without disclosing the scalarisation | MEDIUM | Cherry-picking concern if asked | 1 sentence |
| 11 | Table E.2 lumps Stage 1 LR schedule with Stage 2 (cosine + warmup); **Stage 1 actually uses `ReduceLROnPlateau`** with default `lr=5e-4`, `weight_decay=1e-5` | MEDIUM | Reproducibility table is wrong | Split row across columns |
| 12 | `std_floor_q_deg = 1.0` in Table E.1 — this **only applies to IK mode** (FK ignores it). Listed as if it applies to FK | LOW | Minor irrelevant param | Drop the row or qualify |
| 13 | Table E.1 lists `weight_decay=0` for Stages 1–2; **Stage 1 default has `weight_decay=1e-5`** (Adam L2) | LOW | Minor reproducibility error | Update table row |
| 14 | `aux_loss_weight=0.03` is in the saved Stage-2 ckpt metadata, but for FK it is **never used** by the loss; report Table E.1 lists it as if it weights position/orientation | LOW | Not load-bearing | Update table comment |
| 15 | Dropout (p=0.1) inside every residual block is **not mentioned anywhere** in the report | LOW | Minor omission | 1 row in E.1 |
| 16 | Multitask seed=42 best ckpt has `total_steps=300000`, `step=289500`. Table E.2 claims "1 000 000 (max)" | LOW | Cosmetic mismatch | Update table |

**Of these, items 1–4 are the high-risk ones for the FYP mark band.** They are documented + code-grounded mismatches that can be checked in seconds by anyone with the source.

---

## DETAILED FINDINGS WITH SUGGESTED FIXES

### Finding 1 — Position metric is Mean Euclidean error, not RMSE

**Report (§4.5, Eq 4.1):**
> "The position error is reported as the root-mean-square error in metres … `Epos = √( (1/N) Σ ‖t̂ − t‖² )`."

**Source (eval_model_single_task.py L356, L382, L395):**
```python
pos_err = torch.linalg.norm(pos_pred - pos_true, dim=1)   # per-sample L2
mean_pos_m = sum_pos / max(seen, 1)                       # MEAN of L2
print(f"Mean position error: {mean_pos_m:.6f} m")
print(f"RMSE position error: {rmse_pos_m:.6f} m")          # both printed
```

**Source (adapt_multitask_newest.py L268-269):**
```python
pos_mae = pos_l2_all.mean().item()                        # MEAN of L2
pos_rmse = torch.sqrt((pos_l2_all ** 2).mean()).item()    # RMSE
```

**Eval logs confirm the mismatch:**
| DoF | Mean position (log) | RMSE position (log) | Reported in Table 5.1 |
|---|---|---|---|
| 5 | **0.009270** | 0.010292 | **0.0093** ← Mean |
| 6 | **0.010983** | 0.012117 | **0.0110** ← Mean |

So Table 5.1 reports Mean Euclidean position error throughout, but Eq 4.1 defines RMSE. This is mathematically inconsistent.

**FIX (in §4.5, Eq 4.1):**

Replace:

> *"The position error is reported as the root-mean-square error in metres between the predicted and ground-truth end-effector positions:*
> *Epos = √( (1/N) Σ ‖t̂ − t‖² ),    (4.1)"*

With:

> *"The position error is reported as the mean Euclidean error in metres between the predicted and ground-truth end-effector positions:*
> *E_pos = (1/N) Σᵢ ‖t̂ᵢ − tᵢ‖₂,    (4.1)*
> *which is the average length of the position-error vector across the held-out split. The orientation error is the mean geodesic angle between the predicted and ground-truth quaternions, reported in degrees: E_ori = (1/N) Σᵢ 2·arccos(|⟨r̂ᵢ, rᵢ⟩|) · (180/π)."*

Mean Euclidean error is the standard metric for FK accuracy in robotics (it has a direct physical interpretation: average pose error in mm). Acknowledge it explicitly rather than claim RMSE.

---

### Finding 2 — Stage 1 / Stage 2 FK loss is not what Eq 3.3 describes

**Report (Eq 3.3, §3.5):**
> "ℒ(θ) = (1/K) Σ E[ λ_p ‖t̂ − t‖² + λ_o ℓ_quat(r̂, r) ]"
> where ℓ_quat is a quaternion distance term and λ_p, λ_o balance the components.

Listing E.1 (`quat_distance`) defines `1.0 - dot.abs()`.
Table E.1 lists `λ_p = 0.03` (Stages 1–2), `λ_o = 0.03` (Stages 1–2).

**Source (Stage 1, train_kinematics_nn_pol_pt_2.py L490, L555):**
```python
criterion = nn.MSELoss()      # for FK
...
y_pred = model(x)             # [B, 7] = [3 pos, 4 quat], normalised
loss = criterion(y_pred, y)   # scalar MSE on the entire 7-d normalised vector
```

**Source (Stage 2, train_multitask_separate_weight.py L487-489):**
```python
if mode == "fk":
    yhat = model(x, mask)
    loss = mse(yhat, y)        # nn.MSELoss(); pos_weight/ori_weight unused
```

So for FK in Stages 1 and 2:
- The position and orientation components are not separately weighted; they are concatenated and minimised by a single MSE.
- The orientation term is plain MSE between the 4 quaternion components in normalised space, not a geodesic distance.
- `λ_p` and `λ_o` (and `quat_distance`) are not used in Stage 1/2 FK training.
- The `pos_loss_weight=0.03`, `ori_loss_weight=0.12` stored in the seed=42 Stage-2 best.pt metadata are **inactive** for FK — they are read from the args but the FK branch never uses them.

The actual quaternion geodesic loss `quat_geodesic_loss_rad` in the script is used **only for IK auxiliary loss**, not for FK.

Stage 3 (adapt) is the only stage where Eq 3.3-style separate weighting is real:

**Source (adapt_multitask_newest.py L504-515):**
```python
loss_pos = (dp ** 2).mean()                          # raw metres²
ang = quat_angle_rad(...)                            # geodesic angle, sign-invariant
loss_ori = (ang ** 2).mean()                         # rad²
loss = args.pos_weight * loss_pos + args.ori_weight * loss_ori
```

**FIX (rewrite Eq 3.3 + caption):**

The honest statement is that Stages 1–2 train against a unified MSE on a standardised pose vector, and Stage 3 trains against a weighted physical-units objective. Do not attempt to retrofit a single equation across all three stages.

Replace §3.5 (Loss function) with:

> *"The supervised objective differs slightly across the three training stages, reflecting the different roles each stage plays. Stages 1 and 2 (single-task and shared meta-kinematics) minimise a single mean-squared error on the standardised concatenated pose vector,*
> *L⁽¹,²⁾(θ) = E_(q,p) [ ‖f_θ(q) − p‖² ],    (3.3a)*
> *where p = [t, r] is the position-and-quaternion target, and inputs and targets are per-task standardised before training. This unified MSE is appropriate while the model is being asked to learn the joint structure of the position and orientation outputs simultaneously.*
> *Stage 3 (per-DoF adaptation) operates in raw physical units and weights the two components explicitly,*
> *L⁽³⁾(θ) = λ_p · E ‖t̂ − t‖² + λ_o · E [ θ_geo(r̂, r)² ],    (3.3b)*
> *where t and r are the un-standardised position and quaternion targets, t̂ and r̂ the corresponding model outputs, and θ_geo(r̂, r) = 2·arccos(|⟨r̂, r⟩|) is the geodesic angle between the unit-normalised quaternions in radians. The split of the adaptation loss into raw-unit position MSE and squared geodesic angle is what allows the relative weight of position and orientation to be controlled in metres-and-degrees terms during the short fine-tuning stage."*

Then update Table E.1: under "Position-loss weight", split into Stage-1/2 ("loss is unified MSE; no separate weight") and Stage-3 ("λ_p = 1.0"). Same for the orientation row.

Update Listing E.1 caption to read "Quaternion-geodesic-angle term used inside the **Stage 3 (adaptation)** loss (3.3b)."

---

### Finding 3 — `mask_proj` is missing from the architecture description

**Report (§3.3 and Fig 3.2 caption):**
> "Each residual block applies a linear projection, layer normalisation and a ReLU activation, … Inputs and targets are standardised before training. The shared backbone is reused without modification across all three DoF configurations … the task difference is encoded entirely through the clamped joint inputs and through the fine-tuned parameters produced during adaptation."

**Source (train_multitask_separate_weight.py L280-294, also identical in adapt_multitask_newest.py L105-118):**
```python
class ResidualMLP_Mask(nn.Module):
    def __init__(self, ...):
        self.fc_in     = nn.Linear(7, hidden_dim)
        self.mask_proj = nn.Linear(7, hidden_dim, bias=False)   # <-- not in report
        self.blocks    = nn.ModuleList([ResBlock(hidden_dim) ...])
        self.fc_out    = nn.Linear(hidden_dim, 7)
        nn.init.zeros_(self.mask_proj.weight)                   # init to zero

    def forward(self, x7, mask7):
        h = self.act(self.fc_in(x7) + self.mask_proj(mask7))    # additive mask cond.
        ...
```

The active-DoF mask `m_τ ∈ {0,1}⁷` is fed to the model *as a separate input* and projected into the hidden space, then **added** to the input projection. The mask projection is initialised to zero so that an old un-mask-aware checkpoint can be loaded `strict=False` and the model behaves identically at init; during training the `mask_proj` weights become non-zero (verified — `Σ |w_mask_proj| = 545.3` in the seed=42 best ckpt).

So the model's input is not just "joints zero-clamped to indicate inactive DoF"; the model also receives the binary mask explicitly and is allowed to use it to condition its computation.

**FIX:**

In §3.3, after the existing sentence about "common, fixed-dimensional joint representation":

> *"In addition to the clamped joint vector, the model also receives the binary active-joint mask m_τ ∈ {0,1}⁷ as an explicit auxiliary input. The mask is projected into the hidden space by a learned linear layer with no bias and added to the input projection of the joint vector, so that the model can adapt its internal representation to the active-DoF count rather than relying solely on the zero-clamped joint values. The mask projection is initialised to zero, which preserves backward compatibility with non-mask-aware single-task checkpoints when they are used to warm-start the shared meta-kinematics model in Stage 2."*

Update Fig 3.2 (a) to draw a second input (mask, 7 → 1024) merging into the input of block 0.
Add a row to Table E.1: "Mask conditioning: linear projection of 7-D mask, additive with input projection, zero-initialised."

---

### Finding 4 — Listing E.2 (ResBlock) shows LayerNorm; code has Dropout

**Report (Listing E.2):**
```
class ResBlock(nn.Module):
    def __init__(self, dim=1024):
        ...
        self.fc1   = nn.Linear(dim, dim)
        self.norm  = nn.LayerNorm(dim)            # <-- not in code
        self.act   = nn.ReLU(inplace=True)
        self.fc2   = nn.Linear(dim, dim)
    def forward(self, x):
        y = self.fc1(x)
        y = self.act(self.norm(y))
        y = self.fc2(y)
        return x + y
```

**Source (train_multitask_separate_weight.py L265-277, identical in adapt_multitask_newest.py L90-102 and train_kinematics_nn_pol_pt_2.py L285-296):**
```python
class ResBlock(nn.Module):
    def __init__(self, dim, p_drop: float = 0.1):
        self.fc1  = nn.Linear(dim, dim)
        self.fc2  = nn.Linear(dim, dim)
        self.act  = nn.ReLU()
        self.drop = nn.Dropout(p_drop)            # <-- in code, not in report
    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.drop(h)
        h = self.fc2(h)
        return self.act(h + x)                    # ReLU after the skip
```

Three differences:
- No `LayerNorm` anywhere in the actual block.
- Dropout(0.1) between the two linear layers.
- A second `ReLU` is applied **after** the skip connection.

**FIX:**

Replace Listing E.2 with the actual block:

```python
Listing E.2  Residual block used inside the ResMLP backbone of Figure 3.2(a).

class ResBlock(nn.Module):
    def __init__(self, dim: int = 1024, p_drop: float = 0.1):
        super().__init__()
        self.fc1  = nn.Linear(dim, dim)
        self.fc2  = nn.Linear(dim, dim)
        self.act  = nn.ReLU()
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.drop(h)
        h = self.fc2(h)
        return self.act(h + x)        # ReLU after additive skip
```

In §3.3, replace the description of the residual block:

> "Each residual block applies a linear projection, **a ReLU activation, dropout (p = 0.1) and a second linear projection, with the result added to the block's input through a skip connection and a final ReLU**. ~~Inputs and targets are standardised before training.~~"

Update Table E.1: change "Normalisation: LayerNorm" → "**Regularisation: Dropout (p = 0.1) inside each residual block; no normalisation layer**".

---

### Finding 5 — Stage-1 vs Stage-2 backbone is not literally "the same backbone"

**Report (§3.3):**
> "The shared backbone is reused without modification across all three DoF configurations and across both the meta-kinematics training stage and the per-DoF adaptation stage."

**Report (§4.6):**
> "the three variants — single-task, shared meta-kinematics and per-DoF adapted — share the same ResMLP backbone defined in Chapter 3."

**Source:**
- Stage 1 uses `ResidualMLP` (no `mask_proj`, no mask input).
- Stages 2 and 3 use `ResidualMLP_Mask` (with `mask_proj`, mask input).
- Loading a Stage-1 ckpt into a Stage-2 model is done via `strict=False` and the `mask_proj` parameters get newly initialised (zero) for Stage 2.

So the shared-backbone claim is exactly true between Stage 2 and Stage 3, but **not** between Stage 1 and the rest. Stage 1's checkpoints are *embedded* into the Stage 2 backbone via averaging, not as a 1:1 backbone match.

**FIX (§3.3, §4.6):**

Rewrite the §3.3 sentence:

> "Stages 2 and 3 use one and the same residual MLP backbone, denoted `ResMLP_Mask` (Listing E.2 + the mask projection of §3.3); Stage 3 fine-tunes the parameters that Stage 2 has produced. Stage 1's per-configuration backbone shares the same residual-block stack but does not include the mask projection, because each Stage-1 model only ever sees one DoF setting and does not need to be conditioned on it. The Stage-1 checkpoints are loaded into the Stage-2 backbone with `strict=False`: the residual-block weights are inherited, and the mask projection (which has no Stage-1 counterpart) is initialised to zero so that the Stage-2 model behaves identically to the warm-start Stage-1 model at the very first training step."

Update §4.6 similarly: replace "share the same ResMLP backbone defined in Chapter 3" with "share the same residual-block stack and the same FK output head; Stages 2 and 3 additionally include the mask-conditioning projection introduced for the multi-DoF setting".

---

### Finding 6 — Listing E.3 mis-describes the Stage-2 minibatch sampler

**Report (Listing E.3):**
```
def multitask_step(model, datasets, optimizer, lambda_p, lambda_o):
    losses = []
    for D_k in datasets:                       # 5, 6, 7 DoF datasets
        q_k, p_k = D_k.sample_minibatch()
        ...
        losses.append(lambda_p * Lp + lambda_o * Lo)
    loss = sum(losses) / len(datasets)         # K-task average
```

**Source (train_multitask_separate_weight.py L477-516):**
```python
tname = random.choice(task_names)              # ONE task per step
x, y, mask = next(task_gens[tname])
...
loss = mse(yhat, y)                            # MSE for FK; no per-task average
loss.backward(); opt.step()
```

So a single Stage-2 step touches one DoF configuration, chosen uniformly at random — not all three. (This is a perfectly defensible scheme; in expectation it produces the same gradient as the K-task average up to a 1/K factor in step size.)

**FIX:**

Replace Listing E.3 with the actual loop:

```python
Listing E.3  Multi-task minibatch sampler used for Stage 2 (shared meta-kinematics) training.

def multitask_step(model, task_loaders, optimizer, mse):
    """One optimisation step on a uniformly-sampled DoF configuration."""
    tname = random.choice(list(task_loaders.keys()))   # 5dof / 6dof / 7dof
    x, y, mask = next(task_loaders[tname])             # already standardised
    yhat = model(x, mask)
    loss = mse(yhat, y)                                 # MSE on standardised pose
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return tname, loss.item()
```

Add one sentence to §3.4 (Stage 2): "*At each Stage-2 optimisation step, one of the three DoF configurations is sampled uniformly at random and a single minibatch from that configuration is used to compute the gradient; over a window of T steps each configuration sees ~T/3 minibatches.*"

---

### Finding 7 — Quaternion is not unit-normalised at the model output

**Report (§3.3):**
> "The output head produces seven values corresponding to the predicted position t̂ ∈ ℝ³ and the predicted quaternion r̂ ∈ ℝ⁴, with the quaternion part normalised to unit norm."

**Source:**
The model output is just `self.fc_out(h)` returning a raw 7-D vector. Unit-normalisation happens **only inside `quat_angle_rad` and `quat_geodesic_loss_rad`** at loss/eval time:
```python
q_pred = q_pred / (q_pred.norm(dim=1, keepdim=True).clamp_min(eps))
```

The model itself never normalises.

**FIX (§3.3):**
Replace the sentence with:
> "The output head produces seven values corresponding to the predicted position t̂ ∈ ℝ³ and the predicted quaternion r̂ ∈ ℝ⁴; the quaternion is unit-normalised before any geodesic-distance computation in the loss and the evaluation metrics, but the model output itself is not normalised in-graph."

---

### Finding 8 — Stage-2 standardisation is per-task, not global

**Report (§4.4 / §3.3):**
> "Inputs and targets are standardised before training."

**Source (train_multitask_separate_weight.py L213-227, L703-712):**
Each of the three `MaskedKinematicsDataset` objects (one per DoF) computes its **own** `x_mean, x_std, y_mean, y_std` from its own data and stores them in `task_norm`. So the same physical joint angle q₁ = 0.5 rad has different normalised values in the 5/6/7-DoF datasets.

**FIX (§4.4 add one sentence):**
> "Standardisation is performed independently per DoF configuration: each task's joint and pose statistics are computed from that task's training data, and the per-task statistics are stored in the checkpoint so that evaluation and adaptation re-use the same normalisation. The shared model therefore operates in a per-task standardised input space, which keeps the loss surface comparable across configurations despite the different active-joint counts."

---

### Finding 9 — Stage-2 has no internal held-out split; "best on held-out" is overstated

**Report (§4.4):**
> "validation loss curves are used to monitor training, and the best checkpoint on a held-out split is retained for evaluation."

**Source (train_multitask_separate_weight.py L525-557):**
Stage 2's "evaluation" inside the training loop is a 1-batch quick eval drawn from the **same** infinite training generator. Best checkpoint = lowest avg-of-3-1-batch-losses. There is no held-out test set inside this script.

The actual held-out evaluation that produces Table 5.1's shared-meta numbers happens in a separate script (`eval_multitask.py`) on a deterministic subset of the same data file (or, for adapt, on a disjoint subset of the same file via permutation seed 42).

**FIX (§4.4 honest restatement):**
> "Inside Stage 2, training progress is monitored via a 1-batch quick-eval drawn from each task's data generator at every `eval_every` steps; the model with the lowest mean of these three quick-eval losses is preserved as the best checkpoint. The held-out test metrics reported in Chapter 5 are computed *outside* the training script, on a deterministic permutation subset of each DoF's dataset that is disjoint from the support set used for adaptation. The same permutation seed is used at evaluation time for all three model variants, so the held-out comparisons in Table 5.1 are over the same physical samples for the single-task, shared and adapted models in each configuration."

(If the held-out subset is *not* sample-disjoint from training, this paragraph should be softened further. Look at `compute_norm_from_support` in `adapt_multitask_newest.py`: it splits the data file into support/query via `np.random.RandomState(seed=42).permutation`, so query is disjoint from support but both come from the same data file used in Stage 2 training. This is a single-file evaluation — the user should acknowledge this limitation in §6.4.)

---

### Finding 10 — Stage-3 best-step selection uses a weighted score, not raw metrics

**Source (adapt_multitask_newest.py L329-330, L570-579):**
```python
def score_fk(metrics, score_pos_w, score_ori_w):
    return float(score_pos_w * metrics["pos_mae_m"] + score_ori_w * metrics["ori_deg"])
...
sc = score_fk(mq, args.score_pos_w, args.score_ori_w)
if sc < best_score:                  # best step minimises pos_mae + 0.01*ori_deg
    best_step = it
    best_metrics = dict(mq)           # both pos and ori at this step are reported
```

The reported "BEST" line (which Table 5.1 takes its adapted numbers from) corresponds to whichever evaluation step minimises `1.0 · pos_mae + 0.01 · ori_deg`, **not** the raw best position or best orientation independently. With the default weights, position dominates, but the choice influences which step is "best".

**FIX (§4.6 or §5.3):**
> "For Stage 3, the reported 'best-step' adapted model is the evaluation step that minimises the scalarised score `s = w_p · E_pos + w_o · E_ori` with `w_p = 1.0` and `w_o = 0.01`, which keeps position the dominant criterion while breaking ties using orientation. Reporting the adapted-model row with these weights is consistent across the three DoF configurations and avoids cherry-picking a per-metric best step."

---

### Finding 11 — Table E.2 confuses Stage-1 and Stage-2 LR schedules

**Report (Table E.2):**
| Param | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| LR schedule | Cosine (warmup 2 000) | | Constant |
| LR minimum | 1×10⁻⁵ | | — |

(Suggesting Stage 1 uses cosine + warmup + lr_min.)

**Source:**
- Stage 1 (`train_kinematics_nn_pol_pt_2.py` L494-499): `ReduceLROnPlateau(factor=0.5, patience=10)`, base `lr=5e-4`, `weight_decay=1e-5` (Adam L2).
- Stage 2 (`train_multitask_separate_weight.py` L664-669): `cosine` schedule, `lr=3e-4`, `lr_min=1e-5`, `warmup=2000`. Verified in saved best ckpt metadata (`base_lr=3e-4`, `lr_schedule=cosine`, `warmup_steps=2000`, `total_steps=300000`).
- Stage 3 (`adapt_multitask_newest.py` — no LR schedule, just constant `inner_lr=1e-5`).

**FIX (rewrite Table E.2):**

| Parameter | Stage 1 (single-task) | Stage 2 (shared meta) | Stage 3 (adaptation) |
|---|---|---|---|
| Initial learning rate | 5 × 10⁻⁴ | 3 × 10⁻⁴ | 1 × 10⁻⁵ |
| LR schedule | ReduceLROnPlateau (factor 0.5, patience 10) | Cosine with linear warmup of 2 000 steps | Constant |
| LR minimum | — (set by plateau decay) | 1 × 10⁻⁵ | — |
| Batch size | 4 096 | 4 096 | 8 192 |
| Total steps | up to 1 000 epochs (early-stopped) | 300 000 (best at 289 500) | 100 000 |
| Support / query size | n/a (full train/val/test split, 50/20/30) | n/a | 50 k support / 2 M query |
| Gradient clipping | 1.0 | 1.0 | 1.0 |
| Adam weight decay | 1 × 10⁻⁵ | 0 | 0 |
| Adam ε | 1 × 10⁻⁸ (PyTorch default) | 1 × 10⁻⁸ (PyTorch default) | 1 × 10⁻⁷ |
| L2-to-init regularisation | n/a | n/a | 1 × 10⁻⁶ |
| Random seed | 42 | 42 | 42 |
| Wall-clock 5-DoF | 1.873 hr | 1.159 hr (shared) | 0.365 hr |
| Wall-clock 6-DoF | 4.973 hr | 1.159 hr (shared) | 0.363 hr |
| Wall-clock 7-DoF | 22.12 hr | 1.159 hr (shared) | 0.111 hr |

---

### Finding 12 — `std_floor_q_deg = 1.0` is an IK-only stability device

**Report (Table E.1):**
> "Joint-noise floor (training stability): std_floor_q = 1.0° equivalent"

**Source (train_multitask_separate_weight.py L220-221, adapt_multitask_newest.py L173-175):**
```python
if mode == "ik" and std_floor_q_rad > 0.0:
    y_std = np.maximum(y_std, std_floor_q_rad)
```

`std_floor_q` is only applied when normalising the *target* in IK mode. For FK mode, the target is the pose (not joint angles), so `std_floor_q` has no effect. The CLI argument is still parsed in FK runs but the value is silently ignored.

**FIX:**
Either drop the row from Table E.1 (since the reported runs are FK), or qualify it: "Joint-noise floor (IK target standardisation): std_floor_q = 1.0° equivalent. Not used in FK (the work reported in this thesis)."

---

### Finding 13 — Stage 1 has `weight_decay=1e-5`, not 0

**Report (Table E.1):** "Weight decay: 0.0 (none in Stages 1–2); 1×10⁻⁶ L2 in Stage 3 adaptation"

**Source (train_kinematics_nn_pol_pt_2.py L88-93):**
```python
parser.add_argument("--weight_decay", type=float, default=1e-5, ...)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
```

Default `weight_decay=1e-5` in Adam for Stage 1.

**FIX:**
Change Table E.1 row to: "Weight decay: 1×10⁻⁵ (Adam L2, Stage 1); 0 (Stage 2); 1×10⁻⁶ L2-to-init (Stage 3)."

---

### Finding 14 — `aux_loss_weight` is IK-only

Already noted in Finding 2. Table E.1 should make explicit that `pos_loss_weight=0.03` and `ori_loss_weight=0.12` exist in the Stage-2 ckpt metadata but are inactive in FK.

---

### Findings 15–16 — minor

- **Dropout 0.1**: add a row to Table E.1 (already covered in Finding 4 fix).
- **Stage-2 actual step count**: change Table E.2 row "Total steps … 1 000 000 (max)" to "300 000 (best checkpoint at 289 500)".

---

## SUMMARY OF EDITS BY REPORT SECTION

| Section | Edit |
|---|---|
| §3.3 (Backbone) | Add `mask_proj` paragraph; rewrite ResBlock description (Dropout, no LayerNorm); reword quaternion-normalisation sentence; soften "same backbone" between stages. |
| Fig 3.2 (a) caption | Mention mask input and mask projection. |
| §3.4 (Stage 2) | Add 1 sentence on uniform-task sampling per step. |
| §3.5 / Eq 3.3 | Split into Eq 3.3a (Stages 1–2 unified MSE on standardised pose) and Eq 3.3b (Stage 3 raw-units weighted loss with geodesic squared angle). |
| §4.4 | Add per-task standardisation sentence; rewrite the held-out / monitoring-eval claim. |
| §4.5 / Eq 4.1 | Replace RMSE definition with mean Euclidean error; add explicit orientation metric formula. |
| §4.6 | Reword "same backbone" claim; add Stage-3 scoring scalarisation disclosure. |
| §5.3 / §5.4 | Optionally add 1 sentence in §5.3 noting that "BEST" = step that minimises the scalarised score. |
| Listing E.1 | Caption clarifies it is the **Stage 3** loss term only. |
| Listing E.2 | Replace LayerNorm version with the actual Dropout version. |
| Listing E.3 | Replace per-step K-task average with uniform-task sampling. |
| Table E.1 | Update 'Activation/Normalisation/Regularisation', `mask_proj` row, weight-decay row, `std_floor_q` qualifier. |
| Table E.2 | Replace LR-schedule row with per-stage values; correct Stage 1 schedule, batch size, weight decay; correct Stage-2 step count. |

All of these are **prose / equation / table** changes. None of them invalidates the experimental results — the audit is about making the report describe the experiments accurately.

---

## NOTES THAT DO **NOT** NEED FIXING

- The reported training times (1.873 / 4.973 / 22.12 hr for single-task; 1.159 hr for shared; 0.365 / 0.363 / 0.111 hr for adapt) are consistent with the wall-clocks in the saved logs and the briefing.
- The held-out numbers (0.0093 / 0.0110 / 0.0101 m position; 1.20 / 1.71 / 2.09° orientation, etc.) are consistent with the Mean position/Mean orientation lines in the eval logs once Finding 1 is applied.
- The qualitative narrative — "shared meta-model captures transferable kinematic structure; per-DoF adaptation refines it; the largest cost saving is in 7-DoF" — is well-supported by the data.
- The sample sizes (15 M cap per task, 50 k support / 2 M query for adapt) match the briefing.

---

## ABLATION PLACEHOLDERS

The report has `[fill]` cells in:
- Table 5.3 (Ablation A — random-init Stage-2 vs warm-start)
- Table 5.4 (Ablation B — random-init Stage-3 vs from-shared)
- Table 6.1 (prior-work numerical comparison)
- Figure 6.1 (t-SNE)

These correspond exactly to Tier 4 Experiments C, A and D in the briefing. With the briefing's wall-clock estimates, the only experiment that fits in <24 hr **and** is high-leverage for marks is Experiment A (~7 min for the 7-DoF random-init adaptation, populates Table 5.4 7-DoF row) and Experiment D (~30 min for the t-SNE figure). Experiment B (multi-seed) is ~2.5 hr and the most defensive against the "single-seed" critique already volunteered in §6.4. Experiment C (Ablation A) is ~1.2 hr.

**Recommended order if time is tight:**
1. **Experiment A** — 7 min, fills the highest-cited row of Table 5.4, the killer-comparison row.
2. **Experiment D** — 30 min, fills Figure 6.1 placeholder in §6.1 and removes a "[Student to complete]" tag.
3. **Experiment B** — 2.5 hr, lets you replace "single seed" admissions with mean ± std for the 7-DoF row only (the most-discussed configuration).
4. **Experiment C** — 1.2 hr, fills Ablation A. Lower priority because the §6.4 limitations paragraph already concedes the single-init story.

If only Experiment A and D are run in time, fill the remaining cells with "n/a — single-seed run is the baseline; multi-seed evaluation is identified as the first item of the planned journal extension (§7.3)" and remove the placeholder square brackets.

---

## REFERENCES THAT MIGHT BE QUERIED

A quick examiner-test pass on the bibliography:
- [13] is cited as "Orbit: A unified simulation framework" — this is the **Orbit** paper, which became NVIDIA Isaac Lab. The reference is correct but could note "now released as Isaac Lab".
- [4] KineNN is cited several times — Table 6.1 leaves its position-error number as `[fill from source]`. This needs to be filled before submission (the KineNN paper reports forward-model RMSE on the UR5).
- [3] Köker et al. 2004 — also `[fill from source]` in Table 6.1. The original paper reports IK joint errors, not FK end-effector errors, so the comparison row is awkward as drawn. Consider either dropping that row or rewriting it as "(IK study; not directly comparable)".
- The Cursi et al. 2022 entry in Table 6.1 has no bracketed reference number — add it to the bibliography or drop the row.

---

## END OF AUDIT
