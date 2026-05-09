"""
MetaKinFormer: a joint-as-token Transformer architecture for cross-DoF
forward-kinematics learning on a single robot.

Designed as a drop-in replacement for the residual MLP backbone used in
the existing `train_kinematics_nn_pol_pt_2.py` and `adapt_multitask_newest.py`.

Why this architecture for cross-DoF FK?
---------------------------------------
1. Each joint is naturally a token; a Transformer encoder learns joint-to-joint
   dependencies through self-attention. At 7-DoF the long-range coupling between
   proximal and distal joints is the main difficulty for ResMLP — attention
   models it directly.
2. The active-DoF mask becomes an *attention key-padding mask*: inactive joints
   are zeroed out of attention computation entirely. No need for a separately
   learned `W_mask` projection.
3. Joint position encoding (learned or sinusoidal over joint index) gives the
   model the kinematic chain *order*, not just the joint values.
4. Smaller parameter count than the 17M-parameter ResMLP_Mask: a 4-layer
   transformer with d=64 has ~150k params; even with d=128 the model stays
   < 2M params. Often trains faster and generalises better at small data.

Input/output contract (unchanged from ResMLP_Mask):
    forward(q, mask) -> pose_vec
        q:    (B, 7)  float, joint angles in radians, 0 for inactive joints
        mask: (B, 7)  bool/float, 1 for active joints, 0 for inactive
        pose_vec: (B, 7) float, [x, y, z, qw, qx, qy, qz] standardised per task
"""
from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn


# ----------------------------- Positional encoding -----------------------------

class JointPositionalEncoding(nn.Module):
    """Sinusoidal position embedding indexed by joint number (0..n_joints-1).

    Provides the kinematic-chain *order* signal. We use a fixed sinusoidal
    encoding (not learned) to keep the param count low and to encourage the
    model to attend by joint index rather than memorising arbitrary identifiers.
    """

    def __init__(self, n_joints: int, d_model: int):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even for sinusoidal PE")
        pe = torch.zeros(n_joints, d_model)
        position = torch.arange(0, n_joints, dtype=torch.float).unsqueeze(1)  # (n_joints, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, n_joints, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_joints, d_model)
        return x + self.pe[:, : x.size(1)]


# ----------------------------- Joint embedding -----------------------------

class JointTokenEmbedding(nn.Module):
    """Project (joint_value, active_flag) -> token of dim d_model.

    The 2-D input per joint is:
      - q_i: the joint angle value (clamped to 0 if inactive — caller's responsibility)
      - m_i: the binary active flag (1 = active, 0 = inactive)

    Including the mask flag *inside the token* (in addition to using it for
    attention masking below) lets the model distinguish between "this joint is
    inactive" vs "this joint is at angle 0" — which would otherwise be
    indistinguishable when q_i is clamped to 0.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(2, d_model)

    def forward(self, q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # q: (B, n_joints), mask: (B, n_joints)
        # Stack to (B, n_joints, 2)
        x = torch.stack([q, mask.float()], dim=-1)
        return self.linear(x)  # (B, n_joints, d_model)


# ----------------------------- Main model -----------------------------

class MetaKinFormer(nn.Module):
    """Joint-as-token Transformer for cross-DoF forward kinematics.

    Architecture
    ------------
    Input (B, n_joints) joint angles + (B, n_joints) active mask
        |
        +-- JointTokenEmbedding (Linear 2 -> d_model)
        +-- JointPositionalEncoding (sinusoidal, by joint index)
        |
        +-- TransformerEncoder (N layers, h heads, d_ff = 4*d_model)
        |     - src_key_padding_mask = ~mask  (inactive joints excluded from attention)
        |     - dropout p_drop applied in feedforward + attention
        |
        +-- Pool (concat all token embeddings, then Linear -> 7)
        |
        +-- Output (B, 7): standardised [x, y, z, qw, qx, qy, qz]

    The pooling step concatenates all n_joints token outputs (B, n_joints*d_model)
    rather than mean-pooling, because each joint's contribution to the pose is
    distinct (proximal joints affect global position; distal joints affect
    orientation). Concat lets the final linear head learn the per-joint mapping.

    Parameter count (defaults d_model=128, n_layers=4, h=4):
        joint_embed: 2 * 128 = 256
        pos_embed:   buffer (no params)
        transformer: 4 * (4 * 128 * 128 (qkvo) + 2 * 128 * 512 (ff)) ~ 4 * 197k ~ 790k
        head:        7 * 128 * 7 = 6,272
        TOTAL: ~800k params (vs 17M for ResMLP_Mask — 20x smaller).
    """

    def __init__(
        self,
        n_joints: int = 7,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        out_dim: int = 7,
    ):
        super().__init__()
        self.n_joints = n_joints
        self.d_model = d_model

        self.joint_embed = JointTokenEmbedding(d_model)
        self.pos_embed = JointPositionalEncoding(n_joints, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,  # pre-LN: more stable training, especially at deeper N
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output head: concat all token outputs -> 7-D pose vector
        self.head = nn.Linear(n_joints * d_model, out_dim)

        # Init: zero the final head bias so we start near origin
        nn.init.zeros_(self.head.bias)

    def forward(self, q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        q:    (B, n_joints)  joint angles in radians (caller clamps inactive to 0)
        mask: (B, n_joints)  1 for active, 0 for inactive
        """
        B = q.shape[0]
        x = self.joint_embed(q, mask)          # (B, n_joints, d_model)
        x = self.pos_embed(x)                  # add positional encoding

        # Build src_key_padding_mask: True means PADDED (inactive) — excluded from attention
        # PyTorch's TransformerEncoder treats True as "ignore this key"
        key_pad_mask = ~mask.bool()            # (B, n_joints)

        x = self.encoder(x, src_key_padding_mask=key_pad_mask)  # (B, n_joints, d_model)

        # Concat-pool all tokens
        x = x.reshape(B, -1)                    # (B, n_joints * d_model)
        return self.head(x)                     # (B, out_dim)

    def freeze_attention(self):
        """Helper for ablation: freeze attention layers, only train head."""
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True


# ----------------------------- Smoke test -----------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    model = MetaKinFormer()
    print(model)
    print(f"\nTotal params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable:    {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Test forward pass on a 5-DoF batch (joints 1..5 active, 6..7 zeroed)
    B = 4
    q = torch.randn(B, 7)  # joint angles
    mask = torch.ones(B, 7)
    mask[:, 5:] = 0  # 5-DoF: deactivate joints 6 and 7
    q[:, 5:] = 0      # also clamp the angle values to 0 (caller's job)

    pose = model(q, mask)
    print(f"\nForward pass: input q {q.shape}, mask {mask.shape} -> pose {pose.shape}")
    print(f"Output sample (B=0):  pos = {pose[0, :3].tolist()}, quat = {pose[0, 3:].tolist()}")

    # Test 7-DoF
    mask7 = torch.ones(B, 7)
    pose7 = model(q, mask7)
    print(f"7-DoF pose:           pos = {pose7[0, :3].tolist()}, quat = {pose7[0, 3:].tolist()}")

    # Confirm that swapping mask changes output (sanity check — attention mask works)
    diff = (pose - pose7).abs().mean().item()
    print(f"\n5-DoF vs 7-DoF mean abs diff: {diff:.6f}  (should be > 0 if mask is wired)")

    # Backward pass test
    target = torch.randn_like(pose)
    loss = ((pose - target) ** 2).mean()
    loss.backward()
    print(f"\nLoss = {loss.item():.4f}, backward pass OK")
    print(f"Gradient norm at head: {model.head.weight.grad.norm().item():.4f}")
