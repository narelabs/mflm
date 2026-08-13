"""Mass-Field Language Model — core architecture.

A language model where information propagates through a learned
gravitational field rather than through pairwise dot-product attention.

Each token has:
  - **mass**: a positive scalar (softplus projection) that determines
    how strongly the token radiates into the field.
  - **charge**: a content vector (linear projection of hidden state)
    that carries the actual information.

The field at position *i* is a causal convolution:

    field(i) = Σ_{j ≤ i}  kernel(i − j) · mass(j) · charge(j)

The kernel is a learned per-head decay function — different heads
can learn different interaction ranges.  The entire computation is
O(N · W) where W is the kernel window, not O(N²).

The block is iterated with shared weights (like Universal Transformer).
At each iteration the mass distribution changes because hidden states
change, creating a dynamic, iteratively-refined field.

No Q.  No K.  No attention matrix.  No KV cache.

Mass is SIGNED: positive mass creates a gravitational well (attracts
attention), negative mass creates an anti-gravity hill (repels).  Between
two opposing masses, the field cancels — a Lagrange point — where
information flows without distortion.  The model learns which tokens
to attract toward, which to push away from, and where to leave a
neutral corridor.

Baseline Transformer included for controlled comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------

@dataclass
class MFLMConfig:
    """Configuration for both MFLM and baseline."""
    vocab_size: int = 256        # char-level default, overwritten from data
    d_model: int = 128
    d_ff: int = 512
    n_heads: int = 4             # number of field kernel heads
    max_mass: float = 10.0       # mass range: [-max_mass, +max_mass]
    n_layers: int = 2            # baseline only: number of transformer layers
    max_steps: int = 4           # MFLM only: depth iterations (shared weights)
    field_window: int = 64       # causal kernel width
    max_seq_len: int = 256
    dropout: float = 0.0


# -----------------------------------------------------------------------
# Field block
# -----------------------------------------------------------------------

class FieldBlock(nn.Module):
    """Single field-computation block with shared weights.

    Each forward call:
      1. Project hidden state → mass scalar (softplus, always positive)
      2. Project hidden state → charge vector
      3. Compute field via causal convolution of (mass · charge) with
         learned per-head kernels
      4. Gate the field and add to residual stream
      5. Feed-forward MLP on residual stream

    This block is designed to be iterated multiple times with shared
    weights (the mass/field/MLP evolve across iterations as hidden
    states change).
    """

    def __init__(self, cfg: MFLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.head_dim = cfg.d_model // cfg.n_heads

        # Pre-norm
        self.norm1 = nn.LayerNorm(cfg.d_model)

        # Mass head: d_model → 1, tanh * max_mass gives SIGNED mass
        # Positive mass = gravity well (attracts), negative = anti-gravity (repels)
        # Between opposing masses: Lagrange point where field cancels
        self.mass_proj = nn.Linear(cfg.d_model, 1, bias=True)
        self.max_mass = cfg.max_mass

        # Charge projection: the content vector that propagates through the field
        self.charge_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # Per-head field kernels: learned causal decay functions.
        # Initialized with exponential decay — close tokens have stronger
        # influence, which is a reasonable physics-inspired prior.
        # Shape: (n_heads, field_window)
        init_kernels = torch.zeros(cfg.n_heads, cfg.field_window)
        for h in range(cfg.n_heads):
            # Different heads get different initial decay rates
            rate = 0.05 + 0.05 * h  # head 0: slow decay, head 3: fast decay
            init_kernels[h] = torch.exp(
                -rate * torch.arange(cfg.field_window, dtype=torch.float32)
            )
        self.field_kernels = nn.Parameter(init_kernels)

        # Gate: sigmoid gate controlling how much field enters the stream
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # Output projection (like o_proj in attention)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # MLP — same as baseline transformer for fair comparison
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

        # Diagnostics
        self._last_mass: Tensor | None = None

    def _compute_field(self, h: Tensor, mass: Tensor) -> Tensor:
        """Compute the gravitational field via causal convolution.

        For each position i, the field is:
            field(i) = Σ_{j=max(0,i-W+1)}^{i} kernel(i-j) · mass(j) · charge(j)

        This is implemented as a depthwise causal conv1d where:
        - Input channels = d_model
        - Each group of head_dim channels shares one kernel
        - Causal = left-padded by (window - 1)

        Args:
            h: (B, T, d_model) — normalized hidden states.
            mass: (B, T) — signed mass scalars (positive=attract, negative=repel).

        Returns:
            (B, T, d_model) — field values at each position.
        """
        charge = self.charge_proj(h)  # (B, T, d)

        # Weight charge by mass: heavy tokens radiate strongly
        weighted = mass.unsqueeze(-1) * charge  # (B, T, d)

        # Reshape for conv1d: (B, d, T)
        wt = weighted.transpose(1, 2)

        # Build depthwise kernel: each head's kernel tiled across head_dim channels
        # field_kernels: (n_heads, W)
        # We flip for proper convolution (F.conv1d computes cross-correlation)
        k = self.field_kernels.flip(1)  # (n_heads, W)
        k = k.unsqueeze(1).expand(
            -1, self.head_dim, -1
        )  # (n_heads, head_dim, W)
        k = k.reshape(self.d_model, 1, self.cfg.field_window)  # (d, 1, W)

        # Causal padding: pad left so position i only sees j ≤ i
        padded = F.pad(wt, (self.cfg.field_window - 1, 0))

        # Depthwise convolution: each channel convolved independently
        field = F.conv1d(padded, k, groups=self.d_model)  # (B, d, T)

        return field.transpose(1, 2)  # (B, T, d)

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            h: (B, T, d_model)

        Returns:
            Tuple of (updated h, mass tensor for diagnostics).
        """
        normed = self.norm1(h)

        # 1. Compute SIGNED mass via tanh * max_mass
        # Positive mass = gravity well (token attracts field toward it)
        # Negative mass = anti-gravity (token repels, pushes field away)
        # Zero mass = invisible token, no field contribution
        mass = self.max_mass * torch.tanh(
            self.mass_proj(normed).squeeze(-1)
        )  # (B, T) in [-max_mass, +max_mass]
        self._last_mass = mass.detach()

        # 2. Compute field
        field = self._compute_field(normed, mass)  # (B, T, d)

        # 3. Gated residual: sigmoid gate decides how much field to admit
        gate = torch.sigmoid(self.gate_proj(normed))
        h = h + self.out_proj(gate * field)

        # 4. Feed-forward MLP with residual
        h = h + self.mlp(self.norm2(h))

        return h, mass


# -----------------------------------------------------------------------
# MFLM — the actual model
# -----------------------------------------------------------------------

class MFLM(nn.Module):
    """Mass-Field Language Model.

    Architecture:
      tok_emb + pos_emb
      → FieldBlock × max_steps (SHARED WEIGHTS)
      → LayerNorm → tied LM head

    The same block is iterated max_steps times.  This means the model
    has far fewer unique parameters than a transformer with max_steps
    layers, but performs the same depth of computation.  The mass
    distribution evolves across iterations as hidden states change.
    """

    def __init__(self, cfg: MFLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.block = FieldBlock(cfg)  # ONE block, shared weights
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

        self._step_masses: list[Tensor] = []

    def forward(self, idx: Tensor, return_hiddens: bool = False) -> Tensor:
        """Forward pass.

        Args:
            idx: (B, T) integer token indices.
            return_hiddens: if True, returns the final hidden states before the LM head.

        Returns:
            (B, T, vocab_size) logits, or (B, T, d_model) if return_hiddens=True.
        """
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        h = self.tok_emb(idx) + self.pos_emb(pos)

        self._step_masses = []
        for _ in range(self.cfg.max_steps):
            h, mass = self.block(h)
            self._step_masses.append(mass)

        h = self.norm(h)
        if return_hiddens:
            return h
        return self.head(h)

    def n_params(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters())

    def diagnostics(self) -> list[dict[str, Any]]:
        """Mass diagnostics per iteration step.

        Reports mass polarity: how many tokens attract (+), repel (-),
        and sit in Lagrange neutral zones (|mass| < 1.0).
        """
        out = []
        for i, m in enumerate(self._step_masses):
            n_total = m.numel()
            out.append({
                "step": i,
                "mass_mean": float(m.mean()),
                "mass_std": float(m.std()),
                "mass_min": float(m.min()),
                "mass_max": float(m.max()),
                "frac_positive": float((m > 0).sum() / n_total),
                "frac_negative": float((m < 0).sum() / n_total),
                "frac_lagrange": float((m.abs() < 1.0).sum() / n_total),
            })
        return out


# -----------------------------------------------------------------------
# Baseline Transformer (for controlled comparison)
# -----------------------------------------------------------------------

class _BaselineBlock(nn.Module):
    """Standard pre-norm Transformer block (control)."""

    def __init__(self, cfg: MFLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.head_dim = cfg.d_model // cfg.n_heads

        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def _split(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.cfg.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        h = self.norm1(x)
        q = self._split(self.q_proj(h))
        k = self._split(self.k_proj(h))
        v = self._split(self.v_proj(h))
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn = attn.transpose(1, 2).contiguous().view(*x.shape)
        x = x + self.o_proj(attn)
        return x + self.mlp(self.norm2(x))


class BaselineLM(nn.Module):
    """Standard Transformer LM (control baseline)."""

    def __init__(self, cfg: MFLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(
            [_BaselineBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, idx: Tensor) -> Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        causal = torch.triu(
            torch.ones(T, T, device=idx.device, dtype=torch.bool), diagonal=1
        )
        mask = torch.zeros(T, T, device=idx.device).masked_fill(causal, float("-inf"))
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.norm(x))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
