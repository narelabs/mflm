"""Smoke tests for MFLM architecture."""

from __future__ import annotations
import torch
from mflm.model import MFLM, BaselineLM, MFLMConfig, FieldBlock


def test_field_block_shapes():
    """FieldBlock output shape matches input."""
    cfg = MFLMConfig(vocab_size=64, d_model=32, d_ff=64, n_heads=2,
                     field_window=8)
    block = FieldBlock(cfg)
    h = torch.randn(2, 16, 32)
    h_out, mass = block(h)
    assert h_out.shape == (2, 16, 32), f"got {h_out.shape}"
    assert mass.shape == (2, 16), f"got {mass.shape}"
    assert mass.abs().max() <= 10.0 + 1e-5, "mass must be bounded by max_mass"


def test_mflm_forward_shape():
    """MFLM produces correct logit shape."""
    cfg = MFLMConfig(vocab_size=64, d_model=32, d_ff=64, n_heads=2,
                     max_steps=3, field_window=8, max_seq_len=32)
    model = MFLM(cfg)
    idx = torch.randint(0, 64, (2, 16))
    logits = model(idx)
    assert logits.shape == (2, 16, 64), f"got {logits.shape}"


def test_baseline_forward_shape():
    """BaselineLM produces correct logit shape."""
    cfg = MFLMConfig(vocab_size=64, d_model=32, d_ff=64, n_heads=2,
                     n_layers=2, max_seq_len=32)
    model = BaselineLM(cfg)
    idx = torch.randint(0, 64, (2, 16))
    logits = model(idx)
    assert logits.shape == (2, 16, 64), f"got {logits.shape}"


def test_mflm_causal():
    """Changing a future token must not affect past outputs."""
    cfg = MFLMConfig(vocab_size=64, d_model=32, d_ff=64, n_heads=2,
                     max_steps=2, field_window=8, max_seq_len=32)
    model = MFLM(cfg)
    model.eval()

    idx1 = torch.randint(0, 64, (1, 16))
    idx2 = idx1.clone()
    idx2[0, 10] = (idx1[0, 10] + 1) % 64  # change token at position 10

    with torch.no_grad():
        out1 = model(idx1)
        out2 = model(idx2)

    # Positions 0..9 should be identical
    diff = (out1[0, :10] - out2[0, :10]).abs().max().item()
    assert diff < 1e-5, f"causal violation: diff={diff}"


def test_mass_bounded():
    """Mass must be bounded in [-max_mass, +max_mass]."""
    cfg = MFLMConfig(vocab_size=64, d_model=32, d_ff=64, n_heads=2,
                     max_steps=3, field_window=8, max_seq_len=32,
                     max_mass=5.0)
    model = MFLM(cfg)
    idx = torch.randint(0, 64, (4, 16))
    _ = model(idx)
    for mass in model._step_masses:
        assert mass.abs().max() <= 5.0 + 1e-5, "mass exceeds max_mass"
        # Signed mass: should have both positive AND negative values
        # (at init, random projections produce both signs)


def test_gradient_flows():
    """Gradients flow through the field computation."""
    cfg = MFLMConfig(vocab_size=64, d_model=32, d_ff=64, n_heads=2,
                     max_steps=2, field_window=8, max_seq_len=32)
    model = MFLM(cfg)
    idx = torch.randint(0, 64, (2, 16))
    y = torch.randint(0, 64, (2, 16))
    logits = model(idx)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 64), y.reshape(-1)
    )
    loss.backward()

    # Check that field_kernels got a gradient
    assert model.block.field_kernels.grad is not None, "no grad on kernels"
    assert model.block.field_kernels.grad.abs().sum() > 0, "zero grad on kernels"
    # Check mass_proj got a gradient
    assert model.block.mass_proj.weight.grad is not None, "no grad on mass"


def test_shared_weights():
    """MFLM block weights are truly shared across iterations."""
    cfg = MFLMConfig(vocab_size=64, d_model=32, d_ff=64, n_heads=2,
                     max_steps=4, field_window=8, max_seq_len=32)
    model = MFLM(cfg)
    # Only one block exists
    n_blocks = sum(1 for name, _ in model.named_modules()
                   if isinstance(_, FieldBlock))
    assert n_blocks == 1, f"expected 1 FieldBlock, got {n_blocks}"


if __name__ == "__main__":
    test_field_block_shapes()
    print("✓ field_block_shapes")
    test_mflm_forward_shape()
    print("✓ mflm_forward_shape")
    test_baseline_forward_shape()
    print("✓ baseline_forward_shape")
    test_mflm_causal()
    print("✓ mflm_causal")
    test_mass_bounded()
    print("✓ mass_bounded")
    test_gradient_flows()
    print("✓ gradient_flows")
    test_shared_weights()
    print("✓ shared_weights")
    print("\nAll tests passed.")
