"""Deterministic synthetic MXFP4 tensor set (gpt-oss-style naming)."""

from __future__ import annotations

import numpy as np

__all__ = ["build_synthetic_mxfp4"]


def build_synthetic_mxfp4(
    *,
    n_experts: int = 2,
    hidden: int = 8,
    n_blocks: int = 3,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Return a state dict mirroring gpt-oss-20b's MXFP4 layout.

    Block layout (OCP MXFP4, 32 values per block, packed FP4 = 16 bytes):
        <name>_blocks  uint8  [n_experts, hidden, n_blocks, 16]
        <name>_scales  uint8  [n_experts, hidden, n_blocks]   (E8M0)
        <name>_bias    bf16   [n_experts, hidden]
    """
    rng = np.random.default_rng(seed)
    state: dict[str, np.ndarray] = {}
    base = "model.layers.0.mlp.experts"
    for proj in ("down_proj", "gate_up_proj"):
        state[f"{base}.{proj}_blocks"] = rng.integers(
            0, 256, size=(n_experts, hidden, n_blocks, 16), dtype=np.uint8
        )
        # E8M0 scales: a biased exponent, biased toward typical training-weight
        # exponents (~120 for fp32-like magnitudes).
        state[f"{base}.{proj}_scales"] = rng.integers(
            115, 130, size=(n_experts, hidden, n_blocks), dtype=np.uint8
        )
        # Bias as bf16: cast via uint16 view since numpy lacks native bf16.
        # Caller treats it as a 2-byte tensor with dtype name "BF16".
        bias_f32 = rng.standard_normal((n_experts, hidden), dtype=np.float32)
        state[f"{base}.{proj}_bias"] = bias_f32.view(np.uint16)[..., 1::2].copy()
    return state
