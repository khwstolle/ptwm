"""Deterministic synthetic NVFP4 tensor set (ModelOpt-style naming)."""

from __future__ import annotations

import numpy as np

__all__ = ["build_synthetic_nvfp4"]

_GROUP_SIZE = 16


def build_synthetic_nvfp4(
    *,
    n_layers: int = 1,
    out_features: int = 128,
    in_features: int = 64,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Return a state dict mirroring NVIDIA ModelOpt NVFP4 layout."""
    if in_features % _GROUP_SIZE != 0:
        msg = f"in_features={in_features} not divisible by group_size={_GROUP_SIZE}"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    state: dict[str, np.ndarray] = {}
    n_groups = in_features // _GROUP_SIZE
    for li in range(n_layers):
        base = f"model.layers.{li}.self_attn.q_proj"
        # Packed FP4 nibbles: in_features/2 uint8 per output channel.
        state[f"{base}.weight"] = rng.integers(
            0, 256, size=(out_features, in_features // 2), dtype=np.uint8
        )
        # F8_E4M3 per-block scale, expressed as uint8 here (caller flags
        # the dtype string at compress time as "F8_E4M3" — wire dtype is
        # 1 byte either way).
        state[f"{base}.weight_scale"] = rng.integers(
            64, 192, size=(out_features, n_groups), dtype=np.uint8
        )
        # Scalar F32 globals.
        state[f"{base}.weight_scale_2"] = np.array(
            rng.standard_normal(dtype=np.float32) * 0.01,
            dtype=np.float32,
        )
        state[f"{base}.input_scale"] = np.array(
            rng.standard_normal(dtype=np.float32) * 0.01,
            dtype=np.float32,
        )
    return state
