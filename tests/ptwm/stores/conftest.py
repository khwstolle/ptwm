"""Shared fixtures for the storage-integration tests."""

from __future__ import annotations

from typing import Any

import pytest
import torch

_VIEW = {1: torch.uint8, 2: torch.uint16, 4: torch.uint32, 8: torch.uint64}


def raw_bytes_of(t: torch.Tensor) -> bytes:
    """Extract the raw little-endian bytes of a tensor via a uint view."""
    flat = t.flatten().contiguous()
    view = _VIEW[t.element_size()]
    return bytes(flat.view(view).cpu().numpy().tobytes())


def representative_tensors() -> tuple[
    list[tuple[str, bytes, Any, tuple[int, ...]]], dict[str, torch.Tensor]
]:
    """Return ``(store_tensors, torch_tensors)`` covering production dtypes.

    Covers bf16, fp16, fp32, fp8 (E4M3FN + E5M2), a uint8 "scale" plane, and a
    packed-FP4 value plane (a microscale/scale-style pairing). ``store_tensors``
    is the ``(name, raw, dtype, shape)`` form the writers accept; ``torch_tensors``
    maps name → the expected tensor for the torch-representable dtypes only.
    """
    torch.manual_seed(7)
    torch_tensors: dict[str, torch.Tensor] = {
        "model.embed.weight": torch.randn(48, 24).bfloat16(),
        "model.layers.0.attn.q_proj.weight": torch.randn(32, 16).to(torch.float16),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(20, 12).float(),
        "model.layers.0.act.scale": torch.randint(0, 256, (40,), dtype=torch.uint8),
        "lm_head.weight": (torch.randn(16, 8) * 0.1).to(torch.float8_e4m3fn),
        "model.norm.weight": (torch.randn(10) * 0.1).to(torch.float8_e5m2),
    }
    store: list[tuple[str, bytes, Any, tuple[int, ...]]] = [
        (name, raw_bytes_of(t), t.dtype, tuple(t.shape))
        for name, t in torch_tensors.items()
    ]

    # A packed-FP4 value plane (two 4-bit values per byte). torch may lack the
    # dtype, so it is fed/recovered as raw bytes and excluded from the
    # value-exact torch comparison.
    fp4 = torch.randint(0, 256, (64,), dtype=torch.uint8)
    store.append(
        (
            "model.layers.0.mlp.experts.down_proj_blocks",
            raw_bytes_of(fp4),
            "float4_e2m1fn_x2",
            (64,),
        )
    )
    return store, torch_tensors


@pytest.fixture
def raw_of():
    """Return the :func:`raw_bytes_of` helper as a fixture."""
    return raw_bytes_of


@pytest.fixture
def store_tensors() -> list[tuple[str, bytes, Any, tuple[int, ...]]]:
    return representative_tensors()[0]


@pytest.fixture
def torch_tensors() -> dict[str, torch.Tensor]:
    return representative_tensors()[1]
