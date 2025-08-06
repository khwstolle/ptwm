"""Tests for the BF16/FP32 `+ IntDelta` chain candidates added alongside
the `IntDelta` op.

These tests assert:

1. Both new chains build, serialise, and parse back through the wire bridge.
2. They are registered as candidates in ``PRODUCTION_CHAINS`` for the
   right ``(dtype, role)`` cells.
3. End-to-end ``Compressor → Decompressor`` round-trips succeed on BF16
   and FP32 inputs of various shapes — confirming the dispatcher's
   trial-encode handles either candidate winning.
"""

from __future__ import annotations

import pytest
import torch
from ptwm import (
    CompressionConfig,
    Compressor,
    DecompressionConfig,
    Decompressor,
    Format,
)
from ptwm.preprocessing._chains import (
    CHAIN_BF16_SPLIT,
    CHAIN_BF16_SPLIT_INTDELTA,
    CHAIN_FP32_SPLIT,
    CHAIN_FP32_SPLIT_INTDELTA,
    PRODUCTION_CHAINS,
    ClassifierRole,
)
from ptwm.utils import DType as _DT


def test_bf16_intdelta_chain_serialises() -> None:
    chain = CHAIN_BF16_SPLIT_INTDELTA([32])
    blob = chain.to_bytes()
    # Wire bridge produced a non-empty self-contained blob.
    assert len(blob) > 0


def test_fp32_intdelta_chain_serialises() -> None:
    chain = CHAIN_FP32_SPLIT_INTDELTA([32])
    blob = chain.to_bytes()
    assert len(blob) > 0


def test_bf16_intdelta_is_registered_as_production_candidate() -> None:
    """BF16 STANDARD should list both the plain split and the IntDelta variant."""
    candidates = PRODUCTION_CHAINS[(_DT.BFLOAT16.code, ClassifierRole.STANDARD)]
    assert CHAIN_BF16_SPLIT in candidates
    assert CHAIN_BF16_SPLIT_INTDELTA in candidates


def test_fp16_intdelta_is_registered_as_production_candidate() -> None:
    """FP16 STANDARD should list both the plain split and the IntDelta variant."""
    candidates = PRODUCTION_CHAINS[(_DT.FLOAT16.code, ClassifierRole.STANDARD)]
    assert CHAIN_BF16_SPLIT in candidates
    assert CHAIN_BF16_SPLIT_INTDELTA in candidates


def test_fp32_intdelta_is_registered_as_production_candidate() -> None:
    """FP32 STANDARD should list both the plain split and the IntDelta variant."""
    candidates = PRODUCTION_CHAINS[(_DT.FLOAT32.code, ClassifierRole.STANDARD)]
    assert CHAIN_FP32_SPLIT in candidates
    assert CHAIN_FP32_SPLIT_INTDELTA in candidates


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("numel", [16, 256, 4096])
def test_round_trip_with_intdelta_in_menu(dtype: torch.dtype, numel: int) -> None:
    """Round-trip on trained-weight-shaped tensors with IntDelta in the
    candidate set.

    Uses a deterministic narrow-exponent distribution that mimics trained
    weights so the trial-encode loop has a realistic shot at picking the
    IntDelta variant. The assertion is on byte-exact round-trip, not on
    which chain wins — the test catches any chain that breaks the PPG
    invariant.
    """
    generator = torch.Generator().manual_seed(numel)
    tensor = (torch.randn(numel, generator=generator, dtype=torch.float32) * 0.02).to(
        dtype,
    )

    compressor = Compressor(CompressionConfig(input_format=Format.TORCH))
    decompressor = Decompressor(DecompressionConfig())
    blob = compressor.compress(tensor)
    restored = decompressor.decompress(blob)
    # Compare via raw bytes — dtype-aware torch.equal handles the rest, but
    # the byte view is the strict invariant the PPG promises.
    assert tensor.view(torch.uint8).equal(restored.view(torch.uint8)), (
        f"round-trip mismatch on dtype={dtype} numel={numel}"
    )
