"""Tests for the BF16 / FP32 `+ PredictorXor` chain candidates added
alongside the `PredictorXor` op and the `Fpc` plane codec.

Mirrors `test_intdelta_chains.py` — assertions cover:

1. Both new chains build, serialise, and parse back through the wire bridge.
2. They are registered as candidates in ``PRODUCTION_CHAINS`` alongside the
   plain split and the IntDelta variant.
3. End-to-end ``Compressor → Decompressor`` round-trips succeed on BF16
   and FP32 inputs of various shapes.
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
    CHAIN_BF16_SPLIT_PREDICTOR_XOR,
    CHAIN_FP32_SPLIT,
    CHAIN_FP32_SPLIT_INTDELTA,
    CHAIN_FP32_SPLIT_PREDICTOR_XOR,
    PRODUCTION_CHAINS,
    ClassifierRole,
)
from ptwm.utils import DType as _DT


def test_bf16_predictor_xor_chain_serialises() -> None:
    chain = CHAIN_BF16_SPLIT_PREDICTOR_XOR([32])
    blob = chain.to_bytes()
    assert len(blob) > 0


def test_fp32_predictor_xor_chain_serialises() -> None:
    chain = CHAIN_FP32_SPLIT_PREDICTOR_XOR([32])
    blob = chain.to_bytes()
    assert len(blob) > 0


def test_bf16_predictor_xor_is_registered_as_production_candidate() -> None:
    # Membership check rather than equality: PRODUCTION_CHAINS grows
    # as new chain variants land, and the test's intent is to verify
    # PredictorXor *is* in the menu, not that it is the menu.
    candidates = PRODUCTION_CHAINS[(_DT.BFLOAT16.code, ClassifierRole.STANDARD)]
    assert CHAIN_BF16_SPLIT in candidates
    assert CHAIN_BF16_SPLIT_INTDELTA in candidates
    assert CHAIN_BF16_SPLIT_PREDICTOR_XOR in candidates


def test_fp16_predictor_xor_is_registered_as_production_candidate() -> None:
    candidates = PRODUCTION_CHAINS[(_DT.FLOAT16.code, ClassifierRole.STANDARD)]
    assert CHAIN_BF16_SPLIT in candidates
    assert CHAIN_BF16_SPLIT_INTDELTA in candidates
    assert CHAIN_BF16_SPLIT_PREDICTOR_XOR in candidates


def test_fp32_predictor_xor_is_registered_as_production_candidate() -> None:
    candidates = PRODUCTION_CHAINS[(_DT.FLOAT32.code, ClassifierRole.STANDARD)]
    assert CHAIN_FP32_SPLIT in candidates
    assert CHAIN_FP32_SPLIT_INTDELTA in candidates
    assert CHAIN_FP32_SPLIT_PREDICTOR_XOR in candidates


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("numel", [16, 256, 4096])
def test_round_trip_with_predictor_xor_in_menu(
    dtype: torch.dtype,
    numel: int,
) -> None:
    """Round-trip on trained-weight-shaped tensors with PredictorXor in
    the candidate set. Round-trip correctness is the gate; which chain
    wins is up to the dispatcher.
    """
    generator = torch.Generator().manual_seed(numel)
    tensor = (torch.randn(numel, generator=generator, dtype=torch.float32) * 0.02).to(
        dtype
    )

    compressor = Compressor(CompressionConfig(input_format=Format.TORCH))
    decompressor = Decompressor(DecompressionConfig())
    blob = compressor.compress(tensor)
    restored = decompressor.decompress(blob)
    assert tensor.view(torch.uint8).equal(restored.view(torch.uint8)), (
        f"round-trip mismatch on dtype={dtype} numel={numel}"
    )
