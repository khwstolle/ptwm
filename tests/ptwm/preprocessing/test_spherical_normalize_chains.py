"""Tests for the ``spherical_normalize`` chain candidates.

The op reparameterizes each row of a float tensor into (radius, direction)
and emits an exact XOR residual, so it is lossless by construction. These
tests mirror ``test_predictor_xor_chains.py``:

1. The FP32 / BF16 / FP16 chains build and serialise through the wire bridge.
2. They are registered as candidates in ``PRODUCTION_CHAINS`` alongside the
   plain split chains.
3. The builders decline non-2D shapes (return None); the spherical op only
   runs on 2D tensors. End-to-end ``Compressor → Decompressor`` round-trips
   stay bit-exact on both 2D and 1D tensors — for 1D, other chains carry the
   round-trip; losslessness is the gate and which chain the dispatcher keeps
   is its own choice.
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
    CHAIN_BF16_SPHERICAL_NORMALIZE,
    CHAIN_FP16_SPHERICAL_NORMALIZE,
    CHAIN_FP32_SPHERICAL_NORMALIZE,
    CHAIN_FP32_SPLIT,
    PRODUCTION_CHAINS,
    ClassifierRole,
)
from ptwm.utils import DType as _DT


def test_fp32_spherical_chain_serialises() -> None:
    blob = CHAIN_FP32_SPHERICAL_NORMALIZE([8, 4]).to_bytes()
    assert len(blob) > 0


def test_bf16_spherical_chain_serialises() -> None:
    blob = CHAIN_BF16_SPHERICAL_NORMALIZE([8, 4]).to_bytes()
    assert len(blob) > 0


def test_fp16_spherical_chain_serialises() -> None:
    blob = CHAIN_FP16_SPHERICAL_NORMALIZE([8, 4]).to_bytes()
    assert len(blob) > 0


def test_spherical_chain_declines_non_2d_shapes() -> None:
    # The op only accepts 2D row-structured tensors; the builder returns
    # None for 1D and higher-rank shapes so the dispatcher skips it,
    # bounding trial-encode cost (a model has many 1D norm/bias tensors).
    for builder in (
        CHAIN_FP32_SPHERICAL_NORMALIZE,
        CHAIN_BF16_SPHERICAL_NORMALIZE,
        CHAIN_FP16_SPHERICAL_NORMALIZE,
    ):
        assert builder([128]) is None  # 1D → declined
        assert builder([4, 8, 16]) is None  # 3D → declined
        assert builder([8, 4]) is not None  # 2D → accepted


def test_spherical_is_registered_as_production_candidate() -> None:
    bf16 = PRODUCTION_CHAINS[(_DT.BFLOAT16.code, ClassifierRole.STANDARD)]
    fp16 = PRODUCTION_CHAINS[(_DT.FLOAT16.code, ClassifierRole.STANDARD)]
    fp32 = PRODUCTION_CHAINS[(_DT.FLOAT32.code, ClassifierRole.STANDARD)]
    assert CHAIN_BF16_SPHERICAL_NORMALIZE in bf16
    assert CHAIN_FP16_SPHERICAL_NORMALIZE in fp16
    assert CHAIN_FP32_SPHERICAL_NORMALIZE in fp32
    # The plain split chains stay in the menu — spherical only competes.
    assert CHAIN_FP32_SPLIT in fp32


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(64, 32), (16, 16), (1, 8), (128,)])
def test_round_trip_with_spherical_in_menu(
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> None:
    """Bit-exact round-trip with spherical_normalize in the candidate set."""
    generator = torch.Generator().manual_seed(sum(shape))
    tensor = (torch.randn(*shape, generator=generator, dtype=torch.float32) * 0.02).to(
        dtype
    )

    compressor = Compressor(CompressionConfig(input_format=Format.TORCH))
    decompressor = Decompressor(DecompressionConfig())
    blob = compressor.compress(tensor)
    restored = decompressor.decompress(blob)
    assert tensor.view(torch.uint8).equal(restored.view(torch.uint8)), (
        f"round-trip mismatch on dtype={dtype} shape={shape}"
    )
