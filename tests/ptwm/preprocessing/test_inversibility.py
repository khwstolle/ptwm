"""Hypothesis-based inversibility property tests for the PPG pipeline.

The shipping invariant: for any tensor of a supported dtype,
``Decompressor.decompress(Compressor.compress(t)) == t`` byte-for-byte.
This module fuzzes the input distribution to catch regressions that
fixed-example roundtrip tests in :mod:`tests.ptwm.core.test_roundtrip`
would not surface.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from ptwm import (
    CompressionConfig,
    Compressor,
    DecompressionConfig,
    Decompressor,
    Format,
)


def _roundtrip_tensor(tensor: torch.Tensor) -> None:
    compressor = Compressor(CompressionConfig(input_format=Format.TORCH))
    decompressor = Decompressor(DecompressionConfig())
    restored = decompressor.decompress(compressor.compress(tensor))
    if tensor.dtype.is_floating_point and torch.isnan(tensor).any():
        # NaN payload bits are dtype-defined but every implementation must
        # round-trip the exact bit pattern — compare via raw byte view.
        assert tensor.view(torch.uint8).equal(restored.view(torch.uint8))
    else:
        assert tensor.equal(restored)


# ---------------------------------------------------------------------------
# Float dtypes — fp32 / bf16 / fp16
# ---------------------------------------------------------------------------


_FLOAT_DTYPES = [torch.float32, torch.bfloat16, torch.float16]


@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    numel=st.integers(min_value=1, max_value=4096),
)
@settings(
    deadline=None,
    max_examples=25,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_float_roundtrip(dtype: torch.dtype, seed: int, numel: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    tensor = torch.randn(numel, generator=generator, dtype=torch.float32).to(dtype)
    _roundtrip_tensor(tensor)


# ---------------------------------------------------------------------------
# FP8 dtypes — nibble-split path
# ---------------------------------------------------------------------------


_FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]


@pytest.mark.parametrize("dtype", _FP8_DTYPES)
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    numel=st.integers(min_value=1, max_value=4096),
)
@settings(
    deadline=None,
    max_examples=25,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_fp8_roundtrip(dtype: torch.dtype, seed: int, numel: int) -> None:
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, size=numel, dtype=np.uint8)
    tensor = torch.from_numpy(raw).view(dtype)
    _roundtrip_tensor(tensor)


# ---------------------------------------------------------------------------
# Integer dtypes — byte-split or passthrough
# ---------------------------------------------------------------------------


_INT_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
]


@pytest.mark.parametrize("dtype", _INT_DTYPES)
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    numel=st.integers(min_value=1, max_value=4096),
)
@settings(
    deadline=None,
    max_examples=15,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_int_roundtrip(dtype: torch.dtype, seed: int, numel: int) -> None:
    info = torch.iinfo(dtype)
    generator = torch.Generator().manual_seed(seed)
    tensor = torch.randint(
        info.min, info.max, (numel,), generator=generator, dtype=dtype
    )
    _roundtrip_tensor(tensor)
