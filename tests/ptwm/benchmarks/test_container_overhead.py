"""Size limits for single-tensor ``Compressor.compress`` output.

The fixed bytes per output (header, tables, framing) used to be about 2.7 KB,
so small tensors came out larger than their input. These tests hold the
default (compact) framing to the limits below and keep the full container's
Extension Table trimmed. The sweep matches ``benchmarks/container_overhead``.
"""

from __future__ import annotations

import pytest
import torch
from ptwm._config import Framing

from benchmarks.container_overhead import (
    DTYPES,
    SIZES,
    make_tensor,
    measure,
    tiled_rows,
)

# Largest allowed excess over the raw bytes, and over zstd-19 when the ptwm
# codec payload is smaller than zstd-19's output.
SLACK = 64


def test_constant_64_byte_bf16_tensor_is_at_most_128_bytes() -> None:
    t = torch.full((32,), 0.5, dtype=torch.bfloat16)
    row = measure(t, Framing.COMPACT, "bf16", "constant")
    assert row.out <= 128, f"constant 64 B bf16 tensor -> {row.out} B"


@pytest.mark.parametrize(("dname", "dtype"), DTYPES)
@pytest.mark.parametrize("nbytes", SIZES)
def test_random_normal_overhead_is_bounded(
    dname: str, dtype: torch.dtype, nbytes: int
) -> None:
    row = measure(
        make_tensor(dtype, "normal", nbytes), Framing.COMPACT, dname, "normal"
    )
    assert row.out <= row.raw + SLACK, row
    if row.payload < row.zstd19:
        assert row.out <= row.zstd19 + SLACK, row


@pytest.mark.parametrize(("dname", "dtype"), DTYPES)
@pytest.mark.parametrize("nbytes", SIZES)
def test_constant_overhead_is_bounded(
    dname: str, dtype: torch.dtype, nbytes: int
) -> None:
    row = measure(
        make_tensor(dtype, "constant", nbytes), Framing.COMPACT, dname, "constant"
    )
    assert row.out <= row.raw + SLACK, row
    # Four-plane fp32 frames carry about 120 B of chain and plane metadata.
    assert row.out <= 256, row


def test_tiled_rows_ratio_is_at_least_60() -> None:
    row = measure(tiled_rows(64), Framing.COMPACT, "bf16", "tiled")
    assert row.ratio >= 60, f"(64, 1024) tiled bf16 ratio {row.ratio:.1f}"


@pytest.mark.parametrize(("dname", "dtype"), DTYPES)
def test_container_framing_floor_stays_trimmed(dname: str, dtype: torch.dtype) -> None:
    # v1.1 wrote about 2.5 KB of non-payload bytes per single-tensor
    # container, 2.1 KB of it unreferenced codec entries.
    row = measure(
        make_tensor(dtype, "normal", 4096), Framing.CONTAINER, dname, "normal"
    )
    assert row.non_payload < 1100, row
