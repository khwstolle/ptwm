"""Size limits for single-tensor ``Compressor.compress`` output.

The fixed bytes per output (header, tables, framing) used to be about 2.7 KB,
so small tensors came out larger than their input. These tests hold the
default (compact) framing to the limits below and keep the full container's
Extension Table trimmed. The sweep matches ``benchmarks/container_overhead``.
"""

from __future__ import annotations

import pytest
import torch
from ptwm import CompressionConfig, Compressor, Decompressor, Format
from ptwm._config import Framing

from benchmarks.container_overhead import (
    DTYPES,
    SIZES,
    make_tensor,
    measure,
    non_payload_bytes,
    tiled_rows,
)

# Largest allowed excess over the raw bytes, and over zstd-19 when the ptwm
# codec payload is smaller than zstd-19's output.
SLACK = 64

# Largest compact-frame non-payload bytes (header, compact chain, plane heads
# and payload hash; the frame omits plane CRCs when it stores the hash) per
# dtype. fp32 frames carry four planes, bf16 frames two. Measured maxima over
# SIZES: fp32 55 B, bf16 43 B.
COMPACT_OVERHEAD = {"bf16": 50, "fp32": 62}

# Bound for any fp32 or bf16 chain the compressor picks, including the
# IntDelta and PredictorXor chains for smooth data, which carry an extra node
# and a non-spine edge. Measured maxima for cumulative-sum data: fp32 61 B
# with the default config, 69 B with plane CRCs and no payload hash.
CHAIN_OVERHEAD_LIMIT = 70


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
    assert row.non_payload <= COMPACT_OVERHEAD[dname], row


@pytest.mark.parametrize(("dname", "dtype"), DTYPES)
@pytest.mark.parametrize("nbytes", SIZES)
def test_constant_overhead_is_bounded(
    dname: str, dtype: torch.dtype, nbytes: int
) -> None:
    row = measure(
        make_tensor(dtype, "constant", nbytes), Framing.COMPACT, dname, "constant"
    )
    assert row.out <= row.raw + SLACK, row
    assert row.non_payload <= COMPACT_OVERHEAD[dname], row
    assert row.out <= 256, row


@pytest.mark.parametrize(
    ("hash_", "crc"), [(True, True), (False, True), (False, False)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("n", [64, 1024, 65536, 262144])
def test_smooth_data_overhead_is_bounded(
    hash_: bool, crc: bool, dtype: torch.dtype, n: int
) -> None:
    g = torch.Generator().manual_seed(n)
    t = torch.cumsum(torch.randn(n, generator=g), 0).to(dtype)
    blob = Compressor(
        CompressionConfig(
            input_format=Format.TORCH,
            emit_payload_hash=hash_,
            emit_plane_crc=crc,
        )
    ).compress(t)
    assert blob[:5] == b"\x89PTWS"
    back = Decompressor().decompress(blob)
    assert torch.equal(back.view(torch.uint8), t.view(torch.uint8))
    assert non_payload_bytes(blob) <= CHAIN_OVERHEAD_LIMIT, (dtype, n)


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
