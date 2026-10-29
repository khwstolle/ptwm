"""Size limits for PTWM output.

The fixed bytes per single-tensor output (header, tables, framing) used to be
about 2.7 KB, so small tensors came out larger than their input. These tests
hold the default (compact) framing and the compact container (PTWC) to the
limits below, check that the legacy container keeps the v1.1.0 Extension
Table, and bound the per-tensor overhead of multi-tensor containers. The
sweeps match
``benchmarks/container_overhead``.
"""

from __future__ import annotations

import json

import pytest
import torch
from ptwm import CompressionConfig, Compressor, Decompressor, Format, _rust
from ptwm._config import ContainerLayout, Framing
from ptwm._magic import container_kind
from ptwm.codecs import CodecId

from benchmarks.container_overhead import (
    DTYPES,
    FORMATS,
    SIZES,
    ShapeFn,
    codec_data_bytes,
    compress_checked,
    distinct_shapes,
    make_tensor,
    measure,
    measure_multi,
    non_payload_bytes,
    same_shape,
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
def test_legacy_container_keeps_the_v1_1_table(dname: str, dtype: torch.dtype) -> None:
    # The legacy layout interns all 16 built-in codecs (about 2.1 KB), as
    # v1.1.0 did: the v1.1 transcoder rebuilds that table for exploded
    # stores. A trimmed legacy table (about 1 KB in total) broke them. The
    # upper bound keeps the rest of the legacy overhead from growing.
    row = measure(
        make_tensor(dtype, "normal", 4096),
        Framing.CONTAINER,
        dname,
        "normal",
        ContainerLayout.LEGACY,
    )
    assert row.framing == "ptwm", row
    assert 2000 < row.non_payload < 3200, row


def test_formats_write_the_expected_magic() -> None:
    t = make_tensor(torch.float32, "normal", 4096)
    labels = [
        measure(t, framing, "fp32", "normal", layout).framing
        for framing, layout in FORMATS
    ]
    assert labels == ["ptws", "ptwc", "ptwm"]


@pytest.mark.parametrize(("dname", "dtype"), DTYPES)
@pytest.mark.parametrize("nbytes", [1024, 65536])
def test_codec_data_is_counted_alike_in_every_format(
    dname: str, dtype: torch.dtype, nbytes: int
) -> None:
    # The three formats carry the same coded planes for the same tensor, so
    # the codec data must come out equal. These inputs carry no codec state;
    # the two tests below cover inline and shared state.
    t = make_tensor(dtype, "normal", nbytes)
    data = {
        f"{framing.name}/{layout.name}": codec_data_bytes(
            compress_checked(t, framing, layout)
        )
        for framing, layout in FORMATS
    }
    assert len(set(data.values())) == 1, data


def _payload_only_bytes(blob: bytes) -> int:
    """Sum of the plane payloads of a container, without any codec state."""
    _registry, tensors, _members = _rust.explode_ptwm(blob)
    return sum(
        plane["comp_len"]
        for _name, _key, tensor_json in tensors
        for plane in json.loads(tensor_json)["planes"]
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_inline_state_is_counted_alike_in_every_format(dtype: torch.dtype) -> None:
    # ArithmeticO0 stores its frequency table as inline codec state, so the
    # count must include it in every format. A count that left inline state
    # out of the containers would come out smaller than the frame's.
    t = torch.randn(16384, generator=torch.Generator().manual_seed(3)).to(dtype)
    data = {}
    for framing, layout in FORMATS:
        blob = Compressor(
            CompressionConfig(
                input_format=Format.TORCH,
                framing=framing,
                container_layout=layout,
                codec_menu=[CodecId.ArithmeticO0],
            )
        ).compress(t)
        back = Decompressor().decompress(blob)
        assert torch.equal(back.view(torch.uint8), t.view(torch.uint8))
        data[container_kind(blob)] = codec_data_bytes(blob)
        if framing == Framing.CONTAINER:
            # The input really carries inline state.
            assert codec_data_bytes(blob) > _payload_only_bytes(blob), layout
    assert set(data) == {"ptws", "ptwc", "ptwm"}, data
    assert len(set(data.values())) == 1, data


@pytest.mark.parametrize("layout", [ContainerLayout.LEGACY, ContainerLayout.COMPACT])
def test_shared_state_is_counted(layout: ContainerLayout) -> None:
    # A container with one shared (prelude) state, built through the
    # transcoder: its bytes count as codec data in both container layouts.
    # A single-tensor frame cannot hold shared state.
    t = make_tensor(torch.float32, "normal", 4096)
    blob = compress_checked(t, Framing.CONTAINER, layout)
    registry_json, tensors, members = _rust.explode_ptwm(blob)
    registry = json.loads(registry_json)
    state = bytes(range(200))
    registry["registry"]["shared_state"].append(
        {
            "id": 0,
            "codec": int(CodecId.Order1ScaleAC),
            "state_format_version": 1,
            "applies_to_mask": 1,
            "hash_xxh64": f"{_rust.xxhash64(state):016x}",
            "name": "s",
            "payload": "shared.bin",
        }
    )
    with_state = _rust.implode_ptwm(
        json.dumps(registry),
        [tensor_json for _name, _key, tensor_json in tensors],
        {**members, "shared.bin": state},
    )
    assert container_kind(with_state) == container_kind(blob)
    assert codec_data_bytes(with_state) == codec_data_bytes(blob) + len(state)


@pytest.mark.parametrize(("dname", "dtype"), DTYPES)
@pytest.mark.parametrize("nbytes", SIZES)
def test_compact_container_overhead_is_bounded(
    dname: str, dtype: torch.dtype, nbytes: int
) -> None:
    # A one-tensor PTWC file pays its header, Extension Table, registry,
    # index and footer alone. Measured maxima over SIZES: fp32 141 B,
    # bf16 117 B.
    row = measure(
        make_tensor(dtype, "normal", nbytes),
        Framing.CONTAINER,
        dname,
        "normal",
        ContainerLayout.COMPACT,
    )
    assert row.framing == "ptwc", row
    assert row.non_payload <= 160, row


@pytest.mark.parametrize(
    ("shape_of", "hashes", "limit"),
    [(same_shape, True, 95), (distinct_shapes, True, 100), (same_shape, False, 70)],
    ids=["same-shape", "distinct-shapes", "no-hash-no-crc"],
)
def test_multi_tensor_overhead_per_tensor(
    shape_of: ShapeFn, hashes: bool, limit: int
) -> None:
    # 64 fp32 (64,) bias vectors (synthesis cases A and B). Measured per
    # tensor: 83.1 B, 82.6 B and 59.1 B, against 262 B, 317 B and 238 B for
    # the legacy layout.
    compact = measure_multi(
        "fp32", "float32", shape_of, 64, ContainerLayout.COMPACT, hashes
    )
    legacy = measure_multi(
        "fp32", "float32", shape_of, 64, ContainerLayout.LEGACY, hashes
    )
    assert (compact.layout, legacy.layout) == ("ptwc", "ptwm")
    assert compact.per_tensor <= limit, compact
    assert compact.per_tensor < legacy.per_tensor / 2, (compact, legacy)
