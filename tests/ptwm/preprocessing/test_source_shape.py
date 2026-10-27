"""Source shapes of candidate chains cover the raw byte count exactly (P4).

The decoder sizes every plane from the chain's Source node, and the Rust
writer skips candidates whose Source declares another byte count. These
tests pin :func:`build_chain_blobs` and its three call sites.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest
import torch
from ptwm import (
    CompressionConfig,
    Compressor,
    DecompressionConfig,
    Decompressor,
    Format,
    Framing,
    _rust,
)
from ptwm._core import chain_blob_info, chain_source_byte_len
from ptwm.preprocessing._chains import (
    _CHAIN_DTYPE,
    CHAIN_BYTE_PASSTHROUGH_VALUE,
    CHAIN_FP32_SPLIT,
    ClassifierRole,
    _Op,
    build_chain_blobs,
    chain_storage_bytes,
    chains_for,
    source_shape_for,
)
from ptwm.sharding import compress_ptwm_blob
from ptwm.utils import DType


def _source(blob: bytes) -> tuple[list[int], int]:
    """(dims, chain-internal dtype) of a chain blob's Source node."""
    et_size, table = chain_blob_info(blob)
    ops = dict(table)
    wire = blob[et_size:]
    # Legacy chain header: n_nodes, n_edges, n_terms (u8), 3 × u16 lengths.
    pos = 9
    op_idx, plen = struct.unpack_from("<HB", wire, pos)
    assert ops[op_idx] == _Op.Source
    params = wire[pos + 3 : pos + 3 + plen]
    ndim = params[0]
    dims = list(struct.unpack_from(f"<{ndim}I", params, 1))
    (dtype,) = struct.unpack_from("<H", params, 1 + 4 * ndim)
    return dims, dtype


def _builders(dtype: DType) -> list:
    return chains_for(dtype.code, ClassifierRole.STANDARD)


@pytest.mark.parametrize(
    "dtype",
    [
        DType.FLOAT32,
        DType.BFLOAT16,
        DType.FLOAT16,
        DType.FLOAT8_E4M3FN,
        DType.FLOAT8_E5M2,
        DType.FLOAT4_E2M1FN_X2,
        DType.UINT8,
        DType.FLOAT64,
    ],
)
@pytest.mark.parametrize("nbytes", [0, 8, 4096])
def test_byte_input_source_covers_nbytes(dtype: DType, nbytes: int) -> None:
    blobs = build_chain_blobs(_builders(dtype), shape=None, nbytes=nbytes)
    assert blobs
    for blob in blobs:
        assert chain_source_byte_len(blob) == nbytes
        dims, src_dtype = _source(blob)
        assert dims == [nbytes // chain_storage_bytes(src_dtype)]


@pytest.mark.parametrize(
    ("dtype", "nbytes"), [(DType.FLOAT32, 4094), (DType.BFLOAT16, 1001)]
)
def test_unaligned_byte_input_falls_back_to_passthrough(
    dtype: DType, nbytes: int
) -> None:
    blobs = build_chain_blobs(_builders(dtype), shape=None, nbytes=nbytes)
    expected = CHAIN_BYTE_PASSTHROUGH_VALUE([nbytes]).to_bytes()
    assert blobs == [expected]


@pytest.mark.parametrize(
    ("shape", "nbytes", "dims"),
    [
        ((64, 64), 64 * 64 * 4, [64, 256]),  # int32 tensor, element shape
        ((), 8, [8]),  # int64 scalar
        ((0, 16), 0, [0, 16]),  # empty
    ],
)
def test_passthrough_fallback_counts_bytes(
    shape: tuple[int, ...], nbytes: int, dims: list[int]
) -> None:
    (blob,) = build_chain_blobs([], shape=list(shape), nbytes=nbytes)
    assert _source(blob) == (dims, _CHAIN_DTYPE[DType.UINT8.code])
    assert chain_source_byte_len(blob) == nbytes


def test_typed_shapes_are_unchanged() -> None:
    blobs = build_chain_blobs(
        _builders(DType.BFLOAT16), shape=[64, 32], nbytes=64 * 32 * 2
    )
    assert len(blobs) == len(_builders(DType.BFLOAT16))
    for blob in blobs:
        assert _source(blob)[0] == [64, 32]


def test_source_shape_for_rules() -> None:
    assert source_shape_for(None, 4096, 4) == [1024]
    assert source_shape_for(None, 4094, 4) is None
    assert source_shape_for([], 4, 4) == []
    assert source_shape_for([], 8, 1) == [8]
    assert source_shape_for([3, 5], 60, 4) == [3, 5]
    assert source_shape_for([3, 5], 60, 1) == [3, 20]
    assert source_shape_for([3, 5], 61, 1) is None
    assert source_shape_for([0, 5], 1, 1) is None
    assert source_shape_for(None, 1 << 33, 1) is None  # dim > u32


def test_compressor_byte_input_declares_raw_size(monkeypatch) -> None:
    seen: list[list[bytes]] = []
    real = _rust.compress_compact

    def spy(*args, **kwargs):
        seen.append(list(args[6]))
        return real(*args, **kwargs)

    monkeypatch.setattr(_rust, "compress_compact", spy)
    raw = np.random.default_rng(0).standard_normal(1024).astype(np.float32).tobytes()
    cfg = CompressionConfig(bytearray_dtype="float32", framing=Framing.COMPACT)
    Compressor(cfg).compress(raw)
    (chains,) = seen
    assert chains
    for blob in chains:
        assert chain_source_byte_len(blob) == 4096
    assert CHAIN_FP32_SPLIT([1024]).to_bytes() in chains


@pytest.mark.parametrize("framing", [Framing.COMPACT, Framing.CONTAINER])
@pytest.mark.parametrize(
    ("dtype", "nbytes"),
    [("float32", 4094), ("float32", 4096), ("float64", 1024), ("int32", 1024)],
)
def test_byte_input_round_trips(framing: Framing, dtype: str, nbytes: int) -> None:
    raw = bytes(np.random.default_rng(1).integers(0, 256, nbytes, dtype=np.uint8))
    cfg = CompressionConfig(bytearray_dtype=dtype, framing=framing)
    blob = Compressor(cfg).compress(raw)
    assert Decompressor().decompress(blob) == raw


@pytest.mark.parametrize("framing", [Framing.COMPACT, Framing.CONTAINER])
@pytest.mark.parametrize("fmt", [Format.TORCH, Format.NUMPY])
def test_delta_with_tensor_input_round_trips(framing: Framing, fmt: Format) -> None:
    # Delta encoding used to replace the tensor with its byte residual and
    # then read `.dtype` from the bytes (AttributeError).
    rng = np.random.default_rng(2)
    base = rng.standard_normal((16, 8)).astype(np.float32)
    ref = rng.standard_normal((16, 8)).astype(np.float32)
    data = torch.from_numpy(base) if fmt == Format.TORCH else base
    ref_in = torch.from_numpy(ref) if fmt == Format.TORCH else ref
    cfg = CompressionConfig(
        input_format=fmt, framing=framing, delta_compressed_type="byte"
    )
    blob = Compressor(cfg).compress(data, delta_second_data=ref_in)
    out = Decompressor(DecompressionConfig(delta_second_data=ref.tobytes())).decompress(
        blob
    )
    if fmt == Format.TORCH:
        assert torch.equal(out, data)
    else:
        np.testing.assert_array_equal(out, base)


@pytest.mark.parametrize(
    ("dtype", "shape"),
    [
        (torch.int64, (32, 4)),
        (torch.float64, (8, 8)),
        (torch.int64, ()),
        (torch.int32, (64, 64)),
        (torch.float32, (16, 16)),
    ],
)
def test_sharding_writer_chains_cover_raw_bytes(monkeypatch, dtype, shape) -> None:
    t = (torch.arange(int(np.prod(shape))) % 13).reshape(shape).to(dtype)
    raw = t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    seen: list[list[list[bytes]]] = []
    real = _rust.compress_model

    def spy(records, chains, *args, **kwargs):
        seen.append(chains)
        return real(records, chains, *args, **kwargs)

    monkeypatch.setattr(_rust, "compress_model", spy)
    blob = compress_ptwm_blob([("t", raw, dtype, tuple(shape))])
    for chain in seen[0][0]:
        assert chain_source_byte_len(chain) == len(raw)
    assert bytes(_rust.decode_tensor(blob, "t", False)) == raw
