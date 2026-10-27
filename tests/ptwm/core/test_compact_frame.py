"""Compact single-tensor frame: selection, fallbacks and round trips."""

from __future__ import annotations

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
from ptwm._exceptions import Error, HeaderParseError

COMPACT = b"\x89PTWS"
CONTAINER = b"\x89PTWM"


def _torch_cfg(**kw) -> CompressionConfig:
    return CompressionConfig(input_format=Format.TORCH, **kw)


def test_default_framing_is_compact() -> None:
    t = torch.randn(16, 16).to(torch.bfloat16)
    blob = Compressor(_torch_cfg()).compress(t)
    assert blob[:5] == COMPACT
    back = Decompressor().decompress(blob)
    assert back.dtype == t.dtype
    assert back.shape == t.shape
    assert torch.equal(back.view(torch.uint8), t.view(torch.uint8))


def test_framing_accepts_names() -> None:
    assert Framing("container") is Framing.CONTAINER
    assert Framing("COMPACT") is Framing.COMPACT


def test_container_framing_still_decodes() -> None:
    t = torch.randn(8, 32, dtype=torch.float32)
    blob = Compressor(_torch_cfg(framing=Framing.CONTAINER)).compress(t)
    assert blob[:5] == CONTAINER
    assert torch.equal(Decompressor().decompress(blob), t)


def test_streaming_falls_back_to_container() -> None:
    t = torch.randn(4096).to(torch.bfloat16)
    blob = Compressor(_torch_cfg(is_streaming=True, streaming_chunk=1024)).compress(t)
    assert blob[:5] == CONTAINER
    back = Decompressor().decompress(blob)
    assert torch.equal(back.view(torch.uint8), t.view(torch.uint8))


@pytest.mark.parametrize(
    ("dtype", "shape"),
    [
        (torch.bfloat16, (3, 5, 7)),
        (torch.float16, (64,)),
        (torch.float32, (2, 128)),
        (torch.float8_e4m3fn, (256,)),
        (torch.float8_e5m2, (16, 16)),
        (torch.float32, ()),
        (torch.bfloat16, (0, 16)),
    ],
)
def test_torch_round_trip_keeps_shape_and_dtype(dtype: torch.dtype, shape) -> None:
    g = torch.Generator().manual_seed(3)
    t = torch.randn(shape, generator=g).to(dtype)
    blob = Compressor(_torch_cfg()).compress(t)
    assert blob[:5] == COMPACT
    back = Decompressor().decompress(blob)
    assert back.dtype == t.dtype
    assert back.shape == t.shape
    flat = back.reshape(-1).view(torch.uint8)
    assert torch.equal(flat, t.reshape(-1).view(torch.uint8))


def test_numpy_round_trip() -> None:
    a = np.random.default_rng(0).standard_normal((12, 9)).astype(np.float32)
    blob = Compressor(CompressionConfig(input_format=Format.NUMPY)).compress(a)
    assert blob[:5] == COMPACT
    back = Decompressor().decompress(blob)
    assert isinstance(back, np.ndarray)
    assert back.shape == a.shape
    assert np.array_equal(back, a)


def test_byte_round_trip() -> None:
    data = bytes(range(256)) * 9
    blob = Compressor(CompressionConfig(bytearray_dtype="float32")).compress(data)
    assert blob[:5] == COMPACT
    assert Decompressor().decompress(blob) == data


def test_incompressible_input_is_stored_with_small_overhead() -> None:
    data = np.random.default_rng(1).integers(0, 256, 4000, dtype=np.uint8).tobytes()
    blob = Compressor(CompressionConfig(bytearray_dtype="float32")).compress(data)
    assert len(blob) <= len(data) + 32
    assert Decompressor().decompress(blob) == data


@pytest.mark.parametrize("framing", [Framing.COMPACT, Framing.CONTAINER])
def test_integrity_fields_can_be_disabled(framing: Framing) -> None:
    t = torch.randn(512).to(torch.bfloat16)
    full = Compressor(_torch_cfg(framing=framing)).compress(t)
    bare = Compressor(
        _torch_cfg(framing=framing, emit_payload_hash=False, emit_plane_crc=False)
    ).compress(t)
    # 8 B payload hash, plus 4 B CRC per coded plane (bf16 has two).
    assert len(full) - len(bare) == 8 + 2 * 4
    back = Decompressor().decompress(bare)
    assert torch.equal(back.view(torch.uint8), t.view(torch.uint8))


def test_delta_round_trip() -> None:
    base = torch.randn(1024).to(torch.bfloat16)
    new = base.clone()
    new[::17] += 1
    ref = base.view(torch.uint8).numpy().tobytes()
    cfg = CompressionConfig(bytearray_dtype="bfloat16", delta_compressed_type="file")
    data = new.view(torch.uint8).numpy().tobytes()
    blob = Compressor(cfg).compress(data, delta_second_data=ref)
    assert blob[:5] == COMPACT
    out = Decompressor(DecompressionConfig(delta_second_data=ref)).decompress(blob)
    assert out == data
    with pytest.raises(ValueError, match="delta_second_data is None"):
        Decompressor().decompress(blob)
    with pytest.raises(ValueError, match="BLAKE3"):
        Decompressor(DecompressionConfig(delta_second_data=bytes(len(ref)))).decompress(
            blob
        )


def test_corrupted_frame_is_rejected() -> None:
    t = torch.randn(64, 64).to(torch.bfloat16)
    blob = bytearray(Compressor(_torch_cfg()).compress(t))
    blob[len(blob) // 2] ^= 0xFF
    with pytest.raises(Error):
        Decompressor().decompress(bytes(blob))
    with pytest.raises(Error):
        Decompressor().decompress(bytes(blob[:-3]))


def test_unknown_magic_is_rejected() -> None:
    with pytest.raises(HeaderParseError):
        Decompressor().decompress(b"\x89PTWQ" + bytes(20))
    with pytest.raises(HeaderParseError):
        Decompressor().decompress(b"\x89PT")


def test_decode_compact_reports_metadata() -> None:
    t = torch.full((4, 8), 0.25, dtype=torch.bfloat16)
    blob = Compressor(_torch_cfg()).compress(t)
    raw, shape, dtype_name, _code, input_format, delta, payload_len = (
        _rust.decode_compact(blob)
    )
    assert raw == t.view(torch.uint8).numpy().tobytes()
    assert shape == [4, 8]
    assert dtype_name == "bfloat16"
    assert input_format == Format.TORCH.value
    assert delta is None
    assert 0 < payload_len < len(blob)
