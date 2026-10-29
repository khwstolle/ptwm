"""Compact single-tensor frame: selection, fallbacks and round trips."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch
from ptwm import (
    CompressionConfig,
    Compressor,
    ContainerLayout,
    DecompressionConfig,
    Decompressor,
    Format,
    Framing,
    _rust,
)
from ptwm._exceptions import Error, HeaderParseError, InvalidDTypeError

COMPACT = b"\x89PTWS"
# A full container has the default (compact) layout unless the config pins
# ContainerLayout.LEGACY.
CONTAINER = b"\x89PTWC"
LEGACY_CONTAINER = b"\x89PTWM"


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


@pytest.mark.parametrize(
    ("framing", "magic"),
    [
        ("compact", COMPACT),
        ("COMPACT", COMPACT),
        (1, COMPACT),
        ("container", CONTAINER),
        (2, CONTAINER),
    ],
)
def test_config_coerces_framing(framing, magic: bytes) -> None:
    """A framing name or value selects the same output as the enum.

    The config once kept the raw value, which never equalled
    ``Framing.COMPACT``, so ``framing="compact"`` wrote a full container.
    """
    cfg = _torch_cfg(framing=framing)
    assert isinstance(cfg.framing, Framing)
    assert Compressor(cfg).compress(torch.randn(64))[:5] == magic


def test_config_rejects_unknown_framing() -> None:
    with pytest.raises(ValueError, match="bogus"):
        _torch_cfg(framing="bogus")


@pytest.mark.parametrize(
    ("layout", "magic"),
    [
        (None, CONTAINER),
        (ContainerLayout.COMPACT, CONTAINER),
        (ContainerLayout.LEGACY, LEGACY_CONTAINER),
    ],
)
def test_container_framing_still_decodes(layout, magic: bytes) -> None:
    t = torch.randn(8, 32, dtype=torch.float32)
    kw = {} if layout is None else {"container_layout": layout}
    blob = Compressor(_torch_cfg(framing=Framing.CONTAINER, **kw)).compress(t)
    assert blob[:5] == magic
    assert torch.equal(Decompressor().decompress(blob), t)


@pytest.mark.parametrize(
    ("layout", "magic"),
    [
        (None, CONTAINER),
        (ContainerLayout.COMPACT, CONTAINER),
        (ContainerLayout.LEGACY, LEGACY_CONTAINER),
    ],
)
def test_streaming_falls_back_to_container(layout, magic: bytes) -> None:
    """The compact frame has no chunk table: streaming falls back to a full
    container, compact by default and legacy only when pinned."""
    t = torch.randn(4096).to(torch.bfloat16)
    kw = {} if layout is None else {"container_layout": layout}
    blob = Compressor(
        _torch_cfg(is_streaming=True, streaming_chunk=1024, **kw)
    ).compress(t)
    assert blob[:5] == magic
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


@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.float32, np.int32])
@pytest.mark.parametrize("framing", [Framing.COMPACT, Framing.CONTAINER])
def test_numpy_round_trip_keeps_64_bit_dtypes(dtype, framing: Framing) -> None:
    """float64 and int64 arrays once stored dtype code 0 and INT32."""
    a = np.arange(6, dtype=dtype).reshape(2, 3)
    cfg = CompressionConfig(input_format=Format.NUMPY, framing=framing)
    out = Decompressor().decompress(Compressor(cfg).compress(a))
    assert isinstance(out, np.ndarray)
    assert out.dtype == a.dtype
    np.testing.assert_array_equal(out, a)


def test_numpy_dtype_without_code_is_rejected() -> None:
    """An array whose dtype has no code fails before anything is written."""
    a = np.zeros(4, dtype="datetime64[s]")
    with pytest.raises(InvalidDTypeError, match="unsupported dtype"):
        Compressor(CompressionConfig(input_format=Format.NUMPY)).compress(a)


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
    t = torch.randn(512, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16)
    full = Compressor(_torch_cfg(framing=framing)).compress(t)
    bare = Compressor(
        _torch_cfg(framing=framing, emit_payload_hash=False, emit_plane_crc=False)
    ).compress(t)
    crc_only = Compressor(
        _torch_cfg(framing=framing, emit_payload_hash=False, emit_plane_crc=True)
    ).compress(t)
    # 8 B payload hash; 4 B CRC per coded plane (bf16 has two). The compact
    # frame omits plane CRCs when it stores the payload hash, which already
    # covers every plane of its one tensor.
    if framing is Framing.COMPACT:
        assert len(full) - len(bare) == 8
    else:
        assert len(full) - len(bare) == 8 + 2 * 4
    assert len(crc_only) - len(bare) == 2 * 4
    for blob in (bare, crc_only):
        back = Decompressor().decompress(blob)
        assert torch.equal(back.view(torch.uint8), t.view(torch.uint8))


@pytest.mark.parametrize("tail", [b"\x00", b"\x00" * 7, b"\xff" * 100])
@pytest.mark.parametrize(
    "t",
    [
        torch.randn(64, 64, generator=torch.Generator().manual_seed(0)).to(
            torch.bfloat16
        ),
        torch.randn(64, 64, generator=torch.Generator().manual_seed(0)),
        torch.randint(
            -128, 127, (4096,), generator=torch.Generator().manual_seed(0)
        ).to(torch.int8),
        torch.cumsum(torch.randn(4096, generator=torch.Generator().manual_seed(0)), 0),
    ],
    ids=["bf16", "fp32", "int8", "fp32-smooth"],
)
@pytest.mark.parametrize(
    ("hash_", "crc"), [(True, True), (False, True), (False, False)]
)
def test_appended_bytes_are_rejected(
    t: torch.Tensor, tail: bytes, hash_: bool, crc: bool
) -> None:
    blob = Compressor(_torch_cfg(emit_payload_hash=hash_, emit_plane_crc=crc)).compress(
        t
    )
    assert blob[:5] == b"\x89PTWS"
    with pytest.raises(Error, match="trailing|stored body is"):
        Decompressor().decompress(blob + tail)


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


_SHAPE_BOMB = textwrap.dedent(
    """
    import numpy as np
    from ptwm import CompressionConfig, Compressor, Decompressor, Format, Method
    from ptwm._exceptions import Error


    def varint(n):
        out = bytearray()
        while True:
            b, n = n & 0x7F, n >> 7
            if n:
                out.append(b | 0x80)
            else:
                out.append(b)
                return bytes(out)


    def read_varint(b, i):
        v = shift = 0
        while True:
            c = b[i]
            i += 1
            v |= (c & 0x7F) << shift
            shift += 7
            if c < 0x80:
                return v, i


    cfg = CompressionConfig(input_format=Format.NUMPY, method=Method.ZSTD)
    f = Compressor(cfg).compress(np.zeros((64, 64), dtype=np.uint8))
    assert f[:5] == b"\\x89PTWS"
    _, i = read_varint(f, 6)  # dtype code
    i += 1  # input format
    ndim, i = read_varint(f, i)
    start = i
    for _ in range(ndim):
        _, i = read_varint(f, i)
    # A 32 GiB plane declared by a frame of about 50 bytes.
    bomb = f[:start] + varint(1 << 20) + varint(1 << 15) + f[i:]
    try:
        Decompressor().decompress(bomb)
    except Error as e:
        print("rejected:", e)
    else:
        raise SystemExit("decoded a 32 GiB shape from a 4 KiB payload")
    """
)


def test_declared_shape_does_not_size_allocation() -> None:
    """A frame whose shape claims 32 GiB fails with an error, not an abort.

    The decoder once reserved the declared plane length before decoding,
    so this frame killed the process with SIGABRT. It runs in a child
    process so a regression fails this test instead of the whole session.
    """
    r = subprocess.run(
        [sys.executable, "-c", _SHAPE_BOMB],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    assert "rejected:" in r.stdout


def test_container_readers_name_a_compact_frame() -> None:
    """Container APIs report a frame as a frame, not as a truncated file."""
    blob = Compressor(CompressionConfig(input_format=Format.NUMPY)).compress(
        np.arange(64, dtype=np.float32)
    )
    assert blob[:5] == COMPACT
    for call in (
        lambda: _rust.decode_model(blob),
        lambda: _rust.list_tensor_names(blob),
        lambda: _rust.decode_tensor(blob, "_"),
    ):
        with pytest.raises(Error, match="compact single-tensor frame") as e:
            call()
        assert "truncated" not in str(e.value)


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


@pytest.mark.parametrize("framing", [Framing.COMPACT, Framing.CONTAINER])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(5, 0), (0, 5)])
def test_empty_matrix_round_trips(framing: Framing, dtype: torch.dtype, shape) -> None:
    # The row-normalizing chains reject a matrix with an empty dimension;
    # the builders drop them so the tensor still compresses.
    t = torch.empty(shape, dtype=dtype)
    blob = Compressor(_torch_cfg(framing=framing)).compress(t)
    back = Decompressor().decompress(blob)
    assert back.dtype == dtype
    assert back.shape == t.shape
