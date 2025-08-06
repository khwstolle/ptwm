"""Property tests for the :class:`weights.codecs.Codec` protocol.

Every codec must satisfy ``codec.decode(codec.encode(x), len(x)) == x`` for
arbitrary byte inputs. The codec-specific tagged-blob fallback in Huffman /
rANS (for incompressible input) is exercised implicitly because hypothesis
generates near-uniform random bytes as part of its search.

This file lives at ``tests/weights/test_codecs.py`` rather than inside a
``codecs/`` subpackage because the stdlib ``codecs`` module takes priority
in pytest's rootdir-based import resolution, causing a name collision.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from ptwm.codecs import ZstdCodec, get_codec

_CODEC_NAMES = ["identity", "huffman", "rans", "zstd"]


@pytest.mark.parametrize("codec_name", _CODEC_NAMES)
@given(data=st.binary(min_size=0, max_size=8192))
@settings(deadline=None, max_examples=80)
def test_codec_roundtrip(codec_name: str, data: bytes) -> None:
    codec = get_codec(codec_name)
    blob = codec.encode(data)
    recovered = codec.decode(blob, len(data))
    assert recovered == data


@pytest.mark.parametrize("codec_name", _CODEC_NAMES)
@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"\x00",
        b"\x00" * 1024,  # single-symbol — defeats Huffman's tree build
        bytes(range(256)),  # full alphabet
    ],
)
def test_codec_roundtrip_edge_cases(codec_name: str, data: bytes) -> None:
    codec = get_codec(codec_name)
    blob = codec.encode(data)
    assert codec.decode(blob, len(data)) == data


@pytest.mark.parametrize("level", [1, 3, 9, 19])
@given(data=st.binary(min_size=0, max_size=4096))
@settings(deadline=None, max_examples=30)
def test_zstd_level_roundtrip(level: int, data: bytes) -> None:
    codec = ZstdCodec(level=level)
    blob = codec.encode(data)
    assert codec.decode(blob, len(data)) == data


@pytest.mark.parametrize("codec_name", ["huffman", "rans"])
@given(data=st.binary(min_size=0, max_size=2048))
@settings(deadline=None, max_examples=40)
def test_tagged_blob_decodes_both_paths(codec_name: str, data: bytes) -> None:
    """The 1-byte tag correctly routes raw fallback vs entropy-coded payload."""
    codec = get_codec(codec_name)
    blob = codec.encode(data)
    if blob:
        # First byte is 0 (raw) or 1 (encoded) — anything else is a bug.
        assert blob[0] in (0, 1)
    assert codec.decode(blob, len(data)) == data


@pytest.mark.parametrize("codec_name", ["huffman", "rans"])
def test_large_plane_roundtrip(codec_name: str) -> None:
    """A ~2 MB low-entropy plane must compress, not silently fall back to
    tagged-raw (regression guard for the old u16 per-stream jump-table cap).
    """
    import random

    rng = random.Random(0xC0FFEE)
    # 2 MB: 90 % zeros, 10 % spread over {1..16}. Entropy ≈ 0.6 bits/byte —
    # should compress to well under half its size.
    n = 2 * 1024 * 1024
    data = bytearray(n)
    for i in range(0, n, 10):
        data[i] = rng.randint(1, 16)
    data = bytes(data)

    codec = get_codec(codec_name)
    blob = codec.encode(data)
    # Must not be the raw-fallback tag: we insist on real compression here.
    assert blob[:1] == b"\x01", (
        f"{codec_name}: got raw-fallback tag on compressible 2 MB input "
        f"(blob len {len(blob)})"
    )
    assert len(blob) < len(data) // 2, (
        f"{codec_name}: expected <50 % size, got {len(blob)}/{len(data)}"
    )
    assert codec.decode(blob, len(data)) == data
