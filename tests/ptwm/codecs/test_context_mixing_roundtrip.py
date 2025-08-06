"""End-to-end container roundtrip for the opt-in context_mixing_lite codec.

This is the test class that would have caught a prior decode regression: it
drives a real Compressor -> Decompressor through the container, forcing the
codec via ``codec_menu``. Crucially it asserts the codec is *actually selected*
(the CM-menu container is smaller than the Identity-only one), so the decode
path is genuinely exercised — a roundtrip where Identity won every plane would
be a vacuous guard.
"""

from __future__ import annotations

import numpy as np
import pytest
from ptwm import CompressionConfig, Compressor, DecompressionConfig, Decompressor
from ptwm.codecs import CodecId


@pytest.fixture
def decompressor() -> Decompressor:
    return Decompressor(DecompressionConfig())


def _compress(original: bytes, menu: list[CodecId]) -> bytes:
    comp = Compressor(CompressionConfig(bytearray_dtype="float32", codec_menu=menu))
    return comp.compress(original)


@pytest.mark.parametrize("n", [4096, 8192])
def test_context_mixing_container_roundtrip(decompressor: Decompressor, n: int) -> None:
    # Compressible payload (few distinct float values) so context_mixing_lite
    # beats Identity on >= 1 plane and its decode path actually runs. n floats
    # -> n-byte byte-split planes, which meet the codec's 4096-byte size gate.
    rng = np.random.default_rng(0)
    original = (rng.integers(0, 4, size=n).astype(np.float32) * 0.25).tobytes()

    cm_blob = _compress(original, [CodecId.Identity, CodecId.ContextMixingLite])
    id_blob = _compress(original, [CodecId.Identity])

    # CM must be selected on >= 1 plane — otherwise the roundtrip below would
    # never decode a CM payload and the guard would be vacuous.
    assert len(cm_blob) < len(id_blob), "context_mixing_lite was not selected"

    # The CM-containing container round-trips bit-exactly (lossless).
    assert decompressor.decompress(cm_blob) == original


def test_context_mixing_absent_by_default(decompressor: Decompressor) -> None:
    # Without opting in, the default menu must still round-trip (CM not used).
    original = (
        np.random.default_rng(1).standard_normal(2048).astype(np.float32).tobytes()
    )
    comp = Compressor(CompressionConfig(bytearray_dtype="float32"))
    assert decompressor.decompress(comp.compress(original)) == original
