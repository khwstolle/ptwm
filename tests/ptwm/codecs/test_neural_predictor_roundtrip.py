"""End-to-end container roundtrip for the opt-in neural_predictor codec.

Mirrors test_context_mixing_roundtrip.py: drives a real Compressor ->
Decompressor through the container, forcing the codec via ``codec_menu``.
Non-vacuous — it asserts the codec is *actually selected* (the menu container is
smaller than the Identity-only one), so the decode path genuinely runs
instead of trivially passing through Identity. A roundtrip where Identity won every plane would be a vacuous
guard.
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
def test_neural_predictor_container_roundtrip(
    decompressor: Decompressor, n: int
) -> None:
    # Compressible payload (few distinct float values) so neural_predictor beats
    # Identity on >= 1 byte-split plane and its decode path actually runs. n
    # floats -> n-byte planes, meeting the codec's 4096-byte size gate.
    rng = np.random.default_rng(0)
    original = (rng.integers(0, 4, size=n).astype(np.float32) * 0.25).tobytes()

    np_blob = _compress(original, [CodecId.Identity, CodecId.NeuralPredictor])
    id_blob = _compress(original, [CodecId.Identity])

    # NeuralPredictor must be selected on >= 1 plane — otherwise the roundtrip
    # below would never decode an NP payload and the guard would be vacuous.
    assert len(np_blob) < len(id_blob), "neural_predictor was not selected"

    # The NP-containing container round-trips bit-exactly (lossless).
    assert decompressor.decompress(np_blob) == original


def test_neural_predictor_absent_by_default(decompressor: Decompressor) -> None:
    # Without opting in, the default menu must still round-trip (NP not used).
    original = (
        np.random.default_rng(1).standard_normal(2048).astype(np.float32).tobytes()
    )
    comp = Compressor(CompressionConfig(bytearray_dtype="float32"))
    assert decompressor.decompress(comp.compress(original)) == original
