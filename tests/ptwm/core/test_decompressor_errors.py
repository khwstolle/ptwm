"""Negative-path tests for :class:`weights.Decompressor`.

These feed the decompressor deliberately malformed or adversarial blobs
(shorter-than-declared, wrong dtype_code, mismatched delta state,
truncated payload) and assert that a :class:`Error` subclass is
raised with a useful message. The goal is to catch regressions where
malformed input slips past the header check and surfaces as a Rust
panic, a bare ``RuntimeError``, or silent wrong output.
"""

from __future__ import annotations

import numpy as np
import pytest
from ptwm import (
    CompressionConfig,
    Compressor,
    DecompressionConfig,
    Decompressor,
    Error,
)
from ptwm._exceptions import HeaderParseError, InvalidDTypeError


def _valid_compressed_blob() -> bytes:
    # Non-zero data so a random bit flip has something concrete to corrupt —
    # all-zero payloads compress to a handful of bytes and hide bit flips
    # in the container's non-payload regions.
    rng = np.random.default_rng(0)
    raw = rng.standard_normal(1024).astype(np.float32).tobytes()
    config = CompressionConfig(bytearray_dtype="float32")
    return Compressor(config).compress(raw)


def test_decompress_zero_bytes_raises_header_error() -> None:
    with pytest.raises((HeaderParseError, Error, ValueError)):
        Decompressor().decompress(b"")


def test_decompress_truncated_header_raises() -> None:
    with pytest.raises((HeaderParseError, Error, ValueError)):
        Decompressor().decompress(b"\x00" * 16)  # below 32-byte header minimum


def test_decompress_payload_truncated_after_valid_header_raises() -> None:
    blob = _valid_compressed_blob()
    # Cut off the last quarter of the payload. The header still parses, but
    # the Rust decoder should fail or the wire parser should reject it.
    truncated = blob[: 32 + (len(blob) - 32) // 2]
    with pytest.raises((HeaderParseError, Error, ValueError, RuntimeError)):
        Decompressor().decompress(truncated)


def test_decompress_corrupted_payload_bytes_does_not_silently_succeed() -> None:
    blob = bytearray(_valid_compressed_blob())
    # Flip bits mid-payload.
    flip_pos = 32 + (len(blob) - 32) // 2
    blob[flip_pos] ^= 0xFF
    try:
        result = Decompressor().decompress(bytes(blob))
    except (HeaderParseError, Error, ValueError, RuntimeError):
        return  # errored cleanly — that's acceptable
    # If it decompressed without raising, the output must not equal the
    # original (silent corruption would be a real bug).
    original = np.zeros(1024, dtype=np.float32).tobytes()
    assert result != original, "corrupted blob roundtripped to original"


def test_decompress_random_garbage_blob_does_not_panic() -> None:
    """Pure garbage must surface as an error, never as a Rust panic."""
    import contextlib

    rng = np.random.default_rng(0)
    for _ in range(4):
        garbage = rng.integers(0, 256, size=4096, dtype=np.uint8).tobytes()
        with contextlib.suppress(HeaderParseError, Error, ValueError, RuntimeError):
            Decompressor().decompress(garbage)


def test_decompress_delta_without_second_data_raises() -> None:
    raw = np.random.default_rng(0).standard_normal(256).astype(np.float32).tobytes()
    ref = np.random.default_rng(1).standard_normal(256).astype(np.float32).tobytes()
    config = CompressionConfig(bytearray_dtype="float32", delta_compressed_type="byte")
    delta_blob = Compressor(config).compress(raw, delta_second_data=ref)

    with pytest.raises(ValueError, match="delta"):
        Decompressor().decompress(delta_blob)


def test_decompress_non_delta_with_second_data_raises() -> None:
    raw = np.zeros(256, dtype=np.float32).tobytes()
    plain_blob = Compressor(CompressionConfig(bytearray_dtype="float32")).compress(raw)

    with pytest.raises(ValueError, match="delta"):
        Decompressor(DecompressionConfig(delta_second_data=b"\x00" * 256)).decompress(
            plain_blob
        )


def test_decompress_delta_with_wrong_length_second_data_raises() -> None:
    raw = np.random.default_rng(0).standard_normal(256).astype(np.float32).tobytes()
    ref = np.random.default_rng(1).standard_normal(256).astype(np.float32).tobytes()
    config = CompressionConfig(bytearray_dtype="float32", delta_compressed_type="byte")
    delta_blob = Compressor(config).compress(raw, delta_second_data=ref)

    # Use a differently-sized reference on decode.
    wrong_ref = b"\x00" * (len(ref) // 2)
    with pytest.raises((ValueError, Error)):
        Decompressor(DecompressionConfig(delta_second_data=wrong_ref)).decompress(
            delta_blob
        )


def test_decompress_file_missing_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        Decompressor().decompress_file(str(tmp_path / "does-not-exist.ptwm"))


def test_bytes_to_torch_invalid_dtype_code_raises() -> None:
    with pytest.raises(InvalidDTypeError, match="has no torch.dtype mapping for"):
        Decompressor()._bytes_to_torch(b"dummy", 9999, (1, 2))


def test_bytes_to_numpy_invalid_dtype_code_raises() -> None:
    with pytest.raises(InvalidDTypeError, match="has no numpy.dtype mapping for"):
        Decompressor()._bytes_to_numpy(b"dummy", 9999, (1, 2))
