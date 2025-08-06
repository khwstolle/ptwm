"""Tests for :mod:`ptwm.delta`."""

import os
import struct

import pytest
from ptwm import Error
from ptwm.delta import DeltaMode, decode, encode, reference_hash


def test_xor_roundtrip_random():
    raw = os.urandom(4096)
    ref = os.urandom(4096)
    residual = encode(raw, reference=ref)
    assert decode(residual, reference=ref) == raw


def test_xor_identical_inputs_yield_zero_residual():
    data = b"\x00\xff\x11" * 256
    residual = encode(data, reference=data)
    assert residual == bytes(len(data))


def test_xor_length_mismatch_raises():
    with pytest.raises(Error, match="length mismatch"):
        encode(b"\x01\x02\x03", reference=b"\x01\x02")


def test_xor_empty_inputs_roundtrip():
    residual = encode(b"", reference=b"")
    assert residual == b""
    assert decode(b"", reference=b"") == b""


def test_reference_hash_is_32_bytes_and_deterministic():
    data = os.urandom(8192)
    h1 = reference_hash(data)
    h2 = reference_hash(data)
    assert len(h1) == 32
    assert h1 == h2


def test_reference_hash_distinguishes_inputs():
    assert reference_hash(b"abc") != reference_hash(b"xyz")


def test_decode_with_matching_expected_hash_ok():
    raw = os.urandom(1024)
    ref = os.urandom(1024)
    residual = encode(raw, reference=ref)
    recovered = decode(
        residual, reference=ref, expected_reference_hash=reference_hash(ref)
    )
    assert recovered == raw


def test_decode_with_mismatched_expected_hash_raises():
    raw = os.urandom(1024)
    ref = os.urandom(1024)
    residual = encode(raw, reference=ref)
    wrong_ref = os.urandom(1024)
    with pytest.raises(ValueError, match="reference hash mismatch"):
        decode(
            residual,
            reference=wrong_ref,
            expected_reference_hash=reference_hash(ref),
        )


def test_float_roundtrip_random():
    # Use exact integers (powers of 2 and integer multiples), which are
    # exactly representable as f32 and round-trip through f32 arithmetic
    # without precision loss. Random raw bytes are NOT used here because
    # NaN bit patterns are not preserved through IEEE 754 arithmetic —
    # this is documented in the Rust float_sub_encode/decode caveats.
    raw = struct.pack("<1024f", *[float(i) for i in range(1024)])
    ref = struct.pack("<1024f", *[float(i // 2) for i in range(1024)])
    residual = encode(raw, reference=ref, mode=DeltaMode.FLOAT)
    assert decode(residual, reference=ref, mode=DeltaMode.FLOAT) == raw


def test_float_identical_inputs_yield_zero_residual():
    # Pack 100 known f32 values
    values = [float(i) for i in range(100)]
    data = struct.pack(f"<{len(values)}f", *values)
    residual = encode(data, reference=data, mode=DeltaMode.FLOAT)
    # Every f32 in residual should be 0.0
    zeros = struct.pack(f"<{len(values)}f", *([0.0] * len(values)))
    assert residual == zeros


def test_float_length_not_multiple_of_4_raises():
    with pytest.raises(Error, match="divisible by 4"):
        encode(b"\x00" * 5, reference=b"\x00" * 5, mode=DeltaMode.FLOAT)


def test_float_length_mismatch_raises():
    with pytest.raises(Error, match="length mismatch"):
        encode(b"\x00" * 8, reference=b"\x00" * 4, mode=DeltaMode.FLOAT)


def test_float_composes_with_codec():
    """Float delta residual fed into an entropy coder."""
    from ptwm.codecs import get_codec

    # Two "similar" float buffers — small delta should compress well
    base = struct.pack("<1000f", *[float(i) for i in range(1000)])
    finetune = struct.pack("<1000f", *[float(i) + 0.001 for i in range(1000)])
    residual = encode(finetune, reference=base, mode=DeltaMode.FLOAT)
    codec = get_codec("huffman")
    blob = codec.encode(residual)
    recovered_residual = codec.decode(blob, len(residual))
    assert decode(recovered_residual, reference=base, mode=DeltaMode.FLOAT) == finetune


def test_delta_composes_with_codec():
    """End-to-end: delta residual fed into an entropy coder."""
    from ptwm.codecs import get_codec

    raw = (b"\x00" * 4000) + os.urandom(96)  # mostly zeros + a little noise
    ref = b"\x00" * len(raw)
    residual = encode(raw, reference=ref)
    codec = get_codec("huffman")
    blob = codec.encode(residual)
    recovered_residual = codec.decode(blob, len(residual))
    assert decode(recovered_residual, reference=ref) == raw
    # Delta + entropy coding should beat storing raw bytes with identity.
    assert len(blob) < len(raw)
