"""Delta (reference-frame) compression primitives.

A delta-compressed tensor stores the residual between the tensor bytes and a
reference (e.g. the matching tensor from a base model checkpoint). For
fine-tuned models the residual concentrates on very few bits and compresses
far better than the raw tensor.

This module exposes the delta math only. To obtain a compressed blob,
compose with :mod:`weights.codecs`::

    residual = ptwm.delta.encode(finetune_bytes, reference=base_bytes)
    blob = weights.codecs.get_codec("huffman").encode(residual)

    # inverse:
    residual = weights.codecs.get_codec("huffman").decode(blob, original_len)
    recovered = ptwm.delta.decode(residual, reference=base_bytes)

Use :func:`reference_hash` to record which reference a residual was produced
against; :func:`decode` accepts an optional ``expected_reference_hash`` for
cross-checking at call time.
"""

from __future__ import annotations

from enum import StrEnum

from .. import _rust


class DeltaMode(StrEnum):
    """Supported delta schemes."""

    XOR = "xor"
    FLOAT = "float"  # reserved for the cross-checkpoint float-arithmetic mode


def encode(
    data: bytes | memoryview,
    *,
    reference: bytes | memoryview,
    mode: DeltaMode = DeltaMode.XOR,
) -> bytes:
    """Compute the delta residual between ``data`` and ``reference``.

    ``data`` and ``reference`` must have identical byte length. Returns a
    buffer of the same length; the inverse is :func:`decode`.
    """
    match mode:
        case DeltaMode.XOR:
            return _rust.delta_xor_encode(data, reference)
        case DeltaMode.FLOAT:
            return _rust.delta_float_sub_encode(data, reference)


def decode(
    residual: bytes | memoryview,
    *,
    reference: bytes | memoryview,
    mode: DeltaMode = DeltaMode.XOR,
    expected_reference_hash: bytes | None = None,
) -> bytes:
    """Reconstruct the original bytes from ``residual`` and ``reference``.

    If ``expected_reference_hash`` is given, the BLAKE3 digest of
    ``reference`` is checked against it before reconstruction, guarding
    against silently decoding with the wrong reference.
    """
    if expected_reference_hash is not None:
        actual = reference_hash(reference)
        if actual != expected_reference_hash:
            msg = (
                "reference hash mismatch: expected "
                f"{expected_reference_hash.hex()}, got {actual.hex()}"
            )
            raise ValueError(msg)
    match mode:
        case DeltaMode.XOR:
            return _rust.delta_xor_decode(residual, reference)
        case DeltaMode.FLOAT:
            return _rust.delta_float_sub_decode(residual, reference)


def reference_hash(reference: bytes | memoryview) -> bytes:
    """BLAKE3 digest (32 bytes) of ``reference``. Deterministic."""
    return _rust.blake3_hash(reference)


__all__ = ["DeltaMode", "decode", "encode", "reference_hash"]
