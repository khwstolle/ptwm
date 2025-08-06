"""Plane-level entropy coders.

Exposes the four entropy coders (IDENTITY, Huffman, rANS, ZSTD) behind the
uniform :class:`Codec` protocol, so callers can write
``get_codec(name).encode(plane.data)`` without caring which coder backs the
name.

Each codec operates on a single byte buffer — no preprocessing, no header
framing. Call :func:`ptwm.preprocessing.split` first to obtain planes, feed
each into a codec, and collect the blobs. The inverse runs the coder's
:meth:`Codec.decode` on each blob and passes the recovered bytes to
:func:`ptwm.preprocessing.combine`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol, runtime_checkable

from .. import _rust
from ..preprocessing import Plane


@runtime_checkable
class Codec(Protocol):
    """Plane-level entropy coder."""

    name: str

    def encode(self, data: bytes | memoryview) -> bytes:
        """Encode ``data`` into a codec-specific blob."""
        ...

    def decode(self, blob: bytes | memoryview, original_len: int) -> bytes:
        """Decode ``blob`` back to the original bytes.

        ``original_len`` is the uncompressed length (in bytes) of the data
        passed to :meth:`encode`. Some codecs self-describe the original
        length (e.g. ZSTD) and verify against it; others need it to pre-
        allocate the output (Huffman, rANS).
        """
        ...


@dataclass(frozen=True, slots=True)
class IdentityCodec:
    """Passthrough codec: ``encode`` and ``decode`` return their input."""

    name: str = "identity"

    def encode(self, data: bytes | memoryview) -> bytes:
        """Return ``data`` unchanged."""
        return _rust.identity_encode(data)

    def decode(self, blob: bytes | memoryview, original_len: int) -> bytes:
        """Return ``blob`` unchanged after a length check."""
        return _rust.identity_decode(blob, original_len)


@dataclass(frozen=True, slots=True)
class HuffmanCodec:
    """Canonical-Huffman entropy coder (4-stream interleaved)."""

    name: str = "huffman"

    def encode(self, data: bytes | memoryview) -> bytes:
        """Huffman-encode ``data``; falls back to tagged-raw if not beneficial."""
        return _rust.huffman_encode(data)

    def decode(self, blob: bytes | memoryview, original_len: int) -> bytes:
        """Decode a Huffman blob produced by :meth:`encode`."""
        return _rust.huffman_decode(blob, original_len)


@dataclass(frozen=True, slots=True)
class RansCodec:
    """Range-asymmetric-numeral-systems entropy coder."""

    name: str = "rans"

    def encode(self, data: bytes | memoryview) -> bytes:
        """rANS-encode ``data``; falls back to tagged-raw if not beneficial."""
        return _rust.rans_encode(data)

    def decode(self, blob: bytes | memoryview, original_len: int) -> bytes:
        """Decode an rANS blob produced by :meth:`encode`."""
        return _rust.rans_decode(blob, original_len)


@dataclass(frozen=True, slots=True)
class ZstdCodec:
    """ZSTD entropy coder. ``level`` is the standard 1-22 compression level."""

    level: int = 3

    @property
    def name(self) -> str:
        """Short codec name suffixed by level (e.g. ``zstd-3``)."""
        return f"zstd-{self.level}"

    def encode(self, data: bytes | memoryview) -> bytes:
        """ZSTD-encode ``data`` at ``self.level``."""
        return _rust.zstd_encode(data, self.level)

    def decode(self, blob: bytes | memoryview, original_len: int) -> bytes:
        """Decode a ZSTD blob and verify its length matches ``original_len``."""
        # ZSTD streams self-describe their original length; ``original_len``
        # is accepted for protocol uniformity and cross-checked below.
        out = _rust.zstd_decode(blob)
        if len(out) != original_len:
            msg = (
                f"zstd_decode: expected {original_len} bytes, "
                f"got {len(out)} (blob may be truncated)"
            )
            raise ValueError(msg)
        return out


_BUILTIN_CODECS: dict[str, Codec] = {
    "identity": IdentityCodec(),
    "huffman": HuffmanCodec(),
    "rans": RansCodec(),
    "zstd": ZstdCodec(),
}


def get_codec(name: str) -> Codec:
    """Look up a codec by short name.

    Known names: ``"identity"``, ``"huffman"``, ``"rans"``, ``"zstd"``.
    ``"zstd"`` returns the default level-3 variant; construct
    :class:`ZstdCodec` directly for other levels.
    """
    try:
        return _BUILTIN_CODECS[name]
    except KeyError as e:
        options = ", ".join(sorted(_BUILTIN_CODECS))
        msg = f"Unknown codec {name!r}; known codecs: {options}"
        raise KeyError(msg) from e


def encode_planes(
    planes: Sequence[Plane | bytes | memoryview],
    codec: Codec | str,
) -> list[bytes]:
    """Encode each plane with ``codec``. Returns one blob per plane."""
    resolved: Codec = codec if not isinstance(codec, str) else get_codec(codec)
    return [resolved.encode(p.data if isinstance(p, Plane) else p) for p in planes]


def decode_planes(
    blobs: Sequence[bytes | memoryview],
    original_lens: Sequence[int],
    codec: Codec | str,
) -> list[bytes]:
    """Decode each blob with ``codec``, using the matching ``original_lens[i]``."""
    if len(blobs) != len(original_lens):
        msg = (
            f"blobs / original_lens length mismatch: "
            f"{len(blobs)} vs {len(original_lens)}"
        )
        raise ValueError(msg)
    resolved: Codec = codec if not isinstance(codec, str) else get_codec(codec)
    return [resolved.decode(b, n) for b, n in zip(blobs, original_lens, strict=True)]


class CodecId(IntEnum):
    """Numeric codec identifiers — wire identifiers, never reassign.

    These values are baked into every ``.ptwm`` container's plane and prelude
    records; renumbering them breaks the wire format. Mirrors
    ``crates/ptwm-core/src/codec.rs::CodecId``.
    """

    Identity = 0x0000
    Huffman = 0x0001
    Rans = 0x0003
    Zstd = 0x0004
    ZstdDict = 0x0005
    Fpc = 0x0006
    Tans = 0x0007
    PerGroupCodebook = 0x0010
    Order1ScaleAC = 0x0011
    ArithmeticO0 = 0x0012
    ArithmeticO0Adaptive = 0x0013
    ArithmeticO1 = 0x0014
    ContextMixingLite = 0x0015
    HuffLlm5Bit = 0x0016
    Order1Arithmetic = 0x0017
    NeuralPredictor = 0x0018


__all__ = [
    "Codec",
    "CodecId",
    "HuffmanCodec",
    "IdentityCodec",
    "RansCodec",
    "ZstdCodec",
    "decode_planes",
    "encode_planes",
    "get_codec",
]
