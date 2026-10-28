"""Thin wrapper around the ``weights._core`` PyO3 extension.

The ``weights._rust`` package is the ONLY place in the codebase that imports
``weights._core``. Every other module goes through the wrappers defined here
(and its submodules), so the underlying binding can swap (e.g. for a
subprocess backend) without touching the rest of the codebase.
"""

from __future__ import annotations

from ptwm import _core as _ext

from .._config import ContainerLayout, container_layout_name
from ._errors import translate_errors

Buffer = bytes | bytearray | memoryview
"""A byte buffer the container functions accept. ``bytes`` and a read-only
``mmap`` are borrowed without a copy; every other buffer is copied once."""

__all__ = [
    "ContainerHandle",
    "blake3_hash",
    "builtin_canonical_id",
    "codec_id_registry",
    "compress_compact",
    "compress_model",
    "container_layout",
    "decode_compact",
    "decode_model",
    "decode_tensor",
    "decode_tensor_delta_hash",
    "decode_tensor_info",
    "decode_tensor_shape",
    "delta_float_sub_decode",
    "delta_float_sub_encode",
    "delta_xor_decode",
    "delta_xor_encode",
    "dtype_element_size",
    "explode_ptwm",
    "huffman_decode",
    "huffman_encode",
    "identity_decode",
    "identity_encode",
    "implode_ptwm",
    "list_plane_summaries",
    "list_tensor_names",
    "rans_decode",
    "rans_encode",
    "shannon",
    "translate_errors",
    "xxhash64",
    "zstd_decode",
    "zstd_encode",
]


@translate_errors
def xxhash64(data: bytes | memoryview) -> int:
    """Return the 64-bit xxhash of ``data`` (seed 0)."""
    return _ext.xxhash64(data)


@translate_errors
def shannon(data: bytes | memoryview) -> float:
    """Return the Shannon entropy of ``data`` in bits/byte."""
    return _ext.shannon(data)


# --- Per-plane codec encode / decode ---------------------------------------


@translate_errors
def identity_encode(data: bytes | memoryview) -> bytes:
    return bytes(_ext.identity_encode(data))


@translate_errors
def identity_decode(blob: bytes | memoryview, expected_len: int) -> bytes:
    return bytes(_ext.identity_decode(blob, expected_len))


@translate_errors
def huffman_encode(data: bytes | memoryview) -> bytes:
    return bytes(_ext.huffman_encode(data))


@translate_errors
def huffman_decode(blob: bytes | memoryview, expected_len: int) -> bytes:
    return bytes(_ext.huffman_decode(blob, expected_len))


@translate_errors
def rans_encode(data: bytes | memoryview) -> bytes:
    return bytes(_ext.rans_encode(data))


@translate_errors
def rans_decode(blob: bytes | memoryview, expected_len: int) -> bytes:
    return bytes(_ext.rans_decode(blob, expected_len))


@translate_errors
def zstd_encode(data: bytes | memoryview, level: int = 3) -> bytes:
    return bytes(_ext.zstd_encode(data, level))


@translate_errors
def zstd_decode(blob: bytes | memoryview) -> bytes:
    return bytes(_ext.zstd_decode(blob))


# --- Delta primitives ------------------------------------------------------


@translate_errors
def delta_xor_encode(raw: bytes | memoryview, reference: bytes | memoryview) -> bytes:
    return bytes(_ext.delta_xor_encode(raw, reference))


@translate_errors
def delta_xor_decode(
    residual: bytes | memoryview, reference: bytes | memoryview
) -> bytes:
    return bytes(_ext.delta_xor_decode(residual, reference))


@translate_errors
def delta_float_sub_encode(
    raw: bytes | memoryview, reference: bytes | memoryview
) -> bytes:
    return bytes(_ext.delta_float_sub_encode(raw, reference))


@translate_errors
def delta_float_sub_decode(
    residual: bytes | memoryview, reference: bytes | memoryview
) -> bytes:
    return bytes(_ext.delta_float_sub_decode(residual, reference))


@translate_errors
def blake3_hash(data: bytes | memoryview) -> bytes:
    """BLAKE3 digest (32 bytes) of ``data``."""
    return bytes(_ext.blake3_hash(data))


def codec_id_registry() -> list[tuple[str, int]]:
    """Authoritative ``(name, wire_id)`` list from the Rust ``CodecId``.

    The regression test that locks the Python ``CodecId`` IntEnum to the
    Rust truth consumes this list.
    """
    return [(str(n), int(v)) for n, v in _ext.codec_id_registry()]


def builtin_canonical_id(name: str) -> bytes:
    r"""Return the 32-byte canonical id for the named built-in contribution.

    ``name`` is the short name used in `builtins.rs` (e.g. ``"huffman"``,
    ``"rans"``, ``"order1_scale_ac"``). The derivation matches
    ``CanonicalId::derive``:
    ``blake3(BUILTIN_PUBKEY || 0x00 || name || 0x00 || CARGO_PKG_VERSION)``
    — the zero separators are part of the derivation so authors
    reproducing IDs in other languages get the same bytes.
    """
    return bytes(_ext.builtin_canonical_id(name))


# --- Dtype lookups (Rust-authoritative) -----------------------------------


@translate_errors
def dtype_element_size(dtype_code: int) -> int:
    """Bytes per scalar element for ``dtype_code``."""
    return _ext.dtype_element_size(dtype_code)


# --- Container API (PPG chain-based) ------------------------------------


@translate_errors
def compress_model(
    records: list[
        tuple[
            str,
            int,
            int,
            bytes,
            list[int] | None,
            str | None,
            bytes | None,
        ]
    ],
    chains_per_tensor: list[list[bytes]],
    emit_payload_hash: bool,
    emit_plane_crc: bool,
    method_hint: int = 3,
    streaming_chunk: int | None = None,
    codec_menu: list[int] | None = None,
    *,
    container_layout: ContainerLayout | str = "compact",
) -> bytes:
    r"""Compress a list of tensors into a ``.ptwm`` container.

    Each element of ``records`` is a 7-tuple:
    ``(name, dtype_code, input_format, raw_bytes, shape,
       dtype_name, delta_reference_blake3)``.
    ``chains_per_tensor`` — one list of chain wire-byte blobs per tensor;
    each blob is produced by ``Chain.to_bytes()`` from ``ptwm.preprocessing``.
    Pass a single-element list when no multi-chain trial-encode is needed.
    ``method_hint``: 1=HUFFMAN, 2=ZSTD, 3=MICROSCALE, 4=RANS, 5=IDENTITY.
    ``streaming_chunk``: when set, split each plane into fixed-size chunks.
    ``codec_menu``: optional list of wire-stable ``CodecId`` integers
    restricting the trial-encode menu. ``None`` uses the full menu;
    empty intersection on a plane raises ``InvalidContainer``.
    ``container_layout``: ``"compact"`` (default) writes ``\x89PTWC``
    (readable by v1.2 and later), ``"legacy"`` writes ``\x89PTWM`` (readable
    by v1.1 and later).
    A :class:`ptwm.ContainerLayout` is accepted too. Other values raise
    :class:`ptwm.Error` before anything is compressed (the binding itself
    raises ``ValueError``).
    """
    layout = container_layout_name(container_layout)
    return bytes(
        _ext.compress_model(
            records,
            chains_per_tensor,
            emit_payload_hash,
            emit_plane_crc,
            method_hint,
            streaming_chunk,
            codec_menu,
            container_layout=layout,
        )
    )


@translate_errors
def compress_compact(
    dtype_code: int,
    input_format: int,
    raw_bytes: bytes,
    shape: list[int] | None,
    dtype_name: str | None,
    delta_reference_blake3: bytes | None,
    chains: list[bytes],
    emit_payload_hash: bool,
    emit_plane_crc: bool,
    method_hint: int = 3,
    codec_menu: list[int] | None = None,
) -> bytes | None:
    """Compress one tensor into a compact single-tensor frame.

    The arguments match one record of :func:`compress_model`; the tensor name
    is not stored. Returns ``None`` when the tensor cannot use the compact
    frame (a plane selects shared state); write a full container then.
    """
    got = _ext.compress_compact(
        dtype_code,
        input_format,
        raw_bytes,
        shape,
        dtype_name,
        delta_reference_blake3,
        chains,
        emit_payload_hash,
        emit_plane_crc,
        method_hint,
        codec_menu,
    )
    return bytes(got) if got is not None else None


@translate_errors
def decode_compact(
    blob: bytes,
) -> tuple[bytes, list[int] | None, str | None, int, int, bytes | None, int]:
    """Decode a compact single-tensor frame.

    Returns ``(raw_bytes, shape, dtype_name, dtype_code, input_format,
    delta_reference_blake3, payload_len)``. ``payload_len`` counts codec
    payload plus inline codec state (the raw size for a stored frame), so
    ``len(blob) - payload_len`` is the frame's fixed overhead.
    """
    raw, shape, dtype_name, dtype_code, input_format, delta, payload_len = (
        _ext.decode_compact(blob)
    )
    return (
        bytes(raw),
        shape,
        dtype_name,
        dtype_code,
        input_format,
        bytes(delta) if delta is not None else None,
        payload_len,
    )


@translate_errors
def decode_tensor(blob: Buffer, name: str, skip_missing: bool = False) -> bytes:
    """Decode a single tensor from a ``.ptwm`` container blob.

    Returns raw decompressed bytes.
    """
    return bytes(_ext.decode_tensor(blob, name, skip_missing))


@translate_errors
def decode_model(blob: Buffer, skip_missing: bool = False) -> list[tuple[str, bytes]]:
    """Decode every tensor from a ``.ptwm`` container blob in parallel.

    Returns ``(name, raw_bytes)`` per tensor in write order. Prefer
    this over per-tensor :func:`decode_tensor` for a full-model load — the
    decodes run across the rayon pool with the GIL released.
    """
    return _ext.decode_model(blob, skip_missing)


@translate_errors
def decode_tensor_shape(
    blob: Buffer, name: str, skip_missing: bool = False
) -> tuple[list[int], str] | None:
    """Return ``(shape, dtype_name)`` for a tensor, or ``None``."""
    return _ext.decode_tensor_shape(blob, name, skip_missing)


@translate_errors
def decode_tensor_delta_hash(
    blob: Buffer, name: str, skip_missing: bool = False
) -> bytes | None:
    """Return the 32-byte BLAKE3 hash of the delta reference for ``name``.

    Returns ``None`` if the tensor has no delta dependency.
    """
    got = _ext.decode_tensor_delta_hash(blob, name, skip_missing)
    return bytes(got) if got is not None else None


@translate_errors
def list_tensor_names(blob: Buffer, skip_missing: bool = False) -> list[str]:
    """Return every tensor name stored in a ``.ptwm`` container.

    Names come in write (body) order for both layouts.
    """
    return list(_ext.list_tensor_names(blob, skip_missing))


@translate_errors
def list_plane_summaries(
    blob: Buffer, skip_missing: bool = False
) -> list[tuple[str, int, int, int, int, int]]:
    """Return per-plane summaries for every tensor in a container.

    Each entry: ``(tensor_name, plane_index, role, codec_id,
    state_source, payload_len)``.
    """
    return list(_ext.list_plane_summaries(blob, skip_missing))


@translate_errors
def decode_tensor_info(
    blob: Buffer, name: str, skip_missing: bool = False
) -> tuple[list[int], str, int, int] | None:
    """Return ``(shape, dtype_name, dtype_code, input_format)`` or ``None``."""
    return _ext.decode_tensor_info(blob, name, skip_missing)


def container_layout(blob: Buffer) -> str | None:
    r"""Return the multi-tensor layout of ``blob`` from its magic.

    ``"legacy"`` for ``\x89PTWM`` / ``\x89PTWX``, ``"compact"`` for
    ``\x89PTWC``, ``None`` for anything else (including a ``\x89PTWS``
    single-tensor frame). Only the magic is read; the container is not
    validated.
    """
    return _ext.container_layout(blob)


class ContainerHandle:
    """One opened ``.ptwm`` container (either layout).

    Opening parses the header, Extension Table, prelude, chain registry and
    index once; the methods then query that state. Use a handle instead of
    the free functions (:func:`decode_tensor`, :func:`list_tensor_names`,
    ...) when a container is queried more than once: each free function
    opens the container again.

    ``blob`` is any buffer. ``bytes`` and an ``mmap`` opened with
    ``ACCESS_READ`` (directly or through a ``memoryview``) are borrowed
    without a copy; the handle keeps them alive. Every other buffer is
    copied once, including a read-only view of mutable memory such as
    ``memoryview(bytearray).toreadonly()``, because its owner could still
    change it. The file behind a borrowed ``mmap`` must not change while
    the handle exists.
    ``skip_missing`` opens the container even when non-builtin extensions it
    references are not installed; tensors that need them fail at decode.

    Errors from the binding surface as :class:`ptwm.Error`.
    """

    __slots__ = ("_h",)

    @translate_errors
    def __init__(self, blob: Buffer, skip_missing: bool = False) -> None:
        self._h = _ext.ContainerHandle(blob, skip_missing)

    @property
    def layout(self) -> str:
        """``"legacy"`` or ``"compact"``."""
        return self._h.layout

    @property
    def zero_copy(self) -> bool:
        """Whether the handle borrows ``blob`` (``False``: it holds a copy)."""
        return self._h.zero_copy

    @translate_errors
    def names(self) -> list[str]:
        """Every tensor name, in write (body) order."""
        return list(self._h.names())

    @translate_errors
    def decode(self, name: str) -> bytes:
        """Decode tensor ``name`` and return its raw bytes."""
        return self._h.decode(name)

    @translate_errors
    def decode_model(self) -> list[tuple[str, bytes]]:
        """Decode every tensor in parallel; ``(name, raw_bytes)`` pairs."""
        return self._h.decode_model()

    @translate_errors
    def info(self, name: str) -> tuple[list[int], str, int, int] | None:
        """``(shape, dtype_name, dtype_code, input_format)`` or ``None``."""
        return self._h.info(name)

    @translate_errors
    def delta_hash(self, name: str) -> bytes | None:
        """BLAKE3 hash of the delta reference of ``name``, or ``None``."""
        got = self._h.delta_hash(name)
        return bytes(got) if got is not None else None

    @translate_errors
    def shape(self, name: str) -> tuple[list[int], str] | None:
        """``(shape, dtype_name)`` of ``name``, or ``None``."""
        return self._h.shape(name)

    @translate_errors
    def plane_summaries(self) -> list[tuple[str, int, int, int, int, int]]:
        """One row per plane, tensors in write order.

        Each row is ``(tensor_name, plane_index, role, codec_id,
        state_source, payload_len)``.
        """
        return list(self._h.plane_summaries())

    def __len__(self) -> int:  # noqa: D105
        return len(self._h)

    def __contains__(self, name: object) -> bool:  # noqa: D105
        return isinstance(name, str) and name in self._h

    def __repr__(self) -> str:  # noqa: D105
        return f"ContainerHandle(layout={self.layout!r}, n={len(self)})"


# --- Container ⇄ canonical PTWM-JSON manifest transcoder -------------------


@translate_errors
def explode_ptwm(
    blob: bytes,
) -> tuple[str, list[tuple[str, str, str]], dict[str, bytes]]:
    r"""Explode a ``.ptwm`` container into its canonical PTWM-JSON projection.

    Returns ``(registry_json, tensors, members)`` where ``registry_json`` is
    the canonical (JCS) archive-scoped registry manifest, ``tensors`` is a list
    of ``(name, key, tensor_manifest_json)`` in file order, and ``members`` maps
    member name → bytes. ``implode_ptwm`` reproduces ``blob`` byte-for-byte.

    A legacy (``\x89PTWM``) container gives a ``"ptwm-manifest"`` version 1
    registry manifest; a compact (``\x89PTWC``) container gives a
    ``"ptwm-manifest-c"`` version 2 manifest that lists the Extension Table
    entries and records ``container_flags.layout == "compact"``. Tensors are
    in body order for both.

    The extension already returns ``str``/``bytes`` of the correct type, so the
    result is returned without re-wrapping (avoids copying large payloads).
    """
    return _ext.explode_ptwm(blob)


@translate_errors
def implode_ptwm(
    registry_json: str,
    tensor_jsons: list[str],
    members: dict[str, bytes],
) -> bytes:
    """Reconstruct a ``.ptwm`` container from a manifest projection.

    ``tensor_jsons`` must be in file order. Inverse of :func:`explode_ptwm`.
    The registry manifest selects the layout of the result. A subset of the
    tensors gives a container of those tensors with the archive-wide tables.
    """
    return _ext.implode_ptwm(registry_json, tensor_jsons, members)
