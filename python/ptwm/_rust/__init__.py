"""Thin wrapper around the ``weights._core`` PyO3 extension.

The ``weights._rust`` package is the ONLY place in the codebase that imports
``weights._core``. Every other module goes through the wrappers defined here
(and its submodules), so the underlying binding can swap (e.g. for a
subprocess backend) without touching the rest of the codebase.
"""

from __future__ import annotations

from ptwm import _core as _ext

from ._errors import translate_errors

__all__ = [
    "blake3_hash",
    "builtin_canonical_id",
    "codec_id_registry",
    "compress_model",
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
) -> bytes:
    """Compress a list of tensors into a ``.ptwm`` container.

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
    """
    return bytes(
        _ext.compress_model(
            records,
            chains_per_tensor,
            emit_payload_hash,
            emit_plane_crc,
            method_hint,
            streaming_chunk,
            codec_menu,
        )
    )


@translate_errors
def decode_tensor(blob: bytes, name: str, skip_missing: bool = False) -> bytes:
    """Decode a single tensor from a ``.ptwm`` container blob.

    Returns raw decompressed bytes.
    """
    return bytes(_ext.decode_tensor(blob, name, skip_missing))


@translate_errors
def decode_model(blob: bytes, skip_missing: bool = False) -> list[tuple[str, bytes]]:
    """Decode every tensor from a ``.ptwm`` container blob in parallel.

    Returns ``(name, raw_bytes)`` per tensor in container index order. Prefer
    this over per-tensor :func:`decode_tensor` for a full-model load — the
    decodes run across the rayon pool with the GIL released.
    """
    return _ext.decode_model(blob, skip_missing)


@translate_errors
def decode_tensor_shape(blob: bytes, name: str) -> tuple[list[int], str] | None:
    """Return ``(shape, dtype_name)`` for a tensor, or ``None``."""
    return _ext.decode_tensor_shape(blob, name)


@translate_errors
def decode_tensor_delta_hash(
    blob: bytes, name: str, skip_missing: bool = False
) -> bytes | None:
    """Return the 32-byte BLAKE3 hash of the delta reference for ``name``.

    Returns ``None`` if the tensor has no delta dependency.
    """
    got = _ext.decode_tensor_delta_hash(blob, name, skip_missing)
    return bytes(got) if got is not None else None


@translate_errors
def list_tensor_names(blob: bytes, skip_missing: bool = False) -> list[str]:
    """Return every tensor name stored in a ``.ptwm`` container."""
    return list(_ext.list_tensor_names(blob, skip_missing))


@translate_errors
def list_plane_summaries(
    blob: bytes,
) -> list[tuple[str, int, int, int, int, int]]:
    """Return per-plane summaries for every tensor in a container.

    Each entry: ``(tensor_name, plane_index, role, codec_id,
    state_source, payload_len)``.
    """
    return list(_ext.list_plane_summaries(blob))


@translate_errors
def decode_tensor_info(
    blob: bytes, name: str, skip_missing: bool = False
) -> tuple[list[int], str, int, int] | None:
    """Return ``(shape, dtype_name, dtype_code, input_format)`` or ``None``."""
    return _ext.decode_tensor_info(blob, name, skip_missing)


# --- Container ⇄ canonical PTWM-JSON manifest transcoder -------------------


@translate_errors
def explode_ptwm(
    blob: bytes,
) -> tuple[str, list[tuple[str, str, str]], dict[str, bytes]]:
    """Explode a ``.ptwm`` container into its canonical PTWM-JSON projection.

    Returns ``(registry_json, tensors, members)`` where ``registry_json`` is
    the canonical (JCS) archive-scoped registry manifest, ``tensors`` is a list
    of ``(name, key, tensor_manifest_json)`` in file order, and ``members`` maps
    member name → bytes. ``implode_ptwm`` reproduces ``blob`` byte-for-byte.

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
    """
    return _ext.implode_ptwm(registry_json, tensor_jsons, members)
