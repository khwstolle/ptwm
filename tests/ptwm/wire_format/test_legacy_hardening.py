"""Regression tests for reader hardening on legacy ``\\x89PTWM`` containers."""

from __future__ import annotations

import struct

import pytest
import torch
from ptwm import CompressionConfig, Compressor, Format, Framing, Method, _rust

HEADER_LEN = 91
# Header field offsets (see crates/ptwm-core/src/header.rs).
ET_OFF, ET_LEN, ET_HASH = 11, 19, 27
INDEX_OFF, PRELUDE_OFF = 59, 75


def _legacy_blob() -> bytes:
    tensor = torch.arange(64 * 32, dtype=torch.float32).reshape(64, 32)
    return Compressor(
        CompressionConfig(
            input_format=Format.TORCH,
            method=Method.HUFFMAN,
            framing=Framing.CONTAINER,
        )
    ).compress(tensor)


def _u64(blob: bytes | bytearray, off: int) -> int:
    return struct.unpack_from("<Q", blob, off)[0]


def _foreign_entry() -> bytes:
    """One Extension Table entry for a third-party extension (wasm flavor)
    that is not installed on this host."""
    label = b"io.example.missing"
    return b"".join(
        [
            bytes(range(1, 33)),  # canonical_id
            struct.pack("<H", len(label)),
            label,
            struct.pack("<HH", 0x0001, 1),  # kind=Transform, abi_version
            bytes([0, 0b001]),  # lifecycle=None, flavor_hints=WASM
            struct.pack("<I", 1),
            b"\xa0",  # capabilities: empty CBOR map
            bytes([1]),  # attestation kind: PGP
            struct.pack("<H", 0),  # attestation length
            struct.pack("<H", 0),  # install_hint length
            struct.pack("<QQ", 0, 0),  # embedded wasm offset, length
        ]
    )


def _with_missing_extension(blob: bytes) -> bytes:
    """Append a missing third-party entry to the Extension Table and fix up
    every offset and the table hash. Existing table indices stay valid."""
    blake3 = pytest.importorskip("blake3")
    et_off, et_len = _u64(blob, ET_OFF), _u64(blob, ET_LEN)
    assert et_off == HEADER_LEN
    table = bytearray(blob[et_off : et_off + et_len])
    count = struct.unpack_from("<I", table, 0)[0]
    struct.pack_into("<I", table, 0, count + 1)
    table += _foreign_entry()
    shift = len(table) - et_len

    out = bytearray(blob[:HEADER_LEN]) + table + blob[et_off + et_len :]
    struct.pack_into("<Q", out, ET_LEN, len(table))
    out[ET_HASH : ET_HASH + 32] = blake3.blake3(bytes(table)).digest()
    index_off = _u64(blob, INDEX_OFF) + shift
    struct.pack_into("<Q", out, INDEX_OFF, index_off)
    struct.pack_into("<Q", out, PRELUDE_OFF, _u64(blob, PRELUDE_OFF) + shift)
    # Index: count u64, then (name_hash, tensor_offset, tensor_len) u64 triples.
    n = _u64(out, index_off)
    for i in range(n):
        pos = index_off + 8 + 24 * i + 8
        struct.pack_into("<Q", out, pos, _u64(out, pos) + shift)
    return bytes(out)


def test_shape_and_plane_summaries_honour_skip_missing() -> None:
    blob = _legacy_blob()
    (name,) = _rust.list_tensor_names(blob)
    shape = _rust.decode_tensor_shape(blob, name)
    summaries = _rust.list_plane_summaries(blob)
    assert shape is not None
    assert list(shape[0]) == [64, 32]

    patched = _with_missing_extension(blob)
    # Without skip_missing the reader refuses the file up front.
    with pytest.raises(Exception, match="(?i)missing|flavor"):
        _rust.decode_tensor_shape(patched, name)
    with pytest.raises(Exception, match="(?i)missing|flavor"):
        _rust.list_plane_summaries(patched)
    # With skip_missing the metadata is still readable.
    assert _rust.decode_tensor_shape(patched, name, skip_missing=True) == shape
    assert _rust.list_plane_summaries(patched, skip_missing=True) == summaries
    # The other skip_missing entry points agree.
    assert _rust.list_tensor_names(patched, skip_missing=True) == [name]
    assert _rust.decode_tensor(patched, name, skip_missing=True) == _rust.decode_tensor(
        blob, name
    )


def test_inspect_rejects_extension_table_offset_overflow(tmp_path) -> None:
    from ptwm._core import list_extension_table_entries

    blob = bytearray(_legacy_blob())
    path = tmp_path / "ok.ptwm"
    path.write_bytes(bytes(blob))
    assert list_extension_table_entries(str(path))

    # offset + length wraps around u64: must be a clean ValueError.
    struct.pack_into("<Q", blob, ET_OFF, 2**64 - 8)
    struct.pack_into("<Q", blob, ET_LEN, 16)
    path = tmp_path / "wrap.ptwm"
    path.write_bytes(bytes(blob))
    with pytest.raises(ValueError, match="extension table"):
        list_extension_table_entries(str(path))


def _pad_first_capability(blob: bytes) -> bytes:
    """Append one byte to the first Extension Table entry's capability CBOR
    and grow its length field to match."""
    label_len = struct.unpack_from("<H", blob, 4 + 32)[0]
    # count(u32) + canonical_id + label_len(u16) + label + kind(u16)
    # + abi_version(u16) + lifecycle(u8) + flavor_hints(u8)
    cap_len_pos = 4 + 32 + 2 + label_len + 2 + 2 + 1 + 1
    cap_len = struct.unpack_from("<I", blob, cap_len_pos)[0]
    cap_end = cap_len_pos + 4 + cap_len
    return (
        blob[:cap_len_pos]
        + struct.pack("<I", cap_len + 1)
        + blob[cap_len_pos + 4 : cap_end]
        + b"\x00"
        + blob[cap_end:]
    )


def test_chain_blob_info_reports_parsed_table_size() -> None:
    """chain_blob_info returns the bytes the Extension Table parse consumed,
    not the length of a re-serialization. Padding a capability blob by one
    byte must move the reported chain start by exactly one byte."""
    from ptwm._core import chain_blob_from_legacy, chain_blob_info

    nodes = struct.pack("<HB", 0, 0)  # one node: op 0, no params
    legacy = bytes([1, 0, 0]) + struct.pack("<HHH", len(nodes), 0, 0) + nodes
    blob = bytes(chain_blob_from_legacy(legacy))
    et_size, table = chain_blob_info(blob)

    padded = _pad_first_capability(blob)
    padded_size, padded_table = chain_blob_info(padded)
    assert padded_size == et_size + 1
    assert padded_table == table
    assert padded[padded_size:] == blob[et_size:]
