"""Verify the v1 file carries an Extension Table immediately after the header.

These tests read the fixed-width legacy header, so they pin
``ContainerLayout.LEGACY``; the default layout is compact (``\x89PTWC``).
"""

import torch
from ptwm import (
    CompressionConfig,
    Compressor,
    ContainerLayout,
    Format,
    Framing,
    Method,
)


def test_extension_table_is_present_for_huffman_chain():
    tensor = torch.randn(128, 128, dtype=torch.float32)
    blob = Compressor(
        CompressionConfig(
            input_format=Format.TORCH,
            method=Method.HUFFMAN,
            framing=Framing.CONTAINER,
            container_layout=ContainerLayout.LEGACY,
        )
    ).compress(tensor)
    assert blob[:5] == b"\x89PTWM"

    # Header layout (little-endian):
    #   magic:                  [u8; 9]    bytes  0.. 9
    #   flags:                  u16        bytes  9..11
    #   extension_table_offset: u64        bytes 11..19
    #   extension_table_length: u64        bytes 19..27
    #   extension_table_hash:   [u8; 32]   bytes 27..59
    #   tensor_index_offset:    u64        bytes 59..67
    #   tensor_index_length:    u64        bytes 67..75
    #   shared_prelude_offset:  u64        bytes 75..83
    #   shared_prelude_length:  u64        bytes 83..91

    table_off = int.from_bytes(blob[11:19], "little")
    table_len = int.from_bytes(blob[19:27], "little")
    assert table_off == 91, (
        f"extension table should sit right after the 91-byte header, got offset {table_off}"
    )
    assert table_len > 0, "expected non-empty extension table for a Huffman chain"

    count = int.from_bytes(blob[table_off : table_off + 4], "little")
    assert count >= 1, (
        "expected at least one Extension Table entry referenced by a Huffman chain"
    )


V1_1_PLANE_CODECS = (
    "identity",
    "huffman",
    "rans",
    "zstd",
    "zstd_dict",
    "fpc",
    "tans",
    "order1_scale_ac",
    "arithmetic_o0",
    "arithmetic_o0_adaptive",
    "arithmetic_o1",
    "per_group_codebook",
    "context_mixing_lite",
    "huff_llm_5bit",
    "order1_arithmetic",
    "neural_predictor",
)


def test_legacy_extension_table_holds_every_builtin_codec():
    """The legacy writer interns the chain ops plus all 16 built-in plane
    codecs, as v1.1.0 did, about 2 KB. The v1.1 transcoder rebuilds that
    table for exploded stores and keeps each plane's codec index, so a
    table with only the referenced codecs decoded there with the wrong
    codecs. The compact layout keeps only the referenced codecs."""
    tensor = torch.full((32,), 0.5, dtype=torch.bfloat16)

    def compress(layout: ContainerLayout) -> bytes:
        return Compressor(
            CompressionConfig(
                input_format=Format.TORCH,
                framing=Framing.CONTAINER,
                container_layout=layout,
            )
        ).compress(tensor)

    blob = compress(ContainerLayout.LEGACY)
    assert blob[:5] == b"\x89PTWM"
    table_off = int.from_bytes(blob[11:19], "little")
    table_len = int.from_bytes(blob[19:27], "little")
    table = blob[table_off : table_off + table_len]
    count = int.from_bytes(table[:4], "little")
    assert count > len(V1_1_PLANE_CODECS), f"extension table has {count} entries"
    positions = [
        table.index(f"io.ptwm.builtin.{name}@1.1.0".encode())
        for name in V1_1_PLANE_CODECS
    ]
    assert positions == sorted(positions), "codecs out of built-in order"

    compact = compress(ContainerLayout.COMPACT)
    assert compact[:5] == b"\x89PTWC"
    assert len(compact) < 1024, f"single-tensor container is {len(compact)} bytes"
