"""Verify the v1 file carries an Extension Table immediately after the header."""

import torch
from ptwm import CompressionConfig, Compressor, Format, Method


def test_extension_table_is_present_for_huffman_chain():
    tensor = torch.randn(128, 128, dtype=torch.float32)
    blob = Compressor(
        CompressionConfig(input_format=Format.TORCH, method=Method.HUFFMAN)
    ).compress(tensor)

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
