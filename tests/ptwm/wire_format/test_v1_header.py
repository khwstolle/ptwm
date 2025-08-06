"""Round-trip the v1 wire format from Python."""

import torch
from ptwm import CompressionConfig, Compressor, Decompressor, Format


def test_compress_decompress_round_trip_v1_format():
    tensor = torch.randn(64, 64, dtype=torch.bfloat16)
    compressor = Compressor(CompressionConfig(input_format=Format.TORCH))
    decompressor = Decompressor()

    blob = compressor.compress(tensor)
    restored = decompressor.decompress(blob)

    expected_magic = b"\x89PTWM\r\n\x1a\n"
    assert blob[: len(expected_magic)] == expected_magic, "missing PTWM magic"
    assert torch.equal(tensor, restored)
