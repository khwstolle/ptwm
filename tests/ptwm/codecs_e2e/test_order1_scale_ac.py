"""End-to-end check: MICROSCALE compresses a correlated-scale MXFP4
tensor and beats HUFFMAN."""

from __future__ import annotations

import torch
from ptwm import Compressor, Decompressor, Format, Method
from ptwm._config import CompressionConfig


def synth_mxfp4_bytes(n_blocks: int, seed: int = 0) -> bytes:
    """Block-correlated bytes: scale values drift smoothly along blocks.

    Layout: 16 packed-nibble bytes (values) + 1 scale byte per block.
    The scale is a slow random walk near 0x88, producing high inter-block
    correlation that Order1ScaleAC is designed to exploit.
    """
    g = torch.Generator().manual_seed(seed)
    out = bytearray()
    scale = torch.randint(0x80, 0x90, (1,), generator=g).item()
    for _ in range(n_blocks):
        nibbles = torch.randint(0, 256, (16,), generator=g, dtype=torch.uint8)
        out.extend(nibbles.tolist())
        delta = torch.randint(-2, 3, (1,), generator=g).item()
        scale = (scale + delta) & 0xFF
        out.append(scale)
    return bytes(out)


def test_microscale_roundtrip_byte_equal():
    """MICROSCALE must reconstruct input bit-exactly."""
    raw = synth_mxfp4_bytes(2048, seed=0xCAFE)
    cfg = CompressionConfig(method=Method.MICROSCALE, input_format=Format.BYTE)
    blob = Compressor(cfg).compress(raw)
    restored = Decompressor().decompress(blob)
    assert restored == raw


def test_microscale_smaller_than_huffman_on_correlated_scales():
    """MICROSCALE must not inflate vs HUFFMAN on correlated scale bytes."""
    raw = synth_mxfp4_bytes(2048, seed=0xBEEF)
    cfg_huff = CompressionConfig(method=Method.HUFFMAN, input_format=Format.BYTE)
    cfg_micro = CompressionConfig(method=Method.MICROSCALE, input_format=Format.BYTE)
    blob_huff = Compressor(cfg_huff).compress(raw)
    blob_micro = Compressor(cfg_micro).compress(raw)
    assert len(blob_micro) <= len(blob_huff), (
        f"MICROSCALE {len(blob_micro)} should beat or tie HUFFMAN {len(blob_huff)}"
    )
