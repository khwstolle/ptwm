"""5-cell ablation smoke test on a tiny MXFP4 fixture.

All 5 cells must round-trip bit-exactly, and microscale_full must not inflate
vs the basic Huffman baseline.
"""

from __future__ import annotations

import torch
from ptwm import Compressor, Decompressor, Format, Method
from ptwm._config import CompressionConfig
from ptwm.codecs import CodecId

# Each cell: (label, Method, codec_menu or None).
# None → use the full role-applicable default menu for that method.
CELLS: list[tuple[str, Method, list[CodecId] | None]] = [
    ("huffman_baseline", Method.HUFFMAN, None),
    ("rans_baseline", Method.RANS, None),
    (
        "microscale_value_only",
        Method.MICROSCALE,
        [CodecId.Identity, CodecId.Huffman, CodecId.Rans, CodecId.PerGroupCodebook],
    ),
    (
        "microscale_scale_only",
        Method.MICROSCALE,
        [CodecId.Identity, CodecId.Huffman, CodecId.Rans, CodecId.Order1ScaleAC],
    ),
    ("microscale_full", Method.MICROSCALE, None),
]


def synth_mxfp4_bytes(n_blocks: int, seed: int = 0) -> bytes:
    """MXFP4: 16 packed-nibble bytes + 1 scale byte per block.

    The scale performs a bounded random walk near 0x88, creating the
    inter-block correlation that Order1ScaleAC is designed to exploit.
    """
    g = torch.Generator().manual_seed(seed)
    out = bytearray()
    scale = 0x88
    for _ in range(n_blocks):
        nibbles = torch.randint(0, 256, (16,), generator=g, dtype=torch.uint8)
        out.extend(nibbles.tolist())
        delta = torch.randint(-2, 3, (1,), generator=g).item()
        scale = (scale + delta) & 0xFF
        out.append(scale)
    return bytes(out)


def test_all_cells_succeed_and_microscale_full_wins():
    """All 5 cells succeed; microscale_full ≤ huffman_baseline."""
    raw = synth_mxfp4_bytes(2048, seed=0xABCD)
    sizes: dict[str, int] = {}

    for name, method, menu in CELLS:
        cfg = CompressionConfig(
            method=method,
            input_format=Format.BYTE,
            codec_menu=menu,
        )
        blob = Compressor(cfg).compress(raw)
        restored = Decompressor().decompress(blob)
        assert restored == raw, f"{name} round-trip failed"
        sizes[name] = len(blob)

    assert sizes["microscale_full"] <= sizes["huffman_baseline"], (
        f"microscale_full ({sizes['microscale_full']}) should be ≤ "
        f"huffman_baseline ({sizes['huffman_baseline']}); full results: {sizes}"
    )
