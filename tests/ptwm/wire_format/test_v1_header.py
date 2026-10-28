"""Round-trip the v1 wire format from Python."""

import pytest
import torch
from ptwm import (
    CompressionConfig,
    Compressor,
    ContainerLayout,
    Decompressor,
    Format,
    Framing,
)


def test_compress_decompress_round_trip_v1_format():
    """The legacy layout, pinned explicitly, keeps the v1 magic."""
    tensor = torch.randn(64, 64, dtype=torch.bfloat16)
    compressor = Compressor(
        CompressionConfig(
            input_format=Format.TORCH,
            framing=Framing.CONTAINER,
            container_layout=ContainerLayout.LEGACY,
        )
    )
    decompressor = Decompressor()

    blob = compressor.compress(tensor)
    restored = decompressor.decompress(blob)

    expected_magic = b"\x89PTWM\r\n\x1a\n"
    assert blob[: len(expected_magic)] == expected_magic, "missing PTWM magic"
    assert torch.equal(tensor, restored)


@pytest.mark.parametrize("layout", [None, ContainerLayout.COMPACT, "compact"])
def test_container_framing_defaults_to_compact_layout(layout):
    """``Framing.CONTAINER`` with the default (or an explicit compact) layout
    writes a ``\x89PTWC`` container."""
    tensor = torch.randn(64, 64, dtype=torch.bfloat16)
    kwargs = {} if layout is None else {"container_layout": layout}
    blob = Compressor(
        CompressionConfig(
            input_format=Format.TORCH, framing=Framing.CONTAINER, **kwargs
        )
    ).compress(tensor)

    assert blob[:9] == b"\x89PTWC\r\n\x1a\n"
    assert torch.equal(tensor, Decompressor().decompress(blob))
