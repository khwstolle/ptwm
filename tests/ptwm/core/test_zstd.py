"""End-to-end tests for ``Method.ZSTD`` dispatch.

ZSTD is a forced-codec fallback: it bypasses the Huffman trial-encode
menu and writes each plane with the in-tree Zstd codec. The output is
still a ``.ptwm`` container. These tests exercise the full path
(Compressor + Decompressor) across the supported input formats and a
few dtypes.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ptwm import (
    CompressionConfig,
    Compressor,
    Decompressor,
    Format,
    Method,
)


@pytest.mark.parametrize(
    "bytearray_dtype",
    ["float32", "float16", "bfloat16", "int8"],
)
def test_zstd_byte_roundtrip(bytearray_dtype: str) -> None:
    np_dtype = {
        "float32": np.float32,
        "float16": np.float16,
        "bfloat16": np.float32,  # bf16 has no numpy dtype; use fp32 bytes
        "int8": np.int8,
    }[bytearray_dtype]
    rng = np.random.default_rng(0)
    if np_dtype == np.int8:
        raw = rng.integers(-127, 127, size=2048, dtype=np_dtype).tobytes()
    else:
        raw = rng.standard_normal(2048).astype(np_dtype).tobytes()
    config = CompressionConfig(method=Method.ZSTD, bytearray_dtype=bytearray_dtype)
    compressed = Compressor(config).compress(raw)
    recovered = Decompressor().decompress(compressed)
    assert recovered == raw


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float16, torch.bfloat16],
)
def test_zstd_torch_roundtrip(dtype: torch.dtype) -> None:
    tensor = torch.randn(64, 64, dtype=dtype)
    config = CompressionConfig(method=Method.ZSTD, input_format=Format.TORCH)
    compressed = Compressor(config).compress(tensor)
    recovered = Decompressor().decompress(compressed)
    assert recovered.shape == tensor.shape
    assert recovered.dtype == tensor.dtype
    if dtype == torch.bfloat16:
        assert torch.equal(recovered.view(torch.uint16), tensor.view(torch.uint16))
    else:
        assert torch.equal(recovered, tensor)


def test_zstd_numpy_roundtrip() -> None:
    arr = np.random.default_rng(0).standard_normal(1024).astype(np.float32)
    config = CompressionConfig(method=Method.ZSTD, input_format=Format.NUMPY)
    compressed = Compressor(config).compress(arr)
    recovered = Decompressor().decompress(compressed)
    assert isinstance(recovered, np.ndarray)
    assert np.array_equal(arr, recovered)


def test_zstd_outputs_ptwm() -> None:
    """ZSTD produces a .ptwm container with a forced Zstd plane codec."""
    arr = np.zeros(256, dtype=np.float32).tobytes()
    config = CompressionConfig(method=Method.ZSTD, bytearray_dtype="float32")
    compressed = Compressor(config).compress(arr)
    assert compressed[:5] == b"\x89PTWM"


def test_zstd_empty_input() -> None:
    config = CompressionConfig(method=Method.ZSTD, bytearray_dtype="float32")
    compressed = Compressor(config).compress(b"")
    assert Decompressor().decompress(compressed) == b""
