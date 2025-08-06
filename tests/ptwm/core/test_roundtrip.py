"""Parametrized end-to-end roundtrip tests.

Exercises the full compress → decompress cycle across a range of sizes,
dtypes, formats, methods, and streaming/delta configurations.
"""

import numpy as np
import pytest
import torch
from ptwm import (
    CompressionConfig,
    Compressor,
    DecompressionConfig,
    Decompressor,
    Format,
    Method,
)

# ---------------------------------------------------------------------------
# Basic byte / torch roundtrips across many sizes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        ((0,), torch.bfloat16),
        ((0, 16), torch.float16),
        ((1,), torch.bfloat16),
        ((3, 7, 11), torch.bfloat16),
        ((1, 1, 1, 1), torch.float16),
    ],
)
def test_roundtrip_empty_and_odd_shapes(
    compressor_torch, decompressor, shape, dtype
) -> None:
    # Empty (0-dimension) tensors and odd small shapes must round-trip; empty
    # buffers used to break torch.frombuffer in the decoder.
    original = (torch.randn(shape) * 0.02).to(dtype)
    restored = decompressor.decompress(compressor_torch.compress(original))
    assert restored.shape == original.shape
    assert torch.equal(restored, original)


@pytest.mark.parametrize("size_kb", [255, 256, 257, 511, 512, 513, 1024])
def test_roundtrip_kb(
    compressor_torch,
    compressor_bytes,
    decompressor,
    make_tensor,
    make_random_bytes,
    size_kb,
) -> None:
    original_tensor = make_tensor(size_kb)
    clone = original_tensor.clone()
    compressed = compressor_torch.compress(original_tensor)
    assert torch.equal(clone, decompressor.decompress(compressed))

    original_bytes = make_random_bytes(size_kb)
    compressed = compressor_bytes.compress(original_bytes)
    assert bytearray(original_bytes) == decompressor.decompress(compressed)


@pytest.mark.parametrize("size_mb", [0.99, 1, 1.01, 1.99, 2, 2.1])
def test_roundtrip_mb(
    compressor_torch,
    compressor_bytes,
    decompressor,
    make_tensor,
    make_random_bytes,
    size_mb,
) -> None:
    size_kb = int(size_mb * 1024)
    original_tensor = make_tensor(size_kb)
    clone = original_tensor.clone()
    compressed = compressor_torch.compress(original_tensor)
    assert torch.equal(clone, decompressor.decompress(compressed))

    original_bytes = make_random_bytes(size_kb)
    compressed = compressor_bytes.compress(original_bytes)
    assert bytearray(original_bytes) == decompressor.decompress(compressed)


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "chunk_size", [int(2**19), int(2**20), int(2**21), int(2**23), int(2**24)]
)
def test_streaming_roundtrip(chunk_size, decompressor, make_random_bytes) -> None:
    original_bytes = make_random_bytes(10)
    compressor = Compressor(
        CompressionConfig(is_streaming=True, streaming_chunk=chunk_size)
    )
    compressed = compressor.compress(original_bytes)
    assert bytearray(original_bytes) == decompressor.decompress(compressed)


# ---------------------------------------------------------------------------
# Delta compression (byte-mode)
# ---------------------------------------------------------------------------


def test_delta_buffer_roundtrip(make_random_bytes) -> None:
    compressor = Compressor(CompressionConfig(delta_compressed_type="byte"))
    a, b, c = make_random_bytes(10), make_random_bytes(10), make_random_bytes(10)
    original = a + b
    reference = a + c
    compressed = compressor.compress(original, delta_second_data=reference)
    result = Decompressor(DecompressionConfig(delta_second_data=reference)).decompress(
        compressed
    )
    assert bytearray(original) == result


def test_streaming_delta_buffer_roundtrip(make_random_bytes) -> None:
    compressor = Compressor(
        CompressionConfig(delta_compressed_type="byte", is_streaming=True)
    )
    a, b, c = make_random_bytes(8), make_random_bytes(8), make_random_bytes(8)
    original = a + b
    reference = a + c
    compressed = compressor.compress(original, delta_second_data=reference)
    result = Decompressor(DecompressionConfig(delta_second_data=reference)).decompress(
        compressed
    )
    assert bytearray(original) == result


# ---------------------------------------------------------------------------
# float32 via raw bytes
# ---------------------------------------------------------------------------


def test_float32_bytes_roundtrip(decompressor) -> None:
    compressor = Compressor(CompressionConfig(bytearray_dtype="float32"))
    original = np.random.rand(8 * 1024).astype(np.float32).tobytes()
    result = decompressor.decompress(compressor.compress(original))
    assert np.array_equal(
        np.frombuffer(original, dtype=np.float32),
        np.frombuffer(result, dtype=np.float32),
    )


def test_streaming_float32_bytes_roundtrip(decompressor) -> None:
    compressor = Compressor(
        CompressionConfig(bytearray_dtype="float32", is_streaming=True)
    )
    original = np.random.rand(8 * 1024).astype(np.float32).tobytes()
    result = decompressor.decompress(compressor.compress(original))
    assert np.array_equal(
        np.frombuffer(original, dtype=np.float32),
        np.frombuffer(result, dtype=np.float32),
    )


def test_streaming_delta_float32_roundtrip() -> None:
    size = 8 * 1024
    compressor = Compressor(
        CompressionConfig(
            bytearray_dtype="float32", is_streaming=True, delta_compressed_type="byte"
        )
    )
    a = np.random.rand(size).astype(np.float32)
    b = np.random.rand(size).astype(np.float32)
    c = np.random.rand(size).astype(np.float32)
    original = np.concatenate([a, b]).tobytes()
    reference = np.concatenate([a, c]).tobytes()
    compressed = compressor.compress(original, delta_second_data=reference)
    result = Decompressor(DecompressionConfig(delta_second_data=reference)).decompress(
        compressed
    )
    assert np.array_equal(
        np.frombuffer(original, dtype=np.float32),
        np.frombuffer(result, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# All compression methods across float dtypes and input formats
# ---------------------------------------------------------------------------

_ALL_METHODS = [
    Method.HUFFMAN,
    Method.RANS,
    Method.IDENTITY,
    Method.ZSTD,
]


@pytest.mark.parametrize("method", _ALL_METHODS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("input_format", [Format.BYTE, Format.TORCH])
def test_method_float_roundtrip(
    method: Method, dtype: torch.dtype, input_format: Format
) -> None:
    dtype_name = {
        torch.float32: "float32",
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
    }[dtype]

    element_size = torch.tensor([], dtype=dtype).element_size()
    num_elements = 1024 * 1024 // element_size
    tensor = torch.rand(num_elements, dtype=dtype) * 2 - 1

    compressor = Compressor(
        CompressionConfig(
            method=method,
            input_format=input_format,
            bytearray_dtype=dtype_name,
            threads=1,
        )
    )
    decompressor = Decompressor(DecompressionConfig(threads=1))

    if input_format == Format.BYTE:
        np_array = (
            tensor.view(torch.uint16).numpy()
            if dtype == torch.bfloat16
            else tensor.numpy()
        )
        original_bin = np_array.tobytes()
        compressed = compressor.compress(original_bin)
        result = decompressor.decompress(compressed)
        assert bytes(original_bin) == bytes(result)
    else:
        clone = tensor.clone()
        compressed = compressor.compress(tensor.clone())
        result = decompressor.decompress(compressed)
        assert clone.shape == result.shape
        assert clone.dtype == result.dtype
        assert torch.equal(clone, result)


@pytest.mark.parametrize("method", _ALL_METHODS)
def test_method_numpy_roundtrip(method: Method) -> None:
    arr = np.random.rand(1024).astype(np.float32)
    compressor = Compressor(
        CompressionConfig(
            method=method,
            input_format=Format.NUMPY,
            threads=1,
        )
    )
    compressed = compressor.compress(arr)
    result = Decompressor().decompress(compressed)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(arr, result)
