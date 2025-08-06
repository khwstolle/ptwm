"""Tests for the V2 Compressor and Decompressor API."""

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
from ptwm._exceptions import HeaderParseError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_bytes(n_floats: int = 1024, dtype: str = "bfloat16") -> bytes:
    """Return raw bytes for n_floats random floats of the given dtype."""
    np_dtype = {
        "bfloat16": np.float32,  # bfloat16 not in numpy; use float32 bytes
        "float32": np.float32,
        "float16": np.float16,
    }.get(dtype, np.float32)
    return np.random.rand(n_floats).astype(np_dtype).tobytes()


def roundtrip_bytes(config: CompressionConfig, data: bytes) -> bytes:
    compressed = Compressor(config).compress(data)
    return Decompressor().decompress(compressed)


def roundtrip_tensor(config: CompressionConfig, tensor: torch.Tensor) -> torch.Tensor:
    compressed = Compressor(config).compress(tensor)
    return Decompressor().decompress(compressed)


# ---------------------------------------------------------------------------
# Byte-format roundtrips
# ---------------------------------------------------------------------------


def test_compress_decompress_bfloat16_bytes() -> None:
    data = make_bytes(2048, "bfloat16")
    result = roundtrip_bytes(CompressionConfig(bytearray_dtype="bfloat16"), data)
    assert data == result


def test_compress_decompress_float32_bytes() -> None:
    data = make_bytes(2048, "float32")
    result = roundtrip_bytes(CompressionConfig(bytearray_dtype="float32"), data)
    assert np.array_equal(
        np.frombuffer(data, dtype=np.float32),
        np.frombuffer(result, dtype=np.float32),
    )


def test_compress_decompress_float16_bytes() -> None:
    data = make_bytes(2048, "float16")
    result = roundtrip_bytes(CompressionConfig(bytearray_dtype="float16"), data)
    assert data == result


# ---------------------------------------------------------------------------
# Torch-format roundtrips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_compress_decompress_torch_float(dtype: torch.dtype) -> None:
    tensor = torch.randn(512, 512, dtype=dtype)
    config = CompressionConfig(input_format=Format.TORCH)
    result = roundtrip_tensor(config, tensor)
    assert result.shape == tensor.shape
    assert result.dtype == tensor.dtype
    assert torch.allclose(tensor, result)


@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_compress_decompress_float8_torch(fp8_dtype: torch.dtype) -> None:
    tensor = torch.randn(256, 256).to(fp8_dtype)
    config = CompressionConfig(input_format=Format.TORCH)
    compressed = Compressor(config).compress(tensor)
    result = Decompressor().decompress(compressed)
    assert result.shape == tensor.shape
    assert result.dtype == fp8_dtype
    assert torch.equal(tensor.view(torch.uint8), result.view(torch.uint8))


def test_fp8_compression_ratio() -> None:
    """FP8 nibble split should achieve meaningful compression on weight-like data."""
    # Simulate typical trained weights: clustered exponents, varied mantissas.
    # torch.randn → cast to FP8 produces a realistic exponent distribution.
    tensor = torch.randn(1024, 1024).to(torch.float8_e4m3fn)
    config = CompressionConfig(input_format=Format.TORCH)
    compressed = Compressor(config).compress(tensor)
    ratio = len(compressed) / tensor.nbytes
    # The nibble split + bit reorder should give meaningful compression.
    # We use a conservative threshold (< 0.99) to verify compression actually
    # occurs; real weights with clustered exponents achieve < 0.7.
    assert ratio < 0.99, f"FP8 compression ratio {ratio:.3f} suggests no compression"


@pytest.mark.skipif(
    not hasattr(torch, "float4_e2m1fn_x2"),
    reason="torch.float4_e2m1fn_x2 requires PyTorch >= 2.6",
)
def test_compress_decompress_float4_torch() -> None:
    float4_dtype = torch.float4_e2m1fn_x2
    # Create a packed FP4 tensor by viewing random uint8 data.
    raw = torch.randint(0, 256, (128, 128), dtype=torch.uint8).view(float4_dtype)
    config = CompressionConfig(input_format=Format.TORCH)
    compressed = Compressor(config).compress(raw)
    result = Decompressor().decompress(compressed)
    assert result.shape == raw.shape
    assert result.dtype == float4_dtype
    assert torch.equal(raw.view(torch.uint8), result.view(torch.uint8))


# ---------------------------------------------------------------------------
# Integer/bool Torch-format roundtrips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype",
    [torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64, torch.bool],
)
def test_compress_decompress_torch_integer(dtype: torch.dtype) -> None:
    if dtype == torch.bool:
        tensor = torch.randint(0, 2, (256, 256), dtype=dtype)
    elif dtype in (torch.int8, torch.uint8):
        tensor = torch.randint(0, 127, (256, 256), dtype=dtype)
    else:
        tensor = torch.randint(0, 1000, (256, 256), dtype=dtype)
    config = CompressionConfig(input_format=Format.TORCH)
    result = roundtrip_tensor(config, tensor)
    assert result.shape == tensor.shape
    assert result.dtype == tensor.dtype
    assert torch.equal(tensor, result)


# ---------------------------------------------------------------------------
# NumPy-format roundtrip
# ---------------------------------------------------------------------------


def test_compress_decompress_numpy() -> None:
    arr = np.random.rand(1024).astype(np.float32)
    config = CompressionConfig(input_format=Format.NUMPY)
    compressed = Compressor(config).compress(arr)
    result = Decompressor().decompress(compressed)
    assert isinstance(result, np.ndarray)
    assert np.allclose(arr, result)


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_streaming_compress_decompress() -> None:
    data = np.random.rand(4096).astype(np.float32).tobytes()
    config = CompressionConfig(bytearray_dtype="float32", is_streaming=True)
    compressed = Compressor(config).compress(data)
    result = Decompressor().decompress(compressed)
    assert np.array_equal(
        np.frombuffer(data, dtype=np.float32),
        np.frombuffer(result, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Delta compression
# ---------------------------------------------------------------------------


def test_delta_compress_decompress_byte_mode() -> None:
    a = np.random.rand(512).astype(np.float32).tobytes()
    b = np.random.rand(512).astype(np.float32).tobytes()
    c = np.random.rand(512).astype(np.float32).tobytes()
    original = a + b
    reference = a + c

    config = CompressionConfig(bytearray_dtype="float32", delta_compressed_type="byte")
    compressed = Compressor(config).compress(original, delta_second_data=reference)
    result = Decompressor(DecompressionConfig(delta_second_data=reference)).decompress(
        compressed
    )
    assert original == result


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_invalid_delta_second_data_raises() -> None:
    config = CompressionConfig(delta_compressed_type="byte")
    with pytest.raises(ValueError, match="delta"):
        Compressor(config).compress(b"\x00" * 64)


def test_decompress_invalid_header_raises() -> None:
    with pytest.raises((HeaderParseError, Exception)):
        Decompressor().decompress(b"GARBAGE_DATA_THAT_IS_NOT_A_VALID_HEADER" * 4)


# ---------------------------------------------------------------------------
# Decompressor convenience API
# ---------------------------------------------------------------------------


def test_decompressor_no_config() -> None:
    data = make_bytes(512, "bfloat16")
    compressed = Compressor(CompressionConfig()).compress(data)
    result = Decompressor().decompress(compressed)
    assert data == result


def test_decompress_file(tmp_path) -> None:
    data = make_bytes(512, "bfloat16")
    compressed = Compressor(CompressionConfig()).compress(data)
    p = tmp_path / "test.ptwm"
    p.write_bytes(compressed)
    result = Decompressor().decompress_file(str(p))
    assert data == result


def test_decompress_file_not_found(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        Decompressor().decompress_file(str(tmp_path / "nonexistent.ptwm"))


# ---------------------------------------------------------------------------
# HUFFMAN method explicit
# ---------------------------------------------------------------------------


def test_huffman_method_torch() -> None:
    tensor = torch.randn(256, 256, dtype=torch.bfloat16)
    config = CompressionConfig(method=Method.HUFFMAN, input_format=Format.TORCH)
    compressed = Compressor(config).compress(tensor)
    result = Decompressor().decompress(compressed)
    assert torch.allclose(tensor, result)


# ---------------------------------------------------------------------------
# Large-ish tensor stress (stays under 10 MB to keep tests fast)
# ---------------------------------------------------------------------------


def test_large_bfloat16_tensor_roundtrip() -> None:
    tensor = torch.randn(1024, 1024, dtype=torch.bfloat16)  # 2 MB
    config = CompressionConfig(input_format=Format.TORCH)
    compressed = Compressor(config).compress(tensor)
    result = Decompressor().decompress(compressed)
    assert torch.allclose(tensor, result)


# ---------------------------------------------------------------------------
# Decompressor error paths for delta state mismatches
# ---------------------------------------------------------------------------


def test_decompress_delta_data_without_delta_second_data_raises() -> None:
    """Decompressing delta-compressed data without delta_second_data must raise."""
    a = np.random.rand(256).astype(np.float32).tobytes()
    b = np.random.rand(256).astype(np.float32).tobytes()
    original = a + b
    reference = np.random.rand(512).astype(np.float32).tobytes()

    config = CompressionConfig(bytearray_dtype="float32", delta_compressed_type="byte")
    compressed = Compressor(config).compress(original, delta_second_data=reference)

    with pytest.raises(ValueError, match="delta"):
        Decompressor().decompress(compressed)


def test_decompress_non_delta_data_with_delta_second_data_raises() -> None:
    """Providing delta_second_data for non-delta data must raise."""
    data = make_bytes(256, "float32")
    compressed = Compressor(CompressionConfig(bytearray_dtype="float32")).compress(data)

    with pytest.raises(ValueError, match="delta"):
        Decompressor(
            DecompressionConfig(delta_second_data=b"\x00" * len(data))
        ).decompress(compressed)


# ---------------------------------------------------------------------------
# NumPy float16 roundtrip (covers NUMPY float16 decompressor branch)
# ---------------------------------------------------------------------------


def test_compress_decompress_numpy_float16() -> None:
    arr = np.random.rand(1024).astype(np.float16)
    config = CompressionConfig(input_format=Format.NUMPY)
    compressed = Compressor(config).compress(arr)
    result = Decompressor().decompress(compressed)
    assert isinstance(result, np.ndarray)
    assert np.allclose(arr, result)


# ---------------------------------------------------------------------------
# Torch float32 roundtrip (covers float32 TORCH decompressor branch)
# ---------------------------------------------------------------------------


def test_compress_decompress_float32_torch() -> None:
    tensor = torch.randn(256, 256, dtype=torch.float32)
    config = CompressionConfig(input_format=Format.TORCH)
    compressed = Compressor(config).compress(tensor)
    result = Decompressor().decompress(compressed)
    assert result.shape == tensor.shape
    assert result.dtype == tensor.dtype
    assert torch.allclose(tensor, result)


# ---------------------------------------------------------------------------
# Tensor / ndarray inputs to BYTE-format Compressor (regression)
# ---------------------------------------------------------------------------


def test_compress_tensor_with_byte_format_does_not_raise() -> None:
    """Regression: ``compress(tensor)`` on a BYTE-format Compressor used to
    raise ``TypeError: only integer tensors of a single element can be
    converted to an index`` from the implicit ``bytes(tensor)`` call.
    """
    tensor = torch.randn(64, 64, dtype=torch.float32)
    config = CompressionConfig(bytearray_dtype="float32")  # default: BYTE
    compressed = Compressor(config).compress(tensor)
    result = Decompressor().decompress(compressed)
    np.testing.assert_array_equal(
        np.frombuffer(result, dtype=np.float32),
        tensor.contiguous().reshape(-1).numpy(),
    )


def test_compress_non_contiguous_tensor_byte_format() -> None:
    """A sliced (non-contiguous, storage_offset>0) tensor must serialise its
    own bytes, not the underlying storage's bytes."""
    base = torch.arange(1024, dtype=torch.float32)
    sliced = base[256:768]  # storage_offset=256, len=512
    assert sliced.storage_offset() == 256
    config = CompressionConfig(bytearray_dtype="float32")
    compressed = Compressor(config).compress(sliced)
    result = Decompressor().decompress(compressed)
    np.testing.assert_array_equal(
        np.frombuffer(result, dtype=np.float32),
        sliced.numpy(),
    )


def test_compress_strided_tensor_byte_format() -> None:
    """Non-contiguous (strided) tensor: should compress its logical bytes."""
    base = torch.arange(64 * 4, dtype=torch.float32).reshape(64, 4)
    strided = base[:, ::2]  # non-contiguous
    assert not strided.is_contiguous()
    config = CompressionConfig(bytearray_dtype="float32")
    compressed = Compressor(config).compress(strided)
    result = Decompressor().decompress(compressed)
    np.testing.assert_array_equal(
        np.frombuffer(result, dtype=np.float32),
        strided.contiguous().reshape(-1).numpy(),
    )


def test_compress_ndarray_with_byte_format() -> None:
    arr = np.random.rand(128).astype(np.float32)
    config = CompressionConfig(bytearray_dtype="float32")
    compressed = Compressor(config).compress(arr)
    result = Decompressor().decompress(compressed)
    np.testing.assert_array_equal(np.frombuffer(result, dtype=np.float32), arr)


def test_compress_non_contiguous_ndarray_byte_format() -> None:
    arr = np.random.rand(64, 4).astype(np.float32)[:, ::2]
    assert not arr.flags["C_CONTIGUOUS"]
    config = CompressionConfig(bytearray_dtype="float32")
    compressed = Compressor(config).compress(arr)
    result = Decompressor().decompress(compressed)
    np.testing.assert_array_equal(
        np.frombuffer(result, dtype=np.float32),
        np.ascontiguousarray(arr).reshape(-1),
    )


def test_compress_bytearray_with_byte_format() -> None:
    raw = make_bytes(256, "float32")
    compressed = Compressor(CompressionConfig(bytearray_dtype="float32")).compress(
        bytearray(raw)
    )
    assert Decompressor().decompress(compressed) == raw


def test_compress_memoryview_with_byte_format() -> None:
    """Locks in the buffer-protocol contract on the Python→Rust hop:
    ``compress_model`` accepts any C-contiguous PyBuffer source, not just
    ``bytes``.
    """
    raw = make_bytes(256, "float32")
    mv = memoryview(raw)
    compressed = Compressor(CompressionConfig(bytearray_dtype="float32")).compress(mv)
    assert Decompressor().decompress(compressed) == raw


def test_compress_unsupported_type_raises() -> None:
    config = CompressionConfig(bytearray_dtype="float32")
    with pytest.raises(TypeError, match="Unsupported input type"):
        Compressor(config).compress([1.0, 2.0, 3.0])  # type: ignore[arg-type]


def test_delta_compress_with_tensor_inputs() -> None:
    """Delta path used to call ``bytes(tensor)`` and crash; now it accepts
    tensors directly via the shared bytes-coercion helper.
    """
    a = torch.randn(128, dtype=torch.float32)
    b = torch.randn(128, dtype=torch.float32)
    config = CompressionConfig(bytearray_dtype="float32", delta_compressed_type="byte")
    compressed = Compressor(config).compress(a, delta_second_data=b)
    result = Decompressor(
        DecompressionConfig(delta_second_data=b.numpy().tobytes())
    ).decompress(compressed)
    np.testing.assert_array_equal(
        np.frombuffer(result, dtype=np.float32),
        a.numpy(),
    )


# ---------------------------------------------------------------------------
# Streaming delta decompression
# ---------------------------------------------------------------------------


def test_streaming_delta_decompress() -> None:
    a = np.random.rand(256).astype(np.float32).tobytes()
    b = np.random.rand(256).astype(np.float32).tobytes()
    original = a + b
    reference = np.random.rand(512).astype(np.float32).tobytes()

    config = CompressionConfig(
        bytearray_dtype="float32",
        delta_compressed_type="byte",
        is_streaming=True,
    )
    compressed = Compressor(config).compress(original, delta_second_data=reference)
    result = Decompressor(DecompressionConfig(delta_second_data=reference)).decompress(
        compressed
    )
    assert original == result
