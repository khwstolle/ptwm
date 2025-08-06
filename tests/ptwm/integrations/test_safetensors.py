"""End-to-end safetensors compression/decompression via the CLI entry points."""

import pytest
import torch
from safetensors.torch import safe_open, save_file


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn])
def test_safetensors_roundtrip(dtype: torch.dtype, tmp_path) -> None:
    from ptwm.cli.compress import compress_safetensors_file
    from ptwm.cli.decompress import decompress_safetensors_file

    size = (100, 100)
    half = size[0] // 2
    repetitive = torch.full((half, size[1]), 42.0)
    random = torch.randn((half, size[1]))
    tensor = torch.cat((repetitive, random), dim=0).to(dtype)

    tensor_file = str(tmp_path / f"temp_{dtype}.safetensors")
    metadata = {
        "tensor_dtype": str(dtype),
        "tensor_shape": str(list(tensor.shape)),
    }
    save_file({"tensor": tensor}, tensor_file, metadata=metadata)

    compressed_file = tensor_file.replace(".safetensors", ".ptwm.safetensors")
    compress_safetensors_file(tensor_file, force=True)
    decompress_safetensors_file(compressed_file, force=True)

    with safe_open(tensor_file, framework="pt", device="cpu") as f:
        decompressed = f.get_tensor("tensor")

    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        assert torch.equal(tensor.view(torch.uint8), decompressed.view(torch.uint8))
    else:
        assert torch.allclose(tensor, decompressed)


def test_decompress_safetensors_tensor_not_uint8_or_1d() -> None:
    from ptwm.integrations._safetensors import decompress_safetensors_tensor

    # 2D tensor should be returned as-is
    tensor_2d = torch.zeros((2, 2), dtype=torch.uint8)
    assert decompress_safetensors_tensor(tensor_2d) is tensor_2d

    # Non-uint8 tensor should be returned as-is
    tensor_float = torch.arange(10, dtype=torch.float32)
    assert decompress_safetensors_tensor(tensor_float) is tensor_float


def test_decompress_safetensors_tensor_success() -> None:
    from ptwm._config import CompressionConfig, Format
    from ptwm.core import Compressor
    from ptwm.integrations._safetensors import decompress_safetensors_tensor

    original_tensor = torch.arange(10, dtype=torch.float32)
    config = CompressionConfig(input_format=Format.TORCH)
    compressor = Compressor(config)
    compressed_bytes = compressor.compress(original_tensor)

    compressed_tensor = torch.frombuffer(bytearray(compressed_bytes), dtype=torch.uint8)

    decompressed = decompress_safetensors_tensor(compressed_tensor)

    assert isinstance(decompressed, torch.Tensor)
    assert decompressed.dtype == torch.float32
    assert torch.equal(decompressed, original_tensor)
