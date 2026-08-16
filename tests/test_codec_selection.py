import pytest
import torch

from ptwm import CompressionConfig, Compressor


def test_codec_defaults_to_none():
    assert CompressionConfig().codec is None


def test_device_defaults_to_none():
    assert CompressionConfig().device is None


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_device_mismatch_with_tensor_device_raises():
    t = torch.zeros(64, dtype=torch.bfloat16, device="cuda:0")
    cfg = CompressionConfig(codec="example_codec", device=1)
    with pytest.raises(ValueError, match="device"):
        Compressor(cfg).compress(t)
