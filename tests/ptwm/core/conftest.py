"""Shared fixtures for core roundtrip tests."""

import os

import pytest
import torch
from ptwm import CompressionConfig, Compressor, Decompressor, Format


@pytest.fixture
def compressor_torch() -> Compressor:
    return Compressor(CompressionConfig(input_format=Format.TORCH))


@pytest.fixture
def compressor_bytes() -> Compressor:
    return Compressor(CompressionConfig())


@pytest.fixture
def decompressor() -> Decompressor:
    return Decompressor()


@pytest.fixture
def make_tensor():
    def _make(size_in_kb: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        num_elements = size_in_kb * 1024
        return torch.rand(num_elements, dtype=dtype) * 2 - 1

    return _make


@pytest.fixture
def make_random_bytes():
    def _make(size_in_kb: int) -> bytes:
        return os.urandom(size_in_kb * 1024)

    return _make
