"""TensorIndex tests on ``.ptwm`` containers.

A single ``compress_model`` call bundles many tensors directly, with CBOR
shape metadata carrying the dtype/shape information needed to reconstruct
tensors.
"""

from __future__ import annotations

import pytest
import torch
from ptwm import _rust
from ptwm.preprocessing._chains import (
    CHAIN_BYTE_PASSTHROUGH_VALUE,
    ClassifierRole,
    chains_for,
)
from ptwm.random_access import TensorIndex


def _chain_bytes_for(dtype_code: int, shape: list[int]) -> list[bytes]:
    """Return candidate chain bytes for dtype_code / STANDARD role."""
    builders = list(chains_for(dtype_code, ClassifierRole.STANDARD))
    if not builders:
        builders = [CHAIN_BYTE_PASSTHROUGH_VALUE]
    return [b(shape).to_bytes() for b in builders]


def _write_bundle(path, tensors: dict[str, tuple[torch.Tensor, int, str]]) -> None:
    """Write a ``.ptwm`` bundle.

    Each value is ``(tensor, dtype_code, dtype_name)``. Input format is
    always BYTE (0) since we pass raw bytes.
    """
    records = []
    chains_list = []
    for name, (tensor, dtype_code, dtype_name) in tensors.items():
        contig = tensor.contiguous()
        raw = contig.view(torch.uint8).numpy().tobytes()
        shape = list(tensor.shape)
        records.append((name, dtype_code, 0, raw, shape, dtype_name, None))
        chains_list.append(_chain_bytes_for(dtype_code, shape))
    blob = _rust.compress_model(records, chains_list, True, True)
    path.write_bytes(blob)


@pytest.fixture
def bundle(tmp_path):
    # dtype_code values from DType: BF16=6, FP16=4, FP32=1
    tensors: dict[str, tuple[torch.Tensor, int, str]] = {
        "bf16_weight": (
            torch.randn(16, 64, dtype=torch.bfloat16),
            6,
            "bfloat16",
        ),
        "fp16_weight": (
            torch.randn(8, 32, dtype=torch.float16),
            4,
            "float16",
        ),
        "fp32_weight": (
            torch.randn(32, 32, dtype=torch.float32),
            1,
            "float32",
        ),
    }
    path = tmp_path / "bundle.ptwm"
    _write_bundle(path, tensors)
    return path, tensors


def test_sniff_and_introspection(bundle):
    path, tensors = bundle
    idx = TensorIndex.open(path)
    assert set(idx.names()) == set(tensors)
    assert all(name in idx for name in tensors)
    assert "missing" not in idx
    assert len(idx) == len(tensors)


def test_get_tensor_roundtrip(bundle):
    path, tensors = bundle
    idx = TensorIndex.open(path)
    for name, (expected, _dcode, _dname) in tensors.items():
        recovered = idx.get_tensor(name)
        assert recovered.dtype == expected.dtype
        assert tuple(recovered.shape) == tuple(expected.shape)
        if expected.dtype == torch.bfloat16:
            assert torch.equal(
                recovered.view(torch.uint16), expected.view(torch.uint16)
            )
        else:
            assert torch.equal(recovered, expected)


def test_missing_name_raises_keyerror(bundle):
    path, _ = bundle
    idx = TensorIndex.open(path)
    with pytest.raises(KeyError):
        idx.get_tensor("does_not_exist")


def test_get_bytes_returns_raw(bundle):
    path, tensors = bundle
    idx = TensorIndex.open(path)
    expected, _, _ = tensors["fp32_weight"]
    raw = idx.get_bytes("fp32_weight")
    assert isinstance(raw, bytes)
    assert len(raw) == expected.element_size() * expected.numel()


def test_repr_includes_path_and_count(bundle):
    path, tensors = bundle
    idx = TensorIndex.open(path)
    rendered = repr(idx)
    assert str(path) in rendered
    assert f"n={len(tensors)}" in rendered


def test_missing_shape_metadata_yields_typeerror(tmp_path):
    """Tensor compressed without shape/dtype_name cannot be rebuilt as a torch.Tensor."""
    raw = b"\x11" * 68  # 4 MXFP4 blocks
    dtype_code = 31  # FLOAT4_E2M1FN_X2
    chains = [_chain_bytes_for(dtype_code, [len(raw)])]
    blob = _rust.compress_model(
        [("t0", dtype_code, 0, raw, None, None, None)], chains, False, False
    )
    path = tmp_path / "no_meta.ptwm"
    path.write_bytes(blob)
    idx = TensorIndex.open(path)
    with pytest.raises(TypeError, match="no shape metadata"):
        idx.get_tensor("t0")
    # But raw bytes are still accessible.
    assert idx.get_bytes("t0") == raw
