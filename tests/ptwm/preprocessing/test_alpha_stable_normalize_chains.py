"""Tests for the alpha_stable_normalize chain candidates.

Lossless by construction (robust affine + exact XOR residual). Mirrors
test_spherical_normalize_chains.py: the FP32 / BF16 / FP16 chains build and
serialise, decline non-2D shapes, are registered as STANDARD candidates, and
end-to-end ``Compressor → Decompressor`` round-trips stay bit-exact.
"""

from __future__ import annotations

import pytest
import torch
from ptwm import (
    CompressionConfig,
    Compressor,
    DecompressionConfig,
    Decompressor,
    Format,
)
from ptwm.preprocessing._chains import (
    CHAIN_BF16_ALPHA_STABLE_NORMALIZE,
    CHAIN_FP16_ALPHA_STABLE_NORMALIZE,
    CHAIN_FP32_ALPHA_STABLE_NORMALIZE,
    CHAIN_FP32_SPLIT,
    PRODUCTION_CHAINS,
    ClassifierRole,
)
from ptwm.utils import DType as _DT


def test_chains_serialise() -> None:
    assert len(CHAIN_FP32_ALPHA_STABLE_NORMALIZE([8, 4]).to_bytes()) > 0
    assert len(CHAIN_BF16_ALPHA_STABLE_NORMALIZE([8, 4]).to_bytes()) > 0
    assert len(CHAIN_FP16_ALPHA_STABLE_NORMALIZE([8, 4]).to_bytes()) > 0


def test_declines_non_2d_shapes() -> None:
    # The op only accepts 2D row-structured tensors; the builder returns
    # None for 1D and higher-rank shapes so the dispatcher skips it,
    # bounding trial-encode cost.
    for builder in (
        CHAIN_FP32_ALPHA_STABLE_NORMALIZE,
        CHAIN_BF16_ALPHA_STABLE_NORMALIZE,
        CHAIN_FP16_ALPHA_STABLE_NORMALIZE,
    ):
        assert builder([128]) is None  # 1D → declined
        assert builder([4, 8, 16]) is None  # 3D → declined
        assert builder([8, 4]) is not None  # 2D → accepted


def test_registered_as_production_candidate() -> None:
    bf16 = PRODUCTION_CHAINS[(_DT.BFLOAT16.code, ClassifierRole.STANDARD)]
    fp16 = PRODUCTION_CHAINS[(_DT.FLOAT16.code, ClassifierRole.STANDARD)]
    fp32 = PRODUCTION_CHAINS[(_DT.FLOAT32.code, ClassifierRole.STANDARD)]
    assert CHAIN_BF16_ALPHA_STABLE_NORMALIZE in bf16
    assert CHAIN_FP16_ALPHA_STABLE_NORMALIZE in fp16
    assert CHAIN_FP32_ALPHA_STABLE_NORMALIZE in fp32
    # The plain split chains stay in the menu — alpha_stable only competes.
    assert CHAIN_FP32_SPLIT in fp32


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(64, 32), (16, 16), (1, 8), (128,)])
def test_container_roundtrip_with_alpha_stable_in_menu(
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> None:
    """Bit-exact round-trip with alpha_stable_normalize in the candidate set.

    Routed through ``Format.TORCH`` so the 2D shape + float dtype reach the
    STANDARD classifier and the op actually competes as a candidate.
    """
    generator = torch.Generator().manual_seed(sum(shape))
    tensor = (torch.randn(*shape, generator=generator, dtype=torch.float32) * 0.02).to(
        dtype
    )

    compressor = Compressor(CompressionConfig(input_format=Format.TORCH))
    decompressor = Decompressor(DecompressionConfig())
    blob = compressor.compress(tensor)
    restored = decompressor.decompress(blob)
    assert tensor.view(torch.uint8).equal(restored.view(torch.uint8)), (
        f"round-trip mismatch on dtype={dtype} shape={shape}"
    )


@pytest.mark.parametrize(
    ("torch_dtype", "dt", "dtype_name", "builder"),
    [
        (torch.float32, _DT.FLOAT32, "float32", CHAIN_FP32_ALPHA_STABLE_NORMALIZE),
        (torch.bfloat16, _DT.BFLOAT16, "bfloat16", CHAIN_BF16_ALPHA_STABLE_NORMALIZE),
        (torch.float16, _DT.FLOAT16, "float16", CHAIN_FP16_ALPHA_STABLE_NORMALIZE),
    ],
)
def test_forced_alpha_stable_decode_path(torch_dtype, dt, dtype_name, builder) -> None:
    """Force alpha_stable_normalize as the SOLE candidate chain.

    A self-selecting container round-trip may pick a different chain, leaving
    the op's ``forward → wire → dispatch_builtin → inverse`` path untested.
    Forcing the chain through the
    low-level ``compress_model``/``decode_tensor`` API guarantees that path runs.
    """
    try:
        from ptwm._rust import compress_model, decode_tensor
    except ImportError:
        pytest.skip("ptwm._core extension not built")

    shape = [16, 8]
    gen = torch.Generator().manual_seed(7)
    tensor = (
        (torch.randn(*shape, generator=gen, dtype=torch.float32) * 0.02)
        .to(torch_dtype)
        .contiguous()
    )
    raw = tensor.view(torch.uint8).numpy().tobytes()
    chain = builder(shape)
    assert chain is not None  # builder accepts the 2D shape
    chain_bytes = chain.to_bytes()

    blob = compress_model(
        records=[("t", dt.code, 0, raw, shape, dtype_name, None)],
        chains_per_tensor=[[chain_bytes]],
        emit_payload_hash=True,
        emit_plane_crc=True,
    )
    assert decode_tensor(blob, "t") == raw
