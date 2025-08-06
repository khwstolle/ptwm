"""Tests for SafeOpen Mode A (.ptwm)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from ptwm.integrations import compress_safetensors_file
from ptwm.integrations._safetensors import SafeOpen
from safetensors.torch import save_file


# Inline NVFP4 fixture (mirroring test_compress_safetensors.py's pattern;
# tests/ is not on the Python path so we can't import from
# tests.ptwm.fixtures here).
def _build_synthetic_nvfp4(seed: int = 42) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    base = "model.layers.0.self_attn.q_proj"
    out_features = 128
    in_features = 64
    group_size = 16
    return {
        f"{base}.weight": rng.integers(
            0, 256, size=(out_features, in_features // 2), dtype=np.uint8
        ),
        f"{base}.weight_scale": rng.integers(
            64, 192, size=(out_features, in_features // group_size), dtype=np.uint8
        ),
        f"{base}.weight_scale_2": np.array(
            rng.standard_normal(dtype=np.float32) * 0.01, dtype=np.float32
        ),
        f"{base}.input_scale": np.array(
            rng.standard_normal(dtype=np.float32) * 0.01, dtype=np.float32
        ),
    }


def _make_ptwm(tmp_path: Path) -> Path:
    state = _build_synthetic_nvfp4()
    p = tmp_path / "model.safetensors"
    save_file({k: torch.from_numpy(v) for k, v in state.items()}, str(p))
    out = tmp_path / "out"
    out.mkdir()
    compress_safetensors_file(p, out, mode="a")
    target = out / "model.ptwm"
    assert target.exists()
    return target


def _make_bf16_ptwm(tmp_path: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    """Create a .ptwm with one BF16 tensor and one uint8 tensor."""
    tensors = {
        "bf16_weight": torch.randn(4, 8, dtype=torch.bfloat16),
        "u8_weight": torch.randint(0, 256, (4, 8), dtype=torch.uint8),
    }
    # compress_safetensors_file always names the output "model.ptwm"
    p = tmp_path / "model.safetensors"
    save_file(tensors, str(p))
    out = tmp_path / "bf16_out"
    out.mkdir()
    compress_safetensors_file(p, out, mode="a")
    target = out / "model.ptwm"
    assert target.exists()
    return target, tensors


def test_safe_open_ptwm_keys_and_get_tensor(tmp_path):
    ptwm_path = _make_ptwm(tmp_path)
    f = SafeOpen(str(ptwm_path), framework="pt")
    keys = list(f.keys())
    assert "model.layers.0.self_attn.q_proj.weight" in keys

    expected = _build_synthetic_nvfp4()["model.layers.0.self_attn.q_proj.weight"]
    t = f.get_tensor("model.layers.0.self_attn.q_proj.weight")

    # dtype and shape should match the original tensor (uint8, (128, 32))
    assert t.dtype == torch.uint8
    assert t.shape == torch.from_numpy(expected).shape
    np.testing.assert_array_equal(t.cpu().numpy(), expected)


def test_safe_open_ptwm_float32_tensor(tmp_path):
    """Scalar float32 tensors survive the round-trip with correct dtype."""
    ptwm_path = _make_ptwm(tmp_path)
    f = SafeOpen(str(ptwm_path), framework="pt")
    key = "model.layers.0.self_attn.q_proj.weight_scale_2"
    t = f.get_tensor(key)
    expected = _build_synthetic_nvfp4()[key]
    assert t.dtype == torch.float32
    np.testing.assert_array_equal(t.cpu().numpy(), expected)


def test_lazy_decoding_only_decodes_requested_tensor(tmp_path):
    """decode_tensor must NOT be called during __init__, only from get_tensor."""
    ptwm_path = _make_ptwm(tmp_path)

    import ptwm._rust as _rust_mod

    original = _rust_mod.decode_tensor
    call_log: list[str] = []

    def recording_decode(blob: bytes, name: str) -> bytes:
        call_log.append(name)
        return original(blob, name)

    _rust_mod.decode_tensor = recording_decode
    try:
        f = SafeOpen(str(ptwm_path), framework="pt")
        # __init__ must not have triggered any decode.
        assert call_log == [], f"decode called during __init__: {call_log}"

        target_key = "model.layers.0.self_attn.q_proj.weight"
        _t = f.get_tensor(target_key)
        # Only the requested tensor should have been decoded.
        assert call_log == [target_key], f"unexpected decode calls: {call_log}"
    finally:
        _rust_mod.decode_tensor = original


def test_safe_open_bf16_round_trip(tmp_path):
    """BF16 tensors round-trip through SafeOpen with correct dtype and values."""
    ptwm_path, originals = _make_bf16_ptwm(tmp_path)
    f = SafeOpen(str(ptwm_path), framework="pt")
    t = f.get_tensor("bf16_weight")
    assert t.dtype == torch.bfloat16, f"expected bfloat16, got {t.dtype}"
    assert t.shape == originals["bf16_weight"].shape
    # Bit-exact comparison via uint16 view (avoid float equality pitfalls).
    orig_bits = originals["bf16_weight"].view(torch.uint16).cpu().numpy()
    got_bits = t.view(torch.uint16).cpu().numpy()
    np.testing.assert_array_equal(got_bits, orig_bits)
