from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from ptwm.integrations import compress_safetensors_file
from ptwm.integrations._safetensors import SafeOpen, write_mode_b_shell
from safetensors.torch import safe_open as native_safe_open
from safetensors.torch import save_file


# Inline NVFP4 fixture (tests/ not on Python path).
def _build_synthetic_nvfp4(seed: int = 42) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    base = "model.layers.0.self_attn.q_proj"
    return {
        f"{base}.weight": rng.integers(0, 256, size=(128, 32), dtype=np.uint8),
        f"{base}.weight_scale": rng.integers(64, 192, size=(128, 4), dtype=np.uint8),
        f"{base}.weight_scale_2": np.array(
            rng.standard_normal(dtype=np.float32) * 0.01, dtype=np.float32
        ),
        f"{base}.input_scale": np.array(
            rng.standard_normal(dtype=np.float32) * 0.01, dtype=np.float32
        ),
    }


def _make_ptwm_bytes(tmp_path: Path) -> bytes:
    state = _build_synthetic_nvfp4()
    src = tmp_path / "in.safetensors"
    save_file({k: torch.from_numpy(v) for k, v in state.items()}, str(src))
    work = tmp_path / "work"
    work.mkdir()
    compress_safetensors_file(src, work, mode="a")
    return (work / "model.ptwm").read_bytes()


def _make_bf16_shell(tmp_path: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    """Create a Mode B shell wrapping a .ptwm with a BF16 tensor."""
    tensors = {
        "bf16_weight": torch.randn(4, 8, dtype=torch.bfloat16),
        "u8_weight": torch.randint(0, 256, (4, 8), dtype=torch.uint8),
    }
    # compress_safetensors_file always names the output "model.ptwm"
    src = tmp_path / "model.safetensors"
    save_file(tensors, str(src))
    work = tmp_path / "bf16_work"
    work.mkdir()
    compress_safetensors_file(src, work, mode="a")
    blob = (work / "model.ptwm").read_bytes()
    shell_path = tmp_path / "bf16_shell.safetensors"
    write_mode_b_shell(shell_path, blob)
    return shell_path, tensors


def test_shell_is_valid_safetensors(tmp_path):
    blob = _make_ptwm_bytes(tmp_path)
    out = tmp_path / "shell.safetensors"
    write_mode_b_shell(out, blob)
    # The shell must open with vanilla safetensors (no patches).
    with native_safe_open(str(out), framework="pt") as f:
        assert "__ptwm_payload__" in f.keys()  # noqa: SIM118
        meta = f.metadata()
        assert meta["weights_format"] == "ptwm-shell"
        assert meta["weights_format_version"] == "1"


def test_safe_open_wrapper_returns_embedded_keys(tmp_path):
    blob = _make_ptwm_bytes(tmp_path)
    out = tmp_path / "shell.safetensors"
    write_mode_b_shell(out, blob)
    f = SafeOpen(str(out), framework="pt")
    # The wrapper exposes the embedded ptwm's keyspace, not __ptwm_payload__.
    assert "model.layers.0.self_attn.q_proj.weight" in f.keys()  # noqa: SIM118
    assert "__ptwm_payload__" not in f.keys()  # noqa: SIM118


def test_xxhash64_metadata_present(tmp_path):
    blob = _make_ptwm_bytes(tmp_path)
    out = tmp_path / "shell.safetensors"
    write_mode_b_shell(out, blob)
    with native_safe_open(str(out), framework="pt") as f:
        meta = f.metadata()
        # The hash field is populated and 16 hex digits long.
        h = meta["ptwm_bytes_xxhash64"]
        assert len(h) == 16
        int(h, 16)  # parses as hex


def test_mode_b_get_tensor_dtype_and_shape(tmp_path):
    """Mode B get_tensor returns the correct dtype + shape, not flat uint8."""
    state = _build_synthetic_nvfp4()
    blob = _make_ptwm_bytes(tmp_path)
    out = tmp_path / "shell.safetensors"
    write_mode_b_shell(out, blob)
    f = SafeOpen(str(out), framework="pt")

    # uint8 tensor
    key = "model.layers.0.self_attn.q_proj.weight"
    t = f.get_tensor(key)
    expected = state[key]
    assert t.dtype == torch.uint8
    assert t.shape == torch.from_numpy(expected).shape
    np.testing.assert_array_equal(t.cpu().numpy(), expected)

    # float32 scalar
    key32 = "model.layers.0.self_attn.q_proj.weight_scale_2"
    t32 = f.get_tensor(key32)
    assert t32.dtype == torch.float32
    np.testing.assert_array_equal(t32.cpu().numpy(), state[key32])


def test_mode_b_lazy_decoding(tmp_path):
    """decode_tensor must NOT be called during SafeOpen.__init__ for Mode B."""
    blob = _make_ptwm_bytes(tmp_path)
    out = tmp_path / "shell.safetensors"
    write_mode_b_shell(out, blob)

    import ptwm._rust as _rust_mod

    original = _rust_mod.decode_tensor
    call_log: list[str] = []

    def recording_decode(b: bytes, name: str) -> bytes:
        call_log.append(name)
        return original(b, name)

    _rust_mod.decode_tensor = recording_decode
    try:
        f = SafeOpen(str(out), framework="pt")
        assert call_log == [], f"decode called during __init__: {call_log}"

        target_key = "model.layers.0.self_attn.q_proj.weight"
        _t = f.get_tensor(target_key)
        assert call_log == [target_key], f"unexpected decode calls: {call_log}"
    finally:
        _rust_mod.decode_tensor = original


def test_mode_b_bf16_round_trip(tmp_path):
    """BF16 tensors in a Mode B shell survive SafeOpen with correct dtype."""
    shell_path, originals = _make_bf16_shell(tmp_path)
    f = SafeOpen(str(shell_path), framework="pt")
    t = f.get_tensor("bf16_weight")
    assert t.dtype == torch.bfloat16, f"expected bfloat16, got {t.dtype}"
    assert t.shape == originals["bf16_weight"].shape
    orig_bits = originals["bf16_weight"].view(torch.uint16).cpu().numpy()
    got_bits = t.view(torch.uint16).cpu().numpy()
    np.testing.assert_array_equal(got_bits, orig_bits)
