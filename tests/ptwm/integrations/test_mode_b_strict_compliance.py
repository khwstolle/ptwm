"""Verify Mode B output is openable by the upstream safetensors library
without any monkey-patching from this project."""

import numpy as np
import torch
from ptwm.integrations import compress_safetensors_file
from safetensors.torch import safe_open as native_safe_open
from safetensors.torch import save_file


# Inline fixture (tests/ not on Python path).
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


def test_no_patch_required_to_open_mode_b(tmp_path):
    state = _build_synthetic_nvfp4()
    src = tmp_path / "model.safetensors"
    save_file({k: torch.from_numpy(v) for k, v in state.items()}, str(src))

    out = tmp_path / "out"
    out.mkdir()
    compress_safetensors_file(src, out, mode="b")

    shells = list(out.glob("*.safetensors"))
    assert shells, "Mode B should produce .safetensors files"
    for shell in shells:
        with native_safe_open(str(shell), framework="pt") as f:
            keys = list(f.keys())  # noqa: SIM118 — safe_open isn't a dict
            assert "__ptwm_payload__" in keys
