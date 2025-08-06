"""Integration tests for the :func:`weights.patch_transformers` monkey-patch.

Exercises the ``load_state_dict`` wrapper end-to-end: write a
``.safetensors.ptwm`` file, install the patch, then ask transformers to
load the state dict. Also covers the ``_MIN_TRANSFORMERS_VERSION`` gate.

Marked ``integration`` — the fast unit CI job skips this module; the
integration CI job installs ``transformers`` so the importorskip passes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.integration

transformers = pytest.importorskip("transformers")


@pytest.fixture(autouse=True)
def _isolate_patch_state():
    """Clear patches_applied and restore load_state_dict between tests."""
    from ptwm.utils._patch import patches_applied
    from transformers import modeling_utils

    saved_patches = dict(patches_applied)
    saved_load = modeling_utils.load_state_dict
    patches_applied.clear()
    yield
    patches_applied.clear()
    patches_applied.update(saved_patches)
    modeling_utils.load_state_dict = saved_load


def _write_ptwm(path: Path, tensors: dict[str, torch.Tensor]) -> Path:
    """Compress a safetensors file and rename it to end in .ptwm."""
    from ptwm import CompressionConfig, Compressor, Format
    from safetensors.torch import save_file

    native = path.with_suffix("")  # drop .ptwm
    save_file(tensors, str(native))
    native_bytes = native.read_bytes()
    compressed = Compressor(CompressionConfig(input_format=Format.BYTE)).compress(
        native_bytes
    )
    path.write_bytes(compressed)
    native.unlink()
    return path


def test_patch_transformers_installs_load_state_dict_wrapper() -> None:
    """After calling the patch, load_state_dict must be rebound."""
    from ptwm.integrations import patch_transformers
    from transformers import modeling_utils

    before = modeling_utils.load_state_dict
    patch_transformers()
    after = modeling_utils.load_state_dict
    assert before is not after


def test_patched_loader_decompresses_ptw_transparently(tmp_path) -> None:
    """Asking the patched loader for a .ptwm path returns the original tensors."""
    from ptwm.integrations import patch_transformers
    from transformers import modeling_utils

    tensors = {
        "layer.weight": torch.randn(8, 8, dtype=torch.float32),
        "layer.bias": torch.randn(8, dtype=torch.float32),
    }
    ptwm_path = tmp_path / "model.safetensors.ptwm"
    _write_ptwm(ptwm_path, tensors)

    patch_transformers()
    loaded = modeling_utils.load_state_dict(str(ptwm_path))
    assert set(loaded) == set(tensors)
    for name, expected in tensors.items():
        assert torch.equal(loaded[name], expected)


def test_patched_loader_passthrough_for_non_ptw(tmp_path) -> None:
    """Non-.ptwm paths must delegate to the original loader unchanged."""
    from ptwm.integrations import patch_transformers
    from safetensors.torch import save_file
    from transformers import modeling_utils

    tensors = {"w": torch.randn(4, 4, dtype=torch.float32)}
    native = tmp_path / "plain.safetensors"
    save_file(tensors, str(native))

    patch_transformers()
    loaded = modeling_utils.load_state_dict(str(native))
    assert torch.equal(loaded["w"], tensors["w"])


def test_patch_transformers_rejects_old_transformers(monkeypatch) -> None:
    """Versions older than _MIN_TRANSFORMERS_VERSION must raise RuntimeError."""
    import ptwm.integrations._hf as hf_mod
    from packaging.version import Version

    monkeypatch.setattr(hf_mod, "_transformers_version", lambda: Version("4.0.0"))
    with pytest.raises(RuntimeError, match="requires transformers"):
        hf_mod.patch_transformers()


def test_patch_transformers_requires_transformers(monkeypatch) -> None:
    """If modeling_utils is None the patch must raise ImportError."""
    import ptwm.integrations._hf as hf_mod

    monkeypatch.setattr(hf_mod, "modeling_utils", None)
    with pytest.raises(ImportError, match="Transformers"):
        hf_mod.patch_transformers()
