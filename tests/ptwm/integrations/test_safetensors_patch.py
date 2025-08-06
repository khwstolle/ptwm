"""Integration tests for the :func:`weights.patch_safetensors` monkey-patch.

These exercise the ``SafeOpen`` wrapper and the patch installation itself.
Marked ``integration`` so the fast unit CI job skips them; the integration
CI job installs ``safetensors`` and runs the full suite.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.integration

safetensors_torch = pytest.importorskip("safetensors.torch")


@pytest.fixture(autouse=True)
def _isolate_patch_state():
    """Reset patcher registry + safe_open between tests so order doesn't matter."""
    import safetensors.torch
    from ptwm.utils._patch import patches_applied

    saved_patches = dict(patches_applied)
    saved_safe_open = safetensors.torch.safe_open
    patches_applied.clear()
    yield
    patches_applied.clear()
    patches_applied.update(saved_patches)
    safetensors.torch.safe_open = saved_safe_open


def _save_bundle(tmp_path, tensors: dict[str, torch.Tensor]):
    from safetensors.torch import save_file

    path = tmp_path / "bundle.safetensors"
    save_file(tensors, str(path))
    return path


def test_patch_safetensors_installs_wrapper() -> None:
    """After calling the patch, safetensors.torch.safe_open must be SafeOpen."""
    import safetensors.torch
    from ptwm.integrations import patch_safetensors
    from ptwm.integrations._safetensors import SafeOpen

    patch_safetensors()
    assert safetensors.torch.safe_open is SafeOpen


def test_safeopen_passthrough_for_uncompressed_files(tmp_path) -> None:
    """SafeOpen must return uncompressed tensors verbatim when metadata is absent."""
    from ptwm.integrations._safetensors import SafeOpen

    tensors = {"w": torch.randn(8, 8, dtype=torch.float32)}
    path = _save_bundle(tmp_path, tensors)

    with SafeOpen(str(path), framework="pt", device="cpu") as f:
        recovered = f.get_tensor("w")
    assert torch.equal(recovered, tensors["w"])


def test_safeopen_get_slice_passthrough_for_uncompressed(tmp_path) -> None:
    from ptwm.integrations._safetensors import SafeOpen

    tensors = {"w": torch.randn(4, 4, dtype=torch.float32)}
    path = _save_bundle(tmp_path, tensors)

    with SafeOpen(str(path), framework="pt", device="cpu") as f:
        sl = f.get_slice("w")
        # The slice API is backed by safetensors; we only assert it is usable.
        assert sl is not None


def test_safeopen_get_slice_raises_for_compressed(tmp_path) -> None:
    """get_slice must raise NotImplementedError for tensors flagged compressed."""
    from ptwm.integrations._safetensors import SafeOpen
    from ptwm.utils._safetensors import METADATA_KEY
    from safetensors.torch import save_file

    # Fake a compressed-tensor entry in the metadata header.
    payload = torch.zeros(16, dtype=torch.uint8)
    path = tmp_path / "fake.safetensors"
    metadata = {
        METADATA_KEY: '{"w": {"dtype": "float32", "shape": "[4]"}}',
    }
    save_file({"w": payload}, str(path), metadata=metadata)

    with (
        SafeOpen(str(path), framework="pt", device="cpu") as f,
        pytest.raises(NotImplementedError),
    ):
        f.get_slice("w")


def test_patch_safetensors_is_idempotent() -> None:
    """Calling the patch twice registers only one patch in patches_applied."""
    from ptwm.integrations import patch_safetensors
    from ptwm.utils._patch import patches_applied

    before = len(patches_applied)
    patch_safetensors()
    after_first = len(patches_applied)
    patch_safetensors()
    after_second = len(patches_applied)

    assert after_first == before + 1
    assert after_second == after_first  # second call is a no-op
