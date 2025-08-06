"""Verify patch_transformers() discovers model.ptwm.index.json + .ptwm shards."""

from pathlib import Path

import numpy as np
import pytest
import torch
from ptwm.integrations import compress_safetensors_file, patch_transformers
from safetensors.torch import save_file

pytest.importorskip("transformers")


# Inline fixture (tests/ not on Python path).
def _build_synthetic_nvfp4(seed: int = 42) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    base = "model.layers.0.self_attn.q_proj"
    return {
        f"{base}.weight": rng.integers(0, 256, size=(128, 32), dtype=np.uint8),
        f"{base}.weight_scale": rng.integers(64, 192, size=(128, 4), dtype=np.uint8),
        f"{base}.weight_scale_2": np.array(
            rng.standard_normal() * 0.01, dtype=np.float32
        ),
        f"{base}.input_scale": np.array(rng.standard_normal() * 0.01, dtype=np.float32),
    }


def test_sharded_discovery_finds_ptwm_index(tmp_path):
    src = tmp_path / "model.safetensors"
    save_file(
        {k: torch.from_numpy(v) for k, v in _build_synthetic_nvfp4().items()},
        str(src),
    )
    out = tmp_path / "model"
    out.mkdir()
    # max_shard_size=1 forces sharding (every tensor its own shard).
    compress_safetensors_file(src, out, mode="a", max_shard_size=1)
    assert (out / "model.ptwm.index.json").exists()
    shards = sorted(out.glob("model-*-of-*.ptwm"))
    assert len(shards) >= 2

    patch_transformers()  # install patches

    # The patched discovery code should accept this directory.
    from ptwm.integrations._hf import discover_ptwm_shards

    found = discover_ptwm_shards(out)
    assert len(found) == len(shards)
    assert {Path(p).name for p in found} == {p.name for p in shards}


def test_discover_ptwm_shards_single_file(tmp_path):
    """discover_ptwm_shards returns single model.ptwm when no index present."""
    src = tmp_path / "model.safetensors"
    save_file(
        {k: torch.from_numpy(v) for k, v in _build_synthetic_nvfp4().items()},
        str(src),
    )
    out = tmp_path / "model"
    out.mkdir()
    # A large max_shard_size keeps everything in one file → model.ptwm, no index.
    compress_safetensors_file(src, out, mode="a", max_shard_size=5 * 1024**3)

    from ptwm.integrations._hf import discover_ptwm_shards

    found = discover_ptwm_shards(out)
    assert len(found) == 1
    assert found[0].name == "model.ptwm"


def test_discover_ptwm_shards_empty_dir(tmp_path):
    """discover_ptwm_shards returns empty list when no .ptwm files exist."""
    from ptwm.integrations._hf import discover_ptwm_shards

    found = discover_ptwm_shards(tmp_path)
    assert found == []
