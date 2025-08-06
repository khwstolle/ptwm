"""Portability of explorer-discovered chains.

Asserts the load-bearing invariant behind ``ptwm compress --explore``:
``.ptwm`` files are self-describing with respect to their preprocessing
chains. A discovered chain absent from ``PRODUCTION_CHAINS`` still
round-trips on a decoder that knows nothing of the production table,
because the decoder reads chain bytes from the ``.ptwm`` itself.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from ptwm.classify import HfQuantConfigClassifier
from ptwm.integrations import compress_safetensors_file
from ptwm.integrations._hf import _load_multi_tensor_ptwm
from ptwm.preprocessing._explorer import ExploreOptions
from safetensors.torch import save_file


def _build_nvfp4_safetensors(tmp_path: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    """Synthesise a tiny NVFP4-style payload (FP4 nibbles + uint8 scales)."""
    _GROUP_SIZE = 16
    rng = np.random.default_rng(0)
    out_features, in_features = 64, 64
    n_groups = in_features // _GROUP_SIZE
    state = {
        "model.weight": torch.from_numpy(
            rng.integers(0, 256, size=(out_features, in_features // 2), dtype=np.uint8)
        ),
        "model.weight_scale": torch.from_numpy(
            rng.integers(64, 192, size=(out_features, n_groups), dtype=np.uint8)
        ),
    }
    src = tmp_path / "model.safetensors"
    save_file(state, str(src))
    return src, state


def test_explored_chain_decodes_without_production_chains(tmp_path: Path) -> None:
    """A `.ptwm` written with explored chains decodes with `PRODUCTION_CHAINS` empty.

    The explorer-discovered chain bytes land in the ``.ptwm`` container
    (either inline on the tensor record or in the per-file chain
    registry); the decoder reconstructs them from those bytes alone.
    """
    src, original_state = _build_nvfp4_safetensors(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    opts = ExploreOptions(max_candidates_per_dtype=8, time_budget_ms=2000)
    audit = compress_safetensors_file(
        src,
        out_dir,
        classifier=HfQuantConfigClassifier(quant_algo="NVFP4", exclude_modules=()),
        explore_options=opts,
    )

    # The rest of the test exercises nothing unless the explorer added at
    # least one candidate beyond the production set.
    discovered = audit.discovered_chains()
    assert discovered, (
        "Exploration produced no candidates beyond the production table; "
        "the test cannot prove portability without a discovered chain in "
        "the output. Adjust ExploreOptions or the synthetic input."
    )

    ptwm_path = out_dir / "model.ptwm"
    assert ptwm_path.exists()

    # Decode with PRODUCTION_CHAINS empty. The Rust decoder never reads the
    # Python table, so this acts as a regression guard against a future
    # Python-side dependency on PRODUCTION_CHAINS in the decode path.
    with patch("ptwm.preprocessing._chains.PRODUCTION_CHAINS", {}):
        decoded_state = _load_multi_tensor_ptwm(ptwm_path)

    assert set(decoded_state) == set(original_state)
    for name, original in original_state.items():
        got = decoded_state[name]
        assert got.shape == original.shape, (
            f"{name}: shape mismatch {got.shape} vs {original.shape}"
        )
        assert got.dtype == original.dtype, (
            f"{name}: dtype mismatch {got.dtype} vs {original.dtype}"
        )
        assert torch.equal(got, original), (
            f"{name}: byte-level mismatch after round-trip"
        )
