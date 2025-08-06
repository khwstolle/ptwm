"""CLI integration: classifier flags map to a ClassifierChain."""

from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file


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


def _save(tmp_path: Path) -> Path:
    p = tmp_path / "model.safetensors"
    save_file(
        {k: torch.from_numpy(v) for k, v in _build_synthetic_nvfp4().items()},
        str(p),
    )
    return p


def test_compress_with_classify_rule(tmp_path):
    src = _save(tmp_path)
    out = tmp_path / "out"
    out.mkdir()

    from ptwm.cli.compress import main

    rc = main(
        [
            "--mode",
            "a",
            "--classify-rule",
            "*.weight_scale=scale_block",
            "--classify-rule",
            "*.weight=packed_values",
            "--out",
            str(out),
            str(src),
        ]
    )
    assert rc == 0
    # Single-shard fixture → model.ptwm; sharded → model-NNNNN-of-MMMMM.ptwm.
    assert (out / "model.ptwm").exists() or any(out.glob("model-*-of-*.ptwm"))


def test_compress_with_heuristic(tmp_path):
    src = _save(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    from ptwm.cli.compress import main

    rc = main(
        [
            "--mode",
            "a",
            "--classify-heuristic",
            "--out",
            str(out),
            str(src),
        ]
    )
    assert rc == 0
