"""Latency / size benchmark for the chain explorer.

Compresses a synthetic NVFP4-style model twice — once with production
chains alone, once with ``--explore`` — and reports the size and
encode-time deltas. Marked ``slow`` to keep the regular suite fast; also
runs as a script via ``python -m tests.ptwm.benchmarks.test_explore_savings``.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import torch
from ptwm.classify import HfQuantConfigClassifier
from ptwm.integrations import compress_safetensors_file
from ptwm.preprocessing._explorer import ExploreOptions
from safetensors.torch import save_file


def _build_synthetic(tmp_path: Path) -> Path:
    rng = np.random.default_rng(7)
    out_features, in_features = 256, 256
    group = 16
    n_groups = in_features // group
    state = {
        "model.weight": torch.from_numpy(
            rng.integers(0, 256, size=(out_features, in_features // 2), dtype=np.uint8)
        ),
        "model.weight_scale": torch.from_numpy(
            rng.integers(64, 192, size=(out_features, n_groups), dtype=np.uint8)
        ),
    }
    src = tmp_path / "synthetic.safetensors"
    save_file(state, str(src))
    return src


def _dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _run_one(src: Path, out_dir: Path, *, with_explore: bool) -> tuple[int, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    classifier = HfQuantConfigClassifier(quant_algo="NVFP4", exclude_modules=())
    opts = (
        ExploreOptions(max_candidates_per_dtype=16, time_budget_ms=3000)
        if with_explore
        else None
    )
    t0 = time.perf_counter()
    compress_safetensors_file(
        src,
        out_dir,
        classifier=classifier,
        explore_options=opts,
        use_user_cache=False,
    )
    elapsed = time.perf_counter() - t0
    return _dir_size(out_dir), elapsed


@pytest.mark.slow
def test_explore_savings_report(tmp_path: Path) -> None:
    src = _build_synthetic(tmp_path)
    base_size, base_t = _run_one(src, tmp_path / "baseline", with_explore=False)
    expl_size, expl_t = _run_one(src, tmp_path / "explored", with_explore=True)

    ratio_change = (expl_size - base_size) / base_size if base_size else 0.0
    time_overhead = expl_t - base_t

    report = (
        f"baseline: {base_size} B in {base_t:.2f}s; "
        f"explore: {expl_size} B in {expl_t:.2f}s; "
        f"size Δ {100 * ratio_change:+.3f}%; time Δ {time_overhead:+.2f}s"
    )
    print(report)

    # The trial-encode menu can always fall back to the production
    # candidate the baseline used, so exploration never produces a
    # strictly larger file (within run-to-run noise).
    assert base_size > 0
    assert expl_size > 0
    assert expl_size <= base_size + (base_size // 100)  # within 1% tolerance for noise


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_explore_savings_report(Path(tmp))
