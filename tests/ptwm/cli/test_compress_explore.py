"""Tests for the ``ptwm compress --explore`` CLI surface."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file


def _make_nvfp4_safetensors(tmp_path: Path) -> Path:
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
    return src


def test_compress_help_lists_explore_flags():
    """``ptwm compress --help`` exposes the explore flags."""
    result = subprocess.run(
        [sys.executable, "-m", "ptwm.cli.main", "compress", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--explore" in result.stdout
    assert "--explore-budget-ms" in result.stdout
    assert "--explore-max-candidates" in result.stdout


def test_compress_explore_rejects_non_safetensors_input(tmp_path):
    """--explore on a non-.safetensors input exits with a clear error."""
    raw = tmp_path / "blob.bin"
    raw.write_bytes(b"\x00" * 1024)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ptwm.cli.main",
            "compress",
            str(raw),
            "--explore",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--explore requires a .safetensors file" in result.stderr


def test_compress_explore_writes_ptwm_directory(tmp_path):
    """--explore on a .safetensors input writes a .ptwm directory."""
    src = _make_nvfp4_safetensors(tmp_path)
    out_dir = tmp_path / "out.ptwm"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ptwm.cli.main",
            "compress",
            str(src),
            "--explore",
            "--explore-budget-ms",
            "1500",
            "--explore-max-candidates",
            "6",
            "--out",
            str(out_dir),
            "--force",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    assert out_dir.exists()
    assert out_dir.is_dir()
    assert any(out_dir.rglob("*.ptwm")), "no .ptwm shards in output directory"


def test_compress_explore_prints_summary(tmp_path):
    """The summary line reports candidate counts and the compression ratio."""
    src = _make_nvfp4_safetensors(tmp_path)
    out_dir = tmp_path / "out.ptwm"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ptwm.cli.main",
            "compress",
            str(src),
            "--explore",
            "--explore-budget-ms",
            "1500",
            "--explore-max-candidates",
            "6",
            "--out",
            str(out_dir),
            "--force",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    # Either we discovered something or we honestly say we didn't.
    assert "Explored" in result.stdout
    assert "Output:" in result.stdout
    assert "ratio" in result.stdout
