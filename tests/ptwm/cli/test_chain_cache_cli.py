"""Tests for the `ptwm chains cache` / `export` / `import` subcommands."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch, tmp_path: Path):
    # Per-test cache root. The env var carries into the subprocess
    # invocations below via os.environ inheritance in _run.
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))


def _run(*argv: str, env: dict | None = None) -> subprocess.CompletedProcess:
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    return subprocess.run(
        [sys.executable, "-m", "ptwm.cli.main", *argv],
        check=False,
        capture_output=True,
        text=True,
        env=full_env,
    )


def _make_nvfp4_safetensors(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    state = {
        "model.weight": torch.from_numpy(
            rng.integers(0, 256, size=(64, 32), dtype=np.uint8)
        ),
        "model.weight_scale": torch.from_numpy(
            rng.integers(64, 192, size=(64, 4), dtype=np.uint8)
        ),
    }
    src = tmp_path / "model.safetensors"
    save_file(state, str(src))
    return src


def test_cache_info_on_empty_cache(tmp_path: Path) -> None:
    result = _run("chains", "cache", "info")
    assert result.returncode == 0
    assert "empty" in result.stdout.lower()


def test_cache_clear_on_empty_cache(tmp_path: Path) -> None:
    result = _run("chains", "cache", "clear")
    assert result.returncode == 0
    assert "0" in result.stdout


def test_explore_populates_cache(tmp_path: Path) -> None:
    src = _make_nvfp4_safetensors(tmp_path)
    out_dir = tmp_path / "out.ptwm"
    rc = _run(
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
    )
    assert rc.returncode == 0, rc.stderr
    info_result = _run("chains", "cache", "info")
    assert info_result.returncode == 0
    # Some chain should have been cached for at least one (dtype, role) pair.
    assert "chain(s)" in info_result.stdout
    assert "0:" not in info_result.stdout.replace("(", " ")  # no zero-count lines


def test_export_round_trip(tmp_path: Path) -> None:
    src = _make_nvfp4_safetensors(tmp_path)
    out_dir = tmp_path / "out.ptwm"
    rc = _run(
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
    )
    assert rc.returncode == 0, rc.stderr

    bundle = tmp_path / "bundle.cbor"
    export_result = _run("chains", "export", str(bundle))
    assert export_result.returncode == 0
    assert bundle.exists()
    assert bundle.read_bytes().startswith(b"PTWMEXC\x01")

    # Wipe the cache, then import the bundle. The cache should refill.
    clear_result = _run("chains", "cache", "clear")
    assert clear_result.returncode == 0
    import_result = _run("chains", "import", str(bundle))
    assert import_result.returncode == 0, import_result.stderr
    assert "Imported" in import_result.stdout

    info_result = _run("chains", "cache", "info")
    assert "chain(s)" in info_result.stdout


def test_import_rejects_missing_magic(tmp_path: Path) -> None:
    bad = tmp_path / "bad.cbor"
    bad.write_bytes(b"not a real bundle")
    result = _run("chains", "import", str(bad))
    assert result.returncode != 0
    assert "PTWMEXC" in result.stderr


def test_no_cache_flag_skips_cache_write(tmp_path: Path) -> None:
    src = _make_nvfp4_safetensors(tmp_path)
    out_dir = tmp_path / "out.ptwm"
    rc = _run(
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
        "--no-cache",
    )
    assert rc.returncode == 0, rc.stderr
    info_result = _run("chains", "cache", "info")
    assert "empty" in info_result.stdout.lower()
