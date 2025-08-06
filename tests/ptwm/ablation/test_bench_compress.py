"""ptwm bench compress smoke test."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def isolated_xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    return tmp_path


def test_bench_compress_emits_csv_and_summary(
    isolated_xdg: Path, tmp_path: Path
) -> None:
    from ptwm.ablation._bench import bench_compress

    input_path = tmp_path / "input.bin"
    input_path.write_bytes(b"\x00" * 4096)

    out_dir = tmp_path / "bench-out"
    result = bench_compress(
        input_path=input_path,
        baseline_policy=None,
        variant_policies=[],
        metrics=["ratio", "encode-time"],
        trials=2,
        out_dir=out_dir,
    )

    assert len(result.rows) == 2
    assert (out_dir / "results.csv").exists()
    assert (out_dir / "summary.md").exists()
    csv_text = (out_dir / "results.csv").read_text(encoding="utf-8")
    assert "variant" in csv_text
    assert "baseline" in csv_text
    assert "resolved_policy_toml" in csv_text
