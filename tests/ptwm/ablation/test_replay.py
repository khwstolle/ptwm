"""bench replay smoke test."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def isolated_xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    return tmp_path


def test_replay_round_trip(isolated_xdg: Path, tmp_path: Path) -> None:
    from ptwm.ablation import bench_compress, bench_replay

    input_path = tmp_path / "input.bin"
    input_path.write_bytes(b"\x00" * 1024)
    out = tmp_path / "bench-out"
    bench_compress(input_path, None, [], ["ratio"], 1, out)

    result = bench_replay(out / "results.csv", row=0)
    assert result.row_variant == "baseline"
    assert result.row_trial == 0
    # Synthetic-input replay should match within the v1 envelope.
    assert result.matches, (
        f"expected={result.expected_output_bytes} actual={result.actual_output_bytes}"
    )
