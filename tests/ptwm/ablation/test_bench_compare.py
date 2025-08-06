"""ptwm bench compare smoke tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def isolated_xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    return tmp_path


def test_bench_compare_identical_files(isolated_xdg: Path, tmp_path: Path) -> None:
    from ptwm.ablation import bench_compare, bench_compress

    input_path = tmp_path / "input.bin"
    input_path.write_bytes(b"\x00" * 4096)
    out = tmp_path / "out"
    res = bench_compress(input_path, None, [], ["ratio"], 1, out)
    file_a = res.rows[0].out_file
    file_b = file_a  # same file

    cmp = bench_compare(file_a, file_b)
    assert cmp.size_a == cmp.size_b
    assert cmp.only_in_a == []
    assert cmp.only_in_b == []


def test_bench_compare_result_fields(isolated_xdg: Path, tmp_path: Path) -> None:
    from ptwm.ablation import bench_compare, bench_compress

    input_path = tmp_path / "input.bin"
    input_path.write_bytes(b"\xff" * 2048)
    out = tmp_path / "out"
    res = bench_compress(input_path, None, [], ["ratio"], 1, out)
    ptwm_file = res.rows[0].out_file

    cmp = bench_compare(ptwm_file, ptwm_file)
    # All three lists must be lists (may be empty).
    assert isinstance(cmp.only_in_a, list)
    assert isinstance(cmp.only_in_b, list)
    assert isinstance(cmp.common, list)
    # Self-comparison must have zero unique entries on either side.
    assert cmp.only_in_a == []
    assert cmp.only_in_b == []
    # Size must match file size on disk.
    assert cmp.size_a == Path(ptwm_file).stat().st_size
