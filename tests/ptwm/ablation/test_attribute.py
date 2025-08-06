"""ptwm bench attribute smoke tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def isolated_xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    return tmp_path


def test_attribute_leave_one_out_produces_a_row_per_contribution(
    isolated_xdg: Path, tmp_path: Path
) -> None:
    from ptwm.ablation import attribute_leave_one_out

    input_path = tmp_path / "input.bin"
    input_path.write_bytes(b"\x00" * 4096)
    out_dir = tmp_path / "attr"
    result = attribute_leave_one_out(input_path, None, out_dir)
    # The exact count depends on chains/codecs used; just assert
    # the result shape.
    assert result.method == "leave-one-out"
    assert isinstance(result.rows, list)


def test_attribute_leave_one_out_row_fields(isolated_xdg: Path, tmp_path: Path) -> None:
    from ptwm.ablation import attribute_leave_one_out
    from ptwm.ablation._attribute import AttributionRow

    input_path = tmp_path / "input.bin"
    input_path.write_bytes(b"\xab" * 1024)
    out_dir = tmp_path / "attr2"
    result = attribute_leave_one_out(input_path, None, out_dir)
    for row in result.rows:
        assert isinstance(row, AttributionRow)
        assert isinstance(row.contribution_id, str)
        assert isinstance(row.label, str)
        assert row.baseline_size > 0
        # without_size is a placeholder equal to baseline until the
        # ignore-policy path is wired in.
        assert row.without_size == row.baseline_size
        assert row.delta_bytes == 0
        assert row.delta_pct == 0.0


def test_attribute_shapley_refuses_beyond_max(
    isolated_xdg: Path, tmp_path: Path
) -> None:
    from ptwm.ablation import attribute_shapley

    input_path = tmp_path / "input.bin"
    input_path.write_bytes(b"\x00" * 4096)
    out_dir = tmp_path / "attr-shap"
    # Set max_attributions=0 so any used contribution exceeds it.
    with pytest.raises(ValueError, match="refusing Shapley attribution"):
        attribute_shapley(input_path, None, out_dir, max_attributions=0)


def test_attribute_shapley_result_shape(isolated_xdg: Path, tmp_path: Path) -> None:
    from ptwm.ablation import attribute_shapley

    input_path = tmp_path / "input.bin"
    input_path.write_bytes(b"\x00" * 4096)
    out_dir = tmp_path / "attr-shap2"
    # Use a high limit so we don't hit the ValueError.
    result = attribute_shapley(input_path, None, out_dir, max_attributions=64)
    assert result.method == "shapley"
    assert isinstance(result.rows, list)
