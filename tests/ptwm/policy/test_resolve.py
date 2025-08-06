"""Tests for the ptwm policy CLI surface."""

from __future__ import annotations

from pathlib import Path

from ptwm._rust.policy import PolicyFile


def test_default_empty_resolves(tmp_path: Path) -> None:
    pf = PolicyFile.default_empty()
    resolved = pf.resolve([])
    assert resolved.allowed_count() == 0
    assert resolved.effective_global_count() == 0
    s = resolved.to_toml()
    assert "allowed" in s


def test_load_with_allow_extra(tmp_path: Path) -> None:
    pol = tmp_path / "p.toml"
    pol.write_text(
        '[allow]\nextra = ["blake3:' + "11" * 32 + '"]\n',
        encoding="utf-8",
    )
    pf = PolicyFile.load(str(pol))
    resolved = pf.resolve([])
    assert resolved.allowed_count() == 1
    assert resolved.effective_global_count() == 1
