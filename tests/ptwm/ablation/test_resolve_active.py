"""Tests for ablation.resolve_active."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def isolated_xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    return tmp_path


def test_default_empty_resolves_with_no_trust(isolated_xdg: Path) -> None:
    from ptwm.ablation import resolve_active

    resolved = resolve_active(None)
    assert resolved.allowed_count() == 0
    assert resolved.effective_global_count() == 0


def test_pinned_contribution_appears_in_allowed(isolated_xdg: Path) -> None:
    from ptwm._rust.trust import Keyring, trust_user_keys_path
    from ptwm.ablation import resolve_active

    pin_id = "blake3:" + "11" * 32
    path = trust_user_keys_path()
    k = Keyring.load(path)
    k.add_contribution_hash(pin_id, "pin-for-test")
    k.save()

    resolved = resolve_active(None)
    assert resolved.allowed_count() >= 1


def test_resolve_active_with_policy_file(isolated_xdg: Path, tmp_path: Path) -> None:
    from ptwm.ablation import resolve_active

    pol = tmp_path / "p.toml"
    pol.write_text(
        '[allow]\nextra = ["blake3:' + "22" * 32 + '"]\n',
        encoding="utf-8",
    )
    resolved = resolve_active(str(pol))
    assert resolved.allowed_count() == 1
