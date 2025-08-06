"""End-to-end tests for the ptwm trust CLI surface."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def isolated_xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point XDG_CONFIG_HOME at a clean tempdir for the duration of one test."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    return tmp_path


def test_fresh_install_auto_trusts_bundled(isolated_xdg: Path) -> None:
    from ptwm._rust import trust as _trust

    status = _trust.trust_evaluate_bundled()
    assert status.kind == "FreshInstall"

    _trust.trust_apply_fresh_install()
    status2 = _trust.trust_evaluate_bundled()
    assert status2.kind == "Match"


def test_user_keyring_add_and_remove(isolated_xdg: Path) -> None:
    from ptwm._rust import trust as _trust

    path = _trust.trust_user_keys_path()
    k = _trust.Keyring.load(path)
    k.add_author_key("ed25519:" + "11" * 32, "alice")
    k.save()

    loaded = _trust.Keyring.load(path)
    assert len(loaded.entries()) == 1
    e = loaded.entries()[0]
    assert e.kind == "author_key"
    assert e.label == "alice"
    assert e.pubkey == "ed25519:" + "11" * 32

    removed = loaded.remove("alice")
    assert removed
    loaded.save()

    reread = _trust.Keyring.load(path)
    assert len(reread.entries()) == 0
