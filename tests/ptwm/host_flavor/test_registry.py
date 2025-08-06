"""Tests for the host-flavor contribution registry."""

from __future__ import annotations

import pytest


def _registry():
    from ptwm._rust.host import ContributionRegistry

    return ContributionRegistry()


def test_register_and_count_round_trip() -> None:
    reg = _registry()
    reg.clear()
    reg.register(
        "blake3:" + "11" * 32,
        "classifier",
        "process",
        lambda *args: None,
    )
    assert reg.count() == 1


def test_register_rejects_unknown_kind() -> None:
    reg = _registry()
    reg.clear()
    with pytest.raises(ValueError, match="unknown kind"):
        reg.register("blake3:" + "11" * 32, "not_a_kind", "thread", lambda: None)


def test_register_rejects_unknown_lifecycle() -> None:
    reg = _registry()
    reg.clear()
    with pytest.raises(ValueError, match="unknown lifecycle"):
        reg.register(
            "blake3:" + "11" * 32, "classifier", "not_a_lifecycle", lambda: None
        )


def test_host_registry_entries_lists_registered() -> None:
    from ptwm._rust.host import host_registry_entries

    reg = _registry()
    reg.clear()
    reg.register("blake3:" + "22" * 32, "classifier", "thread", lambda: None)
    entries = host_registry_entries()
    assert any(
        cid == "22" * 32 and kind == "classifier" and lc == "thread"
        for cid, kind, lc in entries
    )
