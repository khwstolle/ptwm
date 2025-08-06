"""Smoke tests for `ptwm._loader.load_all_host_extensions`."""

from __future__ import annotations

from importlib.metadata import EntryPoint
from unittest.mock import patch

import pytest
from ptwm._loader import discover_host_extensions, load_all_host_extensions


def test_discover_returns_iterable() -> None:
    out = list(discover_host_extensions())
    assert isinstance(out, list)
    for name, ep in out:
        assert isinstance(name, str)
        assert isinstance(ep, EntryPoint)


def test_load_all_with_no_entry_points_returns_zero() -> None:
    with patch("ptwm._loader.entry_points") as ep_fn:
        ep_fn.return_value = []
        assert load_all_host_extensions() == 0


def test_load_all_swallows_load_failures(
    capsys: pytest.CaptureFixture[str],
) -> None:
    bad_ep = EntryPoint(
        name="broken", value="nonexistent.module:fn", group="ptwm.extensions"
    )
    with patch("ptwm._loader.entry_points") as ep_fn:
        ep_fn.return_value = [bad_ep]
        loaded = load_all_host_extensions()
    assert loaded == 0
    err = capsys.readouterr().err
    assert "broken" in err
    assert "failed to load" in err


def test_load_all_counts_successful_registrations() -> None:
    register_calls = []

    def fake_register() -> None:
        register_calls.append(1)

    class FakeEP:
        name = "good"

        def load(self) -> object:
            return fake_register

    with patch("ptwm._loader.entry_points") as ep_fn:
        ep_fn.return_value = [FakeEP()]
        assert load_all_host_extensions() == 1
    assert register_calls == [1]


def test_load_all_swallows_register_failures(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def boom() -> None:
        msg = "explode"
        raise RuntimeError(msg)

    class FakeEP:
        name = "boomer"

        def load(self) -> object:
            return boom

    with patch("ptwm._loader.entry_points") as ep_fn:
        ep_fn.return_value = [FakeEP()]
        loaded = load_all_host_extensions()
    assert loaded == 0
    err = capsys.readouterr().err
    assert "boomer" in err
    assert "failed to register" in err
