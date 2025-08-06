"""End-to-end tests for the `ptwm trust` CLI surface."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from ptwm.cli.trust import (
    add_trust_parser,
    cmd_add,
    cmd_list,
    cmd_remove,
    cmd_show,
    cmd_update,
)


@pytest.fixture
def isolated_xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point XDG_CONFIG_HOME at a clean tempdir for the duration of one test."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    return tmp_path


def _build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="cmd", required=True)
    add_trust_parser(sub)
    return root


def test_parser_registers_all_subcommands() -> None:
    root = _build_parser()
    for cmd in ("list", "add", "update", "show", "remove"):
        # argparse raises SystemExit on parse error; success means OK.
        args = root.parse_args(
            ["trust", cmd] if cmd != "remove" else ["trust", cmd, "alice"]
        )
        assert args.cmd == "trust"


def test_cmd_list_on_fresh_install_prints_hint(
    isolated_xdg: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = cmd_list(argparse.Namespace())
    out = capsys.readouterr().out
    assert rc == 0
    assert "bundled keyring not yet imported" in out


def test_cmd_show_reports_status_and_path(
    isolated_xdg: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = cmd_show(argparse.Namespace())
    out = capsys.readouterr().out
    assert rc == 0
    assert "bundled status:" in out
    assert "user keyring path:" in out
    assert "user keyring exists:" in out


def test_cmd_add_bundled_on_fresh_install_imports(
    isolated_xdg: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = cmd_add(argparse.Namespace(bundled=True, key=None, pin=None, label=None))
    out = capsys.readouterr().out
    assert rc == 0
    assert "Bundled keyring imported" in out


def test_cmd_add_bundled_twice_reports_already_active(
    isolated_xdg: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cmd_add(argparse.Namespace(bundled=True, key=None, pin=None, label=None))
    capsys.readouterr()  # discard first output
    rc = cmd_add(argparse.Namespace(bundled=True, key=None, pin=None, label=None))
    out = capsys.readouterr().out
    assert rc == 0
    assert "already active" in out


def test_cmd_add_without_args_returns_error(
    isolated_xdg: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = cmd_add(argparse.Namespace(bundled=False, key=None, pin=None, label=None))
    err = capsys.readouterr().err
    assert rc == 2
    assert "--bundled" in err


def test_cmd_add_key_and_list_round_trip(
    isolated_xdg: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = cmd_add(
        argparse.Namespace(
            bundled=False,
            key="ed25519:" + "11" * 32,
            pin=None,
            label="alice",
        ),
    )
    assert rc == 0
    capsys.readouterr()

    rc = cmd_list(argparse.Namespace())
    out = capsys.readouterr().out
    assert rc == 0
    assert "alice" in out
    assert "ed25519:" + "11" * 32 in out


def test_cmd_remove_when_no_keyring_returns_2(
    isolated_xdg: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = cmd_remove(argparse.Namespace(identifier="alice"))
    out = capsys.readouterr().out
    assert rc == 2
    assert "no user keyring" in out


def test_cmd_remove_unknown_identifier_returns_1(
    isolated_xdg: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Seed the keyring with one entry so the file exists.
    cmd_add(
        argparse.Namespace(
            bundled=False,
            key="ed25519:" + "22" * 32,
            pin=None,
            label="bob",
        ),
    )
    capsys.readouterr()

    rc = cmd_remove(argparse.Namespace(identifier="does-not-exist"))
    err = capsys.readouterr().err
    assert rc == 1
    assert "no entry matched" in err


def test_cmd_update_without_bundled_flag_rejects(
    isolated_xdg: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = cmd_update(argparse.Namespace(bundled=False))
    err = capsys.readouterr().err
    assert rc == 2
    assert "--bundled" in err


def test_cmd_update_on_fresh_install_reports_missing_lock(
    isolated_xdg: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = cmd_update(argparse.Namespace(bundled=True))
    out = capsys.readouterr().out
    assert rc == 2
    assert "No bundled lock" in out
