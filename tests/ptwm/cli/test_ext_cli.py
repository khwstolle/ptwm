"""End-to-end tests for the `ptwm ext` CLI surface."""

from __future__ import annotations

import argparse
import importlib
import tomllib
from pathlib import Path

import pytest
from ptwm.cli import ext as ext_cli
from ptwm.cli.ext import (
    add_ext_parser,
    cmd_init,
    cmd_install,
    cmd_list,
    cmd_pin,
    cmd_remove,
    cmd_show,
    cmd_verify,
)


@pytest.fixture
def isolated_xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Re-point PIN_FILE at a clean tempdir for one test."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    importlib.reload(ext_cli)
    return tmp_path


def _build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="cmd", required=True)
    add_ext_parser(sub)
    return root


def test_parser_registers_all_subcommands() -> None:
    root = _build_parser()
    for cmd, args in (
        ("install", ["./path"]),
        ("list", []),
        ("show", ["some-id"]),
        ("remove", ["some-id"]),
        ("verify", []),
        ("pin", ["some-id", "--to", "1.2.3"]),
    ):
        ns = root.parse_args(["ext", cmd, *args])
        assert ns.cmd == "ext"


def test_init_subcommand_requires_lang() -> None:
    root = _build_parser()
    with pytest.raises(SystemExit):
        root.parse_args(["ext", "init", "demo"])


def test_init_subcommand_rejects_unknown_lang() -> None:
    root = _build_parser()
    with pytest.raises(SystemExit):
        root.parse_args(["ext", "init", "demo", "--lang", "cobol"])


def test_cmd_list_with_no_extensions_prints_user_dir(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = cmd_list(argparse.Namespace(verbose=False))
    out = capsys.readouterr().out
    assert rc == 0
    assert "no extensions installed" in out


def test_cmd_show_with_no_match_returns_1(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = cmd_show(argparse.Namespace(identifier="bogus"))
    err = capsys.readouterr().err
    assert rc == 1
    assert "no extension matching" in err


def test_cmd_remove_with_no_match_returns_1(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = cmd_remove(argparse.Namespace(identifier="bogus"))
    err = capsys.readouterr().err
    assert rc == 1
    assert "no extension matching" in err


def test_cmd_verify_with_no_match_returns_1(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = cmd_verify(argparse.Namespace(identifier="bogus"))
    err = capsys.readouterr().err
    assert rc == 1
    assert "no extension matching" in err


def test_cmd_verify_without_filter_returns_0_when_empty(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # No installed extensions → loop over zero entries → returns 0.
    rc = cmd_verify(argparse.Namespace(identifier=None))
    assert rc == 0


def test_cmd_pin_writes_toml(isolated_xdg: Path) -> None:
    rc = cmd_pin(argparse.Namespace(identifier="demo", to="1.2.3"))
    assert rc == 0

    pin_file = ext_cli.PIN_FILE
    assert pin_file.exists()
    data = tomllib.loads(pin_file.read_text())
    assert data["demo"] == {"version": "1.2.3"}


def test_cmd_pin_updates_existing_entry(isolated_xdg: Path) -> None:
    cmd_pin(argparse.Namespace(identifier="demo", to="1.0.0"))
    cmd_pin(argparse.Namespace(identifier="demo", to="2.0.0"))

    data = tomllib.loads(ext_cli.PIN_FILE.read_text())
    assert data["demo"] == {"version": "2.0.0"}


def test_cmd_install_handles_failure_gracefully(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(_src: str) -> str:
        msg = "fake install failure"
        raise RuntimeError(msg)

    monkeypatch.setattr("ptwm.cli.ext.ext_install", boom)

    rc = cmd_install(argparse.Namespace(source="./does-not-exist"))
    err = capsys.readouterr().err
    assert rc == 1
    assert "install failed" in err
    assert "fake install failure" in err


def test_cmd_init_scaffolds_via_init_extension(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "demo-codec"
    rc = cmd_init(
        argparse.Namespace(
            name="demo-codec",
            lang="rust",
            kind="plane_codec",
            flavor="wasm",
            dir=target,
            description="A demo codec.",
        ),
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "scaffolded" in out
    assert (target / "Cargo.toml").exists()
    assert (target / "manifest.toml").exists()
