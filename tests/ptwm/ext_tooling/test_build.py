"""Tests for `ptwm.ext_tooling.build`."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from ptwm.ext_tooling.build import (
    BuildError,
    _manifest_name,
    build_extension,
    detect_language,
)

_MANIFEST = """\
[bundle]
name = "demo"
version = "0.1.0"
author_pubkey = "ed25519:0000"
"""


def _write_manifest(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    (target / "manifest.toml").write_text(_MANIFEST, encoding="utf-8")


def test_detect_rust(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    (tmp_path / "Cargo.toml").write_text('[package]\nname = "demo"\n')
    assert detect_language(tmp_path) == "rust"


def test_detect_zig(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    (tmp_path / "build.zig").write_text("// zig\n")
    assert detect_language(tmp_path) == "zig"


def test_detect_c(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    (tmp_path / "Makefile").write_text("all:\n\t@true\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "lib.c").write_text("/* c */\n")
    assert detect_language(tmp_path) == "c"


def test_detect_assemblyscript(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    (tmp_path / "package.json").write_text("{}\n")
    assert detect_language(tmp_path) == "assemblyscript"


def test_detect_raises_on_empty(tmp_path: Path) -> None:
    with pytest.raises(BuildError, match="could not detect"):
        detect_language(tmp_path)


def test_manifest_name_round_trip(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    assert _manifest_name(tmp_path) == "demo"


def test_build_extension_propagates_detection_error(tmp_path: Path) -> None:
    with pytest.raises(BuildError):
        build_extension(tmp_path)


def test_build_rust_invokes_cargo_and_copies_wasm(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    (tmp_path / "Cargo.toml").write_text('[package]\nname = "demo"\n')

    # Pre-stage a fake cargo output so the copy step finds something.
    target_dir = tmp_path / "target" / "wasm32-wasip1" / "release"
    target_dir.mkdir(parents=True)
    (target_dir / "demo.wasm").write_bytes(b"\x00asm fake")

    with patch("ptwm.ext_tooling.build._run") as runner:
        out = build_extension(tmp_path, release=True)
    assert runner.call_count == 1
    cmd = runner.call_args.args[0]
    assert cmd[0] == "cargo"
    assert "--release" in cmd
    assert out == tmp_path / "demo.wasm"
    assert out.read_bytes() == b"\x00asm fake"


def test_build_extension_dispatches_to_zig(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    (tmp_path / "build.zig").write_text("// zig\n")
    out_dir = tmp_path / "zig-out" / "bin"
    out_dir.mkdir(parents=True)
    (out_dir / "anything.wasm").write_bytes(b"\x00asm zig")

    with patch("ptwm.ext_tooling.build._run") as runner:
        out = build_extension(tmp_path, release=False)
    cmd = runner.call_args.args[0]
    assert cmd[0] == "zig"
    assert out == tmp_path / "demo.wasm"


def test_build_raises_when_subprocess_fails(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    (tmp_path / "Cargo.toml").write_text('[package]\nname="demo"\n')
    with patch("ptwm.ext_tooling.build.subprocess.run") as run:
        # Mimic a failing cargo invocation.
        proc = type(
            "P",
            (),
            {"returncode": 101, "stderr": "boom"},
        )()
        run.return_value = proc
        with pytest.raises(BuildError, match="boom"):
            build_extension(tmp_path)
