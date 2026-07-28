"""Tests for ptwm ext init."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_init_rust_scaffolds_expected_files(tmp_path: Path) -> None:
    from ptwm.ext_tooling.init import init_extension

    target = tmp_path / "demo-ext"
    init_extension(
        target_dir=target,
        name="demo-ext",
        lang="rust",
        kind="plane_codec",
    )
    assert (target / "Cargo.toml").exists()
    assert (target / "src" / "lib.rs").exists()
    assert (target / "manifest.toml").exists()

    cargo = (target / "Cargo.toml").read_text(encoding="utf-8")
    assert 'name = "demo-ext"' in cargo

    lib = (target / "src" / "lib.rs").read_text(encoding="utf-8")
    assert "ptwm_plane_codec_v1_encode" in lib


def test_init_c_scaffolds_makefile(tmp_path: Path) -> None:
    from ptwm.ext_tooling.init import init_extension

    target = tmp_path / "c-demo"
    init_extension(target_dir=target, name="c-demo", lang="c", kind="transform")
    assert (target / "Makefile").exists()
    assert (target / "src" / "lib.c").exists()
    assert (target / "manifest.toml").exists()


def test_init_rejects_unknown_lang(tmp_path: Path) -> None:
    from ptwm.ext_tooling.init import init_extension

    target = tmp_path / "bogus"
    with pytest.raises(ValueError, match="unsupported --lang"):
        init_extension(target_dir=target, name="bogus", lang="cobol")


def test_init_assemblyscript(tmp_path: Path) -> None:
    from ptwm.ext_tooling.init import init_extension

    target = tmp_path / "as-demo"
    init_extension(
        target_dir=target, name="as-demo", lang="assemblyscript", kind="scorer"
    )
    assert (target / "package.json").exists()
    assert (target / "src" / "index.ts").exists()


def test_init_hardware_backend_scaffolds_correct_native_abi(tmp_path: Path) -> None:
    from ptwm.ext_tooling.init import init_extension

    target = tmp_path / "demo-hw"
    init_extension(
        target_dir=target,
        name="demo-hw",
        lang="rust",
        kind="hardware_backend",
        flavor="native",
    )
    lib = (target / "src" / "lib.rs").read_text(encoding="utf-8")
    assert "ptwm_hardware_backend_v1_dispatch_decode_cuda" in lib
    assert "ptwm_hardware_backend_v1_cuda_stream_handle" in lib
    # The generic template's wrong stateless 2-arg encode/decode shape
    # must NOT appear for this kind.
    assert "ptwm_hardware_backend_v1_encode" not in lib
