"""End-to-end smoke test for ptwm ext list / install (local-dir source)."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def isolated_extensions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point XDG_DATA_HOME at a clean tempdir for the duration of one test."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    return tmp_path


def _make_bundle(tmp_path: Path, name: str, version: str) -> Path:
    src = tmp_path / "src"
    src.mkdir(parents=True, exist_ok=True)
    (src / "manifest.toml").write_text(
        f'[bundle]\nname = "{name}"\nversion = "{version}"\n'
        f'author_pubkey = "ed25519:00000000"\n',
        encoding="utf-8",
    )
    return src


def test_install_local_dir_then_list(isolated_extensions: Path) -> None:
    from ptwm._rust.ext import ext_install, ext_list

    src = _make_bundle(isolated_extensions, "demo", "0.1.0")
    bundle_dir = ext_install(str(src))
    assert bundle_dir
    assert Path(bundle_dir).exists()

    entries = ext_list()
    assert any(e.bundle_name == "demo" and e.bundle_version == "0.1.0" for e in entries)


@pytest.mark.slow
def test_install_rejects_unknown_source(isolated_extensions: Path) -> None:
    from ptwm._rust.ext import ext_install

    # `parse_source` falls through to Pip for unknown strings; this then
    # tries `python -m pip install <name>`. With an obviously-bad name
    # pip fails fast.
    with pytest.raises(
        ValueError, match="definitely-not-a-real-pypi-package-12345-xyz"
    ):
        ext_install("definitely-not-a-real-pypi-package-12345-xyz")
