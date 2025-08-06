"""Smoke test for pack_bundle (build/sign require real toolchains)."""

from __future__ import annotations

from pathlib import Path


def test_pack_archives_manifest_when_present(tmp_path: Path) -> None:
    from ptwm.ext_tooling.pack import pack_bundle

    bundle = tmp_path / "ext"
    bundle.mkdir()
    (bundle / "manifest.toml").write_text(
        '[bundle]\nname = "pack-test"\nversion = "0.1.0"\n'
        'author_pubkey = "ed25519:00"\n',
        encoding="utf-8",
    )
    out = pack_bundle(bundle)
    assert out.exists()
    assert out.name == "pack-test-0.1.0.tar.zst"
