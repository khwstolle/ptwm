"""Tests for `ptwm.ext_tooling.sign`."""

from __future__ import annotations

import secrets
import tomllib
from pathlib import Path

import pytest
from ptwm._rust.trust import SecretKey
from ptwm.ext_tooling.sign import (
    _blake3_hasher,
    _canonical_id_hex,
    _collect_binary_paths,
    sign_bundle,
)

_MANIFEST = """\
[bundle]
name = "demo"
version = "0.1.0"
author_pubkey = "ed25519:placeholder"

[[contributions]]
kind = "plane_codec"
abi_version = 1
flavors = ["wasm"]
lifecycle = "none"
label = "demo.codec"
id = "blake3:placeholder"
"""


@pytest.fixture
def bundle_dir(tmp_path: Path) -> Path:
    (tmp_path / "manifest.toml").write_text(_MANIFEST, encoding="utf-8")
    return tmp_path


def _write_secret_key(path: Path) -> Path:
    path.write_bytes(secrets.token_bytes(32))
    return path


def test_canonical_id_hex_is_64_chars(bundle_dir: Path) -> None:
    seed = _write_secret_key(bundle_dir / "key.bin")
    sk = SecretKey.load(str(seed))
    digest = _canonical_id_hex(sk.public(), "demo", "0.1.0")
    assert len(digest) == 64
    int(digest, 16)  # parses as hex


def test_canonical_id_hex_changes_with_name(bundle_dir: Path) -> None:
    seed = _write_secret_key(bundle_dir / "key.bin")
    sk = SecretKey.load(str(seed))
    pk = sk.public()
    assert _canonical_id_hex(pk, "demo", "0.1.0") != _canonical_id_hex(
        pk, "other", "0.1.0"
    )


def test_blake3_hasher_produces_32_byte_digest() -> None:
    h = _blake3_hasher()
    h.update(b"hello world")
    assert len(h.digest()) == 32


def test_collect_binary_paths_picks_existing_only(bundle_dir: Path) -> None:
    (bundle_dir / "demo.wasm").write_bytes(b"\x00asm fake")
    manifest = tomllib.loads((bundle_dir / "manifest.toml").read_text())
    paths = _collect_binary_paths(bundle_dir, manifest)
    assert paths == [bundle_dir / "demo.wasm"]


def test_collect_binary_paths_empty_when_no_binaries(bundle_dir: Path) -> None:
    manifest = tomllib.loads((bundle_dir / "manifest.toml").read_text())
    assert _collect_binary_paths(bundle_dir, manifest) == []


def test_sign_bundle_writes_signature_and_rewrites_ids(bundle_dir: Path) -> None:
    seed = _write_secret_key(bundle_dir / "key.bin")
    (bundle_dir / "demo.wasm").write_bytes(b"\x00asm fake")

    sig_path = sign_bundle(bundle_dir, seed)
    assert sig_path == bundle_dir / "signature.bin"
    assert sig_path.exists()
    # Ed25519 signatures are 64 bytes.
    assert len(sig_path.read_bytes()) == 64

    rewritten = tomllib.loads((bundle_dir / "manifest.toml").read_text())
    assert rewritten["bundle"]["author_pubkey"].startswith("ed25519:")
    # 64 hex chars after the prefix.
    assert len(rewritten["bundle"]["author_pubkey"]) == len("ed25519:") + 64
    assert rewritten["contributions"][0]["id"].startswith("blake3:")
    assert len(rewritten["contributions"][0]["id"]) == len("blake3:") + 64
