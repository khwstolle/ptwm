"""ptwm ext sign — sign a bundle and rewrite manifest with real ids."""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path

import tomli_w

from ptwm._rust.trust import SecretKey


def sign_bundle(cwd: Path, key_path: Path) -> Path:
    """Compute a signature over manifest + binaries and write ``signature.bin``.

    Also rewrites ``manifest.toml`` so the ``[bundle].author_pubkey`` and the
    ``[[contributions]].id`` fields reflect the *real* key and canonical id
    derived from it.

    Parameters
    ----------
    cwd:
        Root directory of the extension bundle (must contain
        ``manifest.toml``).
    key_path:
        Path to a 32-byte raw Ed25519 seed file.

    Returns
    -------
    Path
        Path to the written ``signature.bin``.
    """
    key = SecretKey.load(str(key_path))
    pubkey = key.public()

    manifest_path = cwd / "manifest.toml"
    manifest_bytes = manifest_path.read_bytes()
    manifest = tomllib.loads(manifest_bytes.decode("utf-8"))

    # Rewrite author_pubkey to the real key.
    manifest["bundle"]["author_pubkey"] = "ed25519:" + pubkey.hex()

    # For each contribution recompute canonical_id from
    # blake3(pubkey || \0 || name || \0 || version).
    bundle_name = manifest["bundle"]["name"]
    bundle_version = manifest["bundle"]["version"]
    for c in manifest.get("contributions", []):
        c["id"] = "blake3:" + _canonical_id_hex(pubkey, bundle_name, bundle_version)

    # Write back the manifest so the signed bytes include the real ids.
    new_manifest_bytes = tomli_w.dumps(manifest).encode("utf-8")
    manifest_path.write_bytes(new_manifest_bytes)

    # Sign new_manifest_bytes || each_binary_bytes_in_declared_order.
    binaries = _collect_binary_paths(cwd, manifest)
    h = _blake3_hasher()
    h.update(new_manifest_bytes)
    for p in binaries:
        h.update(p.read_bytes())
    digest = h.digest()
    signature = key.sign(digest)

    sig_path = cwd / "signature.bin"
    sig_path.write_bytes(signature)
    return sig_path


def _blake3_hasher():  # type: ignore[return]
    """Return a BLAKE3 hasher object with an ``update``/``digest`` interface."""
    if hasattr(hashlib, "blake3"):
        return hashlib.blake3()  # type: ignore[attr-defined]
    try:
        import blake3 as _b3

        return _b3.blake3()
    except ImportError as e:
        msg = (
            "blake3 hashing not available — install the `blake3` Python "
            "package or upgrade to a Python build that includes it"
        )
        raise RuntimeError(msg) from e


def _canonical_id_hex(pubkey: object, name: str, version: str) -> str:
    r"""Compute the 64-char hex canonical id for a bundle.

    The derivation mirrors the Rust side:
    ``blake3(pubkey_bytes || \x00 || name_utf8 || \x00 || version_utf8)``.

    Parameters
    ----------
    pubkey:
        A ``PublicKey`` object (has a ``.hex()`` method returning 64 hex
        chars for the 32 raw bytes).
    name:
        Bundle name string.
    version:
        Bundle version string.
    """
    h = _blake3_hasher()
    h.update(bytes.fromhex(pubkey.hex()))  # type: ignore[union-attr]
    h.update(b"\x00")
    h.update(name.encode("utf-8"))
    h.update(b"\x00")
    h.update(version.encode("utf-8"))
    return h.hexdigest()


def _collect_binary_paths(cwd: Path, manifest: dict) -> list[Path]:  # type: ignore[type-arg]
    """Return paths to existing binaries in the bundle directory."""
    bundle_name = manifest["bundle"]["name"]
    candidates = [
        cwd / f"{bundle_name}.wasm",
        cwd / f"{bundle_name}.so",
        cwd / f"{bundle_name}.dylib",
    ]
    return [p for p in candidates if p.exists()]
