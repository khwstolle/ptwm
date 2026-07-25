"""ptwm ext pack — bundle into <name>-<version>.tar.zst."""

from __future__ import annotations

import io
import tarfile
import tomllib
from pathlib import Path


def pack_bundle(cwd: Path) -> Path:
    """Create a ``<name>-<version>.tar.zst`` archive from the bundle in *cwd*.

    The archive contains ``manifest.toml``, ``signature.bin`` (if present),
    and whichever built binaries exist: ``<name>.wasm``, ``<name>.so``,
    ``<name>.dylib``.

    Parameters
    ----------
    cwd:
        Root directory of the extension bundle.

    Returns
    -------
    Path
        Path to the written ``.tar.zst`` archive (placed inside *cwd*).

    Raises
    ------
    ImportError
        When ``zstandard`` is not installed.  Install it via
        ``pip install ptwm[ext-dev]``.
    """
    try:
        import zstandard as zstd
    except ImportError as e:
        msg = (
            "zstandard is required for ptwm ext pack — "
            "install it via 'pip install ptwm[ext-dev]'"
        )
        raise ImportError(msg) from e

    src = (cwd / "manifest.toml").read_text(encoding="utf-8")
    manifest = tomllib.loads(src)
    name = manifest["bundle"]["name"]
    version = manifest["bundle"]["version"]
    archive_path = cwd / f"{name}-{version}.tar.zst"

    # Build the tar in memory, then zstd-encode.
    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w") as tf:
        members = (
            "manifest.toml",
            "signature.bin",
            f"{name}.wasm",
            f"{name}.so",
            f"{name}.dylib",
        )
        for member in members:
            p = cwd / member
            if p.exists():
                tf.add(p, arcname=member)
    cctx = zstd.ZstdCompressor()
    archive_path.write_bytes(cctx.compress(tar_buf.getvalue()))
    return archive_path
