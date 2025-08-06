"""Pluggable weight-storage backends behind a shared :class:`WeightStore` API.

PTWM's binary container, an LMDB environment, and a WebDataset shard set are
three serializations of the same canonical PTWM-JSON manifest. This package
exposes them behind one protocol and an :func:`open_store` auto-detector.

The LMDB and WebDataset integrations are opt-in extras (``ptwm[lmdb]``,
``ptwm[webdataset]``) and their third-party dependencies are imported lazily.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

from ._common import (
    ExplodedModel,
    WeightStore,
    decode_member_tensor,
    explode_model,
    jcs_canonicalize,
    raw_from_members,
)
from ._lmdb import LmdbWeightStore, write_lmdb
from ._webdataset import (
    SHARDMETA_MEMBER,
    WDS_FORMAT,
    WebDatasetWeightStore,
    write_webdataset,
)

__all__ = [
    "WeightStore",
    "ExplodedModel",
    "explode_model",
    "decode_member_tensor",
    "raw_from_members",
    "jcs_canonicalize",
    "LmdbWeightStore",
    "write_lmdb",
    "WebDatasetWeightStore",
    "write_webdataset",
    "detect_format",
    "open_store",
]


def detect_format(path: str | Path) -> str | None:
    r"""Return ``"ptwm"`` / ``"lmdb"`` / ``"webdataset"`` for ``path`` or ``None``.

    Detection follows the resolved spec: WebDataset via
    ``index.json.ptwm_format == "ptwm-wds"`` (or a ``__shardmeta__.json`` member
    in a single shard); LMDB via the ``\\x00fmt`` key (an env directory contains
    ``data.mdb``); native ``.ptwm`` via the container magic.
    """
    p = Path(path)
    if p.is_dir():
        index = p / "index.json"
        if index.exists():
            try:
                obj = json.loads(index.read_text())
            except (ValueError, OSError):
                obj = {}
            if obj.get("ptwm_format") == WDS_FORMAT:
                return "webdataset"
        if (p / "data.mdb").exists():
            return "lmdb"
        # An LMDB env may also be a single-file (no-subdir) form; fall through.
        return None
    if p.is_file():
        if p.suffix == ".tar" or tarfile.is_tarfile(p):
            try:
                with tarfile.open(p, "r") as tar:
                    if any(m.name == SHARDMETA_MEMBER for m in tar.getmembers()):
                        return "webdataset"
            except (tarfile.TarError, OSError):
                pass
        try:
            with p.open("rb") as fh:
                if fh.read(5) == b"\x89PTWM":
                    return "ptwm"
        except OSError:
            pass
    return None


def open_store(path: str | Path) -> WeightStore:
    """Open ``path`` as a :class:`WeightStore`, auto-detecting the format."""
    fmt = detect_format(path)
    if fmt == "lmdb":
        return LmdbWeightStore.open(path)
    if fmt == "webdataset":
        return WebDatasetWeightStore.open(path)
    if fmt == "ptwm":
        from ..random_access import TensorIndex  # noqa: PLC0415

        return TensorIndex.open(path)
    msg = f"could not detect a PTWM storage format at {path}"
    raise ValueError(msg)
