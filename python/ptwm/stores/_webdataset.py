"""WebDataset storage integration (``ptwm-wds``).

A *native* tar-shard format: no nested ``.ptwm`` container appears in any
archive. Each tensor becomes one WebDataset sample whose members are the
per-tensor manifest JSON and its loose compressed plane payloads. Every shard
carries the archive registry as ``__shardmeta__.json`` so it is independently
decodable for shard-parallel streaming; a sibling ``index.json`` carries the
``weight_map`` for random / whole-model loads.

The ``webdataset`` package is only needed for the optional pipeline-streaming
helper; the writer and the core reader use the standard-library ``tarfile``.
"""

from __future__ import annotations

import io
import json
import tarfile
import threading
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from ..sharding import compress_ptwm_blob, plan_shards
from ._common import (
    ExplodedModel,
    decode_member_tensor,
    explode_model,
    jcs_canonicalize,
    raw_from_members,
    shared_member_keys,
    tensor_member_keys,
)

if TYPE_CHECKING:
    import torch

__all__ = [
    "WebDatasetWeightStore",
    "write_webdataset",
    "WDS_FORMAT",
    "SHARDMETA_MEMBER",
]

WDS_FORMAT = "ptwm-wds"
SHARDMETA_MEMBER = "__shardmeta__.json"
INDEX_NAME = "index.json"
_DEFAULT_MAX_SHARD_SIZE = 2 * 1024**3


def _shard_filename(idx_zero_based: int, total: int) -> str:
    """Return ``weights-{NNNNNN}-of-{MMMMMM}.tar`` (zero-based, brace-friendly)."""
    return f"weights-{idx_zero_based:06d}-of-{total:06d}.tar"


def _add_member(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(data))


# ── Writer ────────────────────────────────────────────────────────────────────


def write_webdataset(
    out_dir: str | Path,
    tensors: Sequence[tuple[str, bytes, Any, tuple[int, ...]]],
    *,
    max_shard_size: int = _DEFAULT_MAX_SHARD_SIZE,
    method_hint: int = 3,
    chains_per_tensor: list[list[bytes]] | None = None,
) -> Path:
    """Write ``tensors`` as a WebDataset shard set rooted at ``out_dir``.

    Returns the directory path. ``tensors`` is a sequence of
    ``(name, raw_bytes, dtype, shape)``; shards are packed by compressed size
    via :func:`ptwm.sharding.plan_shards` so a tensor never straddles a shard.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    blob = compress_ptwm_blob(
        tensors, chains_per_tensor=chains_per_tensor, method_hint=method_hint
    )
    model = explode_model(blob)
    _write_exploded(out_dir, model, max(1, max_shard_size))
    return out_dir


def _write_exploded(out_dir: Path, model: ExplodedModel, max_shard_size: int) -> None:
    shared = shared_member_keys(model.members)
    sized = [(name, model.member_size(name)) for name in model.names()]
    shards = plan_shards(sized, max_shard_size=max_shard_size)
    total = len(shards)

    weight_map: dict[str, dict[str, str]] = {}
    total_size = 0
    for name in model.names():
        meta = json.loads(model.tensor_json(name))
        total_size += int(meta.get("orig_size", 0))
    compressed_total = sum(len(v) for v in model.members.values())

    for shard_idx, shard in enumerate(shards):
        shard_name = _shard_filename(shard_idx, total)
        with tarfile.open(out_dir / shard_name, "w") as tar:
            # Per-shard registry superset first, then any shared state.
            _add_member(tar, SHARDMETA_MEMBER, model.registry_json.encode("utf-8"))
            for k, v in shared.items():
                _add_member(tar, k, v)
            for name, _size in shard:
                key = model.key(name)
                _add_member(tar, f"{key}.json", model.tensor_json(name).encode("utf-8"))
                for member_key, data in tensor_member_keys(model.members, key).items():
                    _add_member(tar, member_key, data)
                weight_map[name] = {"shard": shard_name, "key": key}

    index = {
        "ptwm_format": WDS_FORMAT,
        "manifest_version": 1,
        "registry": json.loads(model.registry_json),
        "weight_map": weight_map,
        "total_size": total_size,
        "compressed_total_size": compressed_total,
    }
    (out_dir / INDEX_NAME).write_text(jcs_canonicalize(index))


# ── Reader ────────────────────────────────────────────────────────────────────


class WebDatasetWeightStore:
    """Read API over a WebDataset shard set (or a single shard for streaming)."""

    def __init__(self, path: str | Path) -> None:
        self._root = Path(path)
        # ``tarfile.TarFile`` is not thread-safe; serialize access to the
        # cached handles so concurrent readers (e.g. DataLoader workers) cannot
        # corrupt a shared handle's seek state.
        self._lock = threading.Lock()
        # Cache open tar handles (member *metadata* only — payloads are read on
        # demand via ``extractfile``, never buffered whole-shard).
        self._open_tars: dict[str, tarfile.TarFile] = {}

        if self._root.is_dir():
            index_path = self._root / INDEX_NAME
            if not index_path.exists():
                msg = f"{self._root} has no {INDEX_NAME}; not a WebDataset store"
                raise FileNotFoundError(msg)
            index = json.loads(index_path.read_text())
            if index.get("ptwm_format") != WDS_FORMAT:
                msg = f"{index_path}: not a {WDS_FORMAT} index"
                raise ValueError(msg)
            self._registry_json = jcs_canonicalize(index["registry"])
            self._weight_map: dict[str, dict[str, str]] = dict(index["weight_map"])
        else:
            # Single shard: stream-only, registry + map built from the (small)
            # JSON members; payloads are not read here.
            with self._lock:
                tar = self._tar(self._root)
                self._registry_json = self._extract(tar, SHARDMETA_MEMBER).decode(
                    "utf-8"
                )
                self._weight_map = {}
                for member_key in tar.getnames():
                    if member_key.endswith(".json") and member_key != SHARDMETA_MEMBER:
                        meta = json.loads(self._extract(tar, member_key))
                        self._weight_map[meta["name"]] = {
                            "shard": self._root.name,
                            "key": member_key[: -len(".json")],
                        }
        self._names = list(self._weight_map.keys())

    @classmethod
    def open(cls, path: str | Path) -> WebDatasetWeightStore:
        """Open a WebDataset store directory (or a single shard)."""
        return cls(path)

    def close(self) -> None:
        """Close any open shard tar handles."""
        with self._lock:
            for tar in self._open_tars.values():
                tar.close()
            self._open_tars.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- internals ----------------------------------------------------------

    def _tar(self, shard_path: Path) -> tarfile.TarFile:
        key = str(shard_path)
        tar = self._open_tars.get(key)
        if tar is None:
            # Long-lived handle: cached for on-demand member reads and closed in
            # ``close()`` / ``__exit__`` (so a context manager isn't applicable).
            tar = tarfile.open(shard_path, "r")  # noqa: SIM115
            self._open_tars[key] = tar
        return tar

    @staticmethod
    def _extract(tar: tarfile.TarFile, member_name: str) -> bytes:
        f = tar.extractfile(member_name)
        if f is None:
            msg = f"member {member_name!r} not found in shard"
            raise KeyError(msg)
        return f.read()

    def _members_for(self, name: str) -> tuple[str, str, dict[str, bytes]]:
        entry = self._weight_map[name]
        shard_path = self._root if self._root.is_file() else self._root / entry["shard"]
        key = entry["key"]
        with self._lock:
            tar = self._tar(shard_path)
            names = tar.getnames()
            registry_json = self._registry_json
            if SHARDMETA_MEMBER in names:
                registry_json = self._extract(tar, SHARDMETA_MEMBER).decode("utf-8")
            tensor_json = self._extract(tar, f"{key}.json").decode("utf-8")
            # Load only this tensor's plane/state members plus any archive-global
            # shared state — never the whole shard.
            members = {
                n: self._extract(tar, n)
                for n in names
                if (n.startswith(f"{key}.") and not n.endswith(".json"))
                or n.startswith(("shared/", "modules/"))
            }
        return registry_json, tensor_json, members

    # -- WeightStore API ----------------------------------------------------

    def names(self) -> list[str]:
        """Return tensor names in storage order."""
        return list(self._names)

    def get_bytes(self, name: str) -> bytes:
        """Return the raw decompressed bytes for ``name``."""
        if name not in self._weight_map:
            raise KeyError(name)
        registry_json, tensor_json, members = self._members_for(name)
        return raw_from_members(registry_json, tensor_json, members)

    def get_tensor(self, name: str) -> torch.Tensor:
        """Decompress ``name`` and return a :class:`torch.Tensor`."""
        if name not in self._weight_map:
            raise KeyError(name)
        registry_json, tensor_json, members = self._members_for(name)
        return decode_member_tensor(registry_json, tensor_json, members)

    def stream_tensors(
        self, names: Iterable[str] | None = None
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield ``(name, tensor)`` pairs, reading shards sequentially.

        Pure streaming: each shard is read once, its samples grouped and decoded
        via the per-shard ``__shardmeta__.json`` registry. No global index or
        container is materialized.
        """
        wanted = set(names) if names is not None else None
        for shard_path in self._iter_shard_paths():
            # Hold the lock only for the (non-thread-safe) tar reads; release it
            # during the CPU-heavy decode so other readers can make progress.
            with self._lock:
                tar = self._tar(shard_path)
                member_names = tar.getnames()
                registry_json = self._registry_json
                if SHARDMETA_MEMBER in member_names:
                    registry_json = self._extract(tar, SHARDMETA_MEMBER).decode("utf-8")
                shared = {
                    n: self._extract(tar, n)
                    for n in member_names
                    if n.startswith(("shared/", "modules/"))
                }
            for member_key in member_names:
                if not member_key.endswith(".json") or member_key == SHARDMETA_MEMBER:
                    continue
                key = member_key[: -len(".json")]
                with self._lock:
                    tensor_json = self._extract(tar, member_key).decode("utf-8")
                name = json.loads(tensor_json)["name"]
                if wanted is not None and name not in wanted:
                    continue
                # Load just this tensor's plane members (+ shared) — not the shard.
                with self._lock:
                    sample = {
                        n: self._extract(tar, n)
                        for n in member_names
                        if n.startswith(f"{key}.") and not n.endswith(".json")
                    }
                sample.update(shared)
                yield name, decode_member_tensor(registry_json, tensor_json, sample)

    def webdataset_pipeline(self) -> Any:
        """Return a ``webdataset.WebDataset`` over the shard URLs.

        Lazily imports the optional ``webdataset`` package (``pip install
        'ptwm[webdataset]'``). Each yielded item is ``(name, tensor)``.
        """
        try:
            import webdataset as wds  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - optional dependency
            msg = (
                "the 'webdataset' package is required for webdataset_pipeline(); "
                "install it with: pip install 'ptwm[webdataset]'"
            )
            raise ImportError(msg) from exc

        registry_json = self._registry_json
        urls = [str(p) for p in self._iter_shard_paths()]

        def _decode(sample: dict[str, Any]) -> tuple[str, Any]:
            tensor_json = sample["json"].decode("utf-8")
            meta = json.loads(tensor_json)
            members = {
                k: v for k, v in sample.items() if isinstance(v, bytes | bytearray)
            }
            # Reconstruct the per-plane member keys from the WebDataset sample.
            key = sample["__key__"]
            named = {f"{key}.{k}": v for k, v in members.items() if k != "json"}
            return meta["name"], decode_member_tensor(registry_json, tensor_json, named)

        return wds.WebDataset(urls).map(_decode)

    def _iter_shard_paths(self) -> list[Path]:
        if self._root.is_file():
            return [self._root]
        return sorted(self._root.glob("weights-*-of-*.tar"))
