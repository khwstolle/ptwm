"""LMDB storage integration (``ptwm-lmdb``).

One memory-mapped environment per model. Two layouts share the same canonical
manifest; **blob is the default** (matches LMDB's ``mmap``/atomic-cell idiom and
whole-tensor reads):

* **blob** — ``t/`` + name → a self-contained single-tensor ``.ptwm`` blob.
  Read is one ``get`` + the existing decoder; no new wire format.
* **exploded** — the canonical manifest spread across keys (registry, per-plane
  payloads, externalized state), enabling per-plane access and shared-codebook
  dedup at the cost of N+1 ``get`` calls per tensor.

LMDB write transactions make in-place per-tensor replacement first-class; a
tensor may be stored as an XOR delta/residual against another tensor already in
the environment (``local_tensor`` reference + BLAKE3 verification).

The ``lmdb`` package is required (``pip install 'ptwm[lmdb]'``); it is imported
lazily so the rest of PTWM has no hard dependency on it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, Self

import torch

from .. import _rust
from ..sharding import compress_ptwm_blob
from ..utils import DType, decode_ptwm_tensor, raw_to_tensor
from ._common import (
    decode_member_tensor,
    explode_model,
    jcs_canonicalize,
    raw_from_members,
)

__all__ = [
    "LmdbWeightStore",
    "write_lmdb",
    "LMDB_FORMAT",
    "KEY_FMT",
    "KEY_IDX",
    "KEY_REG",
]

LMDB_FORMAT = "ptwm-lmdb"
KEY_FMT = b"\x00fmt"
KEY_IDX = b"\x00idx"
KEY_REG = b"\x00reg"
_DEFAULT_MAP_SIZE = 8 * 1024**3


def _require_lmdb() -> Any:
    try:
        import lmdb  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - optional dependency
        msg = (
            "the 'lmdb' package is required for the LMDB store; "
            "install it with: pip install 'ptwm[lmdb]'"
        )
        raise ImportError(msg) from exc
    return lmdb


# ── Writer ────────────────────────────────────────────────────────────────────


def write_lmdb(
    path: str | Path,
    tensors: Sequence[tuple[str, bytes, Any, tuple[int, ...]]],
    *,
    layout: str = "blob",
    map_size: int = _DEFAULT_MAP_SIZE,
    method_hint: int = 3,
    chains_per_tensor: list[list[bytes]] | None = None,
) -> Path:
    """Write ``tensors`` into an LMDB environment at ``path``.

    ``layout`` is ``"blob"`` (default) or ``"exploded"``. Returns ``path``.
    """
    if layout not in ("blob", "exploded"):
        msg = f"unknown LMDB layout {layout!r} (expected 'blob' or 'exploded')"
        raise ValueError(msg)

    lmdb = _require_lmdb()
    path = Path(path)
    blob = compress_ptwm_blob(
        tensors, chains_per_tensor=chains_per_tensor, method_hint=method_hint
    )
    model = explode_model(blob)

    env = lmdb.open(str(path), map_size=map_size, subdir=True)
    try:
        with env.begin(write=True) as txn:
            txn.put(
                KEY_FMT,
                jcs_canonicalize(
                    {
                        "ptwm_format": LMDB_FORMAT,
                        "manifest_version": 1,
                        "layout": layout,
                    }
                ).encode("utf-8"),
            )
            if layout == "blob":
                _write_blob_layout(txn, model)
            else:
                _write_exploded_layout(txn, model)
            txn.put(KEY_IDX, _build_index(model, layout).encode("utf-8"))
    finally:
        env.close()
    return path


def _build_index(model, layout: str) -> str:  # noqa: ANN001
    weight_map: dict[str, dict[str, Any]] = {}
    total_size = 0
    compressed_total = 0
    for name in model.names():
        meta = json.loads(model.tensor_json(name))
        orig = int(meta.get("orig_size", 0))
        comp = model.member_size(name)
        total_size += orig
        compressed_total += comp
        weight_map[name] = {
            "dtype": meta.get("dtype"),
            "shape": meta.get("shape"),
            "orig_size": orig,
            "comp_size": comp,
        }
    return jcs_canonicalize(
        {
            "ptwm_format": LMDB_FORMAT,
            "manifest_version": 1,
            "layout": layout,
            "weight_map": weight_map,
            "total_size": total_size,
            "compressed_total_size": compressed_total,
        }
    )


def _write_blob_layout(txn: Any, model) -> None:  # noqa: ANN001
    shared = {
        k: v for k, v in model.members.items() if k.startswith(("shared/", "modules/"))
    }
    for name in model.names():
        key = model.key(name)
        members = {k: v for k, v in model.members.items() if k.startswith(f"{key}.")}
        members.update(shared)
        single = _rust.implode_ptwm(
            model.registry_json, [model.tensor_json(name)], members
        )
        txn.put(b"t/" + name.encode("utf-8"), single)


def _write_exploded_layout(txn: Any, model) -> None:  # noqa: ANN001
    txn.put(KEY_REG, model.registry_json.encode("utf-8"))
    reg = json.loads(model.registry_json)
    for entry in reg["registry"]["shared_state"]:
        member = model.members[entry["payload"]]
        txn.put(b"reg/state/" + str(entry["id"]).encode("ascii"), member)
    for name in model.names():
        name_b = name.encode("utf-8")
        meta_json = model.tensor_json(name)
        txn.put(b"t/" + name_b + b"/meta", meta_json.encode("utf-8"))
        meta = json.loads(meta_json)
        for i, plane in enumerate(meta["planes"]):
            txn.put(
                b"t/" + name_b + b"/p" + str(i).encode("ascii"),
                model.members[plane["payload"]],
            )
            state_payload = plane.get("state_payload")
            if state_payload is not None:
                txn.put(
                    b"t/" + name_b + b"/p" + str(i).encode("ascii") + b".state",
                    model.members[state_payload],
                )


# ── Reader / mutable store ────────────────────────────────────────────────────


class LmdbWeightStore:
    """Read/update API over an LMDB-backed PTWM model (both layouts)."""

    def __init__(
        self,
        path: str | Path,
        *,
        map_size: int = _DEFAULT_MAP_SIZE,
        readonly: bool = True,
    ) -> None:
        lmdb = _require_lmdb()
        self._path = Path(path)
        self._readonly = readonly
        self._env = lmdb.open(
            str(self._path),
            map_size=map_size,
            subdir=True,
            readonly=readonly,
            lock=not readonly,
        )
        fmt = self._get(KEY_FMT)
        if fmt is None:
            msg = f"{self._path} is not a PTWM LMDB environment (missing \\x00fmt)"
            raise ValueError(msg)
        fmt_obj = json.loads(fmt)
        if fmt_obj.get("ptwm_format") != LMDB_FORMAT:
            msg = f"{self._path}: not a {LMDB_FORMAT} environment"
            raise ValueError(msg)
        self._layout = fmt_obj["layout"]
        idx = self._get(KEY_IDX)
        self._index = json.loads(idx) if idx is not None else {"weight_map": {}}
        self._weight_map = dict(self._index.get("weight_map", {}))
        if self._layout == "exploded":
            reg = self._get(KEY_REG)
            self._registry_json = reg.decode("utf-8") if reg is not None else ""
            # Parse once; reused on every exploded-layout read.
            self._registry = (
                json.loads(self._registry_json)
                if reg is not None
                else {"registry": {"shared_state": []}}
            )

    @classmethod
    def open(cls, path: str | Path, *, readonly: bool = True) -> LmdbWeightStore:
        """Open an LMDB PTWM environment for reading (or updating)."""
        return cls(path, readonly=readonly)

    def close(self) -> None:
        """Close the underlying LMDB environment."""
        self._env.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def layout(self) -> str:
        """Either ``"blob"`` or ``"exploded"``."""
        return self._layout

    # -- internals ----------------------------------------------------------

    def _get(self, key: bytes) -> bytes | None:
        with self._env.begin() as txn:
            v = txn.get(key)
            return bytes(v) if v is not None else None

    def _members_for_exploded(self, name: str) -> tuple[str, str, dict[str, bytes]]:
        name_b = name.encode("utf-8")
        meta_raw = self._get(b"t/" + name_b + b"/meta")
        if meta_raw is None:
            raise KeyError(name)
        meta_json = meta_raw.decode("utf-8")
        meta = json.loads(meta_json)
        members: dict[str, bytes] = {}
        for i, plane in enumerate(meta["planes"]):
            payload = self._get(b"t/" + name_b + b"/p" + str(i).encode("ascii"))
            if payload is None:
                msg = f"missing plane {i} for tensor {name!r}"
                raise KeyError(msg)
            members[plane["payload"]] = payload
            state_payload = plane.get("state_payload")
            if state_payload is not None:
                state = self._get(
                    b"t/" + name_b + b"/p" + str(i).encode("ascii") + b".state"
                )
                if state is None:
                    msg = f"missing plane {i} state for tensor {name!r}"
                    raise KeyError(msg)
                members[state_payload] = state
        for entry in self._registry["registry"]["shared_state"]:
            shared = self._get(b"reg/state/" + str(entry["id"]).encode("ascii"))
            if shared is not None:
                members[entry["payload"]] = shared
        return self._registry_json, meta_json, members

    def _raw_undelta(self, name: str, raw: bytes) -> bytes:
        """Resolve an XOR-delta tensor against its in-store reference."""
        entry = self._weight_map.get(name, {})
        ref_name = entry.get("delta_ref")
        if ref_name is None:
            return raw
        ref_raw = self.get_bytes(ref_name)
        expected = entry.get("delta_hash")
        if expected is not None:
            actual = _rust.blake3_hash(ref_raw).hex()
            if actual != expected:
                msg = (
                    f"delta reference {ref_name!r} for {name!r} failed BLAKE3 "
                    f"verification (expected {expected}, got {actual})"
                )
                raise ValueError(msg)
        return _rust.delta_xor_decode(raw, ref_raw)

    # -- WeightStore API ----------------------------------------------------

    def names(self) -> list[str]:
        """Return tensor names in storage order."""
        return list(self._weight_map.keys())

    def get_bytes(self, name: str) -> bytes:
        """Return the raw decompressed bytes for ``name``."""
        if name not in self._weight_map:
            raise KeyError(name)
        if self._layout == "blob":
            blob = self._get(b"t/" + name.encode("utf-8"))
            if blob is None:
                raise KeyError(name)
            raw = bytes(_rust.decode_tensor(blob, name))
        else:
            registry_json, meta_json, members = self._members_for_exploded(name)
            raw = raw_from_members(registry_json, meta_json, members)
        return self._raw_undelta(name, raw)

    def get_tensor(self, name: str) -> torch.Tensor:
        """Decompress ``name`` and return a :class:`torch.Tensor`."""
        if name not in self._weight_map:
            raise KeyError(name)
        entry = self._weight_map[name]
        # Delta tensors must round-trip through the raw resolver first.
        if entry.get("delta_ref") is not None:
            raw = self.get_bytes(name)
            shape, dtype_name = entry.get("shape"), entry.get("dtype")
            if dtype_name is None:
                return torch.frombuffer(raw, dtype=torch.uint8)
            return raw_to_tensor(raw, shape, dtype_name)
        if self._layout == "blob":
            blob = self._get(b"t/" + name.encode("utf-8"))
            if blob is None:
                raise KeyError(name)
            return decode_ptwm_tensor(blob, name)
        registry_json, meta_json, members = self._members_for_exploded(name)
        return decode_member_tensor(registry_json, meta_json, members)

    def stream_tensors(
        self, names: Iterable[str] | None = None
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield ``(name, tensor)`` pairs."""
        target = list(names) if names is not None else self.names()
        for name in target:
            yield name, self.get_tensor(name)

    # -- mutation -----------------------------------------------------------

    def put_tensor(
        self,
        name: str,
        raw: bytes,
        dtype: Any,
        shape: tuple[int, ...],
        *,
        delta_reference: str | None = None,
        method_hint: int = 3,
    ) -> None:
        """Atomically insert or replace tensor ``name`` (blob layout only).

        When ``delta_reference`` names another tensor already in the store, the
        new tensor is stored as an XOR residual against it (a ``local_tensor``
        reference with BLAKE3 verification), turning small per-step changes into
        small diffs.
        """
        if self._readonly:
            msg = "store opened read-only; reopen with readonly=False to mutate"
            raise RuntimeError(msg)
        if self._layout != "blob":
            msg = "put_tensor is only supported for the 'blob' layout"
            raise NotImplementedError(msg)

        dt = DType.from_dtype(dtype)
        entry: dict[str, Any] = {
            "dtype": dt.dtype_str,
            "shape": list(shape),
            "orig_size": len(raw),
        }
        if delta_reference is not None:
            ref_raw = self.get_bytes(delta_reference)
            residual = _rust.delta_xor_encode(raw, ref_raw)
            stored, store_shape = residual, (len(residual),)
            entry["delta_ref"] = delta_reference
            entry["delta_hash"] = _rust.blake3_hash(ref_raw).hex()
            entry["delta_scheme"] = "xor"
            store_dtype: Any = "uint8"
        else:
            stored, store_shape, store_dtype = raw, shape, dtype

        blob = compress_ptwm_blob(
            [(name, stored, store_dtype, store_shape)], method_hint=method_hint
        )
        entry["comp_size"] = len(blob)
        with self._env.begin(write=True) as txn:
            txn.put(b"t/" + name.encode("utf-8"), blob)
            self._weight_map[name] = entry
            txn.put(KEY_IDX, self._reindex().encode("utf-8"))

    def _reindex(self) -> str:
        total_size = sum(int(e.get("orig_size", 0)) for e in self._weight_map.values())
        compressed = sum(int(e.get("comp_size", 0)) for e in self._weight_map.values())
        return jcs_canonicalize(
            {
                "ptwm_format": LMDB_FORMAT,
                "manifest_version": 1,
                "layout": self._layout,
                "weight_map": self._weight_map,
                "total_size": total_size,
                "compressed_total_size": compressed,
            }
        )
