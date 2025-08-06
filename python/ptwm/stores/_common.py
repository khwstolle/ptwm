"""Shared foundation for the WeightStore integrations.

The LMDB and WebDataset stores are two serializations of the same canonical
PTWM-JSON manifest (see :mod:`ptwm._rust` ``explode_ptwm`` / ``implode_ptwm``).
This module holds the parts they share: the :class:`WeightStore` protocol, the
exploded-model writer helper, the per-tensor decode helper, and an RFC 8785
(JCS) canonicalizer used for the Python-authored manifests (``index.json``).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Protocol, runtime_checkable

import torch

from .. import _rust
from ..utils import raw_to_tensor

__all__ = [
    "WeightStore",
    "ExplodedModel",
    "explode_model",
    "decode_member_tensor",
    "raw_from_members",
    "jcs_canonicalize",
    "shared_member_keys",
    "tensor_member_keys",
]


@runtime_checkable
class WeightStore(Protocol):
    """Read API shared by every PTWM storage backend.

    Generalizes :class:`ptwm.random_access.TensorIndex` so callers can treat a
    native ``.ptwm`` container, an LMDB environment, and a WebDataset shard set
    interchangeably.
    """

    def names(self) -> list[str]:
        """Return tensor names in storage order."""

    def get_bytes(self, name: str) -> bytes:
        """Return the raw decompressed bytes for ``name``."""

    def get_tensor(self, name: str) -> torch.Tensor:
        """Decompress ``name`` and return a :class:`torch.Tensor`."""

    def stream_tensors(
        self, names: Iterable[str] | None = None
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield ``(name, tensor)`` pairs."""


# ── Exploded-model helper ─────────────────────────────────────────────────────


class ExplodedModel:
    """A compressed model projected to its canonical PTWM-JSON form.

    Holds the archive registry manifest, the per-tensor manifests in file order,
    and the loose binary members. The LMDB and WebDataset writers consume this.
    """

    __slots__ = ("registry_json", "tensors", "members", "_by_name")

    def __init__(
        self,
        registry_json: str,
        tensors: list[tuple[str, str, str]],
        members: dict[str, bytes],
    ) -> None:
        self.registry_json = registry_json
        self.tensors = tensors  # (name, key, tensor_manifest_json)
        self.members = members
        self._by_name = {name: (key, j) for (name, key, j) in tensors}

    def names(self) -> list[str]:
        return [name for (name, _, _) in self.tensors]

    def tensor_json(self, name: str) -> str:
        return self._by_name[name][1]

    def key(self, name: str) -> str:
        return self._by_name[name][0]

    def member_size(self, name: str) -> int:
        """Total compressed bytes owned by ``name`` (planes + per-tensor state)."""
        key = self.key(name)
        prefix = f"{key}."
        return sum(len(v) for k, v in self.members.items() if k.startswith(prefix))


def explode_model(blob: bytes) -> ExplodedModel:
    """Explode a multi-tensor ``.ptwm`` container blob into an exploded model."""
    registry_json, tensors, members = _rust.explode_ptwm(blob)
    return ExplodedModel(registry_json, list(tensors), dict(members))


def tensor_member_keys(members: dict[str, bytes], sample_key: str) -> dict[str, bytes]:
    """Return the subset of ``members`` owned by the tensor ``sample_key``."""
    prefix = f"{sample_key}."
    return {k: v for k, v in members.items() if k.startswith(prefix)}


def shared_member_keys(members: dict[str, bytes]) -> dict[str, bytes]:
    """Return the archive-global members (shared state, embedded modules)."""
    return {k: v for k, v in members.items() if k.startswith(("shared/", "modules/"))}


# ── Per-tensor decode ─────────────────────────────────────────────────────────


def raw_from_members(
    registry_json: str,
    tensor_json: str,
    members: dict[str, bytes],
) -> bytes:
    """Reconstruct a one-tensor container and return its raw decompressed bytes.

    ``members`` must contain the tensor's plane/state members plus any
    archive-global shared state referenced by ``registry_json``.
    """
    blob = _rust.implode_ptwm(registry_json, [tensor_json], members)
    name = json.loads(tensor_json)["name"]
    return bytes(_rust.decode_tensor(blob, name))


def decode_member_tensor(
    registry_json: str,
    tensor_json: str,
    members: dict[str, bytes],
) -> torch.Tensor:
    """Reconstruct a single tensor (as a :class:`torch.Tensor`) from members."""
    meta = json.loads(tensor_json)
    raw = raw_from_members(registry_json, tensor_json, members)
    shape = meta.get("shape")
    dtype_name = meta.get("dtype")
    if dtype_name is None:
        return torch.frombuffer(raw, dtype=torch.uint8)
    return raw_to_tensor(raw, shape, dtype_name)


# ── RFC 8785 (JCS) canonicalization ───────────────────────────────────────────


def jcs_canonicalize(obj: Any) -> str:
    """Serialize ``obj`` to a canonical (RFC 8785) JSON string.

    Mirrors the Rust ``serde_jcs``-equivalent serializer in ``ptwm-core``: object
    keys sorted by UTF-16 code unit, no insignificant whitespace, integers in
    shortest form. Floats are rejected to enforce the integers-only-scalar
    manifest schema.
    """
    out: list[str] = []
    _jcs_write(obj, out)
    return "".join(out)


def _jcs_write(obj: Any, out: list[str]) -> None:
    if obj is None:
        out.append("null")
    elif obj is True:
        out.append("true")
    elif obj is False:
        out.append("false")
    elif isinstance(obj, int):
        out.append(str(obj))
    elif isinstance(obj, float):
        msg = "manifest canonicalization: floating-point scalars are not permitted"
        raise ValueError(msg)
    elif isinstance(obj, str):
        out.append(_jcs_string(obj))
    elif isinstance(obj, Sequence) and not isinstance(obj, str | bytes | bytearray):
        out.append("[")
        for i, item in enumerate(obj):
            if i:
                out.append(",")
            _jcs_write(item, out)
        out.append("]")
    elif isinstance(obj, dict):
        keys = sorted(obj.keys(), key=lambda k: k.encode("utf-16-be"))
        out.append("{")
        for i, key in enumerate(keys):
            if i:
                out.append(",")
            if not isinstance(key, str):
                msg = "manifest object keys must be strings"
                raise TypeError(msg)
            out.append(_jcs_string(key))
            out.append(":")
            _jcs_write(obj[key], out)
        out.append("}")
    else:
        msg = f"cannot canonicalize value of type {type(obj).__name__}"
        raise TypeError(msg)


def _jcs_string(s: str) -> str:
    # ``ensure_ascii=False`` produces RFC 8259 minimal escaping over UTF-8,
    # which matches RFC 8785 string serialization.
    return json.dumps(s, ensure_ascii=False)
