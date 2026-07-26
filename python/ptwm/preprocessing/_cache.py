"""User-local cache of explorer-discovered chain templates.

Caches *structural templates* (internal_dtype + op tuple + byte-split planes)
rather than serialised chain wire bytes: exploration discovers a chain on
one tensor with one shape, but the same template applies to any tensor of
the same (dtype, role) regardless of shape.

Keyed by ``(dtype_code, role, signature_bucket)``, where the signature
bucket groups tensors by Shannon entropy (see :func:`signature_bucket_for`).
The keying assumption is that two tensors with similar entropy fingerprints
benefit from the same chain, so an explorer run on one model extends the
candidate set for future runs on models with similar weight distributions,
without pulling in templates discovered on tensors with a very different
byte-statistics profile. Callers that do not have a tensor's bytes on hand
(or do not care to distinguish by entropy) can omit ``signature_bucket``,
which defaults to a single shared bucket.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cbor2

from ._chains import (
    Chain,
    ChainBuilder,
    ClassifierRole,
)

if TYPE_CHECKING:
    pass

__all__ = [
    "CacheEntry",
    "SIGNATURE_BUCKET_WIDTH",
    "cache_dir",
    "cache_info",
    "cache_path_for",
    "clear_cache",
    "load_cached_builders",
    "load_cached_entries",
    "save_cached_entries",
    "signature_bucket_for",
]

# Bumped when the cache wire format changes incompatibly. v2 adds the
# signature bucket to the cache key (see module docstring).
_SCHEMA_VERSION = 2
_MAGIC = b"PTWMCHC\x02"

# Shannon entropy (bits/byte) is bounded [0, 8]; a 0.5-bit bucket width
# gives 17 buckets. Coarse enough to produce hit rates in practice (an
# entropy-signature match need not be exact to indicate similar byte
# statistics); fine enough to actually separate distinct distributions
# (e.g. a near-degenerate all-zero-padded tensor from a well-spread one).
# Not measured against real hit-rate data yet — a reasonable starting
# point, not a tuned constant.
SIGNATURE_BUCKET_WIDTH = 0.5
_N_BUCKETS = int(8.0 / SIGNATURE_BUCKET_WIDTH) + 1


def signature_bucket_for(entropy_bits_per_byte: float) -> int:
    """Map a Shannon-entropy value (bits/byte, expected in [0, 8]) to a bucket.

    Values outside [0, 8] (should not occur for a real byte plane, but
    defends against a caller passing a raw float from elsewhere) are
    clamped rather than raising, so a bad upstream measurement degrades to
    a cache miss instead of an exception.
    """
    clamped = min(max(entropy_bits_per_byte, 0.0), 8.0)
    bucket = int(clamped / SIGNATURE_BUCKET_WIDTH)
    return min(bucket, _N_BUCKETS - 1)


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """One cached chain template, identified structurally."""

    internal_dtype: int
    ops: tuple[int, ...]
    byte_split_planes: int

    def to_builder(self) -> ChainBuilder:
        # Late import: explorer imports preprocessing._chains, which would
        # cycle if pulled at module load.
        from ._explorer import _build_linear_chain  # noqa: PLC0415

        internal_dtype = self.internal_dtype
        ops = self.ops
        planes = self.byte_split_planes

        def _builder(shape: list[int]) -> Chain:
            return _build_linear_chain(internal_dtype, shape, ops, planes)

        return _builder


def cache_dir() -> Path:
    """Return the user-local chain cache directory (XDG-aware)."""
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "ptwm" / "chains" / f"v{_SCHEMA_VERSION}"


def cache_path_for(
    dtype_code: int, role: ClassifierRole, signature_bucket: int = 0
) -> Path:
    return cache_dir() / f"{dtype_code:04x}_{int(role)}_{signature_bucket:02d}.cbor"


def _serialise(entries: Iterable[CacheEntry]) -> bytes:
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "entries": [
            {
                "internal_dtype": e.internal_dtype,
                "ops": list(e.ops),
                "byte_split_planes": e.byte_split_planes,
            }
            for e in entries
        ],
    }
    return _MAGIC + cbor2.dumps(payload)


def _deserialise(blob: bytes) -> list[CacheEntry]:
    if not blob.startswith(_MAGIC):
        msg = "cache file: missing PTWM-chain magic"
        raise ValueError(msg)
    payload = cbor2.loads(blob[len(_MAGIC) :])
    if not isinstance(payload, dict):
        msg = "cache file: payload is not a CBOR map"
        raise ValueError(msg)
    version = payload.get("schema_version")
    # Strict equality so a future bump invalidates old caches deterministically.
    if version != _SCHEMA_VERSION:
        msg = f"cache file: schema_version {version!r} != {_SCHEMA_VERSION}"
        raise ValueError(msg)
    raw_entries = payload.get("entries", [])
    out: list[CacheEntry] = []
    for raw in raw_entries:
        out.append(
            CacheEntry(
                internal_dtype=int(raw["internal_dtype"]),
                ops=tuple(int(o) for o in raw["ops"]),
                byte_split_planes=int(raw["byte_split_planes"]),
            )
        )
    return out


def load_cached_entries(
    dtype_code: int, role: ClassifierRole, signature_bucket: int = 0
) -> list[CacheEntry]:
    """Return cached entries for `(dtype_code, role, signature_bucket)`.

    Returns `[]` on miss or corruption.
    """
    path = cache_path_for(dtype_code, role, signature_bucket)
    if not path.exists():
        return []
    try:
        return _deserialise(path.read_bytes())
    except (ValueError, OSError, cbor2.CBORDecodeError):
        # A corrupt or schema-mismatched cache must not poison the compress
        # path. Treat as a miss; clear_cache() is the deliberate recovery.
        return []


def load_cached_builders(
    dtype_code: int, role: ClassifierRole, signature_bucket: int = 0
) -> list[ChainBuilder]:
    """Return runtime `ChainBuilder`s for `(dtype_code, role, signature_bucket)`."""
    return [
        e.to_builder() for e in load_cached_entries(dtype_code, role, signature_bucket)
    ]


def save_cached_entries(
    dtype_code: int,
    role: ClassifierRole,
    new_entries: Iterable[CacheEntry],
    signature_bucket: int = 0,
) -> int:
    """Union `new_entries` with the existing cache file. Returns count added.

    Idempotent: writing the same entry twice is a no-op.
    """
    existing = {
        (e.internal_dtype, e.ops, e.byte_split_planes): e
        for e in load_cached_entries(dtype_code, role, signature_bucket)
    }
    added = 0
    for e in new_entries:
        key = (e.internal_dtype, e.ops, e.byte_split_planes)
        if key not in existing:
            existing[key] = e
            added += 1
    if added == 0:
        return 0
    path = cache_path_for(dtype_code, role, signature_bucket)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_serialise(existing.values()))
    return added


def clear_cache() -> int:
    """Delete every cache file. Returns the number of files removed."""
    root = cache_dir()
    if not root.exists():
        return 0
    removed = 0
    for f in root.glob("*.cbor"):
        f.unlink()
        removed += 1
    return removed


def cache_info() -> dict[str, int]:
    """Map cache filename → chain count, for diagnostics."""
    root = cache_dir()
    if not root.exists():
        return {}
    out: dict[str, int] = {}
    for f in sorted(root.glob("*.cbor")):
        try:
            entries = _deserialise(f.read_bytes())
            out[f.name] = len(entries)
        except (ValueError, OSError, cbor2.CBORDecodeError):
            out[f.name] = -1
    return out
