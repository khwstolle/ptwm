"""Multi-tensor ``.ptwm`` shard writer (calls into _rust.compress_model)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .. import _rust
from .._config import ContainerLayout, container_layout_name
from ..preprocessing._chains import (
    ClassifierRole,
    build_chain_blobs,
    chains_for,
)
from ..utils import DType

__all__ = ["write_ptwm_shard", "compress_ptwm_blob"]


def _default_chain_bytes(
    dtype_code: int, shape: tuple[int, ...], nbytes: int
) -> list[bytes]:
    """Return chain bytes for ``(dtype_code, STANDARD)``; fallback to passthrough.

    Each chain's Source covers exactly ``nbytes`` (see
    :func:`~ptwm.preprocessing._chains.build_chain_blobs`).
    """
    return build_chain_blobs(
        chains_for(dtype_code, ClassifierRole.STANDARD),
        shape=list(shape),
        nbytes=nbytes,
    )


def compress_ptwm_blob(
    tensors: Sequence[tuple[str, bytes, Any, tuple[int, ...]]],
    *,
    chains_per_tensor: list[list[bytes]] | None = None,
    method_hint: int = 3,
    emit_payload_hash: bool = True,
    emit_plane_crc: bool = True,
    streaming_chunk: int | None = None,
    container_layout: ContainerLayout | str = "compact",
) -> bytes:
    r"""Compress ``tensors`` into a single in-memory ``.ptwm`` container blob.

    ``tensors`` is a sequence of ``(name, raw_bytes, dtype, shape)``.
    ``chains_per_tensor`` — one list of chain wire-byte blobs per tensor;
    each blob is produced by ``Chain.to_bytes()``. When ``None``, chains are
    auto-derived from each tensor's dtype via the STANDARD production table.
    ``method_hint``: 1=HUFFMAN, 2=ZSTD, 3=MICROSCALE, 4=RANS, 5=IDENTITY.
    ``container_layout``: ``"compact"`` (default; ``\x89PTWC``, v1.2 and
    later) or ``"legacy"`` (``\x89PTWM``, readable by every release), as a string or
    :class:`ptwm.ContainerLayout`. Other values raise :class:`ValueError`
    before anything is compressed.
    """
    layout = container_layout_name(container_layout)
    records = []
    auto_chains: list[list[bytes]] = []
    for name, raw, dtype_in, shape in tensors:
        dt = DType.from_dtype(dtype_in)
        records.append(
            (
                name,
                int(dt.code),
                0,  # input_format = BYTE
                raw,
                list(shape),
                dt.dtype_str,
                None,  # delta_ref_blake3
            )
        )
        auto_chains.append(_default_chain_bytes(int(dt.code), shape, len(raw)))

    effective_chains = (
        chains_per_tensor if chains_per_tensor is not None else auto_chains
    )

    return _rust.compress_model(
        records,
        effective_chains,
        emit_payload_hash,
        emit_plane_crc,
        method_hint,
        streaming_chunk,
        container_layout=layout,
    )


def write_ptwm_shard(
    out_path: str | Path,
    tensors: Sequence[tuple[str, bytes, Any, tuple[int, ...]]],
    *,
    chains_per_tensor: list[list[bytes]] | None = None,
    method_hint: int = 3,
    emit_payload_hash: bool = True,
    emit_plane_crc: bool = True,
    streaming_chunk: int | None = None,
    container_layout: ContainerLayout | str = "compact",
) -> None:
    """Write a single ``.ptwm`` container holding ``tensors``.

    See :func:`compress_ptwm_blob` for the parameter semantics. An unknown
    ``container_layout`` raises :class:`ValueError` before ``out_path`` is
    written.
    """
    blob = compress_ptwm_blob(
        tensors,
        chains_per_tensor=chains_per_tensor,
        method_hint=method_hint,
        emit_payload_hash=emit_payload_hash,
        emit_plane_crc=emit_plane_crc,
        streaming_chunk=streaming_chunk,
        container_layout=container_layout,
    )
    Path(out_path).write_bytes(blob)
