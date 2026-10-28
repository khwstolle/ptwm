r"""Magic numbers of the PTWM file formats.

Four formats share the ``\x89PTW`` prefix:

- ``\x89PTWM\r\n\x1a\n``: legacy multi-tensor container (every release reads it).
- ``\x89PTWX\r\n\x1a\n``: legacy container with embedded WASM extensions.
- ``\x89PTWC\r\n\x1a\n``: compact multi-tensor container (v1.2 and later).
- ``\x89PTWS``: compact single-tensor frame written by
  :meth:`ptwm.Compressor.compress` (v1.2 and later).

Compare the full magic, not a 5-byte prefix: the fifth byte is what tells the
formats apart, and readers before v1.2 that check only ``\x89PTWM`` reject the
newer formats by design.
"""

from __future__ import annotations

from typing import Literal

__all__ = [
    "MAGIC_LEN",
    "MAGIC_PTWC",
    "MAGIC_PTWM",
    "MAGIC_PTWS",
    "MAGIC_PTWX",
    "ContainerKind",
    "container_kind",
    "is_container",
]

MAGIC_PTWM = b"\x89PTWM\r\n\x1a\n"
MAGIC_PTWX = b"\x89PTWX\r\n\x1a\n"
MAGIC_PTWC = b"\x89PTWC\r\n\x1a\n"
MAGIC_PTWS = b"\x89PTWS"

MAGIC_LEN = len(MAGIC_PTWM)
"""Bytes to read from the start of a file to tell every format apart."""

ContainerKind = Literal["ptwm", "ptwx", "ptwc", "ptws"]

_CONTAINERS: dict[bytes, ContainerKind] = {
    MAGIC_PTWM: "ptwm",
    MAGIC_PTWX: "ptwx",
    MAGIC_PTWC: "ptwc",
}


def container_kind(buf: bytes | bytearray | memoryview) -> ContainerKind | None:
    """Return the format of ``buf`` from its magic, or ``None``.

    ``buf`` needs only its first :data:`MAGIC_LEN` bytes. Returns ``"ptwm"``,
    ``"ptwx"`` or ``"ptwc"`` for a multi-tensor container, ``"ptws"`` for a
    single-tensor frame, and ``None`` for anything else (including a buffer
    too short to hold a magic).
    """
    head = bytes(memoryview(buf)[:MAGIC_LEN])
    kind = _CONTAINERS.get(head)
    if kind is not None:
        return kind
    if head[: len(MAGIC_PTWS)] == MAGIC_PTWS:
        return "ptws"
    return None


def is_container(buf: bytes | bytearray | memoryview) -> bool:
    """Whether ``buf`` starts with a multi-tensor container magic.

    True for ``\\x89PTWM``, ``\\x89PTWX`` and ``\\x89PTWC``: the formats that
    :class:`ptwm.random_access.TensorIndex`, the stores and the transcoder
    open. False for a ``\\x89PTWS`` single-tensor frame.
    """
    return container_kind(buf) in ("ptwm", "ptwx", "ptwc")
