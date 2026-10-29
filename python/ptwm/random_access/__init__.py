r"""Random-access loader for ``.ptwm`` containers.

:class:`TensorIndex` opens a compressed bundle in either layout (magic
``\x89PTWM``, ``\x89PTWX`` or ``\x89PTWC``) and decompresses individual
tensors on demand. The container is parsed once into a
:class:`ptwm._rust.ContainerHandle`; fetching one tensor then decodes only
that tensor's bytes.

    idx = TensorIndex.open("model.safetensors.ptwm")
    weight = idx.get_tensor("layer_5.weight")
    for name, tensor in idx.stream_tensors(names_of_interest):
        ...
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import torch

from .. import _rust
from .._magic import is_container
from ..utils import raw_to_tensor


@dataclass(frozen=True, slots=True)
class TensorEntry:
    """One entry from a ``.ptwm`` tensor index."""

    name: str


class TensorIndex:
    """Name-indexed random-access loader over a ``.ptwm`` container."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._blob: bytes = self._path.read_bytes()
        if not is_container(self._blob):
            msg = f"{self._path} is not a .ptwm container"
            raise ValueError(msg)
        self._handle = _rust.ContainerHandle(self._blob)
        self._names: list[str] = self._handle.names()
        self._entries_by_name: dict[str, TensorEntry] = {
            n: TensorEntry(name=n) for n in self._names
        }

    @classmethod
    def open(cls, path: str | Path) -> Self:
        """Open ``path`` for random-access tensor lookup."""
        return cls(path)

    # -- inspection ---------------------------------------------------------

    @property
    def path(self) -> Path:
        """Filesystem path this index was opened from."""
        return self._path

    @property
    def layout(self) -> str:
        """Container layout: ``"legacy"`` or ``"compact"``."""
        return self._handle.layout

    def names(self) -> list[str]:
        """Return tensor names in write order.

        This is the order the writer appended the tensors (the order of the
        records in the file), for both layouts. It is not the order of the
        legacy on-disk index, which is sorted by name hash.
        """
        return list(self._names)

    def entries(self) -> list[TensorEntry]:
        """Return every :class:`TensorEntry` in write order."""
        return [self._entries_by_name[n] for n in self._names]

    def entry(self, name: str) -> TensorEntry:
        """Return the :class:`TensorEntry` for ``name`` or raise ``KeyError``."""
        try:
            return self._entries_by_name[name]
        except KeyError:
            raise KeyError(name) from None

    # -- data access --------------------------------------------------------

    def get_bytes(self, name: str) -> bytes:
        """Return the raw decompressed bytes for ``name``."""
        if name not in self._entries_by_name:
            raise KeyError(name)
        return self._handle.decode(name)

    def _get_shape_dtype(self, name: str) -> tuple[list[int], str] | None:
        return self._handle.shape(name)

    def get_tensor(self, name: str) -> torch.Tensor:
        """Decompress ``name`` and return a :class:`torch.Tensor`.

        Shape/dtype come from the CBOR shape metadata written at compression
        time; tensors without metadata raise :class:`TypeError`.
        """
        if name not in self._entries_by_name:
            raise KeyError(name)
        shape_and_dtype = self._get_shape_dtype(name)
        if shape_and_dtype is None:
            msg = (
                f"tensor {name!r} has no shape metadata; cannot reconstruct "
                "a torch.Tensor. Re-compress with shape/dtype info or call "
                "get_bytes()."
            )
            raise TypeError(msg)
        raw = self.get_bytes(name)
        shape, dtype_name = shape_and_dtype
        return raw_to_tensor(raw, shape, dtype_name)

    def stream_tensors(
        self,
        names: Iterable[str] | None = None,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield ``(name, tensor)`` pairs."""
        target = list(names) if names is not None else self.names()
        for name in target:
            yield name, self.get_tensor(name)

    # -- container-like behaviour ------------------------------------------

    def __len__(self) -> int:  # noqa: D105
        return len(self._names)

    def __contains__(self, name: str) -> bool:  # noqa: D105
        return name in self._entries_by_name

    def __iter__(self) -> Iterator[str]:  # noqa: D105
        return iter(self._names)

    def __repr__(self) -> str:  # noqa: D105
        return f"TensorIndex(path={self._path!s}, n={len(self)})"


__all__ = ["TensorEntry", "TensorIndex"]
