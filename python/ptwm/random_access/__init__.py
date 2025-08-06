"""Random-access loader for ``.ptwm`` containers.

:class:`TensorIndex` opens a compressed bundle and decompresses individual
tensors on demand. The index is tiny (a manifest of name → offset/length
records), so opening a bundle and fetching one tensor touches only that
tensor's bytes.

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
from ..utils import raw_to_tensor

_MAGIC = b"\x89PTWM"


@dataclass(frozen=True, slots=True)
class TensorEntry:
    """One entry from a ``.ptwm`` tensor index."""

    name: str


class TensorIndex:
    """Name-indexed random-access loader over a ``.ptwm`` container."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._blob: bytes = self._path.read_bytes()
        if self._blob[: len(_MAGIC)] != _MAGIC:
            msg = f"{self._path} is not a .ptwm container"
            raise ValueError(msg)
        self._names: list[str] = list(_rust.list_tensor_names(self._blob))
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

    def names(self) -> list[str]:
        """Return tensor names in manifest (write) order."""
        return list(self._names)

    def entries(self) -> list[TensorEntry]:
        """Return every :class:`TensorEntry` in manifest order."""
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
        return _rust.decode_tensor(self._blob, name)

    def _get_shape_dtype(self, name: str) -> tuple[list[int], str] | None:
        return _rust.decode_tensor_shape(self._blob, name)

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
