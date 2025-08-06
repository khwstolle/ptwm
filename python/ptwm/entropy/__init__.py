"""Shannon entropy utilities for compressed-tensor analysis.

These functions produce "before entropy coder" measurements on plane bytes.
They are deliberately codec-agnostic: :func:`shannon` runs on arbitrary byte
buffers, and :func:`per_plane` folds over a list of
:class:`~ptwm.preprocessing.Plane` objects.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from .. import _rust

if TYPE_CHECKING:
    from ..preprocessing import Plane


def shannon(data: bytes | memoryview) -> float:
    """Base-2 Shannon entropy of ``data`` in bits/byte. Empty input yields 0.0."""
    return _rust.shannon(data)


def per_plane(planes: Sequence[Plane]) -> list[float]:
    """Shannon entropy per plane, in the order given."""
    return [shannon(plane.data) for plane in planes]


def histogram(data: bytes | memoryview) -> np.ndarray:
    """256-bin byte-frequency histogram as a ``uint32`` NumPy array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    return np.bincount(arr, minlength=256).astype(np.uint32)


def joint_histogram(
    a: bytes | memoryview | np.ndarray,
    b: bytes | memoryview | np.ndarray,
) -> np.ndarray:
    """Return the 256x256 joint byte-frequency histogram of paired byte streams.

    The result has shape ``(256, 256)`` and dtype ``uint64``; ``out[i, j]``
    counts positions where ``a`` has value ``i`` and ``b`` has value ``j``.
    Requires ``len(a) == len(b)``.
    """
    av = (
        np.frombuffer(a, dtype=np.uint8)
        if not isinstance(a, np.ndarray)
        else a.view(np.uint8).ravel()
    )
    bv = (
        np.frombuffer(b, dtype=np.uint8)
        if not isinstance(b, np.ndarray)
        else b.view(np.uint8).ravel()
    )
    if av.size != bv.size:
        msg = f"length mismatch: {av.size} vs {bv.size}"
        raise ValueError(msg)
    # Pack (a, b) into one uint16 index then bincount — avoids the O(N*256)
    # loop a naive histogram2d would take, in pure NumPy.
    idx = av.astype(np.uint32) * 256 + bv.astype(np.uint32)
    flat = np.bincount(idx, minlength=65536).astype(np.uint64)
    return flat.reshape(256, 256)


def mutual_information(joint: np.ndarray) -> float:
    """Mutual information ``I(A; B)`` in bits/byte from a joint histogram.

    ``joint`` must be the ``(256, 256)`` output of :func:`joint_histogram`.
    Computes ``H(A) + H(B) - H(A, B)`` from the marginals and the joint.
    Returns exactly ``0.0`` for empty input.
    """
    if joint.ndim != 2 or joint.shape != (256, 256):
        msg = f"expected (256, 256) joint histogram, got {joint.shape}"
        raise ValueError(msg)
    total = joint.sum()
    if total == 0:
        return 0.0

    def _entropy(hist: np.ndarray) -> float:
        p = hist[hist > 0].astype(np.float64) / float(total)
        return float(-np.sum(p * np.log2(p)))

    h_a = _entropy(joint.sum(axis=1))
    h_b = _entropy(joint.sum(axis=0))
    h_ab = _entropy(joint.ravel())
    return h_a + h_b - h_ab


__all__ = [
    "histogram",
    "joint_histogram",
    "mutual_information",
    "per_plane",
    "shannon",
]
