"""Structured representation of a preprocessed byte plane."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Plane:
    """One plane produced by the preprocessing pipeline.

    ``role`` is a free-form short string identifying the semantic content
    of the plane (e.g. ``"exponent"``, ``"sign_mantissa"``, ``"byte_0"``).
    Used by logging / diagnostics; the compression pipeline does not
    branch on it.
    """

    id: int
    role: str
    data: bytes

    def __len__(self) -> int:
        return len(self.data)


__all__ = ["Plane"]
