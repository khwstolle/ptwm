"""Thin shim exposing the delta_scheme dispatch surface as ptwm._rust.delta_scheme."""

from __future__ import annotations

from ptwm._core import (  # type: ignore[import]
    delta_scheme_decode,
    delta_scheme_encode,
)

__all__ = [
    "delta_scheme_decode",
    "delta_scheme_encode",
]
