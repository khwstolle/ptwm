"""Thin shim re-exporting policy types from the Rust extension."""

from __future__ import annotations

from ptwm._core import PolicyFile, ResolvedPolicy  # type: ignore[import]

__all__ = ["PolicyFile", "ResolvedPolicy"]
