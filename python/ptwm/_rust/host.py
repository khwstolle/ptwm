"""Thin shim exposing the host-flavor registry as ptwm._rust.host."""

from __future__ import annotations

from ptwm._core import ContributionRegistry, host_registry_entries

__all__ = ["ContributionRegistry", "host_registry_entries"]
