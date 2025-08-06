"""Thin shim exposing the install/discovery surface as ptwm._rust.ext."""

from __future__ import annotations

from ptwm._core import (  # type: ignore[import]
    InstalledContribution,
    ext_install,
    ext_list,
    ext_remove,
    ext_user_dir,
)

__all__ = [
    "InstalledContribution",
    "ext_install",
    "ext_list",
    "ext_remove",
    "ext_user_dir",
]
