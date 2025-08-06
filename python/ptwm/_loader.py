"""Discover Python-resident extensions via entry points.

Entry points under the `ptwm.extensions` group are loaded at PTWM
startup. Each one is a callable that, when invoked, calls
ContributionRegistry.register(...) for each contribution it provides.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from importlib.metadata import EntryPoint, entry_points


def discover_host_extensions() -> Iterable[tuple[str, EntryPoint]]:
    """Yield (entry_point_name, ep) for each ptwm.extensions entry point."""
    eps = entry_points(group="ptwm.extensions")
    for ep in eps:
        yield ep.name, ep


def load_all_host_extensions() -> int:
    """Invoke every discovered entry point.

    Returns the number of entry points successfully loaded. Errors from
    a single entry point are logged (via print) but don't stop the
    loader — a broken extension shouldn't take down the whole runtime.
    """
    count = 0
    for name, ep in discover_host_extensions():
        try:
            register_fn = ep.load()
        except Exception as exc:  # noqa: BLE001
            print(  # noqa: T201
                f"warning: ptwm.extensions[{name}] failed to load: {exc}",
                file=sys.stderr,
            )
            continue
        try:
            register_fn()
        except Exception as exc:  # noqa: BLE001
            print(  # noqa: T201
                f"warning: ptwm.extensions[{name}] failed to register: {exc}",
                file=sys.stderr,
            )
            continue
        count += 1
    return count
