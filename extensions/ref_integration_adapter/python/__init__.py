"""Reference contribution for the integration_adapter kind, Python/host flavor."""

from __future__ import annotations

CANONICAL_ID = "blake3:" + "0" * 64
LABEL = "io.ptwm.ref.integration_adapter"


def register() -> None:
    """Entry point called by ptwm._loader.load_all_host_extensions()."""
    from ptwm._rust.host import ContributionRegistry

    reg = ContributionRegistry()
    reg.register(CANONICAL_ID, "integration_adapter", "thread", _impl)


def _impl(state: object, mode: int, data: bytes) -> bytes:
    """Passthrough adapter — returns input unchanged for both convert modes.

    ConvertMode: ImportIntoPtwm = 0, ExportFromPtwm = 1.
    A real integration_adapter transforms data between a host-library format
    and the PTWM internal layout.  This stub is a no-op identity adapter.
    """
    return data
