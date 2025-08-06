"""Reference contribution for the chain_builder kind, Python/host flavor."""

from __future__ import annotations

CANONICAL_ID = "blake3:" + "0" * 64
LABEL = "io.ptwm.ref.chain_builder"


def register() -> None:
    """Entry point called by ptwm._loader.load_all_host_extensions()."""
    from ptwm._rust.host import ContributionRegistry

    reg = ContributionRegistry()
    reg.register(CANONICAL_ID, "chain_builder", "thread", _impl)


def _impl(state: object, dtype_code: int, role: str) -> list[bytes]:
    """Passthrough — returns an empty candidate list.

    A real chain_builder would return ``list[bytes]`` where each element is
    a serialized chain (wire bytes accepted by ``compress_model``).  This
    stub returns nothing so the host falls back to ``PRODUCTION_CHAINS``.

    The actual call shape is determined by the dispatcher that wires it in.
    """
    return []
