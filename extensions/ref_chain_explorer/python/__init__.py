"""Reference contribution for the chain_explorer kind, Python/host flavor."""

from __future__ import annotations

CANONICAL_ID = "blake3:" + "0" * 64
LABEL = "io.ptwm.ref.chain_explorer"


def register() -> None:
    """Entry point called by ptwm._loader.load_all_host_extensions()."""
    from ptwm._rust.host import ContributionRegistry

    reg = ContributionRegistry()
    reg.register(CANONICAL_ID, "chain_explorer", "thread", _impl)


def _impl(
    state: object,
    dtype_code: int,
    role: str,
    seed_candidates: list[bytes],
) -> list[bytes]:
    """Passthrough explorer — returns the seed candidates unchanged.

    A real chain_explorer extends the seed set with BFS-discovered
    candidates.  This stub is a no-op degenerate explorer that simply
    reflects the seeds back so the host can trial-encode them.
    """
    return list(seed_candidates)
