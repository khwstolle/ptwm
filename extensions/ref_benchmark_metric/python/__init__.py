"""Reference contribution for the benchmark_metric kind, Python/host flavor."""

from __future__ import annotations

CANONICAL_ID = "blake3:" + "0" * 64
LABEL = "io.ptwm.ref.benchmark_metric"


def register() -> None:
    """Entry point called by ptwm._loader.load_all_host_extensions()."""
    from ptwm._rust.host import ContributionRegistry

    reg = ContributionRegistry()
    reg.register(CANONICAL_ID, "benchmark_metric", "thread", _impl)


def _impl(state: object, data: bytes) -> float:
    """Stub metric — returns 0.0.

    A real benchmark_metric measures some property of ``data`` (e.g. encode
    time in nanoseconds, compressed size in bytes, etc.) and returns it as
    a ``float``.  This stub always returns 0.0 and is a valid stand-in until
    the dispatcher wires real per-kind calls.
    """
    return 0.0
