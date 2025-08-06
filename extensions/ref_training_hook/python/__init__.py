"""Reference contribution for the training_hook kind, Python/host flavor."""

from __future__ import annotations

CANONICAL_ID = "blake3:" + "0" * 64
LABEL = "io.ptwm.ref.training_hook"


def register() -> None:
    """Entry point called by ptwm._loader.load_all_host_extensions()."""
    from ptwm._rust.host import ContributionRegistry

    reg = ContributionRegistry()
    reg.register(CANONICAL_ID, "training_hook", "thread", _impl)


def _impl(state: object, tensor_name: str, data: bytearray) -> None:
    """No-op pre-encode hook — leaves data unchanged.

    A real training_hook mutates ``data`` in place before the tensor is
    encoded (e.g. to apply quantization, masking, or distillation deltas).
    This stub is a pure no-op; the in-place mutation contract is satisfied
    by returning without modifying ``data``.
    """
    return
