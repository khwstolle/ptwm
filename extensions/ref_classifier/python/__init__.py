"""Reference contribution for the classifier kind, Python/host flavor."""

from __future__ import annotations

CANONICAL_ID = "blake3:" + "0" * 64
LABEL = "io.ptwm.ref.classifier"

_DEFAULT_ROLE = "weight"


def register() -> None:
    """Entry point called by ptwm._loader.load_all_host_extensions()."""
    from ptwm._rust.host import ContributionRegistry

    reg = ContributionRegistry()
    reg.register(CANONICAL_ID, "classifier", "thread", _impl)


def _impl(
    state: object,
    tensor_name: str,
    dtype_code: int,
    shape: list[int],
) -> str:
    """Passthrough classifier — always returns the default role.

    A real classifier inspects tensor_name / dtype_code / shape to assign
    a semantic role (e.g. "weight", "weight_scale", "activation").  This
    stub returns the ``weight`` role unconditionally as the safe default.
    """
    return _DEFAULT_ROLE
