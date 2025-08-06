"""Re-export of the trust bindings from the ``ptwm._core`` PyO3 extension.

The Rust bridge registers ``Keyring``, ``TrustEntry``, ``BundleStatus``, and
the helper functions directly on the flat ``_core`` module.  This shim groups
them under ``ptwm._rust.trust`` so the CLI and tests can use a stable,
namespaced import path:

    from ptwm._rust import trust as _trust
    _trust.trust_evaluate_bundled()
    _trust.Keyring.load(path)
"""

from __future__ import annotations

from ptwm._core import (  # type: ignore[import]
    BundleStatus,
    Keyring,
    PublicKey,
    SecretKey,
    TrustEntry,
    trust_accept_update,
    trust_apply_fresh_install,
    trust_evaluate_bundled,
    trust_user_keys_path,
)

__all__ = [
    "BundleStatus",
    "Keyring",
    "PublicKey",
    "SecretKey",
    "TrustEntry",
    "trust_accept_update",
    "trust_apply_fresh_install",
    "trust_evaluate_bundled",
    "trust_user_keys_path",
]
