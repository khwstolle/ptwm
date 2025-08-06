"""Active-trust-aware policy resolution.

`resolve_active(policy_path)` reads the policy TOML at `policy_path`
(or the default empty policy if None), collects the set of canonical
IDs that are currently trusted by the active keyring (the
bundled-then-pinned state + the user's `keys.toml`), and returns the
ResolvedPolicy.
"""

from __future__ import annotations

import os
from pathlib import Path

from ptwm._rust.policy import PolicyFile, ResolvedPolicy
from ptwm._rust.trust import (
    Keyring,
    trust_evaluate_bundled,
    trust_user_keys_path,
)


def _trusted_ids_for_active_state() -> list[str]:
    """Collect canonical-id hex strings trusted by the active keyring.

    Currently this returns the `contribution_hash` pins from the user's
    `keys.toml`. Author-key entries are also trusted but they bless
    *future* contributions by that author rather than a specific id --
    those are evaluated at extension-table verify time, not here.

    The bundled keyring's contents likewise become trusted only when
    `trust_evaluate_bundled().kind == "Match"`. This currently returns
    the contribution-hash pins; the replay path lifts this into a
    richer set when the harness needs author-key transitive trust.
    """
    pins: list[str] = []

    user_keys_path = trust_user_keys_path()
    if Path(user_keys_path).exists():
        keyring = Keyring.load(user_keys_path)
        for entry in keyring.entries():
            if entry.kind == "contribution_hash" and entry.canonical_id:
                pins.append(entry.canonical_id)

    # Bundled state doesn't carry contribution_hash pins by default in
    # v1, but if the bundled keyring is active and ever ships pins,
    # collect them here too.
    bundled = trust_evaluate_bundled()
    if bundled.kind in ("Match", "FreshInstall"):
        # No bundled pins in v1; placeholder for forward compat.
        pass

    return pins


def resolve_active(policy_path: str | os.PathLike | None = None) -> ResolvedPolicy:
    """Resolve `policy_path` against the *active* trust state.

    `policy_path` may be a path to a TOML policy file or ``None`` to use
    the default empty policy.

    Callers (`ptwm bench compress`) call this once at run start
    and serialize the resulting TOML into every CSV row for
    reproducibility.
    """
    if policy_path is None:
        pf = PolicyFile.default_empty()
    else:
        pf = PolicyFile.load(str(Path(policy_path)))
    trusted = _trusted_ids_for_active_state()
    return pf.resolve(trusted)
