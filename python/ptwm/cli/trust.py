"""CLI for ``ptwm trust`` — manage trusted author keys."""

from __future__ import annotations

import argparse
import os
import sys

from ptwm._rust import trust as _trust


def add_trust_parser(
    subparsers: argparse._SubParsersAction,  # type: ignore[type-arg]
) -> argparse.ArgumentParser:
    """Register the ``trust`` subcommand and its sub-subcommands."""
    p = subparsers.add_parser("trust", help="Manage trusted author keys.")
    sub = p.add_subparsers(dest="trust_cmd", required=True)

    sub.add_parser("list", help="List active trust entries.").set_defaults(
        func=cmd_list
    )

    add_p = sub.add_parser("add", help="Add a trust entry.")
    add_p.add_argument(
        "--bundled",
        action="store_true",
        help="Import the PTWM-shipped bundled keyring.",
    )
    add_p.add_argument(
        "--key",
        type=str,
        metavar="ed25519:HEX",
        help="Trust a raw Ed25519 author public key.",
    )
    add_p.add_argument(
        "--pin",
        type=str,
        metavar="CONTRIBUTION_ID",
        help="Pin trust to a specific contribution hash.",
    )
    add_p.add_argument("--label", type=str, default=None)
    add_p.set_defaults(func=cmd_add)

    upd = sub.add_parser("update", help="Update an existing trust entry.")
    upd.add_argument(
        "--bundled",
        action="store_true",
        help="Accept the latest bundled keyring drift.",
    )
    upd.set_defaults(func=cmd_update)

    sub.add_parser("show", help="Show trust configuration summary.").set_defaults(
        func=cmd_show
    )

    rm = sub.add_parser("remove", help="Remove a trust entry.")
    rm.add_argument(
        "identifier",
        help="Pubkey, contribution_id, or label of the entry to remove.",
    )
    rm.set_defaults(func=cmd_remove)

    return p


def _user_keyring_path() -> str:
    return _trust.trust_user_keys_path()


def cmd_list(args: argparse.Namespace) -> int:  # noqa: ARG001
    """``ptwm trust list`` — show bundled status and user keyring entries."""
    status = _trust.trust_evaluate_bundled()
    if status.kind == "FreshInstall":
        print(
            "(bundled keyring not yet imported;"
            " run `ptwm trust add --bundled` to import)"
        )
    elif status.kind == "Match":
        print("(bundled keyring is active)")
    elif status.kind == "Mismatch":
        print("(bundled keyring drifted; run `ptwm trust update --bundled` to accept)")

    path = _user_keyring_path()
    if not os.path.exists(path):
        print(f"(no user keyring at {path})")
        return 0
    k = _trust.Keyring.load(path)
    for entry in k.entries():
        print(f"- kind={entry.kind} label={entry.label or '-'}", end="")
        if entry.pubkey:
            print(f" pubkey={entry.pubkey}", end="")
        if entry.canonical_id:
            print(f" id={entry.canonical_id}", end="")
        print()
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    """``ptwm trust add`` — add a bundled or custom trust entry."""
    if args.bundled:
        status = _trust.trust_evaluate_bundled()
        if status.kind == "FreshInstall":
            _trust.trust_apply_fresh_install()
            print("Bundled keyring imported.")
        elif status.kind == "Match":
            print("Bundled keyring already active.")
        elif status.kind == "Mismatch":
            print(
                "Bundled keyring drifted. Run `ptwm trust update --bundled` first.",
                file=sys.stderr,
            )
            return 2
        return 0

    if not args.key and not args.pin:
        print("error: provide --bundled, --key, or --pin", file=sys.stderr)
        return 2

    path = _user_keyring_path()
    k = _trust.Keyring.load(path)
    if args.key:
        k.add_author_key(args.key, args.label)
    if args.pin:
        k.add_contribution_hash(args.pin, args.label)
    k.save()
    print(f"User keyring written to {path}")
    return 0


def cmd_update(args: argparse.Namespace) -> int:
    """``ptwm trust update`` — accept bundled keyring drift."""
    if not args.bundled:
        print(
            "error: `ptwm trust update` currently only supports --bundled",
            file=sys.stderr,
        )
        return 2

    status = _trust.trust_evaluate_bundled()
    if status.kind == "FreshInstall":
        print("No bundled lock present; run `ptwm trust add --bundled` instead.")
        return 2
    if status.kind == "Match":
        print("Bundled keyring is already up to date.")
        return 0

    # Mismatch: print diff, then accept.
    print(f"Bundled keyring drift: old={status.old_hash_hex} new={status.new_hash_hex}")
    if status.added:
        print("Added:")
        for e in status.added:
            print(f"  + {e.kind} label={e.label or '-'}")
    if status.removed:
        print("Removed:")
        for e in status.removed:
            print(f"  - {e.kind} label={e.label or '-'}")
    _trust.trust_accept_update()
    print("Update accepted.")
    return 0


def cmd_show(args: argparse.Namespace) -> int:  # noqa: ARG001
    """``ptwm trust show`` — show trust configuration summary."""
    status = _trust.trust_evaluate_bundled()
    print(f"bundled status: {status.kind}")
    if status.kind == "Mismatch":
        print(f"  old_hash: {status.old_hash_hex}")
        print(f"  new_hash: {status.new_hash_hex}")
    path = _user_keyring_path()
    print(f"user keyring path: {path}")
    print(f"user keyring exists: {os.path.exists(path)}")
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    """``ptwm trust remove`` — remove a trust entry by identifier."""
    path = _user_keyring_path()
    if not os.path.exists(path):
        print("(no user keyring)")
        return 2
    k = _trust.Keyring.load(path)
    removed = k.remove(args.identifier)
    if not removed:
        print(f"no entry matched {args.identifier!r}", file=sys.stderr)
        return 1
    k.save()
    print(f"Removed entry matching {args.identifier}.")
    return 0
