"""CLI for ptwm policy — show / validate."""

from __future__ import annotations

import argparse
import sys

from ptwm._rust.policy import PolicyFile


def add_policy_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    p = subparsers.add_parser("policy", help="Inspect / validate a policy file.")
    sub = p.add_subparsers(dest="policy_cmd", required=True)

    show = sub.add_parser("show", help="Resolve and print the policy as TOML.")
    show.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to policy.toml; omit for the default empty policy.",
    )
    show.set_defaults(func=cmd_show)

    val = sub.add_parser("validate", help="Validate a policy file parses and resolves.")
    val.add_argument("path")
    val.set_defaults(func=cmd_validate)

    return p


def _load(path: str | None) -> PolicyFile:
    if path is None:
        return PolicyFile.default_empty()
    return PolicyFile.load(path)


def cmd_show(args: argparse.Namespace) -> int:
    pf = _load(args.path)
    # No active trust state on the CLI surface for v1; pass empty.
    # A future `--trusted` flag can bind this to the real active trust state.
    resolved = pf.resolve([])
    print(resolved.to_toml())
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    try:
        pf = PolicyFile.load(args.path)
        pf.resolve([])
    except Exception as exc:  # noqa: BLE001
        print(f"INVALID: {exc}", file=sys.stderr)
        return 2
    print("OK")
    return 0
