"""CLI for ptwm ext — manage installed extensions."""

from __future__ import annotations

import argparse
import os
import sys
import tomllib
from pathlib import Path

import tomli_w

from ptwm._rust.ext import (
    ext_install,
    ext_list,
    ext_remove,
    ext_user_dir,
)
from ptwm.ext_tooling.init import init_extension

PIN_FILE = (
    Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    / "ptwm"
    / "ext"
    / "pins.toml"
)


def _add_install_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    inst = sub.add_parser("install", help="Install an extension from a source.")
    inst.add_argument(
        "source",
        help="local path | https:// | oci:// | git+...#sha | pip pkg | --bundled",
    )
    inst.set_defaults(func=cmd_install)


def _add_list_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    lst = sub.add_parser("list", help="List installed extensions.")
    lst.add_argument("--verbose", "-v", action="store_true")
    lst.set_defaults(func=cmd_list)


def _add_show_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    show = sub.add_parser("show", help="Show one installed extension's details.")
    show.add_argument("identifier")
    show.set_defaults(func=cmd_show)


def _add_remove_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    rm = sub.add_parser("remove", help="Uninstall an extension.")
    rm.add_argument("identifier")
    rm.set_defaults(func=cmd_remove)


def _add_verify_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    ver = sub.add_parser("verify", help="Re-verify installed signatures.")
    ver.add_argument("identifier", nargs="?")
    ver.set_defaults(func=cmd_verify)


def _add_pin_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    pin = sub.add_parser("pin", help="Pin an extension to a specific version.")
    pin.add_argument("identifier")
    pin.add_argument("--to", required=True, help="Version to pin to.")
    pin.set_defaults(func=cmd_pin)


def _add_init_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    init_p = sub.add_parser(
        "init", help="Scaffold a new extension project from a template."
    )
    init_p.add_argument("name")
    init_p.add_argument(
        "--lang", required=True, choices=("rust", "c", "zig", "assemblyscript")
    )
    init_p.add_argument(
        "--kind",
        default="plane_codec",
        help="Contribution kind (default: plane_codec).",
    )
    init_p.add_argument(
        "--flavor",
        default="wasm",
        choices=("wasm", "native"),
        help="Target flavor the scaffold builds for (default: wasm).",
    )
    init_p.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="Target directory. Default: ./<name>",
    )
    init_p.add_argument("--description")
    init_p.set_defaults(func=cmd_init)


def _add_build_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    build_p = sub.add_parser("build", help="Build the extension's binary (.wasm).")
    build_p.add_argument(
        "--release",
        action="store_true",
        default=True,
        help="Build with optimizations (default).",
    )
    build_p.add_argument(
        "--debug",
        action="store_false",
        dest="release",
        help="Build without optimizations.",
    )
    build_p.add_argument(
        "--flavor",
        default="wasm",
        choices=("wasm", "native"),
        help="Target flavor to build (default: wasm). 'native' is rust-only "
        "and produces a .so/.dylib cdylib instead of a wasm32-wasip1 module.",
    )
    build_p.set_defaults(func=cmd_build)


def _add_sign_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    sign_p = sub.add_parser("sign", help="Sign the bundle with an Ed25519 key.")
    sign_p.add_argument(
        "--key",
        type=Path,
        required=True,
        help="Path to the 32-byte raw Ed25519 seed file.",
    )
    sign_p.set_defaults(func=cmd_sign)


def _add_pack_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    pack_p = sub.add_parser(
        "pack", help="Package the bundle into <name>-<version>.tar.zst."
    )
    pack_p.set_defaults(func=cmd_pack)


def add_ext_parser(
    subparsers: argparse._SubParsersAction,  # type: ignore[type-arg]
) -> argparse.ArgumentParser:
    """Register the ``ext`` subcommand and its sub-subcommands."""
    p = subparsers.add_parser("ext", help="Manage installed extensions.")
    sub = p.add_subparsers(dest="ext_cmd", required=True)

    _add_install_parser(sub)
    _add_list_parser(sub)
    _add_show_parser(sub)
    _add_remove_parser(sub)
    _add_verify_parser(sub)
    _add_pin_parser(sub)
    _add_init_parser(sub)
    _add_build_parser(sub)
    _add_sign_parser(sub)
    _add_pack_parser(sub)

    return p


def cmd_install(args: argparse.Namespace) -> int:
    try:
        bundle_dir = ext_install(args.source)
    except Exception as exc:  # noqa: BLE001
        print(f"install failed: {exc}", file=sys.stderr)
        return 1
    if bundle_dir:
        print(f"Installed → {bundle_dir}")
    else:
        print("Installed (pip-managed; discovery will pick it up at PTWM startup)")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    entries = ext_list()
    if not entries:
        print(f"(no extensions installed in {ext_user_dir()})")
        return 0
    for e in entries:
        if args.verbose:
            print(
                f"- {e.bundle_name}@{e.bundle_version} "
                f"(flavors={','.join(e.installed_flavors) or 'none'}, "
                f"contribs={e.contribution_count})"
            )
            print(f"    path: {e.bundle_dir}")
            print(f"    pubkey: {e.author_pubkey}")
        else:
            print(f"{e.bundle_name}@{e.bundle_version}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    entries = ext_list()
    matches = [
        e
        for e in entries
        if e.bundle_name == args.identifier
        or f"{e.bundle_name}@{e.bundle_version}" == args.identifier
    ]
    if not matches:
        print(f"no extension matching {args.identifier!r}", file=sys.stderr)
        return 1
    for e in matches:
        print(f"name:         {e.bundle_name}")
        print(f"version:      {e.bundle_version}")
        print(f"pubkey:       {e.author_pubkey}")
        print(f"manifest:     {e.manifest_path}")
        print(f"bundle_dir:   {e.bundle_dir}")
        print(f"flavors:      {','.join(e.installed_flavors) or 'none'}")
        print(f"contribs:     {e.contribution_count}")
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    entries = ext_list()
    matches = [
        e
        for e in entries
        if e.bundle_name == args.identifier
        or f"{e.bundle_name}@{e.bundle_version}" == args.identifier
    ]
    if not matches:
        print(f"no extension matching {args.identifier!r}", file=sys.stderr)
        return 1
    removed = 0
    for e in matches:
        if ext_remove(e.bundle_dir):
            removed += 1
            print(f"Removed {e.bundle_name}@{e.bundle_version}")
    return 0 if removed > 0 else 1


def cmd_verify(args: argparse.Namespace) -> int:
    # v1: a stub — list each installed extension and report whether its
    # author pubkey is trusted. The Verifier provides the real check;
    # wiring that into the CLI is a follow-up.
    entries = ext_list()
    if args.identifier:
        entries = [e for e in entries if e.bundle_name == args.identifier]
        if not entries:
            print(f"no extension matching {args.identifier!r}", file=sys.stderr)
            return 1
    for e in entries:
        # Stub: report the pubkey; verifier integration is not yet wired in.
        print(
            f"{e.bundle_name}@{e.bundle_version}: pubkey={e.author_pubkey}"
            " (signature check stubbed)"
        )
    return 0


def cmd_pin(args: argparse.Namespace) -> int:
    PIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    pins: dict[str, object] = {}
    if PIN_FILE.exists():
        pins = tomllib.loads(PIN_FILE.read_text())
    pins[args.identifier] = {"version": args.to}
    PIN_FILE.write_text(tomli_w.dumps(pins))
    print(f"Pinned {args.identifier} to {args.to}")
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    target = Path(args.dir) if args.dir else Path(args.name)
    init_extension(
        target_dir=target,
        name=args.name,
        lang=args.lang,
        kind=args.kind,
        flavor=args.flavor,
        description=args.description,
    )
    print(f"scaffolded {args.kind} ({args.flavor}) extension at {target}")
    print(f"next: cd {target} && ptwm ext build")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    from ptwm.ext_tooling.build import BuildError, build_extension

    try:
        out = build_extension(Path.cwd(), release=args.release, flavor=args.flavor)
    except BuildError as exc:
        print(f"build failed: {exc}", file=sys.stderr)
        return 1
    print(f"built {out}")
    return 0


def cmd_sign(args: argparse.Namespace) -> int:
    from ptwm.ext_tooling.sign import sign_bundle

    try:
        out = sign_bundle(Path.cwd(), Path(args.key))
    except Exception as exc:  # noqa: BLE001
        print(f"sign failed: {exc}", file=sys.stderr)
        return 1
    print(f"signed → {out}")
    return 0


def cmd_pack(args: argparse.Namespace) -> int:
    from ptwm.ext_tooling.pack import pack_bundle

    try:
        out = pack_bundle(Path.cwd())
    except Exception as exc:  # noqa: BLE001
        print(f"pack failed: {exc}", file=sys.stderr)
        return 1
    print(f"packed → {out}")
    return 0
