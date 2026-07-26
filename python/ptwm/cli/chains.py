"""``weights chains`` subcommand: validate and promote explored chains."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ptwm.classify import AuditLog, TensorRole
from ptwm.preprocessing._chains import Chain, ClassifierRole, chains_for


def _role_to_classifier_role(role_str: str) -> ClassifierRole:
    tensor_role = TensorRole(role_str)
    _map = {
        TensorRole.STANDARD: ClassifierRole.STANDARD,
        TensorRole.SCALE_BLOCK: ClassifierRole.SCALE_BLOCK,
        TensorRole.SCALE_GLOBAL: ClassifierRole.SCALE_GLOBAL,
        TensorRole.PACKED_VALUES: ClassifierRole.PACKED_VALUES,
    }
    return _map[tensor_role]


def _validate_chain(chain_bytes: bytes) -> tuple[bool, str]:
    """Deserialise chain bytes and validate basic structural invariants.

    Returns (ok, message).
    """
    try:
        from ptwm.preprocessing._explorer import is_legal_chain

        chain = _deserialise_chain(chain_bytes)
        if chain is None:
            return False, "failed to deserialise chain bytes"
        if not is_legal_chain(chain):
            return False, "chain violates one or more pruning rules"
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, f"validation error: {exc}"


class _Truncated(Exception):
    """Raised internally when chain_bytes is shorter than expected."""


def _deserialise_chain(chain_bytes: bytes) -> Chain | None:
    """Best-effort structural parse of the wire-format chain blob.

    A lightweight sanity parser — the canonical deserialiser lives in Rust.
    Returns None on malformed input.
    """
    try:
        return _parse_chain(chain_bytes)
    except _Truncated:
        return None


def _parse_chain(chain_bytes: bytes) -> Chain:  # noqa: PLR0912
    """Inner parser — raises _Truncated instead of returning None.

    ``chain_bytes`` is a self-contained blob: a local extension
    table (``count: u32`` + entries) followed by the chain wire bytes.  The
    extension table maps ``u16`` table indices to op-ids, allowing this parser
    to resolve node identifiers.
    """
    import struct

    # --- Step 1: parse the extension table prefix and build index→op_id map.
    try:
        from ptwm._core import chain_blob_info  # type: ignore[import]

        et_size, raw_table = chain_blob_info(chain_bytes)
        # raw_table is [(table_idx, op_id), ...]
        idx_to_opid: dict[int, int] = dict(raw_table)
        data = chain_bytes[et_size:]
    except Exception:
        # Fallback for legacy blobs or when the extension is not available:
        # assume no ET prefix and raw op_id encoding.
        idx_to_opid = {}
        data = chain_bytes

    # --- Step 2: parse the chain wire section (post-ET bytes).
    if len(data) < 9:
        raise _Truncated
    n_nodes, n_edges, n_terminals, nodes_len, edges_len, terminals_len = (
        struct.unpack_from("<BBBHHH", data)
    )
    if len(data) < 9 + nodes_len + edges_len + terminals_len:
        raise _Truncated

    # Rebind `chain_bytes` to the chain section for the rest of the parser so
    # the existing offset arithmetic continues to work unchanged.
    chain_bytes = data

    from ptwm.preprocessing._chains import Chain, ChainEdge, ChainNode, TerminalRef

    def need(pos: int, n: int) -> None:
        if pos + n > len(chain_bytes):
            raise _Truncated

    def _role_size(pos: int) -> int:  # noqa: PLR0911
        """Return total wire size (incl. tag byte) of the role at `pos`.

        Mirrors `Role::read` in `crates/ptwm-core/src/types/role.rs`.
        Raises `_Truncated` if the buffer is too short for the indicated tag.
        """
        need(pos, 1)
        tag = chain_bytes[pos]
        if tag == 0x00:  # Scale: tag + format(1)
            need(pos, 2)
            return 2
        if tag == 0x01:  # Value: tag + ValueFormat (1 or 2)
            need(pos, 2)
            sub = chain_bytes[pos + 1]
            if sub == 0:  # Fp4E2m1
                return 2
            if sub == 1:  # IntN { bits }
                need(pos, 3)
                return 3
            raise _Truncated
        if tag == 0x02:  # MantissaByte: tag + index + of
            need(pos, 3)
            return 3
        if tag == 0x03:  # ExponentByte
            return 1
        if tag == 0x04:  # Nibble: tag + kind
            need(pos, 2)
            return 2
        if tag == 0x05:  # IntegerByte: tag + index + of
            need(pos, 3)
            return 3
        if tag == 0x06:  # Residual: tag + ResidualFormat (1 or 3)
            need(pos, 2)
            sub = chain_bytes[pos + 1]
            if sub == 0:  # Xor
                return 2
            if sub == 1:  # FloatDelta { dtype: u16 }
                need(pos, 4)
                return 4
            raise _Truncated
        if tag == 0x07:  # GlobalScale: tag + format(1)
            need(pos, 2)
            return 2
        if tag == 0x08:  # Raw
            return 1
        if tag == 0x09:  # Index
            return 1
        if tag == 0xFF:  # Vendor: tag + tag(u16) + len(u16) + bytes
            need(pos, 5)
            vlen = int.from_bytes(chain_bytes[pos + 3 : pos + 5], "little")
            need(pos, 5 + vlen)
            return 5 + vlen
        raise _Truncated

    # Parse nodes
    nodes: list[ChainNode] = []
    offset = 9
    for _ in range(n_nodes):
        need(offset, 3)
        raw_id, param_len = struct.unpack_from("<HB", chain_bytes, offset)
        offset += 3
        params = chain_bytes[offset : offset + param_len]
        offset += param_len
        # raw_id is a table index; resolve to op_id if possible,
        # otherwise treat it as a raw op_id (legacy fallback path).
        op_id = idx_to_opid.get(raw_id, raw_id)
        nodes.append(ChainNode(op_id=op_id, params=params))

    # Parse edges
    edges: list[ChainEdge] = []
    for _ in range(n_edges):
        need(offset, 5)  # BBBB + has_role byte
        src_node, src_out, dst_node, dst_in = struct.unpack_from(
            "<BBBB", chain_bytes, offset
        )
        offset += 4
        has_role = chain_bytes[offset]
        offset += 1
        role_override = None
        if has_role:
            role_size = _role_size(offset)
            role_override = chain_bytes[offset : offset + role_size]
            offset += role_size
        need(offset, 2)
        vendor_len = struct.unpack_from("<H", chain_bytes, offset)[0]
        offset += 2
        vendor_bytes = chain_bytes[offset : offset + vendor_len]
        offset += vendor_len
        edges.append(
            ChainEdge(
                src_node=src_node,
                src_output_idx=src_out,
                dst_node=dst_node,
                dst_input_idx=dst_in,
                role_override=role_override,
                vendor_bytes=vendor_bytes,
            )
        )

    # Parse terminals
    terminals: list[TerminalRef] = []
    for _ in range(n_terminals):
        need(offset, 3)  # node_idx + output_idx + first role byte
        node_idx, output_idx = struct.unpack_from("<BB", chain_bytes, offset)
        offset += 2
        role_size = _role_size(offset)
        role_bytes = chain_bytes[offset : offset + role_size]
        offset += role_size
        terminals.append(
            TerminalRef(
                node_idx=node_idx,
                output_idx=output_idx,
                role=role_bytes,
            )
        )

    return Chain(nodes=nodes, edges=edges, terminals=terminals)


_OP_NAMES: dict[int, str] = {
    0x0000: "Source",
    0x0001: "Terminal",
    0x0010: "BitReorderIeee16",
    0x0011: "BitReorderIeee32",
    0x0012: "BitReorderFp8E4M3",
    0x0013: "BitReorderFp8E5M2",
    0x0020: "ByteSplit",
    0x0021: "NibbleSplit",
    0x0030: "BytePassthrough",
    0x0040: "MxFp4Deinterleave",
    0x0041: "BlockMicroscalingRepack",
    0x0050: "XorDelta",
    0x0051: "FloatDelta",
    0x0060: "IndexBitwidthPack",
    0x0061: "EntropyEstimate",
    0x0062: "Concat",
    0x0063: "Reshape",
    0x0070: "BurrowsWheeler",
    0x0071: "MoveToFront",
}


def _op_name(op_id: int) -> str:
    return _OP_NAMES.get(op_id, f"Op#0x{op_id:04x}")


def _print_chain_structure(chain: Chain) -> None:
    """Print the parsed chain in a maintainer-readable form."""
    print(f"  nodes ({len(chain.nodes)}):")
    for i, node in enumerate(chain.nodes):
        params_hex = (
            "(" + ", ".join(f"0x{b:02x}" for b in node.params) + ")"
            if node.params
            else "()"
        )
        print(f"    [{i}] {_op_name(node.op_id)} params={params_hex}")
    print(f"  edges ({len(chain.edges)}):")
    for e in chain.edges:
        suffix = " role_override=set" if e.role_override is not None else ""
        print(
            f"    {e.src_node}.out[{e.src_output_idx}] → "
            f"{e.dst_node}.in[{e.dst_input_idx}]{suffix}"
        )
    print(f"  terminals ({len(chain.terminals)}):")
    for t in chain.terminals:
        print(f"    node {t.node_idx}.out[{t.output_idx}]")


def _print_chain_report(
    dtype_code: int,
    classifier_role: ClassifierRole,
    chain_bytes: bytes,
) -> None:
    """Print a maintainer-readable report for one discovered chain.

    The output is *guidance*, not an automatic patch: the maintainer should
    write a named builder in ``python/ptwm/preprocessing/_chains.py``
    following the existing pattern, then add it to ``PRODUCTION_CHAINS``.
    """
    n_existing = len(list(chains_for(dtype_code, classifier_role)))
    chain = _deserialise_chain(chain_bytes)

    print(
        f"# dtype_code={dtype_code:#06x}, role={classifier_role.name} "
        f"(production entries: {n_existing})"
    )
    print(f"# wire bytes ({len(chain_bytes)} B):")
    hex_lines = []
    line: list[str] = []
    for b in chain_bytes:
        line.append(f"0x{b:02x}")
        if len(line) == 12:
            hex_lines.append(", ".join(line))
            line = []
    if line:
        hex_lines.append(", ".join(line))
    for h in hex_lines:
        print(f"#   {h}")
    if chain is not None:
        print("# structure:")
        _print_chain_structure(chain)
    else:
        print("# structure: <unparseable>")
    print(
        "# To promote: write a named builder in "
        "python/ptwm/preprocessing/_chains.py following the existing "
        "CHAIN_* pattern (Source(shape) + the ops above) and register it "
        "in PRODUCTION_CHAINS."
    )


def handle_promote(args: argparse.Namespace) -> None:
    """Handle ``ptwm chains promote`` subcommand."""
    audit_path = Path(args.audit)
    if not audit_path.exists():
        print(f"error: audit log not found: {audit_path}", file=sys.stderr)
        sys.exit(1)

    try:
        audit = AuditLog.from_cbor(audit_path.read_bytes())
    except Exception as exc:  # noqa: BLE001
        print(f"error: could not read audit log: {exc}", file=sys.stderr)
        sys.exit(1)

    discovered = audit.discovered_chains()
    if not discovered:
        print("No discovered chains in audit log.", file=sys.stderr)
        sys.exit(0)

    any_error = False
    for dc in discovered:
        ok, msg = _validate_chain(dc.chain)
        tag = f"dtype={dc.dtype_code:#06x} role={dc.role}"
        if not ok:
            print(f"[SKIP] {tag}: {msg}", file=sys.stderr)
            any_error = True
            continue

        try:
            classifier_role = _role_to_classifier_role(dc.role)
        except (KeyError, ValueError) as exc:
            print(f"[SKIP] {tag}: unknown role mapping: {exc}", file=sys.stderr)
            any_error = True
            continue

        print(
            f"[OK]   {tag}: {dc.n_candidates_tried} candidates tried, "
            f"sample={dc.sample_tensor_name}"
        )
        _print_chain_report(dc.dtype_code, classifier_role, dc.chain)
        print()

    if any_error:
        sys.exit(1)


def handle_cache_info(args: argparse.Namespace) -> None:  # noqa: ARG001
    """Print the user-local chain cache contents."""
    from ptwm.preprocessing._cache import cache_dir, cache_info  # noqa: PLC0415

    info = cache_info()
    root = cache_dir()
    if not info:
        print(f"Chain cache empty (root: {root}).")
        return
    print(f"Chain cache root: {root}")
    for fname, count in info.items():
        marker = "  " if count >= 0 else "! "
        suffix = "" if count >= 0 else " (corrupt or stale schema)"
        print(f"{marker}{fname}: {count} chain(s){suffix}")


def handle_cache_clear(args: argparse.Namespace) -> None:  # noqa: ARG001
    from ptwm.preprocessing._cache import clear_cache  # noqa: PLC0415

    removed = clear_cache()
    print(f"Removed {removed} cache file(s).")


def handle_export(args: argparse.Namespace) -> None:
    """Bundle the current user cache into a single portable file."""
    import cbor2  # noqa: PLC0415

    from ptwm.preprocessing._cache import (  # noqa: PLC0415
        cache_dir,
        load_cached_entries,
    )

    root = cache_dir()
    if not root.exists():
        print("error: chain cache is empty; nothing to export.", file=sys.stderr)
        sys.exit(1)

    bundle: list[dict[str, object]] = []
    for f in sorted(root.glob("*.cbor")):
        try:
            dtype_hex, role_str, bucket_str = f.stem.split("_")
            dtype_code = int(dtype_hex, 16)
            role = ClassifierRole(int(role_str))
            signature_bucket = int(bucket_str)
        except (ValueError, KeyError):
            continue
        for entry in load_cached_entries(dtype_code, role, signature_bucket):
            bundle.append(
                {
                    "dtype_code": dtype_code,
                    "role": int(role),
                    "signature_bucket": signature_bucket,
                    "internal_dtype": entry.internal_dtype,
                    "ops": list(entry.ops),
                    "byte_split_planes": entry.byte_split_planes,
                }
            )

    payload = {"schema_version": 2, "entries": bundle}
    out_path = Path(args.output)
    out_path.write_bytes(b"PTWMEXC\x01" + cbor2.dumps(payload))
    print(f"Exported {len(bundle)} chain(s) to {out_path}.")


def handle_import(args: argparse.Namespace) -> None:
    """Load a portable export bundle into the user cache."""
    import cbor2  # noqa: PLC0415

    from ptwm.preprocessing._cache import (  # noqa: PLC0415
        CacheEntry,
        save_cached_entries,
    )
    from ptwm.preprocessing._explorer import (  # noqa: PLC0415
        _build_linear_chain,
        is_legal_chain,
    )

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"error: import file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    raw = in_path.read_bytes()
    if not raw.startswith(b"PTWMEXC\x01"):
        print(
            f"error: {in_path}: missing PTWMEXC magic; refusing to load.",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        payload = cbor2.loads(raw[len(b"PTWMEXC\x01") :])
    except cbor2.CBORDecodeError as exc:
        print(f"error: {in_path}: malformed CBOR: {exc}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(payload, dict):
        print(f"error: {in_path}: payload is not a CBOR map.", file=sys.stderr)
        sys.exit(1)
    if payload.get("schema_version") != 2:
        print(
            f"error: {in_path}: unsupported schema_version "
            f"{payload.get('schema_version')!r}.",
            file=sys.stderr,
        )
        sys.exit(1)

    by_key: dict[tuple[int, ClassifierRole, int], list[CacheEntry]] = {}
    rejected = 0
    for raw_entry in payload.get("entries", []):
        try:
            dtype_code = int(raw_entry["dtype_code"])
            role = ClassifierRole(int(raw_entry["role"]))
            signature_bucket = int(raw_entry["signature_bucket"])
            internal_dtype = int(raw_entry["internal_dtype"])
            ops = tuple(int(o) for o in raw_entry["ops"])
            planes = int(raw_entry["byte_split_planes"])
        except (KeyError, ValueError, TypeError):
            rejected += 1
            continue
        # Every imported chain runs through `is_legal_chain` — the import file
        # may have been produced by a peer (or a malicious actor) on a
        # different op catalogue; the validator is the boundary.
        try:
            sample = _build_linear_chain(internal_dtype, [1], ops, planes)
        except (KeyError, ValueError, IndexError, AttributeError):
            rejected += 1
            continue
        if not is_legal_chain(sample):
            rejected += 1
            continue
        entry = CacheEntry(
            internal_dtype=internal_dtype, ops=ops, byte_split_planes=planes
        )
        by_key.setdefault((dtype_code, role, signature_bucket), []).append(entry)

    total_added = 0
    for (dtype_code, role, signature_bucket), entries in by_key.items():
        total_added += save_cached_entries(dtype_code, role, entries, signature_bucket)

    print(
        f"Imported {total_added} new chain(s) from {in_path} "
        f"({rejected} rejected by validator)."
    )


def add_chains_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``chains`` subcommand and its sub-subcommands."""
    chains_parser = subparsers.add_parser(
        "chains",
        help="Chain management utilities.",
    )
    chains_sub = chains_parser.add_subparsers(dest="chains_command", required=True)

    promote_parser = chains_sub.add_parser(
        "promote",
        help=(
            "Validate explored chains from an audit log and print a "
            "structural report for promoting them into PRODUCTION_CHAINS by hand."
        ),
    )
    promote_parser.add_argument(
        "audit",
        metavar="AUDIT_LOG",
        help="Path to a CBOR audit log (e.g. produced by compress_safetensors_file).",
    )
    promote_parser.set_defaults(func=handle_promote)

    cache_parser = chains_sub.add_parser(
        "cache",
        help="Inspect or clear the user-local chain cache.",
    )
    cache_sub = cache_parser.add_subparsers(dest="cache_command", required=True)
    info_parser = cache_sub.add_parser("info", help="Print cache contents.")
    info_parser.set_defaults(func=handle_cache_info)
    clear_parser = cache_sub.add_parser("clear", help="Delete every cache entry.")
    clear_parser.set_defaults(func=handle_cache_clear)

    export_parser = chains_sub.add_parser(
        "export",
        help="Export the user-local cache as a portable file.",
    )
    export_parser.add_argument(
        "output",
        metavar="OUTPUT",
        help="Path to write the export bundle to (CBOR).",
    )
    export_parser.set_defaults(func=handle_export)

    import_parser = chains_sub.add_parser(
        "import",
        help="Import chains from an export bundle (validated before storing).",
    )
    import_parser.add_argument(
        "input",
        metavar="INPUT",
        help="Path to a bundle produced by `ptwm chains export`.",
    )
    import_parser.set_defaults(func=handle_import)
