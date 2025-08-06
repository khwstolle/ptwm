"""CLI for ptwm bench."""

from __future__ import annotations

import argparse
from pathlib import Path

from ptwm.ablation._attribute import attribute_leave_one_out, attribute_shapley
from ptwm.ablation._bench import bench_compress
from ptwm.ablation._compare import bench_compare
from ptwm.ablation._replay import bench_replay


def add_bench_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:  # type: ignore[type-arg]
    """Register the ``bench`` sub-command."""
    p = subparsers.add_parser("bench", help="Ablation harness.")
    sub = p.add_subparsers(dest="bench_cmd", required=True)

    c = sub.add_parser("compress", help="Run baseline + variants; emit CSV + summary.")
    c.add_argument("input", type=Path)
    c.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Baseline policy file. Omit for the default empty policy.",
    )
    c.add_argument(
        "--variant",
        type=Path,
        action="append",
        default=[],
        help="Variant policy file (repeatable).",
    )
    c.add_argument(
        "--metric",
        action="append",
        default=["ratio", "encode-time"],
        help="Metric to compute (repeatable). v1 emits ratio + encode-time.",
    )
    c.add_argument("--trials", type=int, default=1)
    c.add_argument("--out", type=Path, required=True)
    c.set_defaults(func=cmd_bench_compress)

    cmp = sub.add_parser(
        "compare",
        help="Diff two .ptwm files' extension tables and on-disk sizes.",
    )
    cmp.add_argument("a", type=Path, help="First .ptwm file.")
    cmp.add_argument("b", type=Path, help="Second .ptwm file.")
    cmp.set_defaults(func=cmd_bench_compare)

    attr = sub.add_parser("attribute", help="Per-contribution attribution.")
    attr.add_argument("input", type=Path, help="Input file to compress.")
    attr.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Baseline policy file. Omit for the default empty policy.",
    )
    attr.add_argument("--out", type=Path, required=True, help="Output directory.")
    attr.add_argument(
        "--shapley",
        action="store_true",
        default=False,
        help="Use Shapley attribution (gated by --max-attributions).",
    )
    attr.add_argument(
        "--leave-one-out",
        action="store_true",
        default=True,
        dest="leave_one_out",
        help="Use leave-one-out attribution (default).",
    )
    attr.add_argument(
        "--max-attributions",
        type=int,
        default=8,
        help="Maximum number of contributions to attribute (default: 8).",
    )
    attr.set_defaults(func=cmd_bench_attribute)

    rep = sub.add_parser("replay", help="Re-run a previous bench compress row.")
    rep.add_argument("results_csv", type=Path)
    rep.add_argument("--row", type=int, required=True)
    rep.add_argument("--out", type=Path, default=None)
    rep.set_defaults(func=cmd_bench_replay)

    return p


def cmd_bench_compress(args: argparse.Namespace) -> int:
    """Execute ``ptwm bench compress``."""
    bench_compress(
        input_path=args.input,
        baseline_policy=args.baseline,
        variant_policies=args.variant,
        metrics=args.metric,
        trials=args.trials,
        out_dir=args.out,
    )
    print(f"Results -> {args.out}/results.csv")
    print(f"Summary -> {args.out}/summary.md")
    return 0


def cmd_bench_compare(args: argparse.Namespace) -> int:
    """Execute ``ptwm bench compare``."""
    r = bench_compare(args.a, args.b)
    print(f"# bench compare {args.a} vs {args.b}")
    print(f"a size: {r.size_a}")
    print(f"b size: {r.size_b}")
    print(f"common: {len(r.common)} entries")
    print(f"only in a: {len(r.only_in_a)}")
    for cid, label in r.only_in_a:
        print(f"  - {label} ({cid[:24]}...)")
    print(f"only in b: {len(r.only_in_b)}")
    for cid, label in r.only_in_b:
        print(f"  + {label} ({cid[:24]}...)")
    return 0


def cmd_bench_attribute(args: argparse.Namespace) -> int:
    """Execute ``ptwm bench attribute``."""
    if args.shapley:
        result = attribute_shapley(
            input_path=args.input,
            baseline_policy=args.baseline,
            out_dir=args.out,
            max_attributions=args.max_attributions,
        )
    else:
        result = attribute_leave_one_out(
            input_path=args.input,
            baseline_policy=args.baseline,
            out_dir=args.out,
            max_attributions=args.max_attributions,
        )
    print(f"# bench attribute ({result.method})")
    for row in result.rows:
        print(
            f"  {row.label}: baseline={row.baseline_size} without={row.without_size} "
            f"delta={row.delta_bytes:+d} ({row.delta_pct:+.2f}%)"
        )
    return 0


def cmd_bench_replay(args: argparse.Namespace) -> int:
    """Execute ``ptwm bench replay``."""
    res = bench_replay(args.results_csv, args.row, args.out)
    print(
        f"replay [{res.row_variant} trial {res.row_trial}]: "
        f"expected={res.expected_output_bytes} actual={res.actual_output_bytes} "
        f"matches={res.matches}"
    )
    print(f"out: {res.out_file}")
    return 0 if res.matches else 1
