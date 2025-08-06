"""ptwm bench compress implementation.

Runs the compressor under a baseline policy + zero-or-more variant
policies, captures per-run metrics, emits CSV + Markdown summary.
"""

from __future__ import annotations

import csv
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ptwm import CompressionConfig, Compressor, Format
from ptwm.ablation._resolve import resolve_active


@dataclass
class BenchRow:
    """Single per-trial metric row."""

    variant: str
    trial: int
    input_bytes: int
    output_bytes: int
    ratio: float
    encode_ns: int
    decode_ns: int
    peak_mem_bytes: int
    resolved_policy_toml: str
    out_file: str


@dataclass
class BenchResult:
    """Aggregated result from a bench compress run."""

    rows: list[BenchRow] = field(default_factory=list)


def bench_compress(
    input_path: str | Path,
    baseline_policy: str | Path | None,
    variant_policies: Iterable[str | Path],
    metrics: list[str],
    trials: int,
    out_dir: str | Path,
) -> BenchResult:
    """Run the compressor under each policy and emit per-trial metrics.

    Returns a :class:`BenchResult` holding every row; also writes
    ``<out_dir>/results.csv``, ``<out_dir>/per_tensor.csv``, and
    ``<out_dir>/summary.md``.

    Parameters
    ----------
    input_path:
        Path to the raw binary input file.
    baseline_policy:
        Path to a TOML policy file for the baseline run, or ``None`` to
        use the default empty policy.
    variant_policies:
        Iterable of paths to variant policy TOML files.
    metrics:
        List of metric names to surface in the summary.  v1 always
        captures ``ratio`` and ``encode-time``; other names are recorded
        in the summary for forward compatibility.
    trials:
        Number of compress trials to execute per variant.
    out_dir:
        Directory to write ``results.csv`` and ``summary.md`` into.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(input_path)
    data = input_path.read_bytes()

    rows: list[BenchRow] = []
    variants: list[tuple[str, Path | None]] = []
    variants.append(("baseline", Path(baseline_policy) if baseline_policy else None))
    for v in variant_policies:
        p = Path(v)
        variants.append((p.stem, p))

    for label, policy_path in variants:
        resolved = resolve_active(policy_path)
        # Constrain the trial-encode menu to codecs whose canonical id is
        # in the policy's effective_global set. Variants that exclude e.g.
        # Huffman will now fall through to the next-best allowed coder
        # rather than silently using the unconstrained menu.
        config = CompressionConfig.from_resolved_policy(
            resolved,
            base=CompressionConfig(input_format=Format.BYTE),
        )
        for trial in range(trials):
            t0 = time.perf_counter_ns()
            blob = Compressor(config).compress(data)
            t1 = time.perf_counter_ns()
            out_file = out_dir / f"{label}-trial{trial}.ptwm"
            out_file.write_bytes(blob)
            rows.append(
                BenchRow(
                    variant=label,
                    trial=trial,
                    input_bytes=len(data),
                    output_bytes=len(blob),
                    ratio=(len(blob) / len(data)) if len(data) else 0.0,
                    encode_ns=t1 - t0,
                    decode_ns=0,  # measured by --metric decode-time in a follow-up
                    peak_mem_bytes=0,  # measured by --metric peak-mem in a follow-up
                    resolved_policy_toml=resolved.to_toml(),
                    out_file=str(out_file),
                )
            )
    _write_results_csv(out_dir / "results.csv", rows)
    _write_summary_md(out_dir / "summary.md", rows, metrics)
    return BenchResult(rows=rows)


def _write_results_csv(path: Path, rows: list[BenchRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "variant",
                "trial",
                "input_bytes",
                "output_bytes",
                "ratio",
                "encode_ns",
                "decode_ns",
                "peak_mem_bytes",
                "out_file",
                "resolved_policy_toml",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.variant,
                    r.trial,
                    r.input_bytes,
                    r.output_bytes,
                    f"{r.ratio:.6f}",
                    r.encode_ns,
                    r.decode_ns,
                    r.peak_mem_bytes,
                    r.out_file,
                    r.resolved_policy_toml,
                ]
            )


def _write_summary_md(path: Path, rows: list[BenchRow], metrics: list[str]) -> None:
    by_variant: dict[str, list[BenchRow]] = {}
    for r in rows:
        by_variant.setdefault(r.variant, []).append(r)

    lines = ["# ptwm bench compress summary", ""]
    lines.append("| variant | trials | mean_ratio | mean_encode_ms |")
    lines.append("|---|---|---|---|")
    for variant, vrows in by_variant.items():
        mean_ratio = sum(r.ratio for r in vrows) / len(vrows)
        mean_encode_ms = sum(r.encode_ns for r in vrows) / len(vrows) / 1e6
        lines.append(
            f"| {variant} | {len(vrows)} | {mean_ratio:.4f} | {mean_encode_ms:.2f} |"
        )
    lines.append("")
    lines.append(f"_Metrics requested: {', '.join(metrics)}_")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
