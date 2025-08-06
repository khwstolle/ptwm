"""Resolved-policy replay for reproducibility.

Reads a `bench compress` results.csv, picks one row by index, and
re-runs the compression. The row's `resolved_policy_toml` field is
the **output** of `ResolvedPolicy.to_toml()` — an audit serialisation,
not a parseable PolicyFile — so replay verifies the size envelope
under the default config rather than reconstituting the original
policy. Reproducing the original constrained run requires the
original `policy.toml` (point ``bench compress`` at it directly).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from ptwm import CompressionConfig, Compressor, Format


@dataclass
class ReplayResult:
    row_variant: str
    row_trial: int
    expected_output_bytes: int
    actual_output_bytes: int
    matches: bool
    out_file: str


def bench_replay(
    results_csv: str | Path,
    row: int,
    out_dir: str | Path | None = None,
) -> ReplayResult:
    """Re-run a single row from a previous ``bench compress`` results CSV.

    Parameters
    ----------
    results_csv:
        Path to the ``results.csv`` produced by :func:`bench_compress`.
    row:
        Zero-based row index to replay.
    out_dir:
        Directory for the replayed ``.ptwm`` file.  Defaults to a
        ``replay/`` sub-directory next to the CSV.

    Returns
    -------
    ReplayResult
        Comparison of the expected (CSV) vs actual output sizes.
    """
    results_csv = Path(results_csv)
    out_dir = Path(out_dir) if out_dir is not None else results_csv.parent / "replay"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(results_csv.open(encoding="utf-8")))
    if row < 0 or row >= len(rows):
        msg = f"row index {row} out of range (csv has {len(rows)} rows)"
        raise IndexError(msg)
    r = rows[row]

    # The CSV doesn't preserve the source-input path or the original
    # policy file path; the replay reconstructs neither and instead does
    # an envelope check: re-encode a synthetic buffer of the original
    # size and confirm the output size matches. Reproducing the policy-
    # constrained encoding requires re-running ``bench compress`` against
    # the original input + policy.
    expected_size = int(r["output_bytes"])
    out_file = out_dir / f"replay-{r['variant']}-trial{r['trial']}.ptwm"

    synthetic_input = b"\x00" * int(r["input_bytes"])
    blob = Compressor(CompressionConfig(input_format=Format.BYTE)).compress(
        synthetic_input
    )
    out_file.write_bytes(blob)

    return ReplayResult(
        row_variant=r["variant"],
        row_trial=int(r["trial"]),
        expected_output_bytes=expected_size,
        actual_output_bytes=len(blob),
        matches=(len(blob) == expected_size),
        out_file=str(out_file),
    )
