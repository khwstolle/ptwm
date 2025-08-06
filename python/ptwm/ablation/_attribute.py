"""ptwm bench attribute -- leave-one-out + Shapley.

Leave-one-out: re-encode N times, each with one contribution-id absent
from the allow set. The marginal contribution of id X is
size(without X) - size(baseline).

Shapley: enumerate all 2^N coalitions, average marginal contributions.
Gated by --max-attributions <= 8 to avoid factorial blowup.

Note: the ignore-policy wiring is not yet implemented. The row shapes
are correct and the baseline sizes are populated; the ``without_size``
/ ``delta_bytes`` / ``delta_pct`` fields stay at zero until the
resolved-policy round-trip is wired in.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path

from ptwm._core import list_extension_table_entries
from ptwm.ablation._bench import bench_compress


@dataclass
class AttributionRow:
    """Per-contribution attribution row."""

    contribution_id: str
    label: str
    baseline_size: int
    without_size: int
    delta_bytes: int
    delta_pct: float


@dataclass
class AttributionResult:
    """Result of a leave-one-out or Shapley attribution run."""

    method: str  # "leave-one-out" or "shapley"
    rows: list[AttributionRow]


def attribute_leave_one_out(
    input_path: str | Path,
    baseline_policy: str | Path | None,
    out_dir: str | Path,
    max_attributions: int = 32,
) -> AttributionResult:
    """O(n) re-encodes -- one per used contribution with that id ignored.

    Parameters
    ----------
    input_path:
        Path to the raw binary input file to compress.
    baseline_policy:
        Path to a TOML policy file for the baseline run, or ``None`` to
        use the default empty policy.
    out_dir:
        Directory for intermediate compressed files and CSV output.
    max_attributions:
        Maximum number of contributions to attribute.  If the baseline
        uses more than this many contributions, only the first
        ``max_attributions`` are attributed.

    Returns
    -------
    AttributionResult
        One row per attributed contribution.  ``delta_bytes`` and
        ``delta_pct`` are zero until the
        ignore-policy path is wired in.
    """
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_dir = out_dir / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    base = bench_compress(
        input_path=input_path,
        baseline_policy=baseline_policy,
        variant_policies=[],
        metrics=["ratio"],
        trials=1,
        out_dir=base_dir,
    )
    base_row = base.rows[0]
    baseline_size = base_row.output_bytes
    base_file = Path(base_row.out_file)
    used_ids = list_extension_table_entries(str(base_file))
    if max_attributions and len(used_ids) > max_attributions:
        used_ids = used_ids[:max_attributions]

    rows: list[AttributionRow] = []
    for cid, label, _kind, _flavor in used_ids:
        # The actual re-encode with the contribution ignored is not yet
        # wired in. For now the "without_size" is a placeholder equal to
        # the baseline so the row shape is correct.
        rows.append(
            AttributionRow(
                contribution_id=cid,
                label=label,
                baseline_size=baseline_size,
                without_size=baseline_size,
                delta_bytes=0,
                delta_pct=0.0,
            )
        )
    return AttributionResult(method="leave-one-out", rows=rows)


def attribute_shapley(
    input_path: str | Path,
    baseline_policy: str | Path | None,
    out_dir: str | Path,
    max_attributions: int = 8,
) -> AttributionResult:
    """Shapley attribution. Gated by max_attributions <= 8.

    Parameters
    ----------
    input_path:
        Path to the raw binary input file to compress.
    baseline_policy:
        Path to a TOML policy file for the baseline run, or ``None`` to
        use the default empty policy.
    out_dir:
        Directory for intermediate compressed files and CSV output.
    max_attributions:
        Maximum number of contributions allowed.  Raises ``ValueError``
        when the baseline uses more contributions than this limit, to
        avoid a 2^n * n encode blowup.

    Returns
    -------
    AttributionResult
        One row per contribution.  ``delta_bytes`` and ``delta_pct`` are
        zero until the ignore-policy path is wired in.

    Raises
    ------
    ValueError
        When the baseline references more contributions than
        ``max_attributions``.
    """
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_dir = out_dir / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    base = bench_compress(
        input_path=input_path,
        baseline_policy=baseline_policy,
        variant_policies=[],
        metrics=["ratio"],
        trials=1,
        out_dir=base_dir,
    )
    base_row = base.rows[0]
    used_ids = list_extension_table_entries(str(Path(base_row.out_file)))
    n = len(used_ids)
    if n > max_attributions:
        msg = (
            f"refusing Shapley attribution with n={n} > "
            f"max_attributions={max_attributions}; "
            f"would run {2**n * n} encodes"
        )
        raise ValueError(msg)

    # Coalition enumeration is not yet implemented.
    # Keep itertools/math imported so a follow-up can add the loop
    # without adding new imports.
    _ = itertools
    _ = math

    rows: list[AttributionRow] = []
    for cid, label, _kind, _flavor in used_ids:
        rows.append(
            AttributionRow(
                contribution_id=cid,
                label=label,
                baseline_size=base_row.output_bytes,
                without_size=base_row.output_bytes,
                delta_bytes=0,
                delta_pct=0.0,
            )
        )
    return AttributionResult(method="shapley", rows=rows)
