"""ptwm bench compare implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ptwm._core import list_extension_table_entries


@dataclass
class CompareResult:
    """Result of comparing extension tables in two `.ptwm` files."""

    only_in_a: list[tuple[str, str]]  # (canonical_id, label)
    only_in_b: list[tuple[str, str]]
    common: list[tuple[str, str]]
    size_a: int
    size_b: int


def bench_compare(a: str | Path, b: str | Path) -> CompareResult:
    """Compare the extension tables and file sizes of two `.ptwm` files.

    Parameters
    ----------
    a:
        Path to the first `.ptwm` file.
    b:
        Path to the second `.ptwm` file.

    Returns
    -------
    CompareResult
        A dataclass describing which contribution IDs are present only in
        *a*, only in *b*, in both, and the on-disk sizes of each file.
    """
    a_path, b_path = Path(a), Path(b)
    a_entries = list_extension_table_entries(str(a_path))
    b_entries = list_extension_table_entries(str(b_path))

    a_ids = {cid: label for cid, label, _, _ in a_entries}
    b_ids = {cid: label for cid, label, _, _ in b_entries}

    only_a = sorted((cid, label) for cid, label in a_ids.items() if cid not in b_ids)
    only_b = sorted((cid, label) for cid, label in b_ids.items() if cid not in a_ids)
    common = sorted((cid, a_ids[cid]) for cid in (a_ids.keys() & b_ids.keys()))

    return CompareResult(
        only_in_a=only_a,
        only_in_b=only_b,
        common=common,
        size_a=a_path.stat().st_size,
        size_b=b_path.stat().st_size,
    )
