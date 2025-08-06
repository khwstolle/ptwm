"""Pack tensors into shards under a size threshold; preserve input order."""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["plan_shards", "shard_filename"]


def shard_filename(idx_one_based: int, total: int, *, suffix: str) -> str:
    """Return ``model-NNNNN-of-MMMMM.<suffix>``."""
    if not 1 <= idx_one_based <= total:
        msg = f"shard index {idx_one_based} out of [1, {total}]"
        raise ValueError(msg)
    return f"model-{idx_one_based:05d}-of-{total:05d}.{suffix}"


def plan_shards(
    sized_tensors: Sequence[tuple[str, int]],
    *,
    max_shard_size: int,
) -> list[list[tuple[str, int]]]:
    """Greedy pack tensors in input order, respecting ``max_shard_size``.

    A single tensor larger than ``max_shard_size`` gets its own shard.
    """
    if max_shard_size <= 0:
        msg = f"max_shard_size must be positive, got {max_shard_size}"
        raise ValueError(msg)
    shards: list[list[tuple[str, int]]] = []
    current: list[tuple[str, int]] = []
    current_size = 0
    for name, size in sized_tensors:
        if current and current_size + size > max_shard_size:
            shards.append(current)
            current = []
            current_size = 0
        current.append((name, size))
        current_size += size
    if current:
        shards.append(current)
    return shards
