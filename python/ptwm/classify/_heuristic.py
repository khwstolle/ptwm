"""Opt-in dtype + suffix heuristic for SCALE_BLOCK detection."""

from __future__ import annotations

from typing import Final

from ._role import TensorClassification, TensorRole

__all__ = ["HeuristicClassifier"]

_SUFFIXES: Final[tuple[str, ...]] = (
    ".weight_scale",
    "_scales",
    ".scale",
)
_DTYPES: Final[frozenset[str]] = frozenset({"uint8", "float8_e4m3fn", "float8_e5m2"})
_SIBLING_SUFFIXES: Final[tuple[str, ...]] = (".weight", "_blocks", "_packed")


def _matched_suffix(name: str) -> str | None:
    for s in _SUFFIXES:
        if name.endswith(s):
            return s
    return None


def _has_sibling(
    candidate_name: str,
    candidate_suffix: str,
    archive_keys: tuple[str, ...],
) -> bool:
    base = candidate_name[: -len(candidate_suffix)]
    return any(f"{base}{sib}" in archive_keys for sib in _SIBLING_SUFFIXES)


class HeuristicClassifier:
    """Conservative dtype + suffix + sibling matcher.

    Returns ``SCALE_BLOCK`` only when all three conditions hold; otherwise
    defers via ``None``.
    """

    __ptwm_canonical_id__ = "io.ptwm.builtin.classifier_heuristic"

    def classify(
        self,
        name: str,
        dtype: str,
        shape: tuple[int, ...],  # noqa: ARG002
        archive_keys: tuple[str, ...],
    ) -> TensorClassification | None:
        if dtype not in _DTYPES:
            return None
        suffix = _matched_suffix(name)
        if suffix is None:
            return None
        if not _has_sibling(name, suffix, archive_keys):
            return None
        return TensorClassification(
            role=TensorRole.SCALE_BLOCK,
            source="heuristic",
            pattern=f"*{suffix}",
        )
