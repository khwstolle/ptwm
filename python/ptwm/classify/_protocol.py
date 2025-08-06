"""Classifier protocol shared by every chain step."""

from __future__ import annotations

from typing import Protocol

from ._role import TensorClassification

__all__ = ["TensorClassifier"]


class TensorClassifier(Protocol):
    """Stateless per-tensor classifier.

    Returns ``None`` to defer to the next chain step.
    """

    def classify(
        self,
        name: str,
        dtype: str,
        shape: tuple[int, ...],
        archive_keys: tuple[str, ...],
    ) -> TensorClassification | None: ...
