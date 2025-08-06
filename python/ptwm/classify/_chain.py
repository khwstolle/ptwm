"""Fixed-order classifier chain."""

from __future__ import annotations

from collections.abc import Sequence

from ._protocol import TensorClassifier
from ._role import TensorClassification, TensorRole

__all__ = ["ClassifierChain"]

_DEFAULT = TensorClassification(role=TensorRole.STANDARD, source="default", pattern="")


class ClassifierChain:
    """Sequence of classifiers; first non-``None`` decision wins."""

    def __init__(self, classifiers: Sequence[TensorClassifier]) -> None:
        self._classifiers: tuple[TensorClassifier, ...] = tuple(classifiers)

    def classify(
        self,
        name: str,
        dtype: str,
        shape: tuple[int, ...],
        archive_keys: tuple[str, ...],
    ) -> TensorClassification:
        for c in self._classifiers:
            r = c.classify(name, dtype, shape, archive_keys)
            if r is not None:
                return r
        return _DEFAULT
