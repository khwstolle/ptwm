"""CLI ``--classify-rule '<glob>=<role>'`` flags compiled into a classifier."""

from __future__ import annotations

import fnmatch
from collections.abc import Sequence

from ._role import TensorClassification, TensorRole

__all__ = ["ExplicitFlagsClassifier"]


class ExplicitFlagsClassifier:
    """In-order pattern → role rules from CLI flags."""

    __ptwm_canonical_id__ = "io.ptwm.builtin.classifier_explicit_flags"

    def __init__(self, rules: list[tuple[str, TensorRole]]) -> None:
        self._rules = list(rules)

    @classmethod
    def from_flag_strings(cls, flags: Sequence[str]) -> ExplicitFlagsClassifier:
        rules: list[tuple[str, TensorRole]] = []
        for raw in flags:
            if "=" not in raw:
                msg = f"--classify-rule {raw!r}: expected '<glob>=<role>'"
                raise ValueError(msg)
            pattern, role_str = raw.split("=", 1)
            try:
                role = TensorRole(role_str)
            except ValueError as e:
                msg = (
                    f"--classify-rule {raw!r}: role {role_str!r} is "
                    f"not one of {sorted(r.value for r in TensorRole)}"
                )
                raise ValueError(msg) from e
            rules.append((pattern, role))
        return cls(rules)

    def classify(
        self,
        name: str,
        dtype: str,  # noqa: ARG002
        shape: tuple[int, ...],  # noqa: ARG002
        archive_keys: tuple[str, ...],  # noqa: ARG002
    ) -> TensorClassification | None:
        for pattern, role in self._rules:
            if fnmatch.fnmatchcase(name, pattern):
                return TensorClassification(role=role, source="flag", pattern=pattern)
        return None
