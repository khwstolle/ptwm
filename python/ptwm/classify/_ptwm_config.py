"""Classifier from a user-authored ``ptwm.toml`` rule list."""

from __future__ import annotations

import fnmatch
import tomllib
from pathlib import Path

from ._role import TensorClassification, TensorRole

__all__ = ["PtwmConfigClassifier"]


class PtwmConfigClassifier:
    """Pattern → role rules from TOML. First match in file order wins."""

    __ptwm_canonical_id__ = "io.ptwm.builtin.classifier_ptwm_config"

    def __init__(self, rules: list[tuple[str, TensorRole]]) -> None:
        self._rules = list(rules)

    @classmethod
    def from_path(cls, path: str | Path) -> PtwmConfigClassifier:
        with Path(path).open("rb") as f:
            doc = tomllib.load(f)
        rules: list[tuple[str, TensorRole]] = []
        for entry in doc.get("rule", []):
            pattern = entry["pattern"]
            role_str = entry["role"]
            try:
                role = TensorRole(role_str)
            except ValueError as e:
                msg = (
                    f"ptwm.toml: rule role={role_str!r} is not one of "
                    f"{sorted(r.value for r in TensorRole)}"
                )
                raise ValueError(msg) from e
            rules.append((pattern, role))
        return cls(rules)

    def classify(
        self,
        name: str,
        dtype: str,  # noqa: ARG002 — protocol-required, unused here
        shape: tuple[int, ...],  # noqa: ARG002
        archive_keys: tuple[str, ...],  # noqa: ARG002
    ) -> TensorClassification | None:
        for pattern, role in self._rules:
            if fnmatch.fnmatchcase(name, pattern):
                return TensorClassification(
                    role=role, source="ptwm_config", pattern=pattern
                )
        return None
