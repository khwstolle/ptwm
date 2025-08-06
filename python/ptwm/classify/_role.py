"""Tensor roles for compressor dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = ["TensorClassification", "TensorRole"]


class TensorRole(StrEnum):
    """Compressor-dispatch role for a single safetensors entry."""

    SCALE_BLOCK = "scale_block"
    SCALE_GLOBAL = "scale_global"
    PACKED_VALUES = "packed_values"
    STANDARD = "standard"


@dataclass(frozen=True, slots=True)
class TensorClassification:
    """Classifier output: role plus the source rule that produced it."""

    role: TensorRole
    source: str
    pattern: str
