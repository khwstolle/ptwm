"""Classifier driven by NVIDIA ModelOpt's ``hf_quant_config.json``."""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import Final

from .._exceptions import UnsupportedQuantConfigError
from ._role import TensorClassification, TensorRole

__all__ = ["HfQuantConfigClassifier"]

_SUPPORTED_ALGOS: Final = frozenset({"NVFP4", "MXFP4"})

# Per-suffix role map for quant_algo == "NVFP4". MXFP4 from ModelOpt has
# no published canonical naming yet — the user must enumerate suffixes
# via ptwm.toml or --classify-rule.
_NVFP4_SUFFIX_ROLES: Final[dict[str, TensorRole]] = {
    ".weight": TensorRole.PACKED_VALUES,
    ".weight_scale": TensorRole.SCALE_BLOCK,
    ".weight_scale_2": TensorRole.SCALE_GLOBAL,
    ".input_scale": TensorRole.SCALE_GLOBAL,
}


class HfQuantConfigClassifier:
    """Classifier built from a ModelOpt ``hf_quant_config.json``."""

    __ptwm_canonical_id__ = "io.ptwm.builtin.classifier_hf_quant_config"

    def __init__(self, *, quant_algo: str, exclude_modules: tuple[str, ...]) -> None:
        if quant_algo not in _SUPPORTED_ALGOS:
            msg = (
                f"hf_quant_config.json: unsupported quant_algo "
                f"{quant_algo!r}; supported = {sorted(_SUPPORTED_ALGOS)}"
            )
            raise UnsupportedQuantConfigError(msg)
        self._quant_algo = quant_algo
        self._exclude_modules = exclude_modules

    @classmethod
    def from_path(cls, path: str | Path) -> HfQuantConfigClassifier:
        with Path(path).open() as f:
            payload = json.load(f)
        try:
            q = payload["quantization"]
            algo = q["quant_algo"]
            exclude = tuple(q.get("exclude_modules", ()))
        except KeyError as e:
            msg = f"hf_quant_config.json: missing required field {e}"
            raise UnsupportedQuantConfigError(msg) from e
        return cls(quant_algo=algo, exclude_modules=exclude)

    def _is_excluded(self, module_name: str) -> bool:
        return any(
            fnmatch.fnmatchcase(module_name, pat) for pat in self._exclude_modules
        )

    def _module_of(self, tensor_name: str, suffix: str) -> str:
        return tensor_name[: -len(suffix)]

    def _is_quantized(self, module: str, archive_keys: tuple[str, ...]) -> bool:
        # NVFP4 modules carry a sibling weight_scale. Use that as the
        # discriminator between quantized and non-quantized modules.
        return f"{module}.weight_scale" in archive_keys

    def classify(
        self,
        name: str,
        dtype: str,  # noqa: ARG002
        shape: tuple[int, ...],  # noqa: ARG002
        archive_keys: tuple[str, ...],
    ) -> TensorClassification | None:
        if self._quant_algo != "NVFP4":
            # MXFP4 from this config alone gives no per-suffix rules.
            return None
        for suffix, role in _NVFP4_SUFFIX_ROLES.items():
            if name.endswith(suffix):
                module = self._module_of(name, suffix)
                if self._is_excluded(module):
                    return None
                if not self._is_quantized(module, archive_keys):
                    return None
                return TensorClassification(
                    role=role,
                    source="hf_quant_config",
                    pattern=f"*{suffix}",
                )
        return None
