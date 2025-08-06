"""Integrations for third-party libraries."""

from ._compress_safetensors import compress_safetensors_file
from ._hf import discover_ptwm_shards, materialize_hf_cache, patch_transformers
from ._safetensors import SafeOpen, patch_safetensors

__all__ = [
    "SafeOpen",
    "compress_safetensors_file",
    "discover_ptwm_shards",
    "materialize_hf_cache",
    "patch_transformers",
    "patch_safetensors",
]
