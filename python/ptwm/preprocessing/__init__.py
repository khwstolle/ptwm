"""Dtype-aware preprocessing for lossless weight compression.

The compression pipeline separates preprocessing (bit reorder + byte/nibble
split) from entropy coding. Preprocessing is expressed as a typed DAG of
ops (a :class:`Chain`) — see :mod:`._chains` for the chain builders and
:mod:`._explorer` for the generative search that extends them.
"""

from ._cache import (
    CacheEntry,
    cache_dir,
    cache_info,
    cache_path_for,
    clear_cache,
    load_cached_builders,
    load_cached_entries,
    save_cached_entries,
)
from ._chains import (
    CHAIN_BF16_SPLIT,
    CHAIN_BYTE_PASSTHROUGH_VALUE,
    CHAIN_F8E4M3_NIBBLE,
    CHAIN_F8E4M3_SCALE_PASSTHROUGH,
    CHAIN_F32_GLOBAL_SCALE,
    CHAIN_FP4_NIBBLE_PASSTHROUGH,
    CHAIN_FP8E4M3_NIBBLE,
    CHAIN_FP8E5M2_NIBBLE,
    CHAIN_FP32_SPLIT,
    CHAIN_U8_SCALE_PASSTHROUGH,
    PRODUCTION_CHAINS,
    Chain,
    ChainBuilder,
    ChainEdge,
    ChainNode,
    ClassifierRole,
    TerminalRef,
    chains_for,
)
from ._explorer import ExploreOptions, explore_chains, is_legal_chain
from ._plane import Plane

__all__ = [
    "CHAIN_BF16_SPLIT",
    "CHAIN_BYTE_PASSTHROUGH_VALUE",
    "CHAIN_F32_GLOBAL_SCALE",
    "CHAIN_F8E4M3_NIBBLE",
    "CHAIN_F8E4M3_SCALE_PASSTHROUGH",
    "CHAIN_FP32_SPLIT",
    "CHAIN_FP4_NIBBLE_PASSTHROUGH",
    "CHAIN_FP8E4M3_NIBBLE",
    "CHAIN_FP8E5M2_NIBBLE",
    "CHAIN_U8_SCALE_PASSTHROUGH",
    "CacheEntry",
    "Chain",
    "ChainBuilder",
    "ChainEdge",
    "ChainNode",
    "ClassifierRole",
    "ExploreOptions",
    "PRODUCTION_CHAINS",
    "Plane",
    "TerminalRef",
    "cache_dir",
    "cache_info",
    "cache_path_for",
    "chains_for",
    "clear_cache",
    "explore_chains",
    "is_legal_chain",
    "load_cached_builders",
    "load_cached_entries",
    "save_cached_entries",
]
