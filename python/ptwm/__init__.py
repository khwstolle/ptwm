"""PTWM — lossless compression for PyTorch model weights.

A typed preprocessing graph (bit reorder, byte/nibble split) feeds two
peer entropy coders — Huffman and rANS — alongside a Zstd plane codec
and a dedicated microscaling-format codec. The on-disk format is a
multi-tensor ``.ptwm`` container with per-tensor random-access lookup.

Public entry points: :class:`Compressor`, :class:`Decompressor`,
:class:`CompressionConfig`, :class:`DecompressionConfig`,
:class:`Method`, :class:`Format`. Third-party integrations
(HuggingFace `transformers`, `safetensors`) live in :mod:`ptwm.integrations`
and are imported from there explicitly, not re-exported here.
"""

from importlib.metadata import PackageNotFoundError, version

from ._config import (
    CompressionConfig,
    DecompressionConfig,
    Format,
    Lossy,
    Method,
)
from ._exceptions import (
    CompressionMethodNotSupportedError,
    Error,
    HeaderParseError,
    InvalidDTypeError,
)
from .core import Compressor, Decompressor
from .integrations import materialize_hf_cache

try:
    __version__ = version("ptwm")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "CompressionConfig",
    "CompressionMethodNotSupportedError",
    "Compressor",
    "DecompressionConfig",
    "Decompressor",
    "Format",
    "Lossy",
    "Method",
    "Error",
    "HeaderParseError",
    "InvalidDTypeError",
    "__version__",
    "materialize_hf_cache",
]
