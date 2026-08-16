"""Thin shim exposing the plane_codec CUDA dispatch surface.

Re-exports the `ptwm._core` plane_codec CUDA functions as
`ptwm._rust.plane_codec_cuda`.
"""

from __future__ import annotations

from ptwm._core import (  # type: ignore[import]
    plane_codec_decode_cuda,
    plane_codec_encode_cuda,
)

__all__ = [
    "plane_codec_decode_cuda",
    "plane_codec_encode_cuda",
]
