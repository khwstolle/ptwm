"""Thin shim exposing the hardware_backend dispatch surface as ptwm._rust.hardware."""

from __future__ import annotations

from ptwm._core import (  # type: ignore[import]
    hardware_backend_cuda_stream_handle,
    hardware_backend_dispatch_decode_cuda,
)

__all__ = [
    "hardware_backend_cuda_stream_handle",
    "hardware_backend_dispatch_decode_cuda",
]
