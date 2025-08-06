"""Integration with safetensors library (Mode A: .ptwm; Mode B: shell)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import torch

from .. import _rust
from .._config import DecompressionConfig
from ..core import Decompressor
from ..utils import decode_ptwm_tensor, get_compressed_tensors_metadata
from ..utils._patch import multi_process_patcher

_PTWM_SHELL_SIGNAL = "ptwm-shell"


def decompress_safetensors_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Decompress a per-tensor compressed entry (legacy path).

    The tensor is expected to be a 1D uint8 array.
    """
    if tensor.dtype != torch.uint8 or tensor.ndim != 1:
        return tensor

    config = DecompressionConfig()
    decompressor = Decompressor(config)
    return decompressor.decompress(tensor.cpu().contiguous().numpy())


def _decode_ptwm_tensor(blob: bytes, name: str) -> torch.Tensor:
    """Decode a single named tensor from a v3 .ptwm blob.

    Thin wrapper over the shared :func:`ptwm.utils.decode_ptwm_tensor` helper,
    kept for backward compatibility with existing call sites.
    """
    return decode_ptwm_tensor(blob, name)


class SafeOpen:
    """``safetensors.safe_open`` wrapper recognising .ptwm + ptwm-shell.

    Lazy decoding: the compressed blob stays in memory; individual tensors
    decode only on :meth:`get_tensor`. This preserves safetensors' lazy-load
    contract for large multi-shard checkpoints.
    """

    def __init__(self, filename: str, framework: str, device: str = "cpu") -> None:
        self._filename = filename
        self._framework = framework
        self._device = device

        path = Path(filename)
        self._mode_a = path.suffix == ".ptwm"
        self._mode_b = False
        # Compressed blob for Mode A / Mode B — held for lazy per-tensor decode.
        self._ptwm_blob: bytes | None = None
        self._ptwm_keys: tuple[str, ...] = ()
        self._underlying = None
        self.compressed_tensors_metadata: dict[str, Any] = {}

        if self._mode_a:
            self._ptwm_blob = path.read_bytes()
            self._ptwm_keys = tuple(_rust.list_tensor_names(self._ptwm_blob))
            return

        # Mode B detection: open as safetensors first, sniff metadata.
        from safetensors.torch import safe_open

        self._underlying = safe_open(filename, framework, device)
        meta = self._underlying.metadata() or {}
        if meta.get("weights_format") == _PTWM_SHELL_SIGNAL:
            self._mode_b = True
            # Extract the embedded .ptwm blob once. safetensors reads the
            # __ptwm_payload__ tensor eagerly; that is safetensors' own
            # behaviour and cannot be deferred further at this layer.
            payload = self._underlying.get_tensor("__ptwm_payload__")
            self._ptwm_blob = bytes(memoryview(payload.cpu().numpy()).cast("B"))
            self._ptwm_keys = tuple(_rust.list_tensor_names(self._ptwm_blob))
        else:
            self.compressed_tensors_metadata = get_compressed_tensors_metadata(meta)

    def keys(self) -> list[str]:
        if self._ptwm_blob is not None:
            return list(self._ptwm_keys)
        return list(self._underlying.keys())

    def metadata(self) -> dict[str, str] | None:
        if self._mode_a:
            return None
        return self._underlying.metadata()

    def get_tensor(self, name: str) -> torch.Tensor:
        if self._ptwm_blob is not None:
            return _decode_ptwm_tensor(self._ptwm_blob, name)
        if name not in self.compressed_tensors_metadata:
            return self._underlying.get_tensor(name)
        return decompress_safetensors_tensor(self._underlying.get_tensor(name))

    def get_slice(self, name: str) -> Any:
        if self._ptwm_blob is not None:
            msg = "get_slice is not supported on .ptwm/ptwm-shell containers"
            raise NotImplementedError(msg)
        if name not in self.compressed_tensors_metadata:
            return self._underlying.get_slice(name)
        raise NotImplementedError

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._underlying is not None:
            return self._underlying.__exit__(exc_type, exc_value, traceback)
        return None

    def __getattr__(self, name: str) -> Any:
        if self._underlying is None:
            msg = (
                f"'SafeOpen' has no attribute {name!r} "
                "(no underlying safetensors file in .ptwm mode)"
            )
            raise AttributeError(msg)
        return getattr(self._underlying, name)


def write_mode_b_shell(out_path: str | Path, ptwm_bytes: bytes) -> None:
    """Wrap ``ptwm_bytes`` as a single-tensor ``.safetensors`` shell.

    The result is a strictly-valid safetensors file with one
    ``__ptwm_payload__`` U8 tensor and
    ``__metadata__["weights_format"] = "ptwm-shell"``.
    """
    from safetensors.torch import save_file

    payload = torch.frombuffer(bytearray(ptwm_bytes), dtype=torch.uint8).clone()
    metadata = {
        "weights_format": _PTWM_SHELL_SIGNAL,
        "weights_format_version": "1",
        "ptwm_bytes_xxhash64": _xxhash64_hex(ptwm_bytes),
    }
    save_file({"__ptwm_payload__": payload}, str(out_path), metadata=metadata)


def _xxhash64_hex(data: bytes) -> str:
    """Return xxhash64 (classic, seed=0) of ``data`` as 16-hex-digit string.

    Matches the Rust side (``xxhash_rust::xxh64::xxh64(&bytes, 0)``) used in
    Order1ScaleAC's prelude entry hashing.
    """
    import xxhash

    return f"{xxhash.xxh64(data).intdigest():016x}"


def _patch_safetensors() -> None:
    import safetensors.torch

    safetensors.torch.safe_open = SafeOpen


def patch_safetensors() -> None:
    multi_process_patcher(_patch_safetensors)
