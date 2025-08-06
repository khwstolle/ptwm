import multiprocessing

import numpy as np
import torch

from .. import _rust
from .._config import CompressionConfig, Format, Method
from .._exceptions import CompressionMethodNotSupportedError
from ..delta import encode as delta_encode
from ..preprocessing._chains import (
    CHAIN_BYTE_PASSTHROUGH_VALUE,
    ClassifierRole,
    chains_for,
)
from ..utils import DType

# Method → header method-hint wire value. HUFFMAN/AUTO use the trial-encode
# menu; the others bias toward a specific codec.
_METHOD_HINTS: dict[int, int] = {
    Method.AUTO.value: 1,
    Method.HUFFMAN.value: 1,
    Method.ZSTD.value: 2,
    Method.MICROSCALE.value: 3,
    Method.RANS.value: 4,
    Method.IDENTITY.value: 5,
}

# Dtypes that are packed into uint8 on the Rust side (single byte per element).
_UINT8_VIEW_DTYPE_CODES = frozenset(
    {
        DType.FLOAT8_E4M3FN.code,
        DType.FLOAT8_E5M2.code,
        DType.FLOAT4_E2M1FN_X2.code,
    }
)


def _to_raw_bytes(
    data: bytes | bytearray | memoryview | np.ndarray | torch.Tensor,
) -> bytes:
    """Reduce any supported input into a flat ``bytes`` buffer.

    Tensors are detached, moved to CPU, made contiguous, and reinterpreted as
    ``uint8`` so dtypes numpy cannot represent (BF16, FP8, FP4) round-trip.
    NumPy arrays go through ``np.ascontiguousarray`` before ``tobytes`` so
    non-contiguous views serialise correctly.
    """
    if isinstance(data, bytes):
        return data
    if isinstance(data, torch.Tensor):
        t = data.detach()
        if t.device.type != "cpu":
            t = t.cpu()
        t = t.contiguous().reshape(-1)
        if t.dtype != torch.uint8:
            t = t.view(torch.uint8)
        return t.numpy().tobytes()
    if isinstance(data, np.ndarray):
        return np.ascontiguousarray(data).tobytes()
    if isinstance(data, bytearray | memoryview):
        return bytes(data)
    msg = f"Unsupported input type for compression: {type(data).__name__}"
    raise TypeError(msg)


class Compressor:
    """Stateless compressor for weights data."""

    def __init__(self, config: CompressionConfig) -> None:
        self.config = config
        self._threads = config.threads or min(multiprocessing.cpu_count(), 16)

    def compress(
        self,
        data: bytes | bytearray | np.ndarray | torch.Tensor,
        delta_second_data: bytes | bytearray | np.ndarray | torch.Tensor | None = None,
    ) -> bytes:
        """Compress the provided data.

        Streaming (``config.is_streaming=True``) enables plane-level chunking:
        each chunk is codec-encoded independently and indexed by
        ``plane_record.chunk_table``. Delta compression XORs the whole buffer
        once; the reference's BLAKE3 digest is recorded as a Dependency so the
        decoder can verify that the caller supplied the matching reference.
        """
        if (
            self.config.delta_compressed_type is not None
            and self.config.delta_compressed_type != 0
            and delta_second_data is None
        ):
            msg = "delta compression required but delta_second_data is None"
            raise ValueError(msg)

        delta_ref_blake3: bytes | None = None
        if delta_second_data is not None:
            ref_bytes = _to_raw_bytes(delta_second_data)
            delta_ref_blake3 = _rust.blake3_hash(ref_bytes)
            data = delta_encode(_to_raw_bytes(data), reference=ref_bytes)

        return self._compress_torch_numpy_byte(data, delta_ref_blake3=delta_ref_blake3)

    def _compress_torch_numpy_byte(
        self,
        data: bytes | bytearray | np.ndarray | torch.Tensor,
        *,
        delta_ref_blake3: bytes | None = None,
    ) -> bytes:
        if self.config.input_format == Format.BYTE:
            dtype_enum = DType.from_dtype(self.config.bytearray_dtype).code
            shape: tuple[int, ...] | None = None
        else:
            dtype_enum = DType.from_dtype(data.dtype).code
            shape = tuple(data.shape)

        # BF16 in TORCH: numpy cannot hold bfloat16, so reinterpret as uint16
        # before going through numpy/memoryview.
        if (
            dtype_enum == DType.BFLOAT16.code
            and self.config.input_format == Format.TORCH
        ):
            data = data.view(torch.uint16)
        # FP8 / packed FP4 on TORCH / NUMPY input: reinterpret as uint8 to
        # hand a flat byte buffer to the Rust compressor.
        elif (
            dtype_enum in _UINT8_VIEW_DTYPE_CODES
            and self.config.input_format != Format.BYTE
        ):
            data = data.view(torch.uint8)

        if self.config.input_format not in (
            Format.TORCH,
            Format.NUMPY,
            Format.BYTE,
        ):
            msg = f"Unsupported input_format: {self.config.input_format}"
            raise ValueError(msg)
        ba_bytes = _to_raw_bytes(data)

        return self._compress_bin(
            ba=ba_bytes,
            shape=shape,
            dtype_code=dtype_enum,
            delta_ref_blake3=delta_ref_blake3,
        )

    def _compress_bin(
        self,
        ba: bytes,
        shape: tuple[int, ...] | None,
        dtype_code: int,
        delta_ref_blake3: bytes | None = None,
    ) -> bytes:
        """Encode a single-tensor ``.ptwm`` via ``compress_model``."""
        method_hint = _METHOD_HINTS.get(self.config.method.value)
        if method_hint is None:
            msg = f"Unsupported compression method: {self.config.method}"
            raise CompressionMethodNotSupportedError(msg)

        dtype_name_str: str | None = None
        for _m in DType:
            if _m.code == int(dtype_code):
                dtype_name_str = _m.dtype_str
                break

        streaming_chunk: int | None = (
            self.config.streaming_chunk if self.config.is_streaming else None
        )

        tensor_shape = list(shape) if shape is not None else None
        builders = list(chains_for(int(dtype_code), ClassifierRole.STANDARD))
        if not builders:
            builders = [CHAIN_BYTE_PASSTHROUGH_VALUE]
        chain_shape = tensor_shape if tensor_shape is not None else [len(ba)]
        # Builders may decline a shape by returning None (e.g. the
        # spherical-normalize chain only accepts 2D tensors); skip those.
        chains = [[c.to_bytes() for b in builders if (c := b(chain_shape)) is not None]]

        codec_menu_ids: list[int] | None
        if self.config.codec_menu is None:
            codec_menu_ids = None
        else:
            codec_menu_ids = [int(c) for c in self.config.codec_menu]

        return _rust.compress_model(
            [
                (
                    "_",
                    int(dtype_code),
                    int(self.config.input_format.value),
                    ba,
                    tensor_shape,
                    dtype_name_str,
                    delta_ref_blake3,
                )
            ],
            chains,
            True,  # emit_payload_hash
            True,  # emit_plane_crc
            method_hint,
            streaming_chunk,
            codec_menu_ids,
        )
