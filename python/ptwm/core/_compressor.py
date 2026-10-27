import multiprocessing

import numpy as np
import torch

from .. import _rust
from .._config import CompressionConfig, Format, Framing, Method
from .._exceptions import CompressionMethodNotSupportedError
from ..delta import encode as delta_encode
from ..preprocessing._chains import (
    ClassifierRole,
    build_chain_blobs,
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
            isinstance(data, torch.Tensor)
            and data.device.type == "cuda"
            and self.config.device is not None
            and self.config.device != data.device.index
        ):
            msg = (
                f"device mismatch: config.device={self.config.device} but the "
                f"input tensor is on cuda:{data.device.index}. Move the tensor "
                f"or change config.device; ptwm will not relocate it implicitly."
            )
            raise ValueError(msg)

        if self.config.codec is not None:
            msg = (
                f"CompressionConfig.codec={self.config.codec!r} is set, but "
                "explicit per-codec dispatch is not wired through "
                "Compressor.compress() yet: the automatic per-plane codec "
                "search always runs regardless of this value, so setting it "
                "would silently do nothing. Leave codec unset until explicit "
                "dispatch lands."
            )
            raise CompressionMethodNotSupportedError(msg)

        if (
            self.config.delta_compressed_type is not None
            and self.config.delta_compressed_type != 0
            and delta_second_data is None
        ):
            msg = "delta compression required but delta_second_data is None"
            raise ValueError(msg)

        if self.config.input_format not in (
            Format.TORCH,
            Format.NUMPY,
            Format.BYTE,
        ):
            msg = f"Unsupported input_format: {self.config.input_format}"
            raise ValueError(msg)

        # Read dtype and shape from the input before any delta encoding:
        # the delta residual is a flat byte buffer, but the record still
        # describes the original tensor.
        if self.config.input_format == Format.BYTE:
            dtype_code = DType.from_dtype(self.config.bytearray_dtype).code
            shape: tuple[int, ...] | None = None
        else:
            dtype_code = DType.from_dtype(data.dtype).code
            shape = tuple(data.shape)
        ba_bytes = _to_raw_bytes(data)

        delta_ref_blake3: bytes | None = None
        if delta_second_data is not None:
            ref_bytes = _to_raw_bytes(delta_second_data)
            delta_ref_blake3 = _rust.blake3_hash(ref_bytes)
            ba_bytes = delta_encode(ba_bytes, reference=ref_bytes)

        return self._compress_bin(
            ba=ba_bytes,
            shape=shape,
            dtype_code=dtype_code,
            delta_ref_blake3=delta_ref_blake3,
        )

    def _compress_bin(
        self,
        ba: bytes,
        shape: tuple[int, ...] | None,
        dtype_code: int,
        delta_ref_blake3: bytes | None = None,
    ) -> bytes:
        """Encode one tensor as a compact frame or a single-tensor container.

        ``config.framing`` selects the layout; see :class:`ptwm.Framing`.
        """
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
        # Each chain's Source shape counts elements of that chain's own
        # Source dtype and covers exactly len(ba) bytes; byte input has no
        # shape, so the element count comes from the byte count.
        chains = [
            build_chain_blobs(
                chains_for(int(dtype_code), ClassifierRole.STANDARD),
                shape=tensor_shape,
                nbytes=len(ba),
            )
        ]

        codec_menu_ids: list[int] | None
        if self.config.codec_menu is None:
            codec_menu_ids = None
        else:
            codec_menu_ids = [int(c) for c in self.config.codec_menu]

        emit_payload_hash = self.config.emit_payload_hash
        emit_plane_crc = self.config.emit_plane_crc

        # The compact frame has no chunk table, so streaming always writes a
        # full container.
        if self.config.framing == Framing.COMPACT and streaming_chunk is None:
            blob = _rust.compress_compact(
                int(dtype_code),
                int(self.config.input_format.value),
                ba,
                tensor_shape,
                dtype_name_str,
                delta_ref_blake3,
                chains[0],
                emit_payload_hash,
                emit_plane_crc,
                method_hint,
                codec_menu_ids,
            )
            if blob is not None:
                return blob

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
            emit_payload_hash,
            emit_plane_crc,
            method_hint,
            streaming_chunk,
            codec_menu_ids,
        )
