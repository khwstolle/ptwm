import multiprocessing
from pathlib import Path

import numpy as np
import torch

from .. import _rust
from .._config import DecompressionConfig, Format
from .._exceptions import HeaderParseError, InvalidDTypeError
from ..delta import decode as delta_decode
from ..utils import DType


def _torch_dtype_for_code(dtype_code: int) -> torch.dtype | None:
    """Map a wire ``dtype_code`` to a concrete ``torch.dtype`` if supported."""
    for m in DType:
        if m.code == int(dtype_code):
            return m.torch_dtype
    return None


def _numpy_dtype_for_code(dtype_code: int) -> np.dtype | None:
    """Map a wire ``dtype_code`` to a concrete numpy dtype if supported."""
    for m in DType:
        if m.code == int(dtype_code) and m.numpy_dtype is not None:
            return np.dtype(m.numpy_dtype)
    return None


class Decompressor:
    """Stateless decompressor for weights data."""

    def __init__(self, config: DecompressionConfig | None = None) -> None:
        self.config = config or DecompressionConfig()
        self._threads = self.config.threads or min(multiprocessing.cpu_count(), 16)

    def decompress(self, data: bytes | memoryview) -> bytes | np.ndarray | torch.Tensor:
        """Decompress a ``.ptwm`` blob.

        The single-tensor wrapper uses name ``"_"``. Raw decompressed bytes
        come from ``decode_tensor``; shape/dtype/input_format (when present)
        come from the CBOR shape-metadata map. Format recovery:

        - No metadata or ``input_format == BYTE`` → raw ``bytes``.
        - ``input_format == TORCH`` → :class:`torch.Tensor` reconstructed
          via ``torch.frombuffer`` then ``reshape(shape)``.
        - ``input_format == NUMPY`` → :class:`numpy.ndarray` via
          ``np.frombuffer`` + ``reshape(shape)``.

        Delta: when the tensor record carries a Dependency with a BLAKE3
        hash, ``config.delta_second_data`` is required; its BLAKE3 must
        match, and the decoded bytes are XOR-ed with it to recover the
        original tensor before any format reconstruction.
        """
        if self.config.codec is not None:
            msg = (
                f"DecompressionConfig.codec={self.config.codec!r} is set, but "
                "explicit per-codec dispatch is not wired through "
                "Decompressor.decompress() yet: decoding always dispatches on "
                "the codec recorded in the container header regardless of "
                "this value, so setting it would silently do nothing. Leave "
                "codec unset until explicit dispatch lands."
            )
            raise ValueError(msg)
        if self.config.device is not None:
            msg = (
                f"DecompressionConfig.device={self.config.device} is set, but "
                "GPU-resident decode dispatch is not wired through "
                "Decompressor.decompress() yet: decoding always runs on CPU "
                "regardless of this value, so setting it would silently do "
                "nothing. Leave device unset until explicit dispatch lands."
            )
            raise ValueError(msg)

        mv_data = memoryview(data)
        magic = b"\x89PTWM"
        if len(mv_data) < len(magic):
            msg = "too short to be a .ptwm container"
            raise HeaderParseError(msg)
        if bytes(mv_data[: len(magic)]) != magic:
            msg = "not a .ptwm container"
            raise HeaderParseError(msg)

        blob = bytes(mv_data)
        skip = self.config.skip_missing
        raw = _rust.decode_tensor(blob, "_", skip)
        expected_hash = _rust.decode_tensor_delta_hash(blob, "_", skip)

        if expected_hash is not None:
            if self.config.delta_second_data is None:
                msg = (
                    "tensor was compressed with delta compression, but "
                    "delta_second_data is None."
                )
                raise ValueError(msg)
            ref_bytes = bytes(self.config.delta_second_data)
            got_hash = _rust.blake3_hash(ref_bytes)
            if got_hash != expected_hash:
                msg = (
                    "delta_second_data BLAKE3 hash does not match the "
                    "reference recorded at compression time."
                )
                raise ValueError(msg)
            raw = delta_decode(raw, reference=ref_bytes)
        elif self.config.delta_second_data is not None:
            msg = (
                "tensor was not compressed using delta compression, but "
                "delta_second_data was provided."
            )
            raise ValueError(msg)

        info = _rust.decode_tensor_info(blob, "_", skip)
        if info is None:
            return raw
        shape, _dtype_name, dtype_code, input_format = info

        if input_format == Format.BYTE.value:
            return raw
        if input_format == Format.TORCH.value:
            return self._bytes_to_torch(raw, dtype_code, shape)
        if input_format == Format.NUMPY.value:
            return self._bytes_to_numpy(raw, dtype_code, shape)
        return raw

    def _bytes_to_torch(
        self, raw: bytes, dtype_code: int, shape: list[int] | tuple[int, ...]
    ) -> torch.Tensor:
        torch_dtype = _torch_dtype_for_code(dtype_code)
        if torch_dtype is None:
            msg = (
                f"dtype_code={dtype_code} has no torch.dtype mapping for "
                "TORCH reconstruction"
            )
            raise InvalidDTypeError(msg)
        # frombuffer rejects a zero-length buffer, so reconstruct empty tensors
        # (any 0-dimension shape, e.g. (0,) or (0, 16)) directly.
        if not raw:
            t = torch.empty(0, dtype=torch_dtype)
            return t.reshape(shape) if shape else t.reshape(())
        # frombuffer requires a writable buffer — bytes are immutable, so
        # allocate a bytearray copy. The raw payload is owned bytes returned
        # from Rust, so the copy is a single memcpy.
        t = torch.frombuffer(bytearray(raw), dtype=torch_dtype)
        return t.reshape(shape) if shape else t.reshape(())

    def _bytes_to_numpy(
        self, raw: bytes, dtype_code: int, shape: list[int] | tuple[int, ...]
    ) -> np.ndarray:
        np_dtype = _numpy_dtype_for_code(dtype_code)
        if np_dtype is None:
            msg = (
                f"dtype_code={dtype_code} has no numpy.dtype mapping for "
                "NUMPY reconstruction"
            )
            raise InvalidDTypeError(msg)
        arr = np.frombuffer(raw, dtype=np_dtype)
        return arr.reshape(tuple(shape)) if shape else arr.reshape(())

    def decompress_container_to_state_dict(self, blob: bytes) -> dict[str, bytes]:
        """Decompress a container into a ``{name: raw_bytes}`` map.

        Iterates the embedded tensor index and decodes each tensor in input
        order. Result keys preserve the tensor names from the source
        safetensors archive.
        """
        skip = self.config.skip_missing
        names = _rust.list_tensor_names(blob, skip)
        return {name: bytes(_rust.decode_tensor(blob, name, skip)) for name in names}

    def decompress_file(self, path: str) -> bytes | np.ndarray | torch.Tensor:
        """Decompress data from a file."""
        file_path = Path(path)
        if not file_path.exists():
            msg = f"The file at {path} was not found."
            raise FileNotFoundError(msg)
        with file_path.open("rb") as f:
            ba = f.read()
        return self.decompress(ba)
