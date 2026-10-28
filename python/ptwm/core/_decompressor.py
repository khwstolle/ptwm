import multiprocessing
from pathlib import Path

import numpy as np
import torch

from .. import _rust
from .._config import DecompressionConfig, Format
from .._exceptions import HeaderParseError, InvalidDTypeError
from .._magic import MAGIC_PTWS, container_kind
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
        r"""Decompress a ``.ptwm`` blob.

        Accepts a compact single-tensor frame (magic ``\x89PTWS``) or a
        full container in either layout (magic ``\x89PTWM``, ``\x89PTWX``
        or ``\x89PTWC``). The single-tensor container uses name ``"_"``. Raw decompressed bytes
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
        if mv_data.nbytes < len(MAGIC_PTWS):
            msg = "too short to be a .ptwm container"
            raise HeaderParseError(msg)

        blob = data if isinstance(data, bytes) else bytes(mv_data)
        kind = container_kind(blob)
        info: tuple[list[int], str, int, int] | None
        if kind == "ptws":
            raw, shape, dtype_name, dtype_code, input_format, expected_hash, _ = (
                _rust.decode_compact(blob)
            )
            info = (
                (shape, dtype_name or "", dtype_code, input_format)
                if shape is not None
                else None
            )
        elif kind is not None:
            # One open serves the decode and both metadata lookups.
            handle = _rust.ContainerHandle(blob, self.config.skip_missing)
            raw = handle.decode("_")
            expected_hash = handle.delta_hash("_")
            info = handle.info("_")
        else:
            msg = "not a .ptwm container"
            raise HeaderParseError(msg)

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
        """Decompress a container (either layout) into ``{name: raw_bytes}``.

        Opens the container once and decodes each tensor in write order, so
        the keys come in the order the tensors were compressed. Result keys
        preserve the tensor names from the source safetensors archive.
        """
        handle = _rust.ContainerHandle(blob, self.config.skip_missing)
        return {name: handle.decode(name) for name in handle.names()}

    def decompress_file(self, path: str) -> bytes | np.ndarray | torch.Tensor:
        """Decompress data from a file."""
        file_path = Path(path)
        if not file_path.exists():
            msg = f"The file at {path} was not found."
            raise FileNotFoundError(msg)
        with file_path.open("rb") as f:
            ba = f.read()
        return self.decompress(ba)
