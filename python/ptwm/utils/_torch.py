"""Utils for handling torch tensors."""

import functools
import struct
import sys
from enum import Enum
from typing import Any

import numpy as np
import torch

from .._config import Format


def multiply_if_max_below(
    tensor: torch.Tensor, max_val: float, multiplier: float, dtype: torch.dtype
) -> tuple[torch.Tensor, bool]:
    """Scale ``tensor`` into integer-friendly range for lossy compression.

    If ``tensor.abs().amax() < max_val``, returns ``(tensor * multiplier).to(dtype)``
    paired with ``True``; otherwise returns the input unchanged with ``False``.
    """
    is_int = False
    with torch.no_grad():
        if tensor.abs().amax().item() < max_val:
            new_tensor = tensor * multiplier
            new_tensor = new_tensor.to(dtype)
            is_int = True
            return new_tensor, is_int
    return tensor, is_int


def divide_int(tensor: torch.Tensor, divisor: float) -> torch.Tensor:
    """Inverse of :func:`multiply_if_max_below`: cast to float32 and divide."""
    with torch.no_grad():
        return tensor.to(torch.float32) / divisor


def get_dtype_bits(dtype: torch.dtype) -> tuple[int, torch.dtype]:
    """Return ``(bit_width, matching_int_dtype)`` for a floating-point dtype."""
    if dtype.is_floating_point:
        bit_size = torch.finfo(dtype).bits
        if bit_size == 32:
            return bit_size, torch.int32
        if bit_size == 16:
            return bit_size, torch.int16
        sys.exit(f"Error: {dtype} is not float 16, 32")
    sys.exit(f"Error: {dtype} is not a floating point type")


def pack_shape(shape: torch.Size | list[int] | tuple[int, ...]) -> bytes:
    """Pack tensor dimensions into bytes with per-dim size indicators.

    Layout: one byte for the dim count, then ``(size_indicator, value)`` pairs
    where the indicator (1/2/4/8) is the byte width of the following value.
    """
    fmt = ["=B"]
    values = [len(shape)]

    for dim in shape:
        if dim < 256:
            fmt.append("BB")
            values.append(1)
        elif dim < 65536:
            fmt.append("BH")
            values.append(2)
        elif dim < 4294967296:
            fmt.append("BI")
            values.append(4)
        else:
            fmt.append("BQ")
            values.append(8)
        values.append(dim)
    return struct.pack("".join(fmt), *values)


def unpack_shape(packed_data: bytes) -> tuple[tuple[int, ...], int]:
    """Inverse of :func:`pack_shape`. Returns ``(dims, bytes_consumed)``."""
    num_dimensions = packed_data[0]
    dimensions = []
    i = 1
    total_bytes_read = 1

    while i < len(packed_data) and len(dimensions) < num_dimensions:
        size_indicator = packed_data[i]
        total_bytes_read += 1
        i += 1
        if size_indicator == 1:
            (dim,) = struct.unpack("B", packed_data[i : i + 1])
            i += 1
            total_bytes_read += 1
        elif size_indicator == 2:
            (dim,) = struct.unpack("H", packed_data[i : i + 2])
            i += 2
            total_bytes_read += 2
        elif size_indicator == 4:
            (dim,) = struct.unpack("I", packed_data[i : i + 4])
            i += 4
            total_bytes_read += 4
        else:
            (dim,) = struct.unpack("Q", packed_data[i : i + 8])
            i += 8
            total_bytes_read += 8
        dimensions.append(dim)
    return tuple(dimensions), total_bytes_read


def is_floating_point(
    data_format_value: int, data: Any, bytearray_dtype: str | None
) -> bool | None:
    """Check if the data is floating point."""
    if data_format_value == Format.TORCH.value:
        return torch.is_floating_point(data)
    if data_format_value == Format.NUMPY.value:
        return np.issubdtype(data.dtype, np.floating)
    if data_format_value == Format.BYTE.value:
        return bytearray_dtype in (
            "float64",
            "float32",
            "float16",
            "bfloat16",
            "float8_e4m3fn",
            "float8_e5m2",
        )
    return None


class DType(Enum):
    """Enum for torch weights dtypes."""

    NONE = (None, None, None, "none", 0)
    FLOAT32 = (torch.float32, np.float32, float, "float32", 1)
    FLOAT = (torch.float, np.float32, float, "float", 2)
    FLOAT64 = (torch.float64, np.float64, float, "float64", 3)
    FLOAT16 = (torch.float16, np.float16, None, "float16", 4)
    HALF = (torch.half, np.float16, None, "half", 5)
    BFLOAT16 = (torch.bfloat16, None, None, "bfloat16", 6)
    COMPLEX32 = (torch.complex32, None, None, "complex32", 7)
    CHALF = (torch.complex32, None, None, "chalf", 8)
    COMPLEX64 = (torch.complex64, np.complex64, complex, "complex64", 9)
    CFLOAT = (torch.cfloat, np.complex64, complex, "cfloat", 10)
    COMPLEX128 = (torch.complex128, np.complex128, complex, "complex128", 11)
    CDOUBLE = (torch.cdouble, np.complex128, complex, "cdouble", 12)
    UINT8 = (torch.uint8, np.uint8, None, "uint8", 13)
    # Torch has limited support (omit it at this stage)
    UINT16 = (None, np.uint16, None, "uint16", 14)
    # Torch has limited support (omit it at this stage)
    UINT32 = (None, np.uint32, None, "uint32", 15)
    # Torch has limited support (omit it at this stage)
    UINT64 = (None, np.uint64, None, "uint64", 16)
    INT8 = (torch.int8, np.int8, None, "int8", 17)
    INT16 = (torch.int16, np.int16, None, "int16", 18)
    SHORT = (torch.int16, np.int16, None, "short", 19)
    INT32 = (torch.int32, np.int32, int, "int32", 20)
    INT = (torch.int32, np.int32, int, "int", 21)
    INT64 = (torch.int64, np.int64, int, "int64", 22)
    LONG = (torch.int64, np.int64, int, "long", 23)
    BOOL = (torch.bool, np.bool_, bool, "bool", 24)
    QUINT8 = (torch.quint8, None, None, "quint8", 25)
    QINT8 = (torch.qint8, None, None, "qint8", 26)
    QINT32 = (torch.qint32, None, None, "qint32", 27)
    QUINT4X2 = (torch.quint4x2, None, None, "quint4x2", 28)
    FLOAT8_E4M3FN = (torch.float8_e4m3fn, None, None, "float8_e4m3fn", 29)
    FLOAT8_E5M2 = (torch.float8_e5m2, None, None, "float8_e5m2", 30)
    # Packed FP4: two 4-bit values per byte (requires PyTorch >= 2.6).
    # The 4-bit weights themselves have low entropy and compress poorly; the
    # FP8 block-scale tensors that accompany MXFP4/NVFP4 models are separate
    # tensors and benefit from FP8 compression (bytes_mode 20/22).
    FLOAT4_E2M1FN_X2 = (
        getattr(torch, "float4_e2m1fn_x2", None),
        None,
        None,
        "float4_e2m1fn_x2",
        31,
    )

    def __init__(
        self,
        torch_dtype: torch.dtype | None,
        numpy_dtype: type | None,
        python_dtype: type | None,
        dtype_str: str,
        code: int,
    ) -> None:
        self.torch_dtype = torch_dtype
        self.numpy_dtype = numpy_dtype
        self.python_dtype = python_dtype
        self.dtype_str = dtype_str
        self.code = code

    @classmethod
    def from_dtype(cls, dtype: Any) -> "DType":
        """Return the enum member matching ``dtype``.

        Accepts ``torch.dtype``, ``numpy.dtype``, Python type, or canonical
        name string. For numpy 2.x / ml_dtypes extension dtypes (BF16, FP8,
        packed FP4 — entries with ``numpy_dtype is None`` because no first-
        class numpy alias exists), a fallback comparison against
        ``dtype.name`` recovers the right member.
        """
        if isinstance(dtype, str):
            dtype = dtype.lower()
        try:
            return cls._from_dtype_cached(dtype)
        except TypeError:
            # Fallback for unhashable dtypes
            return cls._from_dtype_uncached(dtype)

    @classmethod
    @functools.cache
    def _from_dtype_cached(cls, dtype: Any) -> "DType":
        return cls._from_dtype_uncached(dtype)

    @classmethod
    def _from_dtype_uncached(cls, dtype: Any) -> "DType":
        for member in cls:
            if dtype in (
                member.torch_dtype,
                member.numpy_dtype,
                member.python_dtype,
                member.dtype_str,
            ):
                return member
        # Fallback: numpy 2.x / ml_dtypes BF16/FP8/FP4 expose .name matching
        # the canonical dtype_str (e.g. np.dtype('bfloat16').name == "bfloat16").
        name = getattr(dtype, "name", None)
        if isinstance(name, str):
            name_lower = name.lower()
            for member in cls:
                if member.dtype_str == name_lower:
                    return member
        return cls.NONE

    @classmethod
    def from_code(cls, code: int) -> str:
        """Get the enum member name from a code."""
        for member in cls:
            if member.code == code:
                return member.name
        return cls.NONE
