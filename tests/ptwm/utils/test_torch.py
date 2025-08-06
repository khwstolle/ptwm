"""Direct unit tests for :mod:`ptwm.utils._torch`.

The shape pack/unpack helpers, dtype-bit lookups and dtype enum are
used transitively by Compressor/Decompressor but lack dedicated
coverage. These tests exercise the boundaries (zero dims, many dims,
dimension values near each size-indicator boundary) where the format
self-describes via a per-dim size tag.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from ptwm._config import Format
from ptwm.utils import DType, is_floating_point
from ptwm.utils._torch import (
    divide_int,
    get_dtype_bits,
    multiply_if_max_below,
    pack_shape,
    unpack_shape,
)

# ---------------------------------------------------------------------------
# pack_shape / unpack_shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (1,),
        (0,),
        (1, 1),
        (255,),  # boundary: 1-byte max
        (256,),  # boundary: 2-byte min
        (65535,),  # boundary: 2-byte max
        (65536,),  # boundary: 4-byte min
        (4294967295,),  # boundary: 4-byte max
        (4294967296,),  # boundary: 8-byte min
        (18446744073709551615,),  # boundary: 8-byte max
        (2, 3, 4, 5, 6, 7, 8, 9),  # many dims
    ],
)
def test_pack_unpack_roundtrip(shape: tuple[int, ...]) -> None:
    packed = pack_shape(shape)
    recovered, consumed = unpack_shape(packed)
    assert recovered == shape
    assert consumed == len(packed)


@given(
    shape=st.lists(
        st.integers(min_value=0, max_value=18446744073709551615), min_size=0, max_size=8
    )
)
def test_pack_unpack_hypothesis_roundtrip(shape: list[int]) -> None:
    packed = pack_shape(tuple(shape))
    recovered, consumed = unpack_shape(packed)
    assert recovered == tuple(shape)
    assert consumed == len(packed)


def test_pack_shape_error() -> None:
    import struct

    with pytest.raises(struct.error):
        pack_shape((-1,))

    with pytest.raises(struct.error):
        pack_shape((18446744073709551616,))


def test_unpack_stops_at_declared_dim_count() -> None:
    """Trailing bytes after the declared dim count must not be consumed."""
    packed = pack_shape((3, 4))
    padded = packed + b"\xff\xff\xff"  # garbage tail
    recovered, consumed = unpack_shape(padded)
    assert recovered == (3, 4)
    assert consumed == len(packed)


def test_unpack_shape_8_byte_indicator() -> None:
    """Test unpacking a dimension with an 8-byte indicator."""
    # 1 dim, size_indicator=8, value=4294967296
    packed = b"\x01\x08" + struct.pack("Q", 4294967296)
    recovered, consumed = unpack_shape(packed)
    assert recovered == (4294967296,)
    assert consumed == 10


def test_unpack_shape_fallback_indicator() -> None:
    """Test unpacking where size_indicator falls into the else (8-byte) branch."""
    # 1 dim, size_indicator=3 (not 1, 2, or 4), value=12345
    packed = b"\x01\x03" + struct.pack("Q", 12345)
    recovered, consumed = unpack_shape(packed)
    assert recovered == (12345,)
    assert consumed == 10


# ---------------------------------------------------------------------------
# multiply_if_max_below
# ---------------------------------------------------------------------------


def test_multiply_if_max_below_scales_and_casts() -> None:
    t = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32)
    # Max abs is 3.0, which is < 5.0
    result, is_int = multiply_if_max_below(
        tensor=t, max_val=5.0, multiplier=10.0, dtype=torch.int32
    )
    assert is_int is True
    assert result.dtype == torch.int32
    assert torch.equal(result, torch.tensor([10, -20, 30], dtype=torch.int32))


def test_multiply_if_max_below_returns_unchanged_when_above_max() -> None:
    t = torch.tensor([1.0, -2.0, 6.0], dtype=torch.float32)
    # Max abs is 6.0, which is not < 5.0
    result, is_int = multiply_if_max_below(
        tensor=t, max_val=5.0, multiplier=10.0, dtype=torch.int32
    )
    assert is_int is False
    assert result.dtype == torch.float32
    assert result is t


def test_multiply_if_max_below_returns_unchanged_on_boundary() -> None:
    t = torch.tensor([1.0, -2.0, 5.0], dtype=torch.float32)
    # Max abs is 5.0, which is not < 5.0
    result, is_int = multiply_if_max_below(
        tensor=t, max_val=5.0, multiplier=10.0, dtype=torch.int32
    )
    assert is_int is False
    assert result.dtype == torch.float32
    assert result is t


# ---------------------------------------------------------------------------
# get_dtype_bits
# ---------------------------------------------------------------------------


def test_get_dtype_bits_float32() -> None:
    bits, target = get_dtype_bits(torch.float32)
    assert bits == 32
    assert target == torch.int32


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_get_dtype_bits_16bit(dtype: torch.dtype) -> None:
    bits, target = get_dtype_bits(dtype)
    assert bits == 16
    assert target == torch.int16


def test_get_dtype_bits_rejects_non_float() -> None:
    # The helper uses sys.exit on non-float dtypes; assert the exit path.
    with pytest.raises(SystemExit):
        get_dtype_bits(torch.int32)


# ---------------------------------------------------------------------------
# divide_int
# ---------------------------------------------------------------------------


def test_divide_int_produces_float32() -> None:
    t = torch.tensor([10, 20, 30], dtype=torch.int32)
    result = divide_int(t, divisor=10.0)
    assert result.dtype == torch.float32
    assert torch.allclose(result, torch.tensor([1.0, 2.0, 3.0]))


def test_divide_int_negative_divisor() -> None:
    t = torch.tensor([10, -20, 0], dtype=torch.int32)
    result = divide_int(t, divisor=-10.0)
    assert result.dtype == torch.float32
    assert torch.allclose(result, torch.tensor([-1.0, 2.0, -0.0]))


def test_divide_int_multidimensional() -> None:
    t = torch.tensor([[10, 20], [30, 40]], dtype=torch.int32)
    result = divide_int(t, divisor=10.0)
    assert result.dtype == torch.float32
    assert result.shape == (2, 2)
    assert torch.allclose(result, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


def test_divide_int_preserves_float32() -> None:
    t = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)
    original_t = t.clone()
    result = divide_int(t, divisor=10.0)
    assert result.dtype == torch.float32
    assert torch.allclose(result, torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(t, original_t)


def test_divide_int_with_zero_divisor() -> None:
    t = torch.tensor([10, -10, 0], dtype=torch.int32)
    result = divide_int(t, divisor=0.0)
    assert result.dtype == torch.float32
    assert torch.isinf(result[0])
    assert result[0] > 0
    assert torch.isinf(result[1])
    assert result[1] < 0
    assert torch.isnan(result[2])


# ---------------------------------------------------------------------------
# multiply_if_max_below
# ---------------------------------------------------------------------------


def test_multiply_if_max_below_scales_when_below_max() -> None:
    # Amax is 5.0, which is < max_val (10.0)
    t = torch.tensor([-5.0, 2.0, 3.0], dtype=torch.float32)
    result, is_int = multiply_if_max_below(
        t, max_val=10.0, multiplier=2.0, dtype=torch.int32
    )

    assert is_int is True
    assert result.dtype == torch.int32
    # For integer comparison, torch.allclose on integers requires exact match or we can just use equal
    assert torch.equal(result, torch.tensor([-10, 4, 6], dtype=torch.int32))


def test_multiply_if_max_below_unchanged_when_at_max() -> None:
    # Amax is 10.0, which is == max_val (10.0)
    t = torch.tensor([-10.0, 2.0, 3.0], dtype=torch.float32)
    result, is_int = multiply_if_max_below(
        t, max_val=10.0, multiplier=2.0, dtype=torch.int32
    )

    assert is_int is False
    assert result is t


def test_multiply_if_max_below_unchanged_when_above_max() -> None:
    # Amax is 15.0, which is > max_val (10.0)
    t = torch.tensor([-15.0, 2.0, 3.0], dtype=torch.float32)
    result, is_int = multiply_if_max_below(
        t, max_val=10.0, multiplier=2.0, dtype=torch.int32
    )

    assert is_int is False
    assert result is t


# ---------------------------------------------------------------------------
# is_floating_point
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dtype_str", "expected"),
    [
        ("float32", True),
        ("float16", True),
        ("bfloat16", True),
        ("float8_e4m3fn", True),
        ("float8_e5m2", True),
        ("int8", False),
        ("int32", False),
        (None, False),
    ],
)
def test_is_floating_point_byte(dtype_str: str | None, expected: bool) -> None:
    assert is_floating_point(Format.BYTE.value, b"", dtype_str) is expected


def test_is_floating_point_torch_tensor() -> None:
    assert (
        is_floating_point(Format.TORCH.value, torch.zeros(2, dtype=torch.float32), None)
        is True
    )
    assert (
        is_floating_point(Format.TORCH.value, torch.zeros(2, dtype=torch.int32), None)
        is False
    )


def test_is_floating_point_numpy_array() -> None:
    assert (
        is_floating_point(Format.NUMPY.value, np.zeros(2, dtype=np.float32), None)
        is True
    )
    assert (
        is_floating_point(Format.NUMPY.value, np.zeros(2, dtype=np.int32), None)
        is False
    )


def test_is_floating_point_unknown_format_returns_none() -> None:
    assert is_floating_point(9999, b"", None) is None


# ---------------------------------------------------------------------------
# DType lookups
# ---------------------------------------------------------------------------


def test_enum_from_code_roundtrip() -> None:
    for member in DType:
        if member is DType.NONE:
            continue
        assert DType.from_code(member.code) == member.name


def test_enum_from_code_unknown_returns_none() -> None:
    assert DType.from_code(9999) == DType.NONE


def test_enum_from_dtype_string_case_insensitive() -> None:
    assert DType.from_dtype("FLOAT32") == DType.FLOAT32
    assert DType.from_dtype("float32") == DType.FLOAT32


def test_enum_from_dtype_unknown_returns_none() -> None:
    assert DType.from_dtype("not_a_dtype") == DType.NONE


def test_enum_from_dtype_name_fallback() -> None:
    """Numpy 2.x / ml_dtypes BF16/FP8/FP4 dtype objects expose ``.name``
    matching the canonical ``dtype_str``. The fallback path recovers the
    right enum member from the dtype object directly, even though the
    enum's ``numpy_dtype`` slot is ``None`` for these entries."""

    class _FakeNpDtype:
        def __init__(self, name: str) -> None:
            self.name = name

    assert DType.from_dtype(_FakeNpDtype("bfloat16")) is DType.BFLOAT16
    assert DType.from_dtype(_FakeNpDtype("float8_e4m3fn")) is DType.FLOAT8_E4M3FN
    assert DType.from_dtype(_FakeNpDtype("float8_e5m2")) is DType.FLOAT8_E5M2
    # Unknown name still falls through.
    assert DType.from_dtype(_FakeNpDtype("not_a_dtype")) is DType.NONE
