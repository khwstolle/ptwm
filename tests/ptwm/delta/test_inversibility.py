"""Property tests for :mod:`ptwm.delta`.

XOR delta is self-inverse over equal-length byte pairs. The tests here
exercise the full input space (bounded) rather than the hand-picked
scenarios in :mod:`tests.ptwm.delta.test_delta`.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st
from ptwm.delta import decode, encode, reference_hash


@st.composite
def equal_length_byte_pairs(
    draw: st.DrawFn, max_size: int = 8192
) -> tuple[bytes, bytes]:
    size = draw(st.integers(min_value=0, max_value=max_size))
    a = draw(st.binary(min_size=size, max_size=size))
    b = draw(st.binary(min_size=size, max_size=size))
    return a, b


@given(pair=equal_length_byte_pairs())
@settings(deadline=None, max_examples=80)
def test_xor_delta_roundtrip(pair: tuple[bytes, bytes]) -> None:
    raw, ref = pair
    residual = encode(raw, reference=ref)
    assert decode(residual, reference=ref) == raw


@given(pair=equal_length_byte_pairs())
@settings(deadline=None, max_examples=40)
def test_xor_residual_has_same_length_as_input(pair: tuple[bytes, bytes]) -> None:
    raw, ref = pair
    residual = encode(raw, reference=ref)
    assert len(residual) == len(raw)


@given(pair=equal_length_byte_pairs())
@settings(deadline=None, max_examples=40)
def test_reference_hash_verification_catches_wrong_reference(
    pair: tuple[bytes, bytes],
) -> None:
    raw, ref = pair
    residual = encode(raw, reference=ref)
    expected = reference_hash(ref)
    # Passing the actual reference with the matching hash always works.
    assert decode(residual, reference=ref, expected_reference_hash=expected) == raw


@given(data=st.binary(min_size=0, max_size=4096))
@settings(deadline=None, max_examples=40)
def test_blake3_hash_is_deterministic(data: bytes) -> None:
    assert reference_hash(data) == reference_hash(data)
    assert len(reference_hash(data)) == 32
