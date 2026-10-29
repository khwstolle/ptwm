"""Buffer inputs of the binding functions that release the GIL.

These functions borrow ``bytes`` (and read-only ``mmap``) without a copy and
copy every other buffer while the GIL is held, so another thread cannot
change the bytes while Rust code reads them (see
``crates/ptwm-py/src/buffer.rs``).
"""

from __future__ import annotations

import mmap
import os
import threading
from collections.abc import Callable

import numpy as np
import pytest
from ptwm import _core
from ptwm.sharding._writer import _default_chain_bytes
from ptwm.utils import DType

_DATA = np.random.default_rng(0).integers(0, 16, 4096, dtype=np.uint8).tobytes()
_REF = np.random.default_rng(1).integers(0, 256, 4096, dtype=np.uint8).tobytes()


def _readonly_array(b: bytes) -> np.ndarray:
    arr = np.frombuffer(bytearray(b), dtype=np.uint8)
    view = arr.view()
    view.flags.writeable = False
    return view


# Ways to pass the same bytes; each must give the result ``bytes`` gives.
_WRAPPERS: dict[str, Callable[[bytes], object]] = {
    "bytearray": bytearray,
    "memoryview_bytes": memoryview,
    "memoryview_bytearray": lambda b: memoryview(bytearray(b)),
    "readonly_memoryview_bytearray": lambda b: memoryview(bytearray(b)).toreadonly(),
    "numpy_uint8": lambda b: np.frombuffer(bytearray(b), dtype=np.uint8),
    "readonly_numpy_uint8": _readonly_array,
}


def _compress_compact(raw: object) -> bytes | None:
    dt = DType.from_dtype("float32")
    return _core.compress_compact(
        int(dt.code),
        0,
        raw,
        [len(_DATA) // 4],
        dt.dtype_str,
        None,
        _default_chain_bytes(int(dt.code), (len(_DATA) // 4,), len(_DATA)),
        True,
        True,
    )


def _compress_model(raw: object) -> bytes:
    dt = DType.from_dtype("float32")
    shape = [len(_DATA) // 4]
    return _core.compress_model(
        [("t", int(dt.code), 0, raw, shape, dt.dtype_str, None)],
        [_default_chain_bytes(int(dt.code), tuple(shape), len(_DATA))],
        True,
        True,
    )


# name -> (call taking the wrapped buffers, bytes inputs)
_CASES: dict[str, tuple[Callable[..., object], tuple[bytes, ...]]] = {
    "identity_encode": (_core.identity_encode, (_DATA,)),
    "identity_decode": (
        lambda b: _core.identity_decode(b, len(_DATA)),
        (_core.identity_encode(_DATA),),
    ),
    "huffman_encode": (_core.huffman_encode, (_DATA,)),
    "huffman_decode": (
        lambda b: _core.huffman_decode(b, len(_DATA)),
        (_core.huffman_encode(_DATA),),
    ),
    "rans_encode": (_core.rans_encode, (_DATA,)),
    "rans_decode": (
        lambda b: _core.rans_decode(b, len(_DATA)),
        (_core.rans_encode(_DATA),),
    ),
    "zstd_encode": (_core.zstd_encode, (_DATA,)),
    "zstd_decode": (_core.zstd_decode, (_core.zstd_encode(_DATA),)),
    "delta_xor_encode": (_core.delta_xor_encode, (_DATA, _REF)),
    "delta_xor_decode": (
        _core.delta_xor_decode,
        (_core.delta_xor_encode(_DATA, _REF), _REF),
    ),
    "delta_float_sub_encode": (_core.delta_float_sub_encode, (_DATA, _REF)),
    "delta_float_sub_decode": (
        _core.delta_float_sub_decode,
        (_core.delta_float_sub_encode(_DATA, _REF), _REF),
    ),
    "blake3_hash": (_core.blake3_hash, (_DATA,)),
    "compress_compact": (_compress_compact, (_DATA,)),
    "compress_model": (_compress_model, (_DATA,)),
}


@pytest.mark.parametrize("wrapper", sorted(_WRAPPERS))
@pytest.mark.parametrize("name", sorted(_CASES))
def test_buffer_types_match_bytes(name: str, wrapper: str) -> None:
    fn, inputs = _CASES[name]
    expected = fn(*inputs)
    assert expected is not None
    wrap = _WRAPPERS[wrapper]
    assert fn(*(wrap(b) for b in inputs)) == expected


def test_non_contiguous_buffer_is_rejected() -> None:
    strided = np.frombuffer(_DATA, dtype=np.uint8)[::2]
    with pytest.raises(BufferError, match="data: buffer must be C-contiguous"):
        _core.zstd_encode(strided)


def test_non_buffer_argument_names_the_argument() -> None:
    with pytest.raises(TypeError, match="argument 'data': a bytes-like object"):
        _core.zstd_encode(123)


def test_borrow_classification(tmp_path) -> None:
    is_borrowed = _core._buffer_is_borrowed
    b = b"\x00" * 64
    # Immutable owners are borrowed, also through chains of memoryviews.
    assert is_borrowed(b)
    assert is_borrowed(b"")
    assert is_borrowed(memoryview(b))
    assert is_borrowed(memoryview(memoryview(b)))
    assert is_borrowed(memoryview(b).toreadonly())
    # Mutable owners are copied, also behind read-only exports.
    assert not is_borrowed(bytearray(b))
    assert not is_borrowed(memoryview(bytearray(b)))
    assert not is_borrowed(memoryview(bytearray(b)).toreadonly())
    assert not is_borrowed(np.frombuffer(bytearray(b), dtype=np.uint8))
    assert not is_borrowed(_readonly_array(b))
    # A non-contiguous view of immutable memory is copied.
    assert not is_borrowed(memoryview(b)[::2])

    class BytesSub(bytes):
        pass

    # A subclass could override the buffer protocol: copied.
    assert not is_borrowed(BytesSub(b))

    path = tmp_path / "m.bin"
    path.write_bytes(b)
    with (
        path.open("rb") as fh,
        mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm,
    ):
        assert is_borrowed(mm)
        with memoryview(mm) as mv:
            assert is_borrowed(mv)
    with path.open("r+b") as fh, mmap.mmap(fh.fileno(), 0) as mm:
        assert not is_borrowed(mm)
        with memoryview(mm).toreadonly() as mv:
            assert not is_borrowed(mv)


def test_concurrent_writes_do_not_reach_the_reader() -> None:
    # A second thread swaps the whole bytearray between states A and B. Slice
    # assignment is one bytecode, so with the copy taken under the GIL every
    # result encodes A or B; a mix of both means Rust code read the live
    # bytearray while it was written.
    n = 4 << 20
    a = os.urandom(n)
    b = bytes(n)
    zeros = bytes(n)
    ba = bytearray(a)
    stop = threading.Event()

    def mutate() -> None:
        while not stop.is_set():
            ba[:] = b
            ba[:] = a

    t = threading.Thread(target=mutate)
    t.start()
    try:
        for _ in range(20):
            out = _core.delta_xor_encode(ba, zeros)
            got = _core.delta_xor_decode(out, zeros)
            torn = got not in (a, b)
            assert not torn, "delta_xor_encode read a torn buffer"
        for _ in range(3):
            got = _core.zstd_decode(_core.zstd_encode(ba, 19))
            torn = got not in (a, b)
            assert not torn, "zstd_encode read a torn buffer"
    finally:
        stop.set()
        t.join()
