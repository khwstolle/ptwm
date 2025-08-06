"""Container decode-path test for order1_arithmetic.

Forces the standard BF16 split chain (``Source -> BitReorderIeee16 ->
ByteSplit(2)``) so the exponent byte plane is produced and routed to a real
plane codec. The trial menu is restricted to ``[Identity, Order1Arithmetic]``
so the codec wins on a skewed, stationary exponent plane and its real
``dispatch_builtin -> decode`` path is exercised end-to-end.

Non-vacuous: it asserts (a) ``Order1Arithmetic`` was
the *selected* codec on the exponent plane AND (b) the rebuilt tensor is
bit-exact.
"""

from __future__ import annotations

import numpy as np
import pytest


def _codec_ids(blob: bytes, name: str) -> list[int]:
    from ptwm import _rust

    summaries = _rust.list_plane_summaries(blob)
    return [codec_id for tname, _, _, codec_id, _, _ in summaries if tname == name]


def _bf16_skewed_exponent_words(n: int) -> np.ndarray:
    """BF16 words with a skewed, stationary exponent and varied mantissa.

    A BF16 word is ``[sign:1][exp:8][mantissa:7]``. Drawing the 8-bit exponent
    from a small skewed set (heavy mass on 0x7F, the exponent of ~1.0) yields an
    exponent byte plane with low marginal entropy and mild lag-1 structure --
    exactly the regime where the static order-1 table earns its keep. The
    mantissa is random low bits (near-uniform). Built directly as little-endian
    uint16 (no ``ml_dtypes`` dependency); the codec only sees raw bytes.
    """
    rng = np.random.default_rng(0)
    exps = rng.choice(
        np.array([0x7D, 0x7E, 0x7F, 0x80], dtype=np.uint16),
        size=n,
        p=[0.05, 0.15, 0.70, 0.10],
    )
    mant = rng.integers(0, 128, size=n).astype(np.uint16)
    return ((exps << 7) | mant).astype("<u2")  # sign bit 0


def test_forced_order1_arithmetic_decode_path() -> None:
    try:
        from ptwm._rust import compress_model, decode_tensor
    except ImportError:
        pytest.skip("ptwm._core extension not built")

    from ptwm.codecs import CodecId
    from ptwm.preprocessing import _chains
    from ptwm.utils import DType as _DT

    # 512 x 128 = 65536 bf16 values -> exponent plane 64 KiB, well over the
    # codec's MIN_PLANE_BYTES (4096) so the static table amortizes decisively.
    n_rows, n_cols = 512, 128
    words = _bf16_skewed_exponent_words(n_rows * n_cols)
    shape = [n_rows, n_cols]
    raw = words.tobytes()

    chain = _chains.CHAIN_BF16_SPLIT(shape)
    assert chain is not None
    chain_bytes = chain.to_bytes()

    blob = compress_model(
        records=[("t", _DT.BFLOAT16.code, 0, raw, shape, "bfloat16", None)],
        chains_per_tensor=[[chain_bytes]],
        emit_payload_hash=True,
        emit_plane_crc=True,
        codec_menu=[int(CodecId.Identity), int(CodecId.Order1Arithmetic)],
    )

    # (a) The codec must actually have been selected on the exponent plane.
    codec_ids = _codec_ids(blob, "t")
    assert int(CodecId.Order1Arithmetic) in codec_ids, (
        f"order1_arithmetic must be selected on a skewed exponent plane; "
        f"got codec ids {codec_ids}"
    )

    # (b) Bit-exact roundtrip through the dispatch_builtin decode path.
    assert decode_tensor(blob, "t") == raw
