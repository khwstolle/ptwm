"""Container decode-path test for huff_llm_5bit.

Forces the raw-passthrough chain so the codec is exercised end-to-end:
forward (raw 16-bit word plane) -> HuffLlm5Bit encode -> wire ->
dispatch_builtin -> decode -> bit-exact tensor.

Non-vacuous: it asserts (a) the rebuilt bytes equal the original AND (b) that
HuffLlm5Bit was actually the *selected* codec on the raw plane, not just built
successfully.

The trial menu is restricted to ``[Identity, HuffLlm5Bit]`` -- the same
non-vacuous pattern as ``test_context_mixing_container_roundtrip``. HuffLlm5Bit
is a reference/ablation codec that is structurally dominated by the order-1
arithmetic coder in the full menu (an order-1 byte coder reaches the word's
order-0 entropy without Huffman rounding, table headers, or a raw sign bit), so
an unrestricted "wins the whole menu" assertion is unachievable by design.
Restricting the menu to Identity lets HuffLlm5Bit win on a field-compressible
plane and exercises its real ``dispatch_builtin -> decode`` path.
"""

from __future__ import annotations

import numpy as np
import pytest


def _huff_llm_codec_ids(blob: bytes, name: str) -> list[int]:
    from ptwm import _rust

    summaries = _rust.list_plane_summaries(blob)
    return [codec_id for tname, _, _, codec_id, _, _ in summaries if tname == name]


def _bf16_field_compressible_words(n: int) -> np.ndarray:
    """BF16 words with a constant sign+exponent and a varied mantissa.

    A BF16 word is `[sign:1][exp:8][mantissa:7]`. Fixing the top 9 bits and
    varying only the low 7 makes the (little-endian) high byte constant and the
    low byte vary -- exactly the regime where coding the bit-field groups
    independently beats pooling all bytes into one histogram. The bytes are
    built directly as little-endian uint16 (no ``ml_dtypes`` dependency); the
    codec only ever sees raw bytes, and the chain's Source node carries the
    ``bfloat16`` dtype tag.
    """
    # 0x3F80 == 1.0 in bfloat16 (sign 0, exp 0x7F, mantissa 0). Vary only the
    # low 3 mantissa bits with a skewed distribution so Huffman has signal.
    rng = np.random.default_rng(0)
    mant = rng.integers(0, 8, size=n).astype(np.uint16)
    return (np.uint16(0x3F80) | mant).astype("<u2")


def test_forced_huff_llm_decode_path() -> None:
    try:
        from ptwm._rust import compress_model, decode_tensor
    except ImportError:
        pytest.skip("ptwm._core extension not built")

    from ptwm.codecs import CodecId
    from ptwm.preprocessing import _chains
    from ptwm.utils import DType as _DT

    n_rows, n_cols = 256, 64
    words = _bf16_field_compressible_words(n_rows * n_cols)
    shape = [n_rows, n_cols]
    raw = words.tobytes()

    chain = _chains.CHAIN_BF16_HUFF_LLM_RAW(shape)
    assert chain is not None
    chain_bytes = chain.to_bytes()

    blob = compress_model(
        records=[("t", _DT.BFLOAT16.code, 0, raw, shape, "bfloat16", None)],
        chains_per_tensor=[[chain_bytes]],
        emit_payload_hash=True,
        emit_plane_crc=True,
        codec_menu=[int(CodecId.Identity), int(CodecId.HuffLlm5Bit)],
    )

    # (b) The codec must have actually been selected on the raw plane.
    codec_ids = _huff_llm_codec_ids(blob, "t")
    assert int(CodecId.HuffLlm5Bit) in codec_ids, (
        f"huff_llm_5bit must be selected on a field-compressible plane; "
        f"got codec ids {codec_ids}"
    )

    # (a) Bit-exact roundtrip through the dispatch_builtin decode path.
    assert decode_tensor(blob, "t") == raw
