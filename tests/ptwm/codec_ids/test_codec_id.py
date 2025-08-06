"""CodecId IntEnum mirrors the Rust codec registry."""

from __future__ import annotations

import pytest
from ptwm import _rust
from ptwm.codecs import CodecId

# Rust → Python name mapping. The Rust registry uses snake_case; the
# Python IntEnum uses PascalCase. Any Rust id missing here is a sign
# the Python enum drifted and must be updated.
_RUST_NAME_TO_PY = {
    "identity": "Identity",
    "huffman": "Huffman",
    "huffman_nibble": None,  # Reserved on Rust side; not yet on Python.
    "rans": "Rans",
    "zstd": "Zstd",
    "zstd_dict": "ZstdDict",
    "fpc": "Fpc",
    "tans": "Tans",
    "per_group_codebook": "PerGroupCodebook",
    "order1_scale_ac": "Order1ScaleAC",
    "arithmetic_o0": "ArithmeticO0",
    "arithmetic_o0_adaptive": "ArithmeticO0Adaptive",
    "arithmetic_o1": "ArithmeticO1",
    "context_mixing_lite": "ContextMixingLite",
    "huff_llm_5bit": "HuffLlm5Bit",
    "order1_arithmetic": "Order1Arithmetic",
    "neural_predictor": "NeuralPredictor",
}


def test_codec_id_values_match_rust():
    assert CodecId.Identity == 0x0000
    assert CodecId.Huffman == 0x0001
    assert CodecId.Rans == 0x0003
    assert CodecId.Zstd == 0x0004
    assert CodecId.ZstdDict == 0x0005
    assert CodecId.Fpc == 0x0006
    assert CodecId.Tans == 0x0007
    assert CodecId.PerGroupCodebook == 0x0010
    assert CodecId.Order1ScaleAC == 0x0011


def test_codec_id_is_intenum():
    assert int(CodecId.Order1ScaleAC) == 0x0011
    assert CodecId.Order1ScaleAC + 0 == 0x0011


@pytest.mark.parametrize(("rust_name", "wire_id"), _rust.codec_id_registry())
def test_python_enum_matches_rust_registry(rust_name: str, wire_id: int):
    """Live cross-check: pull Rust truth at runtime, assert Python mirrors."""
    py_name = _RUST_NAME_TO_PY.get(rust_name)
    if py_name is None:
        # Reserved Rust id with no Python member yet — that's allowed,
        # but the mapping must be explicit (not a silent KeyError).
        assert rust_name in _RUST_NAME_TO_PY, (
            f"Rust registry has unknown id {rust_name!r}; "
            f"add an entry to _RUST_NAME_TO_PY"
        )
        return
    assert hasattr(CodecId, py_name), (
        f"Rust id {rust_name!r}=0x{wire_id:04x} has no Python member"
    )
    assert int(getattr(CodecId, py_name)) == wire_id, (
        f"CodecId.{py_name} = 0x{int(getattr(CodecId, py_name)):04x} "
        f"does not match Rust 0x{wire_id:04x}"
    )


def test_no_python_id_missing_from_rust_registry():
    rust_ids = {wire_id for _, wire_id in _rust.codec_id_registry()}
    for member in CodecId:
        assert int(member) in rust_ids, (
            f"Python CodecId.{member.name} = 0x{int(member):04x} "
            f"not present in Rust registry"
        )
