"""Tests for the huff_llm_5bit raw-passthrough chains."""

from __future__ import annotations

import pytest
from ptwm.codecs import CodecId
from ptwm.preprocessing import _chains
from ptwm.preprocessing._chains import ClassifierRole, chains_for
from ptwm.preprocessing._explorer import is_legal_chain
from ptwm.utils import DType as _DT


def test_codec_id_has_huff_llm() -> None:
    assert CodecId.HuffLlm5Bit == 0x0016


@pytest.mark.parametrize(
    "builder",
    [_chains.CHAIN_BF16_HUFF_LLM_RAW, _chains.CHAIN_FP16_HUFF_LLM_RAW],
)
def test_raw_chain_is_legal(builder) -> None:
    chain = builder([64, 64])
    assert chain is not None
    # is_legal_chain validates the chain structure against the pruning rules.
    assert is_legal_chain(chain)


def test_raw_chains_registered() -> None:
    bf16 = chains_for(_DT.BFLOAT16.code, ClassifierRole.STANDARD)
    fp16 = chains_for(_DT.FLOAT16.code, ClassifierRole.STANDARD)
    assert _chains.CHAIN_BF16_HUFF_LLM_RAW in bf16
    assert _chains.CHAIN_FP16_HUFF_LLM_RAW in fp16
