"""Tests for the --codec-menu CLI flag parser."""

from __future__ import annotations

import pytest
from ptwm.cli.compress import _parse_codec_menu
from ptwm.codecs import CodecId


def test_parse_none_returns_none():
    """None input returns None."""
    assert _parse_codec_menu(None) is None


def test_parse_single_codec():
    """Single codec name parses correctly."""
    assert _parse_codec_menu("huffman") == [CodecId.Huffman]


def test_parse_multiple_codecs_case_insensitive():
    """Multiple codecs parse with case insensitivity."""
    out = _parse_codec_menu("Identity,HUFFMAN,order1-scale-ac")
    assert out == [CodecId.Identity, CodecId.Huffman, CodecId.Order1ScaleAC]


def test_parse_unknown_codec_raises():
    """Unknown codec name raises ValueError."""
    with pytest.raises(ValueError, match="unknown codec"):
        _parse_codec_menu("not-a-real-codec")


def test_parse_empty_string_returns_empty_list():
    """Empty string returns empty list."""
    assert _parse_codec_menu("") == []


def test_parse_handles_whitespace():
    """Leading/trailing whitespace is stripped."""
    out = _parse_codec_menu("  huffman , rans  ")
    assert out == [CodecId.Huffman, CodecId.Rans]


def test_parse_all_codecs():
    """All codec names parse correctly."""
    input_str = "identity,huffman,rans,zstd,per-group-codebook,order1-scale-ac"
    out = _parse_codec_menu(input_str)
    assert out == [
        CodecId.Identity,
        CodecId.Huffman,
        CodecId.Rans,
        CodecId.Zstd,
        CodecId.PerGroupCodebook,
        CodecId.Order1ScaleAC,
    ]


def test_parse_duplicate_codecs():
    """Duplicate codec names are allowed and kept."""
    out = _parse_codec_menu("huffman,huffman,rans")
    assert out == [CodecId.Huffman, CodecId.Huffman, CodecId.Rans]


def test_parse_preserves_order():
    """Codec order is preserved."""
    out = _parse_codec_menu("zstd,huffman,identity")
    assert out == [CodecId.Zstd, CodecId.Huffman, CodecId.Identity]


def test_parse_mixed_case():
    """Mixed case is handled correctly."""
    out = _parse_codec_menu("HuFFmAn,RaNs,ZsTd")
    assert out == [CodecId.Huffman, CodecId.Rans, CodecId.Zstd]


def test_parse_with_internal_spaces_in_names():
    """Codec names with hyphens parse correctly."""
    out = _parse_codec_menu("per-group-codebook")
    assert out == [CodecId.PerGroupCodebook]


def test_parse_extra_commas():
    """Extra commas (empty tokens) are skipped."""
    out = _parse_codec_menu("huffman,,rans,")
    assert out == [CodecId.Huffman, CodecId.Rans]
