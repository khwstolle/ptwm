import pytest
from ptwm.classify import ExplicitFlagsClassifier, TensorRole


def test_parses_glob_role_pairs():
    c = ExplicitFlagsClassifier.from_flag_strings(
        ["*_scales=scale_block", "*_blocks=packed_values"]
    )
    out = c.classify("foo_scales", "U8", (10,), ())
    assert out.role is TensorRole.SCALE_BLOCK
    assert out.pattern == "*_scales"
    assert out.source == "flag"


def test_first_flag_wins():
    c = ExplicitFlagsClassifier.from_flag_strings(
        ["*_scales=scale_block", "*=standard"]
    )
    out = c.classify("foo_scales", "U8", (10,), ())
    assert out.pattern == "*_scales"


def test_no_match_returns_none():
    c = ExplicitFlagsClassifier.from_flag_strings(["*_blocks=packed_values"])
    assert c.classify("foo_scales", "U8", (10,), ()) is None


def test_invalid_role_rejected():
    with pytest.raises(ValueError, match="invented"):
        ExplicitFlagsClassifier.from_flag_strings(["*=invented"])


def test_missing_separator_rejected():
    with pytest.raises(ValueError, match="expected '<glob>=<role>'"):
        ExplicitFlagsClassifier.from_flag_strings(["malformed"])
