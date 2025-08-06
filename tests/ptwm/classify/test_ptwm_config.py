import textwrap

import pytest
from ptwm.classify import PtwmConfigClassifier, TensorRole


def _write(tmp_path, body):
    p = tmp_path / "ptwm.toml"
    p.write_text(body)
    return p


def test_first_match_wins(tmp_path):
    cfg = _write(
        tmp_path,
        textwrap.dedent(
            """
            [meta]
            quant_algo = "MXFP4"

            [[rule]]
            pattern = "*_scales"
            role = "scale_block"

            [[rule]]
            pattern = "*_blocks"
            role = "packed_values"
            """
        ).strip(),
    )
    c = PtwmConfigClassifier.from_path(cfg)
    out = c.classify(
        "model.layers.0.mlp.experts.down_proj_scales", "U8", (32, 8, 90), ()
    )
    assert out.role is TensorRole.SCALE_BLOCK
    assert out.source == "ptwm_config"
    assert out.pattern == "*_scales"


def test_no_match_returns_none(tmp_path):
    cfg = _write(
        tmp_path,
        textwrap.dedent(
            """
            [[rule]]
            pattern = "*_unrelated"
            role = "scale_global"
            """
        ).strip(),
    )
    c = PtwmConfigClassifier.from_path(cfg)
    assert c.classify("model.layers.0.weight", "BF16", (10, 10), ()) is None


def test_unknown_role_rejected(tmp_path):
    cfg = _write(
        tmp_path,
        textwrap.dedent(
            """
            [[rule]]
            pattern = "*"
            role = "invented_role"
            """
        ).strip(),
    )
    with pytest.raises(ValueError, match="invented_role"):
        PtwmConfigClassifier.from_path(cfg)


def test_empty_rules_returns_none(tmp_path):
    cfg = _write(tmp_path, "")
    c = PtwmConfigClassifier.from_path(cfg)
    assert c.classify("anything", "F32", (1,), ()) is None
