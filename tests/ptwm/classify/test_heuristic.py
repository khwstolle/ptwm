from ptwm.classify import HeuristicClassifier, TensorRole


def test_modelopt_style_scale_recognised():
    keys = (
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.weight_scale",
    )
    c = HeuristicClassifier()
    out = c.classify(
        "model.layers.0.self_attn.q_proj.weight_scale",
        "float8_e4m3fn",
        (8192, 512),
        keys,
    )
    assert out is not None
    assert out.role is TensorRole.SCALE_BLOCK
    assert out.source == "heuristic"


def test_oai_style_scale_recognised():
    keys = (
        "model.layers.0.mlp.experts.down_proj_blocks",
        "model.layers.0.mlp.experts.down_proj_scales",
    )
    c = HeuristicClassifier()
    out = c.classify(
        "model.layers.0.mlp.experts.down_proj_scales",
        "uint8",
        (32, 8, 90),
        keys,
    )
    assert out is not None
    assert out.role is TensorRole.SCALE_BLOCK


def test_wrong_dtype_returns_none():
    keys = ("foo.weight", "foo.weight_scale")
    c = HeuristicClassifier()
    assert c.classify("foo.weight_scale", "float32", (10, 10), keys) is None


def test_no_sibling_weight_returns_none():
    keys = ("foo.weight_scale",)
    c = HeuristicClassifier()
    assert c.classify("foo.weight_scale", "float8_e4m3fn", (10, 10), keys) is None


def test_unrelated_name_returns_none():
    keys = ("foo.weight", "foo.bias")
    c = HeuristicClassifier()
    assert c.classify("foo.bias", "bfloat16", (10,), keys) is None
