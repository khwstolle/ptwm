import json

import pytest
from ptwm._exceptions import UnsupportedQuantConfigError
from ptwm.classify import HfQuantConfigClassifier, TensorRole


def _write(tmp_path, payload):
    p = tmp_path / "hf_quant_config.json"
    p.write_text(json.dumps(payload))
    return p


def test_nvfp4_translation_rules(tmp_path):
    cfg = _write(
        tmp_path,
        {
            "producer": {"name": "modelopt", "version": "0.23.0"},
            "quantization": {
                "quant_algo": "NVFP4",
                "kv_cache_quant_algo": "FP8",
                "group_size": 16,
                "exclude_modules": ["lm_head"],
            },
        },
    )
    c = HfQuantConfigClassifier.from_path(cfg)
    base = "model.layers.0.self_attn.q_proj"
    keys = (
        f"{base}.weight",
        f"{base}.weight_scale",
        f"{base}.weight_scale_2",
        f"{base}.input_scale",
    )
    assert (
        c.classify(f"{base}.weight", "U8", (8192, 4096), keys).role
        is TensorRole.PACKED_VALUES
    )
    assert (
        c.classify(f"{base}.weight_scale", "F8_E4M3", (8192, 512), keys).role
        is TensorRole.SCALE_BLOCK
    )
    assert (
        c.classify(f"{base}.weight_scale_2", "F32", (), keys).role
        is TensorRole.SCALE_GLOBAL
    )
    assert (
        c.classify(f"{base}.input_scale", "F32", (), keys).role
        is TensorRole.SCALE_GLOBAL
    )


def test_excluded_modules_return_none(tmp_path):
    cfg = _write(
        tmp_path,
        {
            "producer": {"name": "modelopt", "version": "0.23.0"},
            "quantization": {
                "quant_algo": "NVFP4",
                "kv_cache_quant_algo": None,
                "group_size": 16,
                "exclude_modules": ["lm_head"],
            },
        },
    )
    c = HfQuantConfigClassifier.from_path(cfg)
    assert (
        c.classify("lm_head.weight", "BF16", (32000, 4096), ("lm_head.weight",)) is None
    )


def test_unknown_algo_refuses(tmp_path):
    cfg = _write(
        tmp_path,
        {
            "producer": {"name": "modelopt", "version": "0.23.0"},
            "quantization": {
                "quant_algo": "INVENTED_ALGO",
                "kv_cache_quant_algo": None,
                "group_size": 16,
                "exclude_modules": [],
            },
        },
    )
    with pytest.raises(UnsupportedQuantConfigError):
        HfQuantConfigClassifier.from_path(cfg)


def test_non_quantized_modules_return_none(tmp_path):
    """Modules without a sibling weight_scale aren't quantized; return None."""
    cfg = _write(
        tmp_path,
        {
            "producer": {"name": "modelopt", "version": "0.23.0"},
            "quantization": {
                "quant_algo": "NVFP4",
                "kv_cache_quant_algo": None,
                "group_size": 16,
                "exclude_modules": [],
            },
        },
    )
    c = HfQuantConfigClassifier.from_path(cfg)
    keys = ("model.embed_tokens.weight",)  # no sibling weight_scale
    assert c.classify("model.embed_tokens.weight", "BF16", (32000, 4096), keys) is None
