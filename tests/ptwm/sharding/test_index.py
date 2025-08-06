import json

import pytest
from ptwm.sharding import PtwmIndex


def test_round_trip(tmp_path):
    idx = PtwmIndex(
        total_size=102538378240,
        compressed_total_size=56210334752,
        weight_map={
            "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00007.ptwm",
            "model.layers.0.self_attn.q_proj.weight_scale": "model-00001-of-00007.ptwm",
            "model.layers.99.weight": "model-00007-of-00007.ptwm",
        },
    )
    p = tmp_path / "model.ptwm.index.json"
    idx.write(p)
    raw = json.loads(p.read_text())
    assert raw["metadata"]["format"] == "ptwm"
    assert raw["metadata"]["format_version"] == 1
    assert raw["metadata"]["total_size"] == 102538378240
    assert raw["metadata"]["compressed_total_size"] == 56210334752
    assert raw["weight_map"]["model.layers.99.weight"] == "model-00007-of-00007.ptwm"

    loaded = PtwmIndex.read(p)
    assert loaded == idx


def test_safetensors_compat_extraction(tmp_path):
    idx = PtwmIndex(
        total_size=100,
        compressed_total_size=50,
        weight_map={
            "a": "model-00001-of-00002.ptwm",
            "b": "model-00002-of-00002.ptwm",
            "c": "model-00001-of-00002.ptwm",
        },
    )
    p = tmp_path / "model.ptwm.index.json"
    idx.write(p)
    raw = json.loads(p.read_text())
    # Standard safetensors-style shard extraction works on this index.
    shards = sorted(set(raw["weight_map"].values()))
    assert shards == ["model-00001-of-00002.ptwm", "model-00002-of-00002.ptwm"]


def test_unknown_format_raises(tmp_path):
    bad = tmp_path / "x.json"
    bad.write_text(json.dumps({"metadata": {"format": "other"}, "weight_map": {}}))
    with pytest.raises(ValueError, match="format"):
        PtwmIndex.read(bad)
