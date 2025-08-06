import numpy as np

from . import build_synthetic_mxfp4


def test_fixture_shapes_and_keys():
    state = build_synthetic_mxfp4(n_experts=2, hidden=8, n_blocks=3)
    base = "model.layers.0.mlp.experts"
    expected = {
        f"{base}.down_proj_blocks",
        f"{base}.down_proj_scales",
        f"{base}.down_proj_bias",
        f"{base}.gate_up_proj_blocks",
        f"{base}.gate_up_proj_scales",
        f"{base}.gate_up_proj_bias",
    }
    assert set(state.keys()) == expected
    # blocks: 16 bytes per block × n_blocks × hidden × n_experts
    assert state[f"{base}.down_proj_blocks"].shape == (2, 8, 3, 16)
    assert state[f"{base}.down_proj_scales"].shape == (2, 8, 3)
    assert state[f"{base}.down_proj_bias"].shape == (2, 8)


def test_fixture_is_deterministic():
    a = build_synthetic_mxfp4(n_experts=2, hidden=8, n_blocks=3, seed=42)
    b = build_synthetic_mxfp4(n_experts=2, hidden=8, n_blocks=3, seed=42)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
