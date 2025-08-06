import numpy as np

from . import build_synthetic_nvfp4


def test_fixture_shapes_and_dtypes():
    state = build_synthetic_nvfp4(n_layers=1, out_features=128, in_features=64)
    base = "model.layers.0.self_attn.q_proj"
    expected = {
        f"{base}.weight",
        f"{base}.weight_scale",
        f"{base}.weight_scale_2",
        f"{base}.input_scale",
    }
    assert expected.issubset(set(state.keys()))
    # group_size=16 → in_features/group_size = 4
    assert state[f"{base}.weight"].shape == (128, 32)  # in_features/2
    assert state[f"{base}.weight_scale"].shape == (128, 4)
    assert state[f"{base}.weight_scale_2"].shape == ()
    assert state[f"{base}.input_scale"].shape == ()


def test_fixture_is_deterministic():
    a = build_synthetic_nvfp4(seed=7)
    b = build_synthetic_nvfp4(seed=7)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
