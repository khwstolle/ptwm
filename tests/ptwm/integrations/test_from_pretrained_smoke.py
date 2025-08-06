"""Smoke test: patch_transformers() + from_pretrained on a synthetic model.

The dummy model is small (no GPU, no network). Verifies the patched
resolution + load_state_dict flow round-trips state_dict bit-exactly.

Two paths are exercised:

1. **BYTE-format path** (``test_from_pretrained_round_trips_through_ptwm``):
   Compresses ``model.safetensors`` as a raw BYTE blob →
   ``model.safetensors.ptwm``, materialises it back via
   ``materialize_hf_cache``, and loads with ``from_pretrained``. This is the
   documented fallback for frozen Docker images / sealed wheels.

2. **Multi-tensor path** (``test_load_multi_tensor_ptwm_returns_state_dict``
   and ``test_from_pretrained_multi_tensor_ptwm``):
   Compresses with ``compress_safetensors_file(mode="a")`` to produce a
   multi-tensor ``.ptwm`` container, then exercises the
   ``load_state_dict`` wrapper (via ``_load_multi_tensor_ptwm``) directly
   and via ``from_pretrained``.
"""

from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")

from transformers import PretrainedConfig, PreTrainedModel


class _TinyConfig(PretrainedConfig):
    model_type = "tiny_test"

    def __init__(self, hidden=8, **kw):
        super().__init__(**kw)
        self.hidden = hidden


class _TinyModel(PreTrainedModel):
    config_class = _TinyConfig

    def __init__(self, config: _TinyConfig):
        super().__init__(config)
        self.linear = torch.nn.Linear(config.hidden, config.hidden, bias=True)
        self.post_init()  # required for transformers ≥ 5.x (initialises all_tied_weights_keys)


def _save_synthetic(tmp_path: Path) -> Path:
    cfg = _TinyConfig(hidden=8)
    model = _TinyModel(cfg)
    model_dir = tmp_path / "model_src"
    model_dir.mkdir()
    cfg.save_pretrained(str(model_dir))
    model.save_pretrained(str(model_dir), safe_serialization=True)
    return model_dir


def _register_tiny_model() -> None:
    """Register dummy model/config with transformers' AUTO registries (idempotent)."""
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.auto.modeling_auto import MODEL_MAPPING

    if "tiny_test" not in CONFIG_MAPPING:
        CONFIG_MAPPING.register("tiny_test", _TinyConfig)
        MODEL_MAPPING.register(_TinyConfig, _TinyModel)


# ---------------------------------------------------------------------------
# Test 1: legacy BYTE-format path via materialize_hf_cache
# ---------------------------------------------------------------------------


def test_from_pretrained_round_trips_through_ptwm(tmp_path):
    from ptwm._config import CompressionConfig, Format, Method
    from ptwm.core import Compressor
    from ptwm.integrations import materialize_hf_cache, patch_transformers

    src_dir = _save_synthetic(tmp_path)
    src_safe = next(src_dir.glob("*.safetensors"))

    out_dir = tmp_path / "model_ptwm"
    out_dir.mkdir()
    # Copy non-tensor files (config.json) so from_pretrained can find them.
    for f in src_dir.iterdir():
        if f.suffix in (".json", ".txt"):
            (out_dir / f.name).write_bytes(f.read_bytes())

    # Compress the safetensors file as a BYTE-format blob so that
    # materialize_hf_cache can reconstruct the native model.safetensors.
    # This mirrors the HF-cache snapshot use-case described in the spec.
    cfg = CompressionConfig(input_format=Format.BYTE, method=Method.HUFFMAN)
    raw_bytes = src_safe.read_bytes()
    compressed = Compressor(cfg).compress(raw_bytes)
    ptwm_path = out_dir / (src_safe.name + ".ptwm")  # model.safetensors.ptwm
    ptwm_path.write_bytes(compressed)

    # Install the HF integration patch (exercises the patch code path).
    patch_transformers()

    # Decompress .ptwm → model.safetensors so that from_pretrained can find it.
    materialize_hf_cache(out_dir)

    # Register the dummy model class for from_pretrained to pick up.
    _register_tiny_model()

    loaded = _TinyModel.from_pretrained(str(out_dir))
    original = _TinyModel.from_pretrained(str(src_dir))
    for (n_a, p_a), (n_b, p_b) in zip(
        loaded.state_dict().items(),
        original.state_dict().items(),
        strict=True,
    ):
        assert n_a == n_b
        torch.testing.assert_close(p_a, p_b)


# ---------------------------------------------------------------------------
# Test 2: multi-tensor path — unit-style exercise of _load_multi_tensor_ptwm
# ---------------------------------------------------------------------------


def test_load_multi_tensor_ptwm_returns_state_dict(tmp_path):
    """_load_multi_tensor_ptwm decodes a multi-tensor container correctly."""
    from ptwm.integrations._compress_safetensors import compress_safetensors_file
    from ptwm.integrations._hf import _load_multi_tensor_ptwm

    src_dir = _save_synthetic(tmp_path)
    src_safe = next(src_dir.glob("*.safetensors"))

    out_dir = tmp_path / "model_multi"
    # compress_safetensors_file(mode="a") writes model.ptwm (single shard)
    compress_safetensors_file(src_safe, out_dir, mode="a")

    ptwm_path = out_dir / "model.ptwm"
    assert ptwm_path.exists(), "compress_safetensors_file did not write model.ptwm"

    state = _load_multi_tensor_ptwm(ptwm_path)

    # The tiny model has linear.weight and linear.bias.
    assert "linear.weight" in state, f"missing linear.weight; keys={list(state)}"
    assert "linear.bias" in state, f"missing linear.bias; keys={list(state)}"

    # Verify shape and dtype round-trip.
    import safetensors.torch as st

    reference = st.load_file(str(src_safe))
    for name, ref_tensor in reference.items():
        assert name in state, f"tensor {name!r} missing from decoded state"
        got = state[name]
        assert got.shape == ref_tensor.shape, (
            f"{name}: shape mismatch {got.shape} vs {ref_tensor.shape}"
        )
        assert got.dtype == ref_tensor.dtype, (
            f"{name}: dtype mismatch {got.dtype} vs {ref_tensor.dtype}"
        )
        torch.testing.assert_close(got, ref_tensor)


# ---------------------------------------------------------------------------
# Test 3: multi-tensor path via from_pretrained + patch_transformers() patch
# ---------------------------------------------------------------------------


def test_from_pretrained_multi_tensor_ptwm(tmp_path):
    """from_pretrained loads a multi-tensor .ptwm directory via the patch.

    ``patch_transformers()`` installs two patches:
    1. ``_get_resolved_checkpoint_files`` → discovers ``model.ptwm`` shards.
    2. ``load_state_dict`` → calls ``_load_multi_tensor_ptwm`` for .ptwm blobs.

    Together these should allow ``from_pretrained`` to load a directory
    containing only ``model.ptwm`` + ``config.json``, with no intermediate
    materialisation step.

    If transformers' internals call additional methods on the path before
    ``load_state_dict`` runs (e.g. ``is_safetensors_compatible``), those are
    outside the reach of our patches and the test is expected to surface that
    constraint.  In that case the failure message will explain what blocked.
    """
    from ptwm.integrations._compress_safetensors import compress_safetensors_file
    from ptwm.integrations._hf import patch_transformers

    src_dir = _save_synthetic(tmp_path)
    src_safe = next(src_dir.glob("*.safetensors"))

    out_dir = tmp_path / "model_multi_hf"
    # Copy non-tensor files so from_pretrained finds config.json.
    out_dir.mkdir()
    for f in src_dir.iterdir():
        if f.suffix in (".json", ".txt"):
            (out_dir / f.name).write_bytes(f.read_bytes())

    # Compress as multi-tensor .ptwm.
    compress_safetensors_file(src_safe, out_dir, mode="a")

    # Install patches before calling from_pretrained.
    patch_transformers()
    _register_tiny_model()

    loaded = _TinyModel.from_pretrained(str(out_dir))
    original = _TinyModel.from_pretrained(str(src_dir))

    for (n_a, p_a), (n_b, p_b) in zip(
        loaded.state_dict().items(),
        original.state_dict().items(),
        strict=True,
    ):
        assert n_a == n_b
        torch.testing.assert_close(p_a, p_b)
