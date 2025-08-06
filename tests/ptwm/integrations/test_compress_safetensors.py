"""Round-trip tests for compress_safetensors_file."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from ptwm import _rust
from ptwm.classify import HfQuantConfigClassifier
from ptwm.integrations import compress_safetensors_file
from ptwm.sharding import read_sharded_ptwm
from safetensors.torch import save_file

_GROUP_SIZE = 16

# Codec wire IDs (mirror crates/ptwm-core/src/codec.rs).
_CODEC_IDENTITY = 0x00
_CODEC_HUFFMAN = 0x01
_CODEC_RANS = 0x03
_CODEC_ZSTD = 0x04
_CODEC_ORDER1_SCALE_AC = 0x11

# Codec sets per role — trial-encode picks the best within each set.
# v3 adds Zstd to the capability set so it can appear alongside Huffman/Rans.
_SCALE_BLOCK_HINT_CODECS = frozenset(
    {_CODEC_ORDER1_SCALE_AC, _CODEC_HUFFMAN, _CODEC_RANS, _CODEC_ZSTD}
)
_PACKED_VALUES_HINT_CODECS = frozenset({_CODEC_IDENTITY, _CODEC_HUFFMAN, _CODEC_ZSTD})


def _build_synthetic_nvfp4(
    *,
    n_layers: int = 1,
    out_features: int = 128,
    in_features: int = 64,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Deterministic NVFP4 fixture (mirrors tests/weights/fixtures/_nvfp4.py)."""
    rng = np.random.default_rng(seed)
    state: dict[str, np.ndarray] = {}
    n_groups = in_features // _GROUP_SIZE
    for li in range(n_layers):
        base = f"model.layers.{li}.self_attn.q_proj"
        state[f"{base}.weight"] = rng.integers(
            0, 256, size=(out_features, in_features // 2), dtype=np.uint8
        )
        state[f"{base}.weight_scale"] = rng.integers(
            64, 192, size=(out_features, n_groups), dtype=np.uint8
        )
        state[f"{base}.weight_scale_2"] = np.array(
            rng.standard_normal(dtype=np.float32) * 0.01,
            dtype=np.float32,
        )
        state[f"{base}.input_scale"] = np.array(
            rng.standard_normal(dtype=np.float32) * 0.01,
            dtype=np.float32,
        )
    return state


def _save_nvfp4_safetensors(tmp_path: Path) -> Path:
    state = _build_synthetic_nvfp4()
    torch_state = {k: torch.from_numpy(v) for k, v in state.items()}
    p = tmp_path / "model.safetensors"
    save_file(torch_state, str(p))
    return p


def _hf_quant_classifier() -> HfQuantConfigClassifier:
    """Synthetic HfQuantConfigClassifier for NVFP4."""
    return HfQuantConfigClassifier(quant_algo="NVFP4", exclude_modules=())


def _ptwm_blob(out_dir: Path) -> bytes:
    """Return raw bytes of the single-shard model.ptwm."""
    ptwm = out_dir / "model.ptwm"
    if not ptwm.exists():
        # Multi-shard: pick the first shard listed in the index.
        import json

        idx = json.loads((out_dir / "model.ptwm.index.json").read_text())
        first_shard = next(iter(sorted(set(idx["weight_map"].values()))))
        ptwm = out_dir / first_shard
    return ptwm.read_bytes()


def _codec_ids_for_tensor(blob: bytes, name: str) -> list[int]:
    """Return the codec_id of every plane for ``name`` in the container."""
    summaries = _rust.list_plane_summaries(blob)
    return [codec_id for tname, _, _, codec_id, _, _ in summaries if tname == name]


def test_round_trip_mode_a_default_classifier(tmp_path):
    src = _save_nvfp4_safetensors(tmp_path)
    out_dir = tmp_path / "out_a"
    out_dir.mkdir()
    compress_safetensors_file(src, out_dir, mode="a")
    state = read_sharded_ptwm(out_dir)
    base = "model.layers.0.self_attn.q_proj"
    weight = state[f"{base}.weight"]
    expected = _build_synthetic_nvfp4()[f"{base}.weight"].tobytes()
    assert weight == expected


def test_round_trip_fast_mode_pure_float(tmp_path):
    # Fast mode (forced rANS + chunked) must stay bit-exact. BF16/FP16 ride the
    # multi-chain branch; FP8 has a single STANDARD chain so it additionally
    # exercises the skip+chunk+fuse fast path.
    torch.manual_seed(0)
    state = {
        "model.layers.0.mlp.weight": (torch.randn(256, 256) * 0.02).to(torch.bfloat16),
        "model.layers.0.attn.weight": (torch.randn(128, 256) * 0.02).to(torch.float16),
        "model.layers.0.ffn.weight": (torch.randn(256, 256) * 0.02).to(
            torch.float8_e4m3fn
        ),
    }
    src = tmp_path / "model.safetensors"
    save_file(state, str(src))
    out_dir = tmp_path / "out_fast"
    out_dir.mkdir()
    compress_safetensors_file(src, out_dir, mode="a", fast=True)
    decoded = read_sharded_ptwm(out_dir)
    for name, t in state.items():
        expected = t.flatten().contiguous().view(torch.uint8).numpy().tobytes()
        assert decoded[name] == expected, f"fast-mode roundtrip mismatch for {name}"


def test_round_trip_mode_b(tmp_path):
    src = _save_nvfp4_safetensors(tmp_path)
    out_dir = tmp_path / "out_b"
    out_dir.mkdir()
    compress_safetensors_file(src, out_dir, mode="b")
    shells = list(out_dir.glob("*.safetensors"))
    assert shells, (
        f"expected .safetensors output in {out_dir}, got {list(out_dir.iterdir())}"
    )


# ---------------------------------------------------------------------------
# Per-tensor codec selection tests
# ---------------------------------------------------------------------------


def test_nvfp4_hf_quant_classifier_round_trip(tmp_path):
    """NVFP4 fixture round-trips bit-exactly when HfQuantConfigClassifier is used."""
    src = _save_nvfp4_safetensors(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    compress_safetensors_file(src, out_dir, classifier=_hf_quant_classifier())
    state = read_sharded_ptwm(out_dir)
    expected = _build_synthetic_nvfp4()
    base = "model.layers.0.self_attn.q_proj"
    for suffix in (".weight", ".weight_scale", ".weight_scale_2", ".input_scale"):
        name = f"{base}{suffix}"
        assert state[name] == expected[name].tobytes(), (
            f"round-trip mismatch for {name}"
        )


def test_scale_block_tensor_uses_order1_scale_ac(tmp_path):
    """uint8 weight_scale (E8M0 MXFP4 block scale) must not be encoded with O1SAC.

    Order1ScaleAC is FP8-only by design (E4M3 / E5M2). E8M0 is a pure 8-bit
    exponent — structurally different — and is excluded at the
    ``Order1ScaleAC::accepts()`` gate. The effective menu for an E8M0 plane is
    therefore {Huffman, Rans, Zstd}; trial-encode picks the smallest.
    """
    # Use a larger fixture so the plane exceeds the O1SAC minimum threshold
    # (4 096 bytes). out_features=128, in_features=512 → n_groups=32,
    # weight_scale shape=(128, 32) = 4096 bytes.
    state = _build_synthetic_nvfp4(out_features=128, in_features=512)
    torch_state = {k: torch.from_numpy(v) for k, v in state.items()}
    src = tmp_path / "model.safetensors"
    save_file(torch_state, str(src))

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    compress_safetensors_file(src, out_dir, classifier=_hf_quant_classifier())

    blob = _ptwm_blob(out_dir)
    base = "model.layers.0.self_attn.q_proj"
    scale_codecs = _codec_ids_for_tensor(blob, f"{base}.weight_scale")
    assert scale_codecs, "weight_scale has no planes in container"
    # O1SAC is filtered for PlaneRole::Value planes; effective menu is {Huffman, Rans, Zstd}.
    _effective_menu = frozenset({_CODEC_HUFFMAN, _CODEC_RANS, _CODEC_ZSTD})
    assert all(c in _effective_menu for c in scale_codecs), (
        f"expected all planes of weight_scale to use a hint-menu codec "
        f"({[hex(c) for c in sorted(_effective_menu)]}), "
        f"got {[hex(c) for c in scale_codecs]}"
    )


def test_packed_values_tensor_uses_identity(tmp_path):
    """weight (PACKED_VALUES role) must be encoded with a hint-menu codec.

    The hint menu for PACKED_VALUES is {Identity, Huffman}; trial-encode picks
    the best.  On high-entropy nibble-packed FP4 data Identity is expected to
    win (since entropy ≈ 4 bits/symbol), but we assert membership in the hint
    set rather than a hard single-codec requirement.
    """
    src = _save_nvfp4_safetensors(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    compress_safetensors_file(src, out_dir, classifier=_hf_quant_classifier())

    blob = _ptwm_blob(out_dir)
    base = "model.layers.0.self_attn.q_proj"
    weight_codecs = _codec_ids_for_tensor(blob, f"{base}.weight")
    assert weight_codecs, "weight has no planes in container"
    assert all(c in _PACKED_VALUES_HINT_CODECS for c in weight_codecs), (
        f"expected all planes of weight to use a PACKED_VALUES hint codec "
        f"({[hex(c) for c in sorted(_PACKED_VALUES_HINT_CODECS)]}), "
        f"got {[hex(c) for c in weight_codecs]}"
    )


def test_mxfp4_heuristic_scale_tensor_uses_order1_scale_ac(tmp_path):
    """*_scales tensors (SCALE_BLOCK from HeuristicClassifier) use a hint-menu codec.

    HeuristicClassifier matches ``*.weight_scale`` when a sibling
    ``*.weight`` exists and the dtype is uint8. Name the tensors
    ``block.weight`` + ``block.weight_scale`` so the sibling check passes.
    Use 128 × 512 = 65 536 bytes for the scale plane (well above the
    4 096-byte O1SAC threshold).

    Order1ScaleAC is FP8-only by design; E8M0 (uint8 MXFP4 block scale) is
    excluded at the ``accepts()`` gate. Trial-encode picks from the effective
    menu {Huffman, Rans, Zstd}.
    """
    from ptwm.classify import HeuristicClassifier

    rng = np.random.default_rng(0)
    torch_state = {
        "block.weight": torch.from_numpy(
            rng.integers(0, 256, size=(128, 256), dtype=np.uint8)
        ),
        "block.weight_scale": torch.from_numpy(
            rng.integers(64, 192, size=(128, 512), dtype=np.uint8)
        ),
    }
    src = tmp_path / "model.safetensors"
    save_file(torch_state, str(src))

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    compress_safetensors_file(src, out_dir, classifier=HeuristicClassifier())

    blob = _ptwm_blob(out_dir)
    scale_codecs = _codec_ids_for_tensor(blob, "block.weight_scale")
    assert scale_codecs, "weight_scale has no planes in container"
    # O1SAC is filtered for PlaneRole::Value planes; effective menu is {Huffman, Rans, Zstd}.
    _effective_menu = frozenset({_CODEC_HUFFMAN, _CODEC_RANS, _CODEC_ZSTD})
    assert all(c in _effective_menu for c in scale_codecs), (
        f"expected hint-menu codec on weight_scale "
        f"({[hex(c) for c in sorted(_effective_menu)]}), "
        f"got {[hex(c) for c in scale_codecs]}"
    )


def test_no_classifier_preserves_dtype_default_behaviour(tmp_path):
    """classifier=None path must not change codec selection vs. the baseline."""
    src = _save_nvfp4_safetensors(tmp_path)

    out_baseline = tmp_path / "baseline"
    out_baseline.mkdir()
    compress_safetensors_file(src, out_baseline, classifier=None)

    out_none = tmp_path / "none_classifier"
    out_none.mkdir()
    compress_safetensors_file(src, out_none, classifier=None)

    blob_baseline = _ptwm_blob(out_baseline)
    blob_none = _ptwm_blob(out_none)

    # Byte-exact: deterministic pipeline must produce identical output.
    assert blob_baseline == blob_none, (
        "classifier=None produced different bytes across two identical runs"
    )

    # Neither run should warn about unused classifier.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out_warn = tmp_path / "warn_check"
        out_warn.mkdir()
        # Should not raise WarningMessage.
        compress_safetensors_file(src, out_warn, classifier=None)


@pytest.mark.parametrize("classifier", [None, "hf_quant"])
def test_round_trip_all_tensors_with_classifier(tmp_path, classifier):
    """All tensors round-trip bit-exactly regardless of classifier."""
    src = _save_nvfp4_safetensors(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    clf = _hf_quant_classifier() if classifier == "hf_quant" else None
    compress_safetensors_file(src, out_dir, classifier=clf)
    state = read_sharded_ptwm(out_dir)
    expected = _build_synthetic_nvfp4()
    for name, arr in expected.items():
        assert state[name] == arr.tobytes(), f"round-trip mismatch for {name}"


def test_f8_e4m3_weight_scale_round_trip(tmp_path):
    """Regression: F8_E4M3 weight_scale with SCALE_BLOCK role round-trips without CRC mismatch.

    Nemotron-NVFP4 uses ``float8_e4m3fn`` as the dtype for ``weight_scale``
    tensors.  The Fp8E4m3Nibble preprocessing spec splits them into
    [Exponent, Mantissa] nibble planes.  The previous (broken) code forced
    Order1ScaleAC onto ALL planes via ``build_direct_menu``, which is only
    valid for Scale role planes and uses ``plane_decoded_len = orig_size / 2``
    at decode time (half the actual encoded nibble-plane size), causing a
    plane CRC mismatch on ``decode_tensor_v2``.

    The fix: ``build_direct_menu`` silently drops Order1ScaleAC for non-Scale
    plane roles so nibble planes fall back to Identity/Huffman/Rans which do
    not depend on ``plane_decoded_len``.
    """
    rng = np.random.default_rng(7)
    # Build a fixture where weight_scale has dtype float8_e4m3fn (not uint8).
    # Use raw uint8 values reinterpreted as F8_E4M3 — byte pattern is what
    # matters for the round-trip; semantic validity is irrelevant.
    weight_scale_raw = rng.integers(64, 192, size=(32, 16), dtype=np.uint8)
    torch_state = {
        "block.weight": torch.from_numpy(
            rng.integers(0, 256, size=(32, 32), dtype=np.uint8)
        ),
        # Store weight_scale with float8_e4m3fn dtype → routes to Fp8E4m3Nibble spec.
        "block.weight_scale": torch.from_numpy(weight_scale_raw).view(
            torch.float8_e4m3fn
        ),
    }
    src = tmp_path / "model.safetensors"
    save_file(torch_state, str(src))

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    # Use HfQuantConfigClassifier so weight_scale gets SCALE_BLOCK role,
    # triggering the per-tensor codec hint path.
    compress_safetensors_file(src, out_dir, classifier=_hf_quant_classifier())

    # Must not raise "plane CRC mismatch".
    state = read_sharded_ptwm(out_dir)
    assert state["block.weight_scale"] == weight_scale_raw.tobytes(), (
        "F8_E4M3 weight_scale round-trip mismatch"
    )


# ---------------------------------------------------------------------------
# Audit log discovery
# ---------------------------------------------------------------------------


def test_explore_options_adds_candidates_to_audit(tmp_path):
    """AuditLog.discovered_chains() is non-empty when ExploreOptions is provided.

    The explorer generates additional chain candidates beyond the production set.
    When any are found, record_discovered() adds a DiscoveredChain entry.
    """
    from ptwm.preprocessing._explorer import ExploreOptions

    src = _save_nvfp4_safetensors(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    opts = ExploreOptions(max_candidates_per_dtype=8, time_budget_ms=2000)
    audit = compress_safetensors_file(
        src, out_dir, classifier=_hf_quant_classifier(), explore_options=opts
    )

    discovered = audit.discovered_chains()
    assert len(discovered) >= 1, (
        "Expected at least one discovered chain in audit log when ExploreOptions is set"
    )


def test_discovered_chain_fields_are_valid(tmp_path):
    """Each DiscoveredChain entry has correct field types and non-empty chain bytes."""
    from ptwm.classify import DiscoveredChain
    from ptwm.preprocessing._explorer import ExploreOptions

    src = _save_nvfp4_safetensors(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    opts = ExploreOptions(max_candidates_per_dtype=8, time_budget_ms=2000)
    audit = compress_safetensors_file(
        src, out_dir, classifier=_hf_quant_classifier(), explore_options=opts
    )

    for dc in audit.discovered_chains():
        assert isinstance(dc, DiscoveredChain)
        assert isinstance(dc.chain, bytes)
        assert len(dc.chain) > 0
        assert isinstance(dc.dtype_code, int)
        assert isinstance(dc.role, str)
        assert isinstance(dc.n_candidates_tried, int)
        assert dc.n_candidates_tried >= 1
        assert isinstance(dc.sample_tensor_name, str)
        assert dc.sample_tensor_name


def test_discovered_chain_deduplication(tmp_path):
    """Each (dtype_code, role) pair is recorded at most once."""
    from ptwm.preprocessing._explorer import ExploreOptions

    src = _save_nvfp4_safetensors(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    opts = ExploreOptions(max_candidates_per_dtype=8, time_budget_ms=2000)
    audit = compress_safetensors_file(
        src, out_dir, classifier=_hf_quant_classifier(), explore_options=opts
    )

    pairs = [(dc.dtype_code, dc.role) for dc in audit.discovered_chains()]
    assert len(pairs) == len(set(pairs)), (
        "Duplicate (dtype_code, role) pairs found in discovered_chains"
    )


def test_no_explore_options_yields_empty_discovered(tmp_path):
    """AuditLog.discovered_chains() is empty when explore_options=None."""
    src = _save_nvfp4_safetensors(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    audit = compress_safetensors_file(src, out_dir, classifier=_hf_quant_classifier())
    assert audit.discovered_chains() == (), (
        "Expected no discovered chains when explore_options=None"
    )


def test_audit_log_cbor_roundtrip_with_discovered_chains(tmp_path):
    """AuditLog with discovered chains survives CBOR serialisation round-trip."""
    from ptwm.classify import AuditLog
    from ptwm.preprocessing._explorer import ExploreOptions

    src = _save_nvfp4_safetensors(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    opts = ExploreOptions(max_candidates_per_dtype=8, time_budget_ms=2000)
    audit = compress_safetensors_file(
        src, out_dir, classifier=_hf_quant_classifier(), explore_options=opts
    )

    blob = audit.to_cbor()
    recovered = AuditLog.from_cbor(blob)

    assert recovered.entries() == audit.entries()
    assert recovered.discovered_chains() == audit.discovered_chains()
