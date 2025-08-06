"""Conformance tests for the container ⇄ manifest transcoder.

``implode(explode(blob)) == blob`` must hold byte-for-byte for every container
the compression engine produces, across the production dtypes/chains/codecs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from ptwm import _rust
from ptwm.integrations import compress_safetensors_file
from ptwm.sharding import compress_ptwm_blob
from ptwm.stores import jcs_canonicalize
from safetensors.torch import save_file


def _ptwm_blobs_under(root: Path) -> list[bytes]:
    """Collect every native ``.ptwm`` container blob under ``root``."""
    blobs: list[bytes] = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            data = p.read_bytes()
            if data[:5] == b"\x89PTWM":
                blobs.append(data)
    return blobs


def _assert_identity(blob: bytes) -> None:
    registry_json, tensors, members = _rust.explode_ptwm(blob)
    rebuilt = _rust.implode_ptwm(
        registry_json, [tj for (_, _, tj) in tensors], dict(members)
    )
    assert rebuilt == blob, "implode(explode(blob)) != blob"


def test_identity_representative(store_tensors):
    blob = compress_ptwm_blob(store_tensors, method_hint=3)
    _assert_identity(blob)


def test_identity_per_method(store_tensors):
    # bf16/fp16/fp32/fp8 across each dominant-codec hint the engine emits.
    for method_hint in (1, 2, 3, 4, 5):
        blob = compress_ptwm_blob(store_tensors, method_hint=method_hint)
        _assert_identity(blob)


def _build_mxfp4_state() -> dict[str, np.ndarray]:
    """A small gpt-oss-style MXFP4 state dict (blocks + E8M0 scales + bias)."""
    rng = np.random.default_rng(11)
    base = "model.layers.0.mlp.experts"
    state: dict[str, np.ndarray] = {}
    for proj in ("down_proj", "gate_up_proj"):
        state[f"{base}.{proj}_blocks"] = rng.integers(
            0, 256, size=(2, 8, 3, 16), dtype=np.uint8
        )
        state[f"{base}.{proj}_scales"] = rng.integers(
            115, 130, size=(2, 8, 3), dtype=np.uint8
        )
        bias = rng.standard_normal((2, 8), dtype=np.float32)
        state[f"{base}.{proj}_bias"] = bias.view(np.uint16)[..., 1::2].copy()
    return state


def test_identity_microscale(tmp_path):
    # Run the real engine over an MXFP4 (microscale/scale) state dict, then
    # assert byte-exact transcode of every produced container.
    state = _build_mxfp4_state()
    src = tmp_path / "model.safetensors"
    save_file({k: torch.from_numpy(v) for k, v in state.items()}, str(src))
    out = tmp_path / "out"
    out.mkdir()
    compress_safetensors_file(src, out, mode="a")

    blobs = _ptwm_blobs_under(out)
    assert blobs, "no .ptwm container produced"
    for blob in blobs:
        _assert_identity(blob)


def test_manifests_are_canonical_jcs(store_tensors):
    # Rust emits JCS; re-canonicalizing via the Python serializer is a no-op.
    blob = compress_ptwm_blob(store_tensors, method_hint=3)
    registry_json, tensors, _members = _rust.explode_ptwm(blob)
    assert jcs_canonicalize(json.loads(registry_json)) == registry_json
    for _name, _key, tjson in tensors:
        assert jcs_canonicalize(json.loads(tjson)) == tjson


def test_registry_hash_is_hex(store_tensors):
    blob = compress_ptwm_blob(store_tensors, method_hint=3)
    registry_json, _tensors, _members = _rust.explode_ptwm(blob)
    reg = json.loads(registry_json)
    assert reg["ptwm_format"] == "ptwm-manifest"
    h = reg["registry_hash_blake3"]
    assert isinstance(h, str)
    bytes.fromhex(h)  # valid hex


def test_every_member_is_referenced(store_tensors):
    blob = compress_ptwm_blob(store_tensors, method_hint=3)
    registry_json, tensors, members = _rust.explode_ptwm(blob)
    referenced: set[str] = set()
    reg = json.loads(registry_json)
    for entry in reg["registry"]["shared_state"]:
        referenced.add(entry["payload"])
    for _name, _key, tjson in tensors:
        meta = json.loads(tjson)
        for plane in meta["planes"]:
            referenced.add(plane["payload"])
            if plane.get("state_payload"):
                referenced.add(plane["state_payload"])
    assert set(members) == referenced
