"""Stores and the transcoder over compact (``\\x89PTWC``) containers.

The LMDB and WebDataset stores rebuild tensors through ``implode_ptwm``. A
compact container explodes to a ``"ptwm-manifest-c"`` registry manifest, and
every store path must rebuild tensors from it as it does from a legacy one.
"""

from __future__ import annotations

import json
import tarfile

import pytest
import torch
from ptwm import _rust
from ptwm._exceptions import Error
from ptwm.sharding import compress_ptwm_blob, write_ptwm_shard
from ptwm.stores import (
    LmdbWeightStore,
    WebDatasetWeightStore,
    explode_model,
    jcs_canonicalize,
    raw_from_members,
    write_lmdb,
    write_webdataset,
)
from ptwm.stores._common import (
    manifest_container_layout,
    shared_member_keys,
    tensor_member_keys,
)
from ptwm.stores._lmdb import KEY_REG
from ptwm.stores._webdataset import INDEX_NAME, SHARDMETA_MEMBER

MAGIC = {"legacy": b"\x89PTWM", "compact": b"\x89PTWC"}
LAYOUTS = ("legacy", "compact")


def _reference_raw(store_tensors, layout: str) -> dict[str, bytes]:
    blob = compress_ptwm_blob(store_tensors, method_hint=3, container_layout=layout)
    return {name: bytes(_rust.decode_tensor(blob, name)) for name, *_ in store_tensors}


@pytest.mark.parametrize("layout", LAYOUTS)
def test_compress_ptwm_blob_honours_the_layout(tmp_path, store_tensors, layout):
    blob = compress_ptwm_blob(store_tensors, method_hint=3, container_layout=layout)
    assert blob[:9].startswith(MAGIC[layout])
    out = tmp_path / "m.ptwm"
    write_ptwm_shard(out, store_tensors, container_layout=layout)
    assert out.read_bytes()[:5] == MAGIC[layout]
    # Both layouts decode to the same bytes.
    assert _reference_raw(store_tensors, layout) == _reference_raw(
        store_tensors, "legacy"
    )


def test_unknown_container_layout_is_rejected(store_tensors):
    with pytest.raises(Error, match="container_layout"):
        compress_ptwm_blob(store_tensors, container_layout="bogus")


def test_compact_manifest_form_and_identity(store_tensors):
    blob = compress_ptwm_blob(store_tensors, method_hint=3, container_layout="compact")
    registry_json, tensors, members = _rust.explode_ptwm(blob)
    reg = json.loads(registry_json)
    assert reg["ptwm_format"] == "ptwm-manifest-c"
    assert reg["manifest_version"] == 2
    assert reg["container_flags"] == {"embedded_wasm": False, "layout": "compact"}
    assert "registry_hash_blake3" not in reg
    assert set(reg["registry"]) == {"chains", "extension_table", "shared_state"}
    for entry in reg["registry"]["extension_table"]:
        assert len(entry) == 1
        assert next(iter(entry)) in {"op", "codec", "named", "full_b64"}
    # The Rust JCS output and the Python canonicalizer agree.
    assert jcs_canonicalize(reg) == registry_json
    # Tensors come in body (write) order.
    assert [name for (name, _, _) in tensors] == [n for n, *_ in store_tensors]

    rebuilt = _rust.implode_ptwm(registry_json, [tj for (_, _, tj) in tensors], members)
    assert rebuilt == blob

    model = explode_model(blob)
    assert model.container_layout == "compact"
    assert manifest_container_layout(registry_json) == "compact"
    legacy = compress_ptwm_blob(store_tensors, method_hint=3)
    assert explode_model(legacy).container_layout == "legacy"


@pytest.mark.parametrize("method_hint", [1, 2, 3, 4, 5])
def test_compact_identity_per_method(store_tensors, method_hint):
    blob = compress_ptwm_blob(
        store_tensors, method_hint=method_hint, container_layout="compact"
    )
    registry_json, tensors, members = _rust.explode_ptwm(blob)
    rebuilt = _rust.implode_ptwm(registry_json, [tj for (_, _, tj) in tensors], members)
    assert rebuilt == blob


@pytest.mark.parametrize("layout", LAYOUTS)
def test_raw_from_members_rebuilds_one_tensor(store_tensors, layout):
    blob = compress_ptwm_blob(store_tensors, method_hint=3, container_layout=layout)
    want = _reference_raw(store_tensors, layout)
    model = explode_model(blob)
    shared = shared_member_keys(model.members)
    for name in model.names():
        members = tensor_member_keys(model.members, model.key(name))
        members.update(shared)
        single = _rust.implode_ptwm(
            model.registry_json, [model.tensor_json(name)], members
        )
        assert single[:5] == MAGIC[layout]
        raw = raw_from_members(model.registry_json, model.tensor_json(name), members)
        assert raw == want[name]


def test_implode_rejects_a_compact_manifest_with_another_layout(store_tensors):
    blob = compress_ptwm_blob(store_tensors, container_layout="compact")
    registry_json, tensors, members = _rust.explode_ptwm(blob)
    reg = json.loads(registry_json)
    reg["container_flags"]["layout"] = "legacy"
    with pytest.raises(Error, match="container_flags.layout"):
        _rust.implode_ptwm(json.dumps(reg), [tj for (_, _, tj) in tensors], members)


# ── LMDB ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("store_layout", ["blob", "exploded"])
def test_lmdb_compact_round_trip(tmp_path, store_tensors, torch_tensors, store_layout):
    path = tmp_path / "m.lmdb"
    write_lmdb(path, store_tensors, layout=store_layout, container_layout="compact")
    want = _reference_raw(store_tensors, "compact")
    store = LmdbWeightStore.open(path)
    try:
        assert set(store.names()) == {n for n, *_ in store_tensors}
        if store_layout == "blob":
            for name, *_ in store_tensors:
                assert store._get(b"t/" + name.encode())[:5] == MAGIC["compact"]
        else:
            reg = json.loads(store._get(KEY_REG))
            assert reg["ptwm_format"] == "ptwm-manifest-c"
        for name, *_ in store_tensors:
            assert store.get_bytes(name) == want[name]
        for name, expected in torch_tensors.items():
            got = store.get_tensor(name)
            assert got.dtype == expected.dtype
            assert torch.equal(got.view(torch.uint8), expected.view(torch.uint8))
    finally:
        store.close()


def test_lmdb_put_tensor_in_either_layout(tmp_path, raw_of):
    tensors = [("w", raw_of(torch.zeros(8).float()), torch.float32, (8,))]
    path = tmp_path / "m.lmdb"
    write_lmdb(path, tensors, layout="blob", container_layout="compact")

    new = torch.arange(8, dtype=torch.float32)
    step = new + 1
    with LmdbWeightStore.open(path, readonly=False) as store:
        store.put_tensor("w", raw_of(new), torch.float32, (8,))
        assert store._get(b"t/w")[:5] == MAGIC["legacy"]
        store.put_tensor(
            "v", raw_of(new), torch.float32, (8,), container_layout="compact"
        )
        assert store._get(b"t/v")[:5] == MAGIC["compact"]
        store.put_tensor(
            "d",
            raw_of(step),
            torch.float32,
            (8,),
            delta_reference="v",
            container_layout="compact",
        )
        assert store._get(b"t/d")[:5] == MAGIC["compact"]
    with LmdbWeightStore.open(path) as store:
        assert torch.equal(store.get_tensor("w"), new)
        assert torch.equal(store.get_tensor("v"), new)
        assert torch.equal(store.get_tensor("d"), step)


# ── WebDataset ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("max_shard_size", [2 * 1024**3, 1])
def test_webdataset_compact_round_trip(tmp_path, store_tensors, max_shard_size):
    wd = tmp_path / "m.wds"
    write_webdataset(
        wd, store_tensors, max_shard_size=max_shard_size, container_layout="compact"
    )
    want = _reference_raw(store_tensors, "compact")

    index = json.loads((wd / INDEX_NAME).read_text())
    assert index["manifest_version"] == 1  # describes the store
    assert index["registry"]["ptwm_format"] == "ptwm-manifest-c"
    shards = sorted(wd.glob("weights-*-of-*.tar"))
    assert (len(shards) > 1) == (max_shard_size == 1)
    for shard in shards:
        with tarfile.open(shard, "r") as tar:
            f = tar.extractfile(SHARDMETA_MEMBER)
            assert f is not None
            meta = json.loads(f.read())
            assert meta["container_flags"]["layout"] == "compact"

    store = WebDatasetWeightStore.open(wd)
    for name, *_ in store_tensors:
        assert store.get_bytes(name) == want[name]
    streamed = {name for name, _ in store.stream_tensors()}
    assert streamed == {n for n, *_ in store_tensors}

    # One shard opened on its own streams with its own registry copy.
    single = WebDatasetWeightStore.open(shards[0])
    for name, tensor in single.stream_tensors():
        assert tensor.numel() > 0
        assert name in want
