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
from ptwm import ContainerLayout, _rust
from ptwm._exceptions import Error
from ptwm._magic import MAGIC_LEN, MAGIC_PTWC, MAGIC_PTWM, container_kind
from ptwm.sharding import compress_ptwm_blob, write_ptwm_shard
from ptwm.sharding._writer import _default_chain_bytes
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
from ptwm.stores._lmdb import KEY_FMT, KEY_REG
from ptwm.stores._webdataset import INDEX_NAME, SHARDMETA_MEMBER
from ptwm.utils import DType

MAGIC = {"legacy": MAGIC_PTWM, "compact": MAGIC_PTWC}
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
    assert out.read_bytes()[:MAGIC_LEN] == MAGIC[layout]
    # Both layouts decode to the same bytes.
    assert _reference_raw(store_tensors, layout) == _reference_raw(
        store_tensors, "legacy"
    )


def test_unknown_container_layout_is_rejected(store_tensors, tmp_path):
    # Every public writer raises ValueError, before touching the filesystem.
    with pytest.raises(ValueError, match="container_layout"):
        compress_ptwm_blob(store_tensors, container_layout="bogus")
    out = tmp_path / "x.ptwm"
    with pytest.raises(ValueError, match="container_layout"):
        write_ptwm_shard(out, store_tensors, container_layout="bogus")
    assert not out.exists()
    out_dir = tmp_path / "wds"
    with pytest.raises(ValueError, match="container_layout"):
        write_webdataset(out_dir, store_tensors, container_layout="bogus")
    assert not out_dir.exists()


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
    legacy = compress_ptwm_blob(store_tensors, method_hint=3, container_layout="legacy")
    assert explode_model(legacy).container_layout == "legacy"
    # The default layout is compact.
    default = compress_ptwm_blob(store_tensors, method_hint=3)
    assert explode_model(default).container_layout == "compact"


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
        assert single[:MAGIC_LEN] == MAGIC[layout]
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
                assert store._get(b"t/" + name.encode())[:MAGIC_LEN] == MAGIC["compact"]
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
        store.put_tensor(
            "w", raw_of(new), torch.float32, (8,), container_layout="legacy"
        )
        assert store._get(b"t/w")[:MAGIC_LEN] == MAGIC["legacy"]
        store.put_tensor(
            "v", raw_of(new), torch.float32, (8,), container_layout="compact"
        )
        assert store._get(b"t/v")[:MAGIC_LEN] == MAGIC["compact"]
        store.put_tensor(
            "d",
            raw_of(step),
            torch.float32,
            (8,),
            delta_reference="v",
            container_layout="compact",
        )
        assert store._get(b"t/d")[:MAGIC_LEN] == MAGIC["compact"]
    with LmdbWeightStore.open(path) as store:
        assert torch.equal(store.get_tensor("w"), new)
        assert torch.equal(store.get_tensor("v"), new)
        assert torch.equal(store.get_tensor("d"), step)


@pytest.mark.parametrize("layout", ["legacy", "compact"])
def test_lmdb_put_tensor_inherits_store_layout(tmp_path, raw_of, layout):
    # put_tensor without container_layout uses the layout the store was
    # written with, so a compact store does not silently gain legacy blobs.
    tensors = [("w", raw_of(torch.zeros(8).float()), torch.float32, (8,))]
    path = tmp_path / "m.lmdb"
    write_lmdb(path, tensors, layout="blob", container_layout=layout)

    new = torch.arange(8, dtype=torch.float32)
    with LmdbWeightStore.open(path, readonly=False) as store:
        fmt = json.loads(store._get(KEY_FMT))
        if layout == "legacy":
            # The legacy fmt record is unchanged from earlier releases.
            assert "container_layout" not in fmt
        else:
            assert fmt["container_layout"] == "compact"
        store.put_tensor("w", raw_of(new), torch.float32, (8,))
        store.put_tensor("d", raw_of(new + 1), torch.float32, (8,), delta_reference="w")
        assert store._get(b"t/w")[:MAGIC_LEN] == MAGIC[layout]
        assert store._get(b"t/d")[:MAGIC_LEN] == MAGIC[layout]
    # The layout survives a reopen.
    with LmdbWeightStore.open(path, readonly=False) as store:
        store.put_tensor("v", raw_of(new), torch.float32, (8,))
        assert store._get(b"t/v")[:MAGIC_LEN] == MAGIC[layout]
    with LmdbWeightStore.open(path) as store:
        assert torch.equal(store.get_tensor("w"), new)
        assert torch.equal(store.get_tensor("v"), new)
        assert torch.equal(store.get_tensor("d"), new + 1)


def test_compress_model_unknown_layout_raises_ptwm_error():
    # The public wrapper translates the binding's ValueError to ptwm.Error.
    with pytest.raises(Error, match="container_layout"):
        _rust.compress_model([], [], True, True, container_layout="bogus")
    with pytest.raises(ValueError, match="container_layout"):
        _rust._ext.compress_model([], [], True, True, container_layout="bogus")


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


def test_stores_accept_the_layout_enum(tmp_path, store_tensors, raw_of):
    path = tmp_path / "m.lmdb"
    write_lmdb(
        path, store_tensors, layout="blob", container_layout=ContainerLayout.COMPACT
    )
    with LmdbWeightStore.open(path, readonly=False) as store:
        fmt = json.loads(store._get(KEY_FMT))
        assert fmt["container_layout"] == "compact"
        for name, *_ in store_tensors:
            assert container_kind(store._get(b"t/" + name.encode())) == "ptwc"
        store.put_tensor(
            "x",
            raw_of(torch.ones(4)),
            torch.float32,
            (4,),
            container_layout=ContainerLayout.LEGACY,
        )
        assert container_kind(store._get(b"t/x")) == "ptwm"

    wd = tmp_path / "m.wds"
    write_webdataset(wd, store_tensors, container_layout=ContainerLayout.COMPACT)
    index = json.loads((wd / INDEX_NAME).read_text())
    assert index["registry"]["ptwm_format"] == "ptwm-manifest-c"


# ── Defaults ──────────────────────────────────────────────────────────────────


def test_new_writes_default_to_the_compact_layout(tmp_path, store_tensors):
    """Every multi-tensor writer defaults to ``\\x89PTWC``; ``"legacy"`` is
    still available on request (see the layout-parametrized tests above)."""
    assert container_kind(compress_ptwm_blob(store_tensors)) == "ptwc"
    out = tmp_path / "m.ptwm"
    write_ptwm_shard(out, store_tensors)
    assert container_kind(out.read_bytes()) == "ptwc"

    # The binding wrapper defaults to the compact layout too.
    name, raw, dtype, shape = store_tensors[0]
    dt = DType.from_dtype(dtype)
    records = [(name, int(dt.code), 0, raw, list(shape), dt.dtype_str, None)]
    chains = [_default_chain_bytes(int(dt.code), shape, len(raw))]
    blob = _rust.compress_model(records, chains, True, True)
    assert container_kind(blob) == "ptwc"

    path = tmp_path / "m.lmdb"
    write_lmdb(path, store_tensors, layout="blob")
    with LmdbWeightStore.open(path) as store:
        assert json.loads(store._get(KEY_FMT))["container_layout"] == "compact"

    wd = tmp_path / "m.wds"
    write_webdataset(wd, store_tensors)
    index = json.loads((wd / INDEX_NAME).read_text())
    assert index["registry"]["ptwm_format"] == "ptwm-manifest-c"
