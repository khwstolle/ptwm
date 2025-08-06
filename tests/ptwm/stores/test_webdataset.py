"""WebDataset store: shard layout, per-shard registry, index, streaming."""

from __future__ import annotations

import json
import tarfile

from ptwm import _rust
from ptwm.sharding import compress_ptwm_blob
from ptwm.stores import WebDatasetWeightStore, detect_format, write_webdataset
from ptwm.stores._webdataset import INDEX_NAME, SHARDMETA_MEMBER, WDS_FORMAT


def _reference_raw(store_tensors):
    blob = compress_ptwm_blob(store_tensors, method_hint=3)
    return {name: bytes(_rust.decode_tensor(blob, name)) for name, *_ in store_tensors}


def test_shard_and_index_layout(tmp_path, store_tensors):
    wd = tmp_path / "m.wds"
    write_webdataset(wd, store_tensors, max_shard_size=2 * 1024**3)

    shards = sorted(wd.glob("weights-*-of-*.tar"))
    assert shards, "no shards written"
    # No nested .ptwm container appears anywhere in a shard.
    for shard in shards:
        with tarfile.open(shard, "r") as tar:
            members = tar.getnames()
            assert members[0] == SHARDMETA_MEMBER
            for info in tar.getmembers():
                if info.name.endswith(".bin"):
                    f = tar.extractfile(info)
                    assert f is not None
                    assert f.read(5) != b"\x89PTWM"

    index = json.loads((wd / INDEX_NAME).read_text())
    assert index["ptwm_format"] == WDS_FORMAT
    assert set(index["weight_map"]) == {n for n, *_ in store_tensors}
    assert index["total_size"] > 0


def test_index_is_jcs(tmp_path, store_tensors):
    wd = tmp_path / "m.wds"
    write_webdataset(wd, store_tensors)
    from ptwm.stores import jcs_canonicalize

    raw = (wd / INDEX_NAME).read_text()
    assert jcs_canonicalize(json.loads(raw)) == raw


def test_multi_shard_no_straddle(tmp_path, store_tensors):
    ref = _reference_raw(store_tensors)
    wd = tmp_path / "m.wds"
    # A tiny threshold forces many shards.
    write_webdataset(wd, store_tensors, max_shard_size=256)
    shards = sorted(wd.glob("weights-*-of-*.tar"))
    assert len(shards) > 1, "expected multiple shards"

    index = json.loads((wd / "index.json").read_text())
    # Every tensor's members live entirely within its assigned shard.
    by_shard: dict[str, set[str]] = {}
    for loc in index["weight_map"].values():
        by_shard.setdefault(loc["shard"], set()).add(loc["key"])
    for shard in shards:
        with tarfile.open(shard, "r") as tar:
            names = set(tar.getnames())
        keys_here = by_shard.get(shard.name, set())
        for key in keys_here:
            assert f"{key}.json" in names

    store = WebDatasetWeightStore.open(wd)
    for name in ref:
        assert store.get_bytes(name) == ref[name]


def test_single_shard_streaming(tmp_path, store_tensors):
    ref = _reference_raw(store_tensors)
    wd = tmp_path / "m.wds"
    write_webdataset(wd, store_tensors, max_shard_size=2 * 1024**3)
    shard = next(iter(sorted(wd.glob("weights-*-of-*.tar"))))

    # detect a bare shard as webdataset, open it stream-only.
    assert detect_format(shard) == "webdataset"
    store = WebDatasetWeightStore.open(shard)
    streamed = dict(store.stream_tensors())
    assert set(streamed) == set(ref)


def test_stream_subset(tmp_path, store_tensors):
    wd = tmp_path / "m.wds"
    write_webdataset(wd, store_tensors)
    store = WebDatasetWeightStore.open(wd)
    wanted = [store_tensors[0][0], store_tensors[1][0]]
    got = dict(store.stream_tensors(wanted))
    assert set(got) == set(wanted)
