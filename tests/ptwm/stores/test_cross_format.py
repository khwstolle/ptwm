"""Cross-validate tensor recovery across PTWM, WebDataset, and LMDB."""

from __future__ import annotations

import pytest
import torch
from ptwm import _rust
from ptwm.sharding import compress_ptwm_blob
from ptwm.stores import (
    LmdbWeightStore,
    WebDatasetWeightStore,
    open_store,
    write_lmdb,
    write_webdataset,
)


def _reference_raw(store_tensors) -> dict[str, bytes]:
    blob = compress_ptwm_blob(store_tensors, method_hint=3)
    return {name: bytes(_rust.decode_tensor(blob, name)) for name, *_ in store_tensors}


def _all_stores(tmp_path, store_tensors):
    """Build all three formats (LMDB blob+exploded, WebDataset) and open them."""
    lb = tmp_path / "model.lmdb"
    write_lmdb(lb, store_tensors, layout="blob")
    le = tmp_path / "model_exploded.lmdb"
    write_lmdb(le, store_tensors, layout="exploded")
    wd = tmp_path / "model.wds"
    write_webdataset(wd, store_tensors, max_shard_size=2 * 1024**3)
    return {
        "lmdb_blob": LmdbWeightStore.open(lb),
        "lmdb_exploded": LmdbWeightStore.open(le),
        "webdataset": WebDatasetWeightStore.open(wd),
    }


def test_byte_exact_recovery_all_formats(tmp_path, store_tensors):
    ref = _reference_raw(store_tensors)
    stores = _all_stores(tmp_path, store_tensors)
    for label, store in stores.items():
        assert sorted(store.names()) == sorted(ref), label
        for name in ref:
            assert store.get_bytes(name) == ref[name], f"{label}:{name}"


def test_value_exact_recovery_all_formats(tmp_path, store_tensors, torch_tensors):
    stores = _all_stores(tmp_path, store_tensors)
    for label, store in stores.items():
        for name, expected in torch_tensors.items():
            got = store.get_tensor(name)
            assert got.dtype == expected.dtype, f"{label}:{name}"
            assert got.shape == expected.shape, f"{label}:{name}"
            assert torch.equal(got, expected), f"{label}:{name}"


def test_streaming_matches_random(tmp_path, store_tensors):
    ref = _reference_raw(store_tensors)
    wd = tmp_path / "model.wds"
    write_webdataset(wd, store_tensors, max_shard_size=2 * 1024**3)
    store = WebDatasetWeightStore.open(wd)
    streamed = dict(store.stream_tensors())
    assert sorted(streamed) == sorted(ref)


def test_open_store_autodetect(tmp_path, store_tensors):
    lb = tmp_path / "a.lmdb"
    write_lmdb(lb, store_tensors)
    wd = tmp_path / "a.wds"
    write_webdataset(wd, store_tensors)
    ptwm_path = tmp_path / "a.ptwm"
    ptwm_path.write_bytes(compress_ptwm_blob(store_tensors))

    assert isinstance(open_store(lb), LmdbWeightStore)
    assert isinstance(open_store(wd), WebDatasetWeightStore)
    # native container resolves to a TensorIndex
    from ptwm.random_access import TensorIndex

    assert isinstance(open_store(ptwm_path), TensorIndex)


def test_open_store_rejects_unknown(tmp_path):
    junk = tmp_path / "junk.bin"
    junk.write_bytes(b"not a ptwm thing")
    with pytest.raises(ValueError, match="could not detect"):
        open_store(junk)
