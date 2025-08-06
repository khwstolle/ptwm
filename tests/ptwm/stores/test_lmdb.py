"""LMDB store: layouts, detection, atomic update, and delta-against-reference."""

from __future__ import annotations

import json

import pytest
import torch
from ptwm import _rust
from ptwm.stores import LmdbWeightStore, detect_format, write_lmdb
from ptwm.stores._lmdb import KEY_FMT, KEY_IDX, KEY_REG, LMDB_FORMAT


def test_blob_layout_keys(tmp_path, store_tensors):
    path = tmp_path / "m.lmdb"
    write_lmdb(path, store_tensors, layout="blob")
    assert detect_format(path) == "lmdb"
    store = LmdbWeightStore.open(path)
    assert store.layout == "blob"

    fmt = json.loads(store._get(KEY_FMT))
    assert fmt["ptwm_format"] == LMDB_FORMAT
    assert fmt["layout"] == "blob"
    assert store._get(KEY_REG) is None  # blob layout has no shared registry key
    # Every tensor is a self-contained single-tensor container.
    for name, *_ in store_tensors:
        blob = store._get(b"t/" + name.encode())
        assert blob[:5] == b"\x89PTWM"


def test_exploded_layout_keys(tmp_path, store_tensors):
    path = tmp_path / "m.lmdb"
    write_lmdb(path, store_tensors, layout="exploded")
    store = LmdbWeightStore.open(path)
    assert store.layout == "exploded"

    assert json.loads(store._get(KEY_FMT))["layout"] == "exploded"
    assert store._get(KEY_REG) is not None
    idx = json.loads(store._get(KEY_IDX))
    assert set(idx["weight_map"]) == {n for n, *_ in store_tensors}
    # Per-plane payloads exist as separate keys.
    first = store_tensors[0][0].encode()
    assert store._get(b"t/" + first + b"/meta") is not None
    assert store._get(b"t/" + first + b"/p0") is not None


def test_invalid_layout_rejected(tmp_path, store_tensors):
    with pytest.raises(ValueError, match="unknown LMDB layout"):
        write_lmdb(tmp_path / "x.lmdb", store_tensors, layout="bogus")


def test_atomic_update(tmp_path, raw_of):
    tensors = [("w", raw_of(torch.zeros(8).float()), torch.float32, (8,))]
    path = tmp_path / "m.lmdb"
    write_lmdb(path, tensors, layout="blob")

    new = torch.arange(8, dtype=torch.float32)
    with LmdbWeightStore.open(path, readonly=False) as store:
        store.put_tensor("w", raw_of(new), torch.float32, (8,))

    store = LmdbWeightStore.open(path)
    assert torch.equal(store.get_tensor("w"), new)


def test_delta_against_reference(tmp_path, raw_of):
    base = torch.randn(64).float()
    # A near-identical tensor: a small XOR residual against ``base``.
    nxt = base.clone()
    nxt[0] = base[0] + 1.0
    tensors = [("base", raw_of(base), torch.float32, (64,))]
    path = tmp_path / "m.lmdb"
    write_lmdb(path, tensors, layout="blob")

    with LmdbWeightStore.open(path, readonly=False) as store:
        store.put_tensor(
            "next",
            raw_of(nxt),
            torch.float32,
            (64,),
            delta_reference="base",
        )

    store = LmdbWeightStore.open(path)
    # The delta tensor recovers its full value via the in-store reference.
    assert store.get_bytes("next") == raw_of(nxt)
    assert torch.equal(store.get_tensor("next"), nxt)

    # The stored residual is recorded in the index with a verifiable hash.
    entry = store._weight_map["next"]
    assert entry["delta_ref"] == "base"
    assert entry["delta_scheme"] == "xor"
    assert entry["delta_hash"] == _rust.blake3_hash(raw_of(base)).hex()


def test_delta_detects_corrupt_reference(tmp_path, raw_of):
    base = torch.randn(16).float()
    nxt = base.clone()
    tensors = [("base", raw_of(base), torch.float32, (16,))]
    path = tmp_path / "m.lmdb"
    write_lmdb(path, tensors, layout="blob")
    with LmdbWeightStore.open(path, readonly=False) as store:
        store.put_tensor(
            "next", raw_of(nxt), torch.float32, (16,), delta_reference="base"
        )
        # Overwrite the reference so the recorded BLAKE3 no longer matches.
        store.put_tensor("base", raw_of(torch.randn(16).float()), torch.float32, (16,))

    store = LmdbWeightStore.open(path)
    with pytest.raises(ValueError, match="BLAKE3"):
        store.get_bytes("next")


def test_readonly_rejects_mutation(tmp_path, store_tensors):
    path = tmp_path / "m.lmdb"
    write_lmdb(path, store_tensors, layout="blob")
    store = LmdbWeightStore.open(path)
    with pytest.raises(RuntimeError, match="read-only"):
        store.put_tensor("w", b"\x00" * 4, torch.float32, (1,))
