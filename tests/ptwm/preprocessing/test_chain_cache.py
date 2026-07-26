"""Tests for the user-local chain cache.

The cache stores *structural templates* (internal_dtype + op tuple + planes)
rather than baked-in wire bytes; a template discovered on a (64, 64) tensor
still applies to a (1024, 1024) tensor of the same dtype.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ptwm.preprocessing import (
    SIGNATURE_BUCKET_WIDTH,
    CacheEntry,
    ClassifierRole,
    cache_dir,
    cache_info,
    cache_path_for,
    clear_cache,
    load_cached_builders,
    load_cached_entries,
    save_cached_entries,
    signature_bucket_for,
)


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch, tmp_path: Path):
    # Redirect XDG_CACHE_HOME so the test cannot touch the developer's real
    # cache. The cache module reads the env var dynamically per call.
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))


def test_cache_dir_respects_xdg(tmp_path: Path) -> None:
    root = cache_dir()
    assert root.is_relative_to(tmp_path / "xdg")
    assert root.name.startswith("v")


def test_save_and_load_round_trip() -> None:
    entries = [
        CacheEntry(internal_dtype=0x000F, ops=(0x0010, 0x0020), byte_split_planes=2),
        CacheEntry(internal_dtype=0x000F, ops=(0x0030,), byte_split_planes=1),
    ]
    added = save_cached_entries(0x0006, ClassifierRole.STANDARD, entries)
    assert added == 2
    loaded = load_cached_entries(0x0006, ClassifierRole.STANDARD)
    assert set(loaded) == set(entries)


def test_save_is_idempotent() -> None:
    entry = CacheEntry(internal_dtype=0x000F, ops=(0x0010, 0x0020), byte_split_planes=2)
    assert save_cached_entries(0x0006, ClassifierRole.STANDARD, [entry]) == 1
    assert save_cached_entries(0x0006, ClassifierRole.STANDARD, [entry]) == 0
    loaded = load_cached_entries(0x0006, ClassifierRole.STANDARD)
    assert loaded == [entry]


def test_load_miss_returns_empty_list() -> None:
    assert load_cached_entries(0x1234, ClassifierRole.STANDARD) == []


def test_load_corrupt_file_returns_empty(tmp_path: Path) -> None:
    # A corrupt cache must not poison the compress path.
    path = cache_path_for(0x0006, ClassifierRole.STANDARD)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a valid cache file")
    assert load_cached_entries(0x0006, ClassifierRole.STANDARD) == []


def test_load_schema_mismatch_returns_empty(monkeypatch) -> None:
    import cbor2

    path = cache_path_for(0x0006, ClassifierRole.STANDARD)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 99, "entries": []}
    path.write_bytes(b"PTWMCHC\x02" + cbor2.dumps(payload))
    assert load_cached_entries(0x0006, ClassifierRole.STANDARD) == []


def test_clear_cache_removes_all_files() -> None:
    save_cached_entries(
        0x0006,
        ClassifierRole.STANDARD,
        [CacheEntry(internal_dtype=0x000F, ops=(0x0030,), byte_split_planes=1)],
    )
    save_cached_entries(
        0x0007,
        ClassifierRole.STANDARD,
        [CacheEntry(internal_dtype=0x000F, ops=(0x0030,), byte_split_planes=1)],
    )
    assert clear_cache() == 2
    assert load_cached_entries(0x0006, ClassifierRole.STANDARD) == []
    assert load_cached_entries(0x0007, ClassifierRole.STANDARD) == []


def test_load_cached_builders_apply_runtime_shape() -> None:
    entry = CacheEntry(internal_dtype=0x000F, ops=(0x0010, 0x0020), byte_split_planes=2)
    save_cached_entries(0x0006, ClassifierRole.STANDARD, [entry])
    builders = load_cached_builders(0x0006, ClassifierRole.STANDARD)
    assert len(builders) == 1
    chain_small = builders[0]([4, 8])
    chain_large = builders[0]([1024, 1024])
    # Same op structure; only Source.params differs because the shape is
    # encoded inline.
    assert [n.op_id for n in chain_small.nodes] == [n.op_id for n in chain_large.nodes]
    assert chain_small.nodes[0].params != chain_large.nodes[0].params


def test_cache_info_reports_counts() -> None:
    save_cached_entries(
        0x0006,
        ClassifierRole.STANDARD,
        [
            CacheEntry(internal_dtype=0x000F, ops=(0x0030,), byte_split_planes=1),
            CacheEntry(
                internal_dtype=0x000F, ops=(0x0010, 0x0020), byte_split_planes=2
            ),
        ],
    )
    info = cache_info()
    assert any("0006_0_00.cbor" in k and v == 2 for k, v in info.items())


def test_signature_bucket_for_clamps_to_valid_range() -> None:
    assert signature_bucket_for(-1.0) == 0
    assert signature_bucket_for(0.0) == 0
    assert signature_bucket_for(100.0) == signature_bucket_for(8.0)


def test_signature_bucket_for_separates_low_and_high_entropy() -> None:
    low = signature_bucket_for(0.1)
    high = signature_bucket_for(7.9)
    assert low != high


def test_signature_bucket_for_boundary_is_half_open() -> None:
    # Bucket edges fall on the lower bucket; the next bucket starts exactly
    # at the next multiple of the bucket width.
    edge = signature_bucket_for(SIGNATURE_BUCKET_WIDTH)
    just_below = signature_bucket_for(SIGNATURE_BUCKET_WIDTH - 1e-9)
    assert edge == just_below + 1


def test_different_signature_buckets_are_isolated() -> None:
    entry_low = CacheEntry(internal_dtype=0x000F, ops=(0x0030,), byte_split_planes=1)
    entry_high = CacheEntry(
        internal_dtype=0x000F, ops=(0x0010, 0x0020), byte_split_planes=2
    )
    save_cached_entries(0x0006, ClassifierRole.STANDARD, [entry_low], 0)
    save_cached_entries(0x0006, ClassifierRole.STANDARD, [entry_high], 5)

    assert load_cached_entries(0x0006, ClassifierRole.STANDARD, 0) == [entry_low]
    assert load_cached_entries(0x0006, ClassifierRole.STANDARD, 5) == [entry_high]
    # The default bucket (0) must not see entries written under a different
    # bucket, and vice versa.
    assert load_cached_entries(0x0006, ClassifierRole.STANDARD, 1) == []


def test_cache_path_for_encodes_signature_bucket() -> None:
    path_bucket_0 = cache_path_for(0x0006, ClassifierRole.STANDARD, 0)
    path_bucket_5 = cache_path_for(0x0006, ClassifierRole.STANDARD, 5)
    assert path_bucket_0 != path_bucket_5
    assert path_bucket_0.name == "0006_0_00.cbor"
    assert path_bucket_5.name == "0006_0_05.cbor"
