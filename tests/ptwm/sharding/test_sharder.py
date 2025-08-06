from ptwm.sharding import plan_shards, shard_filename


def test_filename_format():
    assert shard_filename(1, 7, suffix="ptwm") == "model-00001-of-00007.ptwm"
    assert shard_filename(7, 7, suffix="ptwm") == "model-00007-of-00007.ptwm"
    assert (
        shard_filename(1, 2, suffix="safetensors") == "model-00001-of-00002.safetensors"
    )


def test_plan_shards_packs_within_threshold():
    # 5 tensors, each 1 MiB compressed; threshold 2 MiB → expect 3 shards
    # (sizes [2, 2, 1] MiB).
    sizes = [(f"t{i}", 1 << 20) for i in range(5)]
    shards = plan_shards(sizes, max_shard_size=2 << 20)
    assert len(shards) == 3
    assert [len(s) for s in shards] == [2, 2, 1]


def test_plan_shards_single_shard_below_threshold():
    sizes = [(f"t{i}", 1 << 10) for i in range(3)]
    shards = plan_shards(sizes, max_shard_size=1 << 30)
    assert len(shards) == 1


def test_plan_shards_oversize_tensor_gets_its_own_shard():
    # A single tensor larger than the threshold cannot be split; it
    # gets its own shard.
    sizes = [("a", 1 << 10), ("big", 4 << 20), ("c", 1 << 10)]
    shards = plan_shards(sizes, max_shard_size=2 << 20)
    names_per_shard = [[name for name, _ in s] for s in shards]
    assert ["a"] in names_per_shard or ["a", "c"] in names_per_shard
    assert ["big"] in names_per_shard


def test_plan_shards_preserves_input_order():
    sizes = [(f"t{i}", 1 << 20) for i in range(4)]
    shards = plan_shards(sizes, max_shard_size=2 << 20)
    flat_names = [name for shard in shards for name, _ in shard]
    assert flat_names == ["t0", "t1", "t2", "t3"]
