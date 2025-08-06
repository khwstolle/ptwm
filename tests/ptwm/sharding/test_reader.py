import numpy as np
from ptwm.sharding import PtwmIndex, read_sharded_ptwm, write_ptwm_shard


def test_round_trip_two_shards(tmp_path):
    shard1 = tmp_path / "model-00001-of-00002.ptwm"
    shard2 = tmp_path / "model-00002-of-00002.ptwm"
    write_ptwm_shard(
        shard1,
        [("a", np.arange(64, dtype=np.uint8).tobytes(), np.uint8, (64,))],
    )
    write_ptwm_shard(
        shard2,
        [("b", np.arange(32, dtype=np.uint8).tobytes(), np.uint8, (32,))],
    )

    PtwmIndex(
        total_size=64 + 32,
        compressed_total_size=shard1.stat().st_size + shard2.stat().st_size,
        weight_map={
            "a": "model-00001-of-00002.ptwm",
            "b": "model-00002-of-00002.ptwm",
        },
    ).write(tmp_path / "model.ptwm.index.json")

    state = read_sharded_ptwm(tmp_path)
    assert set(state.keys()) == {"a", "b"}
    np.testing.assert_array_equal(
        np.frombuffer(state["a"], dtype=np.uint8),
        np.arange(64, dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        np.frombuffer(state["b"], dtype=np.uint8),
        np.arange(32, dtype=np.uint8),
    )


def test_single_file_no_index(tmp_path):
    shard = tmp_path / "model.ptwm"
    write_ptwm_shard(
        shard,
        [("only", np.arange(8, dtype=np.uint8).tobytes(), np.uint8, (8,))],
    )
    state = read_sharded_ptwm(tmp_path)
    assert set(state.keys()) == {"only"}
