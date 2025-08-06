import numpy as np
from ptwm import _rust
from ptwm.sharding import write_ptwm_shard


def test_writes_single_shard_with_three_tensors(tmp_path):
    tensors = [
        ("a", np.arange(64, dtype=np.uint8).tobytes(), np.uint8, (64,)),
        ("b", np.arange(32, dtype=np.uint8).tobytes(), np.uint8, (32,)),
        ("c", np.arange(16, dtype=np.uint8).tobytes(), np.uint8, (16,)),
    ]
    out = tmp_path / "shard.ptwm"
    write_ptwm_shard(out, tensors)
    assert out.exists()
    assert out.stat().st_size > 0

    # Verify the blob contains all three tensors with the right names.
    blob = out.read_bytes()
    names = _rust.list_tensor_names(blob)
    assert sorted(names) == ["a", "b", "c"]
    # Round-trip the first tensor.
    raw = _rust.decode_tensor(blob, "a")
    assert bytes(raw) == np.arange(64, dtype=np.uint8).tobytes()
