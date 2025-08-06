"""Read a sharded ``.ptwm`` directory into a ``{name: raw_bytes}`` map."""

from __future__ import annotations

from pathlib import Path

from .. import _rust
from ._index import PtwmIndex

__all__ = ["read_sharded_ptwm"]


def read_sharded_ptwm(model_dir: str | Path) -> dict[str, bytes]:
    """Open every shard listed in ``model.ptwm.index.json``, return the union.

    If no index file is present, falls back to a single ``model.ptwm``
    in the directory.

    Parameters
    ----------
    model_dir:
        Directory containing ``.ptwm`` shard files and optionally a
        ``model.ptwm.index.json`` index.

    Returns
    -------
    dict[str, bytes]
        Flat mapping of tensor name → raw decompressed bytes, merged
        across all shards.

    Raises
    ------
    FileNotFoundError
        If neither ``model.ptwm.index.json`` nor ``model.ptwm`` exists.
    ValueError
        If the same tensor name appears in more than one shard.
    """
    root = Path(model_dir)
    index_path = root / "model.ptwm.index.json"
    if index_path.exists():
        idx = PtwmIndex.read(index_path)
        shard_names = sorted(set(idx.weight_map.values()))
    else:
        single = root / "model.ptwm"
        if not single.exists():
            msg = f"{root}: no model.ptwm or model.ptwm.index.json"
            raise FileNotFoundError(msg)
        shard_names = ["model.ptwm"]

    state: dict[str, bytes] = {}
    for shard_name in shard_names:
        blob = (root / shard_name).read_bytes()
        # decode_model decodes every tensor in the shard across the rayon pool
        # with the GIL released, returning (name, bytes) in container order —
        # far faster than a per-tensor decode_tensor loop for a full load.
        for name, payload in _rust.decode_model(blob):
            if name in state:
                msg = (
                    f"tensor {name!r} appears in multiple shards; "
                    "shards must partition the tensor name space"
                )
                raise ValueError(msg)
            state[name] = bytes(payload)
    return state
