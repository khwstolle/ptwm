"""CLI: ``--container-layout`` selects the layout of written containers.

New ``.ptwm`` files default to the compact layout (``\\x89PTWC``), and the
random-access tools open them. ``--container-layout legacy`` keeps writing
``\\x89PTWM`` files for readers before v1.2.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
import torch
from ptwm import Decompressor
from ptwm._magic import container_kind
from ptwm.cli.compress import add_compress_parser, compress_file, handle_compress
from ptwm.cli.compress import main as compress_main
from ptwm.random_access import TensorIndex
from safetensors.torch import save_file

KIND = {"compact": "ptwc", "legacy": "ptwm"}


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_compress_parser(parser.add_subparsers(dest="command"))
    return parser.parse_args(["compress", *argv])


def _raw_file(tmp_path: Path) -> tuple[Path, bytes]:
    data = np.random.default_rng(0).standard_normal(2048).astype(np.float32)
    path = tmp_path / "weights.bin"
    path.write_bytes(data.tobytes())
    return path, data.tobytes()


def _safetensors_file(tmp_path: Path) -> Path:
    g = torch.Generator().manual_seed(0)
    path = tmp_path / "model.safetensors"
    save_file(
        {
            "a.weight": torch.randn(64, 32, generator=g).to(torch.bfloat16),
            "b.weight": torch.randn(128, generator=g),
        },
        str(path),
    )
    return path


def test_parser_defaults_to_compact() -> None:
    assert _parse(["x"]).container_layout == "compact"
    assert _parse(["x", "--container-layout", "LEGACY"]).container_layout == "legacy"
    with pytest.raises(SystemExit):
        _parse(["x", "--container-layout", "bogus"])


@pytest.mark.parametrize("layout", [None, "compact", "legacy"])
def test_file_compression_writes_an_openable_container(
    tmp_path: Path, layout: str | None
) -> None:
    src, raw = _raw_file(tmp_path)
    argv = [str(src), "--file_compression", "--force", "--dtype", "float32"]
    if layout is not None:
        argv += ["--container-layout", layout]
    handle_compress(_parse(argv))

    out = src.with_suffix(".bin.ptwm")
    blob = out.read_bytes()
    assert container_kind(blob) == KIND[layout or "compact"]
    assert Decompressor().decompress(blob) == raw
    index = TensorIndex.open(out)
    assert index.layout == (layout or "compact")
    assert len(index.names()) == 1


@pytest.mark.parametrize("layout", ["compact", "legacy"])
def test_compress_file_function_takes_the_layout(tmp_path: Path, layout: str) -> None:
    src, raw = _raw_file(tmp_path)
    compress_file(str(src), dtype="float32", force=True, container_layout=layout)
    blob = src.with_suffix(".bin.ptwm").read_bytes()
    assert container_kind(blob) == KIND[layout]
    assert Decompressor().decompress(blob) == raw


@pytest.mark.parametrize("layout", [None, "legacy"])
def test_standalone_main_mode_a_shards(tmp_path: Path, layout: str | None) -> None:
    src = _safetensors_file(tmp_path)
    out = tmp_path / "out"
    argv = ["--mode", "a", "--out", str(out), str(src)]
    if layout is not None:
        argv += ["--container-layout", layout]
    assert compress_main(argv) == 0
    shards = sorted(out.glob("*.ptwm"))
    assert shards
    for shard in shards:
        assert container_kind(shard.read_bytes()) == KIND[layout or "compact"]
        assert set(TensorIndex.open(shard).names()) <= {"a.weight", "b.weight"}


@pytest.mark.parametrize("layout", [None, "legacy"])
def test_store_formats_take_the_layout(tmp_path: Path, layout: str | None) -> None:
    pytest.importorskip("lmdb")
    from ptwm.stores import LmdbWeightStore  # noqa: PLC0415

    src = _safetensors_file(tmp_path)
    out = tmp_path / "m.lmdb"
    argv = [str(src), "--format", "lmdb", "--out", str(out)]
    if layout is not None:
        argv += ["--container-layout", layout]
    handle_compress(_parse(argv))
    store = LmdbWeightStore.open(out)
    blob = store._get(b"t/a.weight")
    assert container_kind(blob) == KIND[layout or "compact"]
