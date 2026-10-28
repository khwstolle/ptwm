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
            # Large enough to compress in the legacy layout too, whose
            # Extension Table of all 16 built-in codecs takes about 2 KB.
            "a.weight": torch.randn(256, 128, generator=g).to(torch.bfloat16),
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


def _assert_tensors_match(src: Path, out: Path) -> None:
    from ptwm.integrations import SafeOpen  # noqa: PLC0415
    from safetensors.torch import load_file  # noqa: PLC0415

    want = load_file(str(src))
    got = SafeOpen(str(out), "pt")
    for name, tensor in want.items():
        assert torch.equal(got.get_tensor(name), tensor)


@pytest.mark.parametrize("layout", [None, "compact", "legacy"])
def test_delta_compression_round_trips(tmp_path: Path, layout: str | None) -> None:
    """``--delta`` reads the reference file; it used to pass the reference
    path string to the compressor and fail with ``TypeError``."""
    from ptwm import DecompressionConfig  # noqa: PLC0415

    rng = np.random.default_rng(1)
    ref = rng.standard_normal(2048).astype(np.float32)
    cur = ref + rng.standard_normal(2048).astype(np.float32) * 1e-3
    src = tmp_path / "weights.bin"
    src.write_bytes(cur.tobytes())
    ref_path = tmp_path / "base.bin"
    ref_path.write_bytes(ref.tobytes())

    argv = [str(src), "--delta", str(ref_path), "--force", "--dtype", "float32"]
    argv += ["--verification"]
    if layout is not None:
        argv += ["--container-layout", layout]
    handle_compress(_parse(argv))

    out = tmp_path / "weights_delta_base.bin.ptwm"
    blob = out.read_bytes()
    assert container_kind(blob) == KIND[layout or "compact"]
    dec = Decompressor(DecompressionConfig(delta_second_data=ref_path.read_bytes()))
    assert dec.decompress(blob) == cur.tobytes()


@pytest.mark.parametrize("layout", [None, "legacy"])
def test_webdataset_store_takes_the_layout(tmp_path: Path, layout: str | None) -> None:
    from ptwm.stores import WebDatasetWeightStore  # noqa: PLC0415

    src = _safetensors_file(tmp_path)
    out = tmp_path / "m.wds"
    argv = [str(src), "--format", "webdataset", "--out", str(out)]
    if layout is not None:
        argv += ["--container-layout", layout]
    handle_compress(_parse(argv))
    from ptwm.stores._common import manifest_container_layout  # noqa: PLC0415

    with WebDatasetWeightStore.open(out) as store:
        assert manifest_container_layout(store._registry_json) == (layout or "compact")
        want = torch.randn(256, 128, generator=torch.Generator().manual_seed(0))
        assert torch.equal(store.get_tensor("a.weight"), want.to(torch.bfloat16))


@pytest.mark.parametrize("layout", [None, "legacy"])
def test_mode_b_writes_the_out_file(tmp_path: Path, layout: str | None) -> None:
    """``--mode b --out F`` writes F itself and leaves no staging directory."""
    from safetensors import safe_open  # noqa: PLC0415

    src = _safetensors_file(tmp_path)
    out = tmp_path / "b.ptwm.safetensors"
    argv = [str(src), "--mode", "b", "--out", str(out), "--force"]
    if layout is not None:
        argv += ["--container-layout", layout]
    handle_compress(_parse(argv))

    assert out.is_file()
    assert not (tmp_path / "b.ptwm__staging").exists()
    with safe_open(str(out), "pt") as f:
        payload = f.get_tensor("__ptwm_payload__").numpy().tobytes()
    assert container_kind(payload) == KIND[layout or "compact"]
    _assert_tensors_match(src, out)


@pytest.mark.parametrize("layout", [None, "legacy"])
def test_mode_a_writes_the_out_directory(tmp_path: Path, layout: str | None) -> None:
    src = _safetensors_file(tmp_path)
    out = tmp_path / "a_out"
    argv = [str(src), "--mode", "a", "--out", str(out), "--force"]
    if layout is not None:
        argv += ["--container-layout", layout]
    handle_compress(_parse(argv))
    shard = out / "model.ptwm"
    assert container_kind(shard.read_bytes()) == KIND[layout or "compact"]
    _assert_tensors_match(src, shard)


@pytest.mark.parametrize("layout", [None, "legacy"])
def test_directory_file_compression_takes_the_layout(
    tmp_path: Path, layout: str | None
) -> None:
    src = _safetensors_file(tmp_path)
    argv = [str(tmp_path), "--file_compression", "--force"]
    if layout is not None:
        argv += ["--container-layout", layout]
    handle_compress(_parse(argv))
    blob = (tmp_path / (src.name + ".ptwm")).read_bytes()
    assert container_kind(blob) == KIND[layout or "compact"]
    assert Decompressor().decompress(blob) == src.read_bytes()


@pytest.mark.parametrize("layout", [None, "legacy"])
def test_directory_per_tensor_shell_takes_the_layout(
    tmp_path: Path, layout: str | None
) -> None:
    src = _safetensors_file(tmp_path)
    argv = [str(tmp_path), "--force"]
    if layout is not None:
        argv += ["--container-layout", layout]
    handle_compress(_parse(argv))
    out = tmp_path / "model.ptwm.safetensors"

    from ptwm.integrations._safetensors import (  # noqa: PLC0415
        decompress_safetensors_tensor,
    )
    from safetensors import safe_open  # noqa: PLC0415
    from safetensors.torch import load_file  # noqa: PLC0415

    want = load_file(str(src))
    with safe_open(str(out), "pt") as f:
        assert set(f.keys()) == set(want)
        for name, tensor in want.items():
            got = decompress_safetensors_tensor(f.get_tensor(name))
            raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
            if isinstance(got, torch.Tensor):
                got = got.contiguous().view(torch.uint8).numpy().tobytes()
            assert bytes(got) == raw


def test_directory_compression_with_several_processes(tmp_path: Path) -> None:
    """``--max_processes 2`` runs the workers in spawned children, so a
    parent that already ran the Rust and torch thread pools does not hang."""
    torch.randn(256, 256) @ torch.randn(256, 256)
    srcs = []
    for i in range(3):
        sub = _safetensors_file(tmp_path)
        srcs.append(sub.rename(tmp_path / f"m{i}.safetensors"))
    argv = [str(tmp_path), "--file_compression", "--force", "--max_processes", "2"]
    argv += ["--container-layout", "legacy"]
    handle_compress(_parse(argv))
    for src in srcs:
        blob = (tmp_path / (src.name + ".ptwm")).read_bytes()
        assert container_kind(blob) == "ptwm"
        assert Decompressor().decompress(blob) == src.read_bytes()


def _builtin_label_versions(blob: bytes) -> set[bytes]:
    """Versions of every ``io.ptwm.builtin.<name>@<version>`` label in ``blob``."""
    import re  # noqa: PLC0415

    return set(re.findall(rb"io\.ptwm\.builtin\.[a-z0-9_]+@([0-9.]+)", blob))


@pytest.mark.parametrize("layout", [None, "compact", "legacy"])
def test_per_tensor_shell_honours_the_layout(
    tmp_path: Path, layout: str | None
) -> None:
    """The default per-tensor path for a ``.safetensors`` file writes v1.1
    readable ``\\x89PTWM`` containers with ``--container-layout legacy``. It
    used to write ``\\x89PTWS`` frames whatever the flag said."""
    from safetensors import safe_open  # noqa: PLC0415

    src = _safetensors_file(tmp_path)
    argv = [str(src), "--force"]
    if layout is not None:
        argv += ["--container-layout", layout]
    handle_compress(_parse(argv))
    out = tmp_path / "model.ptwm.safetensors"
    whole = out.read_bytes()

    from ptwm.integrations._safetensors import (  # noqa: PLC0415
        decompress_safetensors_tensor,
    )
    from safetensors.torch import load_file  # noqa: PLC0415

    want = load_file(str(src))
    kinds = set()
    with safe_open(str(out), "pt") as f:
        assert set(f.keys()) == set(want)
        for name, tensor in want.items():
            stored = f.get_tensor(name)
            if stored.dtype != torch.uint8:
                # Stored raw: compression did not make it smaller.
                assert torch.equal(stored, tensor), name
                continue
            kinds.add(container_kind(stored.numpy().tobytes()))
            got = decompress_safetensors_tensor(stored)
            if isinstance(got, torch.Tensor):
                got = got.contiguous().view(torch.uint8).numpy().tobytes()
            assert bytes(got) == tensor.contiguous().view(torch.uint8).numpy().tobytes()
    assert kinds
    if layout == "legacy":
        assert kinds == {"ptwm"}
        assert b"\x89PTWS" not in whole
        assert b"\x89PTWC" not in whole
        # A v1.1 reader recognises only the built-in ids of 1.1.0.
        assert _builtin_label_versions(whole) == {b"1.1.0"}
    else:
        assert kinds <= {"ptws", "ptwc"}
        assert b"\x89PTWM" not in whole
