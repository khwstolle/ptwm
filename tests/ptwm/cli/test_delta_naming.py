"""CLI: names of ``--delta`` outputs for every input extension shape.

``ptwm compress X --delta R`` used to name its output ``X[:-4] + "_delta_"
+ R + ".ptwm"`` and ``ptwm decompress`` always restored ``<prefix>.bin``.
Only ``.bin`` inputs round-tripped; other names were mangled or collided.
``.bin`` inputs keep the old names so existing files still decompress.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
from ptwm.cli.compress import add_compress_parser, handle_compress
from ptwm.cli.decompress import add_decompress_parser, handle_decompress
from ptwm.cli.utils import delta_compressed_name, delta_restored_name

REF = "base.bin"

# input name -> compressed name
NAMES = {
    "weights.bin": "weights_delta_base.bin.ptwm",
    "m.v2.bin": "m.v2_delta_base.bin.ptwm",
    "x_delta_y.bin": "x_delta_y_delta_base.bin.ptwm",
    "model.safetensors": "model.safetensors.delta.base.bin.ptwm",
    "w.pt": "w.pt.delta.base.bin.ptwm",
    "a.b.c.pth": "a.b.c.pth.delta.base.bin.ptwm",
    "weights": "weights.delta.base.bin.ptwm",
    "x_delta_y.pt": "x_delta_y.pt.delta.base.bin.ptwm",
}


def _parse(add, command: str, argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add(parser.add_subparsers(dest="command"))
    return parser.parse_args([command, *argv])


def _compress(src: Path, ref: Path, *extra: str) -> None:
    argv = [str(src), "--delta", str(ref), "--dtype", "float32", *extra]
    handle_compress(_parse(add_compress_parser, "compress", argv))


def _decompress(blob: Path, ref: Path, *extra: str) -> None:
    argv = [str(blob), "--delta", str(ref), *extra]
    handle_decompress(_parse(add_decompress_parser, "decompress", argv))


def _pair(tmp_path: Path, name: str) -> tuple[Path, Path, bytes]:
    rng = np.random.default_rng(3)
    ref = rng.standard_normal(512).astype(np.float32)
    cur = (ref + rng.standard_normal(512).astype(np.float32) * 1e-3).tobytes()
    src = tmp_path / name
    src.write_bytes(cur)
    ref_path = tmp_path / REF
    ref_path.write_bytes(ref.tobytes())
    return src, ref_path, cur


@pytest.mark.parametrize(("name", "compressed"), NAMES.items())
def test_names_are_inverse(name: str, compressed: str) -> None:
    assert delta_compressed_name(name, REF) == compressed
    assert delta_restored_name(compressed, REF) == name


@pytest.mark.parametrize("name", NAMES)
def test_cli_round_trip_restores_the_input_name(tmp_path: Path, name: str) -> None:
    src, ref, data = _pair(tmp_path, name)
    _compress(src, ref, "--force")
    blob = tmp_path / NAMES[name]
    assert blob.is_file()
    src.unlink()
    _decompress(blob, ref)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [name, REF, NAMES[name]]
    )
    assert src.read_bytes() == data


def test_short_names_do_not_collide(tmp_path: Path) -> None:
    """``a.pt`` and ``b.pt`` both used to compress to ``_delta_<ref>.ptwm``."""
    a, ref, data_a = _pair(tmp_path, "a.pt")
    b = tmp_path / "b.pt"
    b.write_bytes(data_a[::-1])
    _compress(a, ref, "--force")
    _compress(b, ref, "--force")
    assert (tmp_path / "a.pt.delta.base.bin.ptwm").is_file()
    assert (tmp_path / "b.pt.delta.base.bin.ptwm").is_file()


def test_legacy_name_without_matching_reference_name(tmp_path: Path) -> None:
    """A legacy ``_delta_`` file still restores ``<prefix>.bin`` when the
    reference was renamed since compression."""
    src, ref, data = _pair(tmp_path, "weights.bin")
    _compress(src, ref, "--force")
    renamed = ref.rename(tmp_path / "renamed.bin")
    src.unlink()
    _decompress(tmp_path / "weights_delta_base.bin.ptwm", renamed)
    assert src.read_bytes() == data


@pytest.mark.parametrize(
    "name", [n for n in NAMES if not n.endswith(".bin")] + ["a.delta.b.pt"]
)
def test_new_name_without_matching_reference_name(name: str) -> None:
    """A ``.delta.`` file restores its input name when the reference was
    renamed since compression. The fallback used to split at ``_delta_``
    or keep the whole ``.delta.<ref>`` part."""
    compressed = delta_compressed_name(name, REF)
    assert delta_restored_name(compressed, "renamed.bin") == name


def test_new_name_with_delta_in_input_and_renamed_reference(tmp_path: Path) -> None:
    """``my_delta_x.pt`` restored to ``my.bin`` (and with ``--force``
    overwrote that file) when the reference was renamed."""
    src, ref, data = _pair(tmp_path, "my_delta_x.pt")
    _compress(src, ref, "--force")
    renamed = ref.rename(tmp_path / "renamed.bin")
    other = tmp_path / "my.bin"
    other.write_bytes(b"keep me")
    src.unlink()
    _decompress(tmp_path / "my_delta_x.pt.delta.base.bin.ptwm", renamed, "--force")
    assert src.read_bytes() == data
    assert other.read_bytes() == b"keep me"


def test_decompress_asks_before_overwriting_the_restored_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The overwrite prompt used to check ``<compressed>`` minus ``.ptwm``,
    not the file that is written, so an existing input was overwritten."""
    src, ref, _ = _pair(tmp_path, "weights.bin")
    _compress(src, ref, "--force")
    src.write_bytes(b"keep me")
    prompts: list[str] = []

    def no(prompt: str) -> str:
        prompts.append(prompt)
        return "n"

    monkeypatch.setattr("builtins.input", no)
    _decompress(tmp_path / "weights_delta_base.bin.ptwm", ref)
    assert prompts == [f"{src} already exists; overwrite (y/n)? "]
    assert src.read_bytes() == b"keep me"
