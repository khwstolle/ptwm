"""CLI: names of ``--delta`` outputs for every input extension shape.

``ptwm compress X --delta R`` used to name its output ``X[:-4] + "_delta_"
+ R + ".ptwm"`` and ``ptwm decompress`` always restored ``<prefix>.bin``.
Only ``.bin`` inputs round-tripped; other names were mangled or collided.
``.bin`` inputs keep the old names so existing files still decompress.
A name with more than one ``.delta.`` or ``_delta_`` has no known split, so
decompression needs ``--output`` for it.
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
    "model.safetensors": "model.safetensors.delta.base.bin.ptwm",
    "w.pt": "w.pt.delta.base.bin.ptwm",
    "a.b.c.pth": "a.b.c.pth.delta.base.bin.ptwm",
    "weights": "weights.delta.base.bin.ptwm",
}

# (input name, reference name) pairs whose compressed name holds more than
# one separator.
AMBIGUOUS = [
    # Another input whose reference name contains `_delta_`.
    ("model.pt", "base_delta_v1.bin"),
    # Another input whose reference name contains `.delta.`. The last
    # `.delta.` used to give `model.pt.delta.base`.
    ("model.pt", "base.delta.v1.bin"),
    # Another input whose name contains `.delta.` or `_delta_`.
    ("a.delta.b.pt", "ref.bin"),
    ("x_delta_y.pt", "ref.bin"),
    # A `.bin` input whose stem contains `.delta.` or `_delta_`.
    ("a.delta.b.bin", "ref.bin"),
    ("x_delta_y.bin", "ref.bin"),
    # A `.bin` input whose reference name contains `_delta_`.
    ("w.bin", "base_delta_v1.bin"),
    # The same name from two inputs: a `.bin` input whose reference name
    # contains `.delta.`, and another input whose name contains `_delta_`.
    ("a.bin", "b.delta.c"),
    ("a_delta_b", "c"),
]


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


def _pair(tmp_path: Path, name: str, ref_name: str = REF) -> tuple[Path, Path, bytes]:
    rng = np.random.default_rng(3)
    ref = rng.standard_normal(512).astype(np.float32)
    cur = (ref + rng.standard_normal(512).astype(np.float32) * 1e-3).tobytes()
    src = tmp_path / name
    src.write_bytes(cur)
    ref_path = tmp_path / ref_name
    ref_path.write_bytes(ref.tobytes())
    return src, ref_path, cur


@pytest.mark.parametrize(("name", "compressed"), NAMES.items())
def test_names_are_inverse(name: str, compressed: str) -> None:
    assert delta_compressed_name(name, REF) == compressed
    assert delta_restored_name(compressed) == name


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


def test_input_with_delta_is_refused_without_output(tmp_path: Path) -> None:
    """``my_delta_x.pt`` restored to ``my.bin`` (and with ``--force``
    overwrote that file) when the reference was renamed. The name holds two
    separators, so the CLI refuses, exits with status 1 and writes nothing,
    with the original reference name too."""
    src, ref, _ = _pair(tmp_path, "my_delta_x.pt")
    _compress(src, ref, "--force")
    other = tmp_path / "my.bin"
    other.write_bytes(b"keep me")
    src.unlink()
    blob = tmp_path / "my_delta_x.pt.delta.base.bin.ptwm"
    for new_ref_name in (None, "renamed.bin"):
        reference = ref if new_ref_name is None else ref.rename(tmp_path / new_ref_name)
        with pytest.raises(SystemExit) as exc:
            _decompress(blob, reference, "--force")
        assert exc.value.code == 1
        assert not src.exists()
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


@pytest.mark.parametrize(("input_name", "reference"), AMBIGUOUS)
def test_ambiguous_name_is_refused(input_name: str, reference: str) -> None:
    """A name with more than one separator reads more than one way. The
    reference name given at decompression does not settle it: a renamed
    reference can match the end of the name (``model.pt`` with
    ``base.delta.v1.bin`` renamed to ``v1.bin``)."""
    compressed = delta_compressed_name(input_name, reference)
    with pytest.raises(ValueError, match="cannot tell the original file name"):
        delta_restored_name(compressed)


@pytest.mark.parametrize(("input_name", "reference"), AMBIGUOUS)
def test_ambiguous_name_round_trips_with_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    input_name: str,
    reference: str,
) -> None:
    """Compress names the flag to use, and ``--output`` restores the file."""
    src, ref, data = _pair(tmp_path, input_name, reference)
    _compress(src, ref, "--force")
    assert f"--output {input_name}" in capsys.readouterr().err
    blob = tmp_path / delta_compressed_name(input_name, reference)
    src.unlink()
    _decompress(blob, ref, "--output", str(src))
    assert src.read_bytes() == data


def test_matching_suffix_of_renamed_reference_is_refused(tmp_path: Path) -> None:
    """``model.pt`` with reference ``base.delta.v1``, then the reference
    renamed to ``v1``: the name ends in ``.delta.v1.ptwm``, so a match on
    the given reference name restored ``model.pt.delta.base``."""
    src, ref, _ = _pair(tmp_path, "model.pt", "base.delta.v1")
    _compress(src, ref, "--force")
    blob = tmp_path / "model.pt.delta.base.delta.v1.ptwm"
    src.unlink()
    with pytest.raises(SystemExit) as exc:
        _decompress(blob, ref.rename(tmp_path / "v1"), "--force")
    assert exc.value.code == 1
    assert not (tmp_path / "model.pt.delta.base").exists()
    assert not src.exists()


def test_bin_input_with_empty_stem_round_trips(tmp_path: Path) -> None:
    """The input ``.bin`` compresses to ``_delta_<ref>.ptwm``: one separator,
    at the start. Restore refused it with a wrong "more than one" message."""
    assert delta_restored_name(delta_compressed_name(".bin", REF)) == ".bin"
    src, ref, data = _pair(tmp_path, ".bin")
    _compress(src, ref, "--force")
    src.unlink()
    _decompress(tmp_path / "_delta_base.bin.ptwm", ref)
    assert src.read_bytes() == data


def test_name_without_input_part_is_refused() -> None:
    with pytest.raises(ValueError, match="no input name before"):
        delta_restored_name(".delta.base.bin.ptwm")


def test_output_without_delta_is_refused(tmp_path: Path) -> None:
    blob = tmp_path / "w.bin.ptwm"
    blob.write_bytes(b"")
    argv = [str(blob), "--output", str(tmp_path / "w.bin")]
    with pytest.raises(SystemExit) as exc:
        handle_decompress(_parse(add_decompress_parser, "decompress", argv))
    assert exc.value.code == 1


def test_unambiguous_name_prints_no_output_hint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src, ref, _ = _pair(tmp_path, "w.pt")
    _compress(src, ref, "--force")
    assert "--output" not in capsys.readouterr().err


def test_cli_refuses_ambiguous_restore_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI reports the ambiguity, exits with status 1 and leaves both
    candidate files alone, even with --force. It used to exit 0."""
    src = tmp_path / "model.pt"
    rng = np.random.default_rng(5)
    ref_arr = rng.standard_normal(512).astype(np.float32)
    src.write_bytes((ref_arr + 1e-3).tobytes())
    ref = tmp_path / "base_delta_v1.bin"
    ref.write_bytes(ref_arr.tobytes())
    _compress(src, ref)
    blob = tmp_path / "model.pt.delta.base_delta_v1.bin.ptwm"
    assert blob.exists()

    src.unlink()
    bystander = tmp_path / "model.pt.delta.base.bin"
    bystander.write_bytes(b"keep")
    renamed = ref.rename(tmp_path / "renamed.bin")
    with pytest.raises(SystemExit) as exc:
        _decompress(blob, renamed, "--force")

    assert exc.value.code == 1
    assert "cannot tell the original file name" in capsys.readouterr().err
    assert bystander.read_bytes() == b"keep"
    assert not src.exists()
