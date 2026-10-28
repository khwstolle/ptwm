"""Built-in Extension Table ids of other releases.

Built-in canonical ids are derived from the release version, so files and
chain blobs written by another release carry other ids. The reader maps them
to the current ids when the label and id re-derive consistently.
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path

import pytest
from ptwm import _rust

FIXTURES = Path(__file__).resolve().parents[3] / "crates/ptwm-core/tests/fixtures"
BUILTIN_PUBKEY = b"\xbb" * 32
_LABEL = re.compile(rb"io\.ptwm\.builtin\.([a-z0-9_]+)@([0-9.]+)")


def _derive(name: str, version: str) -> bytes:
    blake3 = pytest.importorskip("blake3")
    return blake3.blake3(
        BUILTIN_PUBKEY + b"\x00" + name.encode() + b"\x00" + version.encode()
    ).digest()


def _rederive(blob: bytes, to: str) -> bytes:
    """Rewrite every built-in entry's id and label to version ``to`` (which
    must have the length of the current version). No hash is fixed up, so
    use it only on blobs without a header hash (chain blobs)."""
    out = bytearray(blob)
    count = 0
    for m in _LABEL.finditer(blob):
        name, ver = m.group(1).decode(), m.group(2).decode()
        assert len(ver) == len(to)
        id_start = m.start() - 34
        assert bytes(out[id_start : id_start + 32]) == _derive(name, ver)
        out[id_start : id_start + 32] = _derive(name, to)
        out[m.start(2) : m.end(2)] = to.encode()
        count += 1
    assert count > 0
    return bytes(out)


def _fp32_chain_blob(shape: list[int]) -> bytes:
    from ptwm.preprocessing import _chains

    return _chains.CHAIN_FP32_SPLIT(shape).to_bytes()


def _other_version() -> str:
    import ptwm

    return "1.0.0" if ptwm.__version__ != "1.0.0" else "1.0.1"


def test_chain_blob_info_maps_old_builtin_ids() -> None:
    from ptwm._core import chain_blob_info

    blob = _fp32_chain_blob([16])
    old = _rederive(blob, _other_version())
    assert old != blob
    assert chain_blob_info(old) == chain_blob_info(blob)


@pytest.mark.parametrize(
    ("layout", "magic"), [("legacy", b"\x89PTWM"), ("compact", b"\x89PTWC")]
)
def test_compress_model_accepts_chain_blobs_of_other_releases(
    layout: str, magic: bytes
) -> None:
    """A chain blob cached by another release compresses to the same
    container as the current blob. The legacy layout writes full canonical
    ids and labels into the Extension Table, so it is the case that needs the
    ids normalized; the compact layout stores built-ins by wire id."""
    import torch

    shape = [8, 16]
    raw = torch.linspace(-1, 1, 128, dtype=torch.float32).numpy().tobytes()
    blob = _fp32_chain_blob(shape)
    old = _rederive(blob, _other_version())

    def compress(chain: bytes) -> bytes:
        return _rust.compress_model(
            records=[("w", 3, 0, raw, shape, "float32", None)],
            chains_per_tensor=[[chain]],
            emit_payload_hash=True,
            emit_plane_crc=True,
            container_layout=layout,
        )

    current = compress(blob)
    assert current.startswith(magic)
    assert compress(old) == current
    assert _rust.decode_tensor(current, "w") == raw


def test_forged_chain_blob_entry_is_rejected() -> None:
    """A built-in label whose id does not re-derive is not a built-in."""
    from ptwm._core import chain_blob_info

    old = bytearray(_rederive(_fp32_chain_blob([16]), _other_version()))
    first = _LABEL.search(old)
    assert first is not None
    old[first.start() - 34 : first.start() - 2] = b"\x5a" * 32
    _, table = chain_blob_info(bytes(old))
    _, full = chain_blob_info(_fp32_chain_blob([16]))
    assert len(table) == len(full) - 1


def test_v1_0_0_fixture_decodes_and_transcodes() -> None:
    old = (FIXTURES / "legacy_v1_0_0.ptwm").read_bytes()
    cur = (FIXTURES / "legacy_v1_1_0.ptwm").read_bytes()
    assert dict(_rust.decode_model(old)) == dict(_rust.decode_model(cur))

    registry_json, tensors, members = _rust.explode_ptwm(old)
    rebuilt = _rust.implode_ptwm(registry_json, [t[2] for t in tensors], members)
    assert rebuilt == old
    reg = json.loads(registry_json)
    import ptwm

    if ptwm.__version__ != "1.0.0":
        assert reg["container_flags"]["builtin_version"] == "1.0.0"
    assert "extension_table_b64" not in reg["registry"]

    # One tensor at a time, as the exploded stores do.
    name, _key, tjson = tensors[0]
    one = _rust.implode_ptwm(registry_json, [tjson], members)
    assert _rust.decode_tensor(one, name) == _rust.decode_tensor(old, name)


def test_implode_rejects_unknown_manifest_fields() -> None:
    blob = (FIXTURES / "legacy_trimmed.ptwm").read_bytes()
    registry_json, tensors, members = _rust.explode_ptwm(blob)
    reg = json.loads(registry_json)
    reg["container_flags"]["layout"] = "compact"
    with pytest.raises(Exception, match="unknown field"):
        _rust.implode_ptwm(json.dumps(reg), [t[2] for t in tensors], members)
    reg = json.loads(registry_json)
    reg["manifest_version"] = 2
    with pytest.raises(Exception, match="unsupported manifest"):
        _rust.implode_ptwm(json.dumps(reg), [t[2] for t in tensors], members)


def test_label_layout_assumption() -> None:
    """``_rederive`` assumes ``canonical_id[32] | label_len u16 | label``."""
    blob = _fp32_chain_blob([16])
    m = _LABEL.search(blob)
    assert m is not None
    (label_len,) = struct.unpack_from("<H", blob, m.start() - 2)
    assert label_len == m.end() - m.start()


def test_list_extension_table_entries_reports_current_builtin_ids() -> None:
    """Built-in entries of an older file are listed with the current ids, so
    ``ptwm.ablation.bench_compare`` sees the same contributions in both files."""
    from ptwm._core import list_extension_table_entries
    from ptwm.ablation import bench_compare

    old = FIXTURES / "legacy_v1_0_0.ptwm"
    cur = FIXTURES / "legacy_v1_1_0.ptwm"
    old_entries = list_extension_table_entries(str(old))
    cur_entries = list_extension_table_entries(str(cur))
    assert old_entries
    assert {cid for cid, *_ in old_entries} <= {cid for cid, *_ in cur_entries}

    import ptwm

    for _cid, label, _kind, _flavor in old_entries:
        assert label.endswith(f"@{ptwm.__version__}"), label

    result = bench_compare(old, cur)
    assert result.only_in_a == []
    assert result.common
