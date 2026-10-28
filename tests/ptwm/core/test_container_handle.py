r"""ContainerHandle, the container_layout setting and the ``_magic`` helpers.

Covers synthesis section 9 item 13: layout coercion, the handle agreeing with
the free functions in both layouts, write-order names (E14), and the
Decompressor / TensorIndex accepting ``\x89PTWM`` and ``\x89PTWC``.
"""

from __future__ import annotations

import mmap

import numpy as np
import pytest
import torch
from ptwm import (
    CompressionConfig,
    Compressor,
    ContainerLayout,
    Decompressor,
    Format,
    Framing,
    Method,
    _rust,
)
from ptwm._config import container_layout_name
from ptwm._exceptions import Error
from ptwm._magic import (
    MAGIC_LEN,
    MAGIC_PTWC,
    MAGIC_PTWM,
    MAGIC_PTWS,
    MAGIC_PTWX,
    container_kind,
    is_container,
)
from ptwm.integrations._hf import _load_multi_tensor_ptwm, _open_multi_tensor
from ptwm.integrations._safetensors import SafeOpen
from ptwm.random_access import TensorIndex
from ptwm.sharding import compress_ptwm_blob, write_ptwm_shard
from ptwm.sharding._writer import _default_chain_bytes
from ptwm.utils import DType, decode_ptwm_tensor

LAYOUTS = ("legacy", "compact")
MAGIC = {"legacy": MAGIC_PTWM, "compact": MAGIC_PTWC}

# Deliberately not sorted, and not in name-hash order.
NAMES = ("zeta.weight", "alpha.bias", "mid.weight", "beta", "omega.scale")


def _tensors() -> dict[str, torch.Tensor]:
    torch.manual_seed(3)
    return {
        "zeta.weight": torch.randn(24, 16).bfloat16(),
        "alpha.bias": torch.randn(16).float(),
        "mid.weight": torch.randn(8, 12).to(torch.float16),
        "beta": torch.randn(5, 3).float(),
        "omega.scale": torch.randint(0, 256, (40,), dtype=torch.uint8),
    }


def _raw(t: torch.Tensor) -> bytes:
    view = {1: torch.uint8, 2: torch.uint16, 4: torch.uint32}[t.element_size()]
    return t.flatten().contiguous().view(view).numpy().tobytes()


def _blob(layout: str) -> bytes:
    tensors = _tensors()
    items = [
        (n, _raw(tensors[n]), tensors[n].dtype, tuple(tensors[n].shape)) for n in NAMES
    ]
    return compress_ptwm_blob(items, method_hint=3, container_layout=layout)


def _delta_blob(layout: str) -> tuple[bytes, bytes]:
    """A one-tensor container whose record carries a delta reference hash."""
    raw = bytes(range(256)) * 4
    ref_hash = bytes(range(32))
    dt = DType.from_dtype(torch.uint8)
    chains = _default_chain_bytes(int(dt.code), (1024,), len(raw))
    blob = _rust.compress_model(
        [("d", int(dt.code), 0, raw, [1024], dt.dtype_str, ref_hash)],
        [chains],
        True,
        True,
        container_layout=layout,
    )
    return blob, ref_hash


# ── ContainerLayout coercion ─────────────────────────────────────────────────


def test_container_layout_enum_coerces_strings() -> None:
    assert ContainerLayout("compact") is ContainerLayout.COMPACT
    assert ContainerLayout("LEGACY") is ContainerLayout.LEGACY
    assert ContainerLayout("Compact") is ContainerLayout.COMPACT
    with pytest.raises(ValueError, match="ptwc"):
        ContainerLayout("ptwc")
    with pytest.raises(ValueError, match="1"):
        ContainerLayout(1)


def test_container_layout_name() -> None:
    assert container_layout_name(ContainerLayout.COMPACT) == "compact"
    assert container_layout_name("LEGACY") == "legacy"
    with pytest.raises(ValueError, match="container_layout"):
        container_layout_name("bogus")


def test_compression_config_coerces_container_layout() -> None:
    # The default stays LEGACY until the switch to the compact layout.
    assert CompressionConfig().container_layout is ContainerLayout.LEGACY
    cfg = CompressionConfig(container_layout="compact")
    assert cfg.container_layout is ContainerLayout.COMPACT
    cfg = CompressionConfig(container_layout="Legacy")
    assert cfg.container_layout is ContainerLayout.LEGACY
    with pytest.raises(ValueError, match="bogus"):
        CompressionConfig(container_layout="bogus")


@pytest.mark.parametrize("layout", LAYOUTS)
def test_compressor_honours_container_layout(layout: str) -> None:
    tensor = torch.randn(32, 16)
    cfg = CompressionConfig(
        input_format=Format.TORCH,
        method=Method.HUFFMAN,
        framing=Framing.CONTAINER,
        container_layout=layout,
    )
    blob = Compressor(cfg).compress(tensor)
    assert blob[:MAGIC_LEN] == MAGIC[layout]
    # Decompressor reads both layouts.
    out = Decompressor().decompress(blob)
    assert torch.equal(out, tensor)


def test_compact_frame_ignores_container_layout() -> None:
    tensor = torch.randn(32, 16)
    cfg = CompressionConfig(input_format=Format.TORCH, container_layout="compact")
    blob = Compressor(cfg).compress(tensor)
    assert blob[: len(MAGIC_PTWS)] == MAGIC_PTWS
    assert torch.equal(Decompressor().decompress(blob), tensor)


@pytest.mark.parametrize(
    "layout", [ContainerLayout.LEGACY, ContainerLayout.COMPACT, "COMPACT"]
)
def test_writers_accept_the_enum_and_any_case(tmp_path, layout) -> None:
    expected = MAGIC[container_layout_name(layout)]
    assert _rust.container_layout(_blob(container_layout_name(layout))) is not None
    items = [("w", bytes(64), torch.float32, (16,))]
    assert compress_ptwm_blob(items, container_layout=layout)[:MAGIC_LEN] == expected
    out = tmp_path / "s.ptwm"
    write_ptwm_shard(out, items, container_layout=layout)
    assert out.read_bytes()[:MAGIC_LEN] == expected
    dt = DType.from_dtype(torch.float32)
    chains = _default_chain_bytes(int(dt.code), (16,), 64)
    blob = _rust.compress_model(
        [("w", int(dt.code), 0, bytes(64), [16], dt.dtype_str, None)],
        [chains],
        True,
        True,
        container_layout=layout,
    )
    assert blob[:MAGIC_LEN] == expected


def test_binding_rejects_unknown_layout_with_value_error() -> None:
    with pytest.raises(ValueError, match="container_layout"):
        _rust._ext.compress_model([], [], True, True, container_layout="ptwc")
    with pytest.raises(Error, match="container_layout"):
        _rust.compress_model([], [], True, True, container_layout="ptwc")


# ── _magic ───────────────────────────────────────────────────────────────────


def test_container_kind() -> None:
    assert container_kind(MAGIC_PTWM + b"rest") == "ptwm"
    assert container_kind(MAGIC_PTWX) == "ptwx"
    assert container_kind(bytearray(MAGIC_PTWC)) == "ptwc"
    assert container_kind(memoryview(MAGIC_PTWS + b"\x00\x01")) == "ptws"
    # A 5-byte prefix is not enough for the 9-byte magics.
    assert container_kind(MAGIC_PTWM[:5]) is None
    assert container_kind(b"\x89PTWQ\r\n\x1a\n") is None
    assert container_kind(b"") is None
    assert container_kind(b"\x89PTW") is None


def test_is_container() -> None:
    assert is_container(MAGIC_PTWM)
    assert is_container(MAGIC_PTWX)
    assert is_container(MAGIC_PTWC)
    assert not is_container(MAGIC_PTWS + bytes(8))
    assert not is_container(b"not a container")


@pytest.mark.parametrize("layout", LAYOUTS)
def test_rust_container_layout(layout: str) -> None:
    blob = _blob(layout)
    assert _rust.container_layout(blob) == layout
    assert _rust.container_layout(bytearray(blob)) == layout
    assert container_kind(blob) == ("ptwm" if layout == "legacy" else "ptwc")


def test_rust_container_layout_of_other_bytes() -> None:
    assert _rust.container_layout(MAGIC_PTWX) == "legacy"
    assert _rust.container_layout(MAGIC_PTWS + bytes(16)) is None
    assert _rust.container_layout(b"") is None
    assert _rust.container_layout(b"\x89PTWM") is None


# ── ContainerHandle ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("layout", LAYOUTS)
def test_handle_matches_the_free_functions(layout: str) -> None:
    blob = _blob(layout)
    h = _rust.ContainerHandle(blob)
    assert h.layout == layout
    assert len(h) == len(NAMES)
    assert "alpha.bias" in h
    assert "missing" not in h
    assert 3 not in h
    assert h.names() == _rust.list_tensor_names(blob)
    assert h.decode_model() == _rust.decode_model(blob)
    assert h.plane_summaries() == _rust.list_plane_summaries(blob)
    for name in NAMES:
        assert h.decode(name) == _rust.decode_tensor(blob, name)
        assert h.info(name) == _rust.decode_tensor_info(blob, name)
        assert h.shape(name) == _rust.decode_tensor_shape(blob, name)
        assert h.delta_hash(name) == _rust.decode_tensor_delta_hash(blob, name)
        assert h.delta_hash(name) is None


@pytest.mark.parametrize("layout", LAYOUTS)
def test_handle_names_and_decode_model_in_write_order(layout: str) -> None:
    blob = _blob(layout)
    h = _rust.ContainerHandle(blob)
    assert h.names() == list(NAMES)
    assert [n for n, _ in h.decode_model()] == list(NAMES)
    rows = h.plane_summaries()
    assert list(dict.fromkeys(r[0] for r in rows)) == list(NAMES)
    assert _rust.list_tensor_names(blob) == list(NAMES)


def test_handle_decodes_the_same_bytes_in_both_layouts() -> None:
    legacy = _rust.ContainerHandle(_blob("legacy"))
    compact = _rust.ContainerHandle(_blob("compact"))
    tensors = _tensors()
    for name in NAMES:
        assert legacy.decode(name) == compact.decode(name) == _raw(tensors[name])
        assert legacy.shape(name) == compact.shape(name)
        assert legacy.info(name) == compact.info(name)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_handle_delta_hash(layout: str) -> None:
    blob, ref_hash = _delta_blob(layout)
    h = _rust.ContainerHandle(blob)
    assert h.delta_hash("d") == ref_hash
    assert _rust.decode_tensor_delta_hash(blob, "d") == ref_hash


def test_handle_borrows_read_only_buffers_and_copies_writable_ones(tmp_path) -> None:
    blob = _blob("compact")
    assert _rust.ContainerHandle(blob).zero_copy
    assert _rust.ContainerHandle(memoryview(blob)).zero_copy

    ba = bytearray(blob)
    h = _rust.ContainerHandle(ba)
    assert not h.zero_copy
    expected = h.decode("beta")
    # The copy is independent of later writes to the bytearray.
    ba[:] = bytes(len(ba))
    assert h.decode("beta") == expected

    path = tmp_path / "m.ptwm"
    path.write_bytes(blob)
    with (
        path.open("rb") as fh,
        mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm,
    ):
        h = _rust.ContainerHandle(mm)
        assert h.zero_copy
        assert h.decode("beta") == expected
        del h


def test_handle_copies_read_only_views_of_mutable_memory(tmp_path) -> None:
    # A read-only export does not make the owner immutable: the handle must
    # copy, so later writes to the owner cannot reach the reader.
    blob = _blob("compact")
    expected = _rust.ContainerHandle(blob).decode("beta")

    ba = bytearray(blob)
    h = _rust.ContainerHandle(memoryview(ba).toreadonly())
    assert not h.zero_copy
    ba[:] = bytes(len(ba))
    assert h.decode("beta") == expected
    assert "beta" in h.names()

    arr = np.frombuffer(bytearray(blob), dtype=np.uint8)
    view = arr.view()
    view.flags.writeable = False
    h = _rust.ContainerHandle(view)
    assert not h.zero_copy
    arr[:] = 0
    assert h.decode("beta") == expected

    # Views of immutable memory are still borrowed, through any chain of
    # memoryviews.
    assert _rust.ContainerHandle(memoryview(memoryview(blob))).zero_copy
    assert _rust.ContainerHandle(memoryview(blob).toreadonly()).zero_copy

    # A writable mmap is copied, also behind a read-only memoryview.
    path = tmp_path / "w.ptwm"
    path.write_bytes(blob)
    with path.open("r+b") as fh, mmap.mmap(fh.fileno(), 0) as mm:
        h = _rust.ContainerHandle(memoryview(mm).toreadonly())
        assert not h.zero_copy
        mm[:] = bytes(len(mm))
        assert h.decode("beta") == expected
        del h

    # A read-only mmap behind a memoryview is borrowed.
    path.write_bytes(blob)
    with (
        path.open("rb") as fh,
        mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm,
        memoryview(mm) as mv,
    ):
        h = _rust.ContainerHandle(mv)
        assert h.zero_copy
        assert h.decode("beta") == expected
        del h


def test_free_functions_copy_read_only_views_of_mutable_memory() -> None:
    blob = _blob("legacy")
    view = memoryview(bytearray(blob)).toreadonly()
    assert _rust.list_tensor_names(view) == list(NAMES)
    assert _rust.decode_tensor(view, "beta") == _rust.decode_tensor(blob, "beta")


def test_handle_keeps_the_buffer_alive() -> None:
    h = _rust.ContainerHandle(bytes(_blob("legacy")))
    # The only reference to the bytes object is the handle's buffer export.
    assert h.decode("alpha.bias") == _raw(_tensors()["alpha.bias"])


def test_handle_errors_surface_as_ptwm_error() -> None:
    with pytest.raises(Error):
        _rust.ContainerHandle(b"not a container")
    h = _rust.ContainerHandle(_blob("compact"))
    with pytest.raises(Error, match="not found"):
        h.decode("missing")
    with pytest.raises(Error, match="not found"):
        h.shape("missing")


# ── Python callers ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("layout", LAYOUTS)
def test_decompress_container_to_state_dict(layout: str) -> None:
    blob = _blob(layout)
    got = Decompressor().decompress_container_to_state_dict(blob)
    assert list(got) == list(NAMES)
    tensors = _tensors()
    for name in NAMES:
        assert got[name] == _raw(tensors[name])


@pytest.mark.parametrize("layout", LAYOUTS)
def test_tensor_index_opens_both_layouts(tmp_path, layout: str) -> None:
    path = tmp_path / "m.ptwm"
    path.write_bytes(_blob(layout))
    idx = TensorIndex.open(path)
    assert idx.layout == layout
    assert idx.names() == list(NAMES)
    tensors = _tensors()
    for name in NAMES:
        assert torch.equal(idx.get_tensor(name), tensors[name])


def test_tensor_index_rejects_a_compact_frame(tmp_path) -> None:
    path = tmp_path / "t.ptwm"
    path.write_bytes(
        Compressor(CompressionConfig(input_format=Format.TORCH)).compress(
            torch.randn(8)
        )
    )
    with pytest.raises(ValueError, match="not a .ptwm container"):
        TensorIndex.open(path)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_decode_ptwm_tensor_takes_bytes_or_a_handle(layout: str) -> None:
    blob = _blob(layout)
    h = _rust.ContainerHandle(blob)
    tensors = _tensors()
    for name in NAMES:
        assert torch.equal(decode_ptwm_tensor(blob, name), tensors[name])
        assert torch.equal(decode_ptwm_tensor(h, name), tensors[name])


@pytest.mark.parametrize("layout", LAYOUTS)
def test_hf_loader_reads_both_layouts(tmp_path, layout: str) -> None:
    path = tmp_path / "model.safetensors.ptwm"
    blob = _blob(layout)
    path.write_bytes(blob)
    assert _open_multi_tensor(blob) is not None
    state = _load_multi_tensor_ptwm(path)
    assert list(state) == list(NAMES)
    tensors = _tensors()
    for name in NAMES:
        assert torch.equal(state[name], tensors[name])
    # A single-tensor blob is not a multi-tensor container.
    single = Compressor(
        CompressionConfig(framing=Framing.CONTAINER, container_layout=layout)
    ).compress(b"\x00" * 64)
    assert _open_multi_tensor(single) is None
    assert _open_multi_tensor(b"\x89PTWS" + bytes(32)) is None


@pytest.mark.parametrize("layout", LAYOUTS)
def test_safe_open_mode_a_reads_both_layouts(tmp_path, layout: str) -> None:
    path = tmp_path / "model.ptwm"
    path.write_bytes(_blob(layout))
    tensors = _tensors()
    with SafeOpen(str(path), framework="pt") as f:
        assert f.keys() == list(NAMES)
        for name in NAMES:
            assert torch.equal(f.get_tensor(name), tensors[name])
