# ruff: noqa: T201
r"""Fixed overhead of PTWM output: single-tensor and multi-tensor.

Single tensor: sweeps bf16 and fp32 tensors from 64 B to 1 MB, constant and
random-normal, through ``Compressor.compress`` in three output formats:

- ``ptws``: the default compact frame (``Framing.COMPACT``);
- ``ptwc``: a full container in the compact layout (``Framing.CONTAINER``,
  ``ContainerLayout.COMPACT``, the default layout);
- ``ptwm``: a full container in the legacy layout (``Framing.CONTAINER``,
  ``ContainerLayout.LEGACY``), the format readers before v1.2 need.

For each it reports output size, non-payload bytes, ratio (raw / output) and
zstd-19 on the same raw bytes. Every output is decoded and checked bit-exact,
including shape and dtype.

Multi-tensor: writes ``n`` small tensors into one container with
``compress_ptwm_blob`` in both layouts and reports the non-payload bytes per
tensor (index entry, tensor record, plane records, and the file's fixed
sections divided by ``n``). Every tensor is decoded and checked.

Needs no shards::

    python -m benchmarks.container_overhead

"Non-payload" is the output size minus the data a codec produced: plane
payloads, inline codec state and shared (prelude) codec state. The same rule
applies to every format. A stored compact frame counts the raw bytes as
payload.

``tests/ptwm/benchmarks/test_container_overhead.py`` runs the same sweeps in
CI and asserts the size limits, so a regression in the fixed overhead fails
the build.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import zstandard
from ptwm import _rust
from ptwm._config import CompressionConfig, ContainerLayout, Format, Framing
from ptwm._magic import container_kind
from ptwm.core import Compressor, Decompressor
from ptwm.sharding._writer import compress_ptwm_blob

SIZES = (64, 256, 1024, 4096, 16384, 65536, 1 << 20)
DTYPES = (("bf16", torch.bfloat16), ("fp32", torch.float32))
KINDS = ("constant", "normal")

# (framing, container layout) for each single-tensor output format, in report
# order. The layout applies to a full container: ``Framing.CONTAINER``, or the
# container the compact framing falls back to.
FORMATS: tuple[tuple[Framing, ContainerLayout], ...] = (
    (Framing.COMPACT, ContainerLayout.COMPACT),
    (Framing.CONTAINER, ContainerLayout.COMPACT),
    (Framing.CONTAINER, ContainerLayout.LEGACY),
)

ShapeFn = Callable[[int], tuple[int, ...]]
"""Shape of tensor ``i`` in a multi-tensor case."""

# Tensor counts of the multi-tensor sweep.
MULTI_COUNTS = (1, 8, 64, 512)


def same_shape(_i: int) -> tuple[int, ...]:
    """Shape of tensor ``i`` when every tensor has the same shape."""
    return (64,)


def distinct_shapes(i: int) -> tuple[int, ...]:
    """Shape of tensor ``i`` when every tensor has its own shape."""
    return (64 + i,)


# Multi-tensor cases: (label, dtype, shape of tensor i).
MULTI_CASES = (
    ("fp32 same", "float32", same_shape),
    ("fp32 distinct", "float32", distinct_shapes),
    ("bf16 same", "bfloat16", same_shape),
)


@dataclass(frozen=True)
class Row:
    framing: str
    dtype: str
    kind: str
    raw: int
    out: int
    non_payload: int
    zstd19: int

    @property
    def ratio(self) -> float:
        return self.raw / self.out

    @property
    def zstd_ratio(self) -> float:
        return self.raw / self.zstd19

    @property
    def payload(self) -> int:
        return self.out - self.non_payload


@dataclass(frozen=True)
class MultiRow:
    layout: str
    case: str
    hashes: bool
    n: int
    out: int
    non_payload: int

    @property
    def per_tensor(self) -> float:
        return self.non_payload / self.n


def make_tensor(
    dtype: torch.dtype, kind: str, nbytes: int, seed: int = 1
) -> torch.Tensor:
    """A 1-D tensor of ``nbytes`` raw bytes: constant 0.5, or seeded normal."""
    n = nbytes // torch.empty(0, dtype=dtype).element_size()
    if kind == "constant":
        return torch.full((n,), 0.5, dtype=dtype)
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, generator=g).to(dtype)


def tiled_rows(rows: int, seed: int = 1) -> torch.Tensor:
    """``rows`` copies of one random 1,024-element bf16 vector."""
    g = torch.Generator().manual_seed(seed)
    row = torch.randn(1024, generator=g).to(torch.bfloat16)
    return row.unsqueeze(0).repeat(rows, 1).contiguous()


def raw_bytes(t: torch.Tensor) -> bytes:
    return t.contiguous().view(torch.uint8).numpy().tobytes()


def codec_data_bytes(blob: bytes) -> int:
    """Bytes of ``blob`` that a codec produced (see the module docstring).

    For a container: every plane payload, every inline codec state and every
    shared prelude state, read from the transcoder's manifest projection.
    """
    if container_kind(blob) == "ptws":
        # Payload plus inline state; the raw size for a stored frame.
        return _rust.decode_compact(blob)[6]
    registry_json, tensors, members = _rust.explode_ptwm(blob)
    # Both manifest forms keep the shared states under "registry".
    shared = json.loads(registry_json)["registry"].get("shared_state", [])
    total = sum(len(members[s["payload"]]) for s in shared)
    for _name, _key, tensor_json in tensors:
        for plane in json.loads(tensor_json)["planes"]:
            total += plane["comp_len"]
            if plane.get("state_b64"):
                total += len(base64.b64decode(plane["state_b64"]))
            elif plane.get("state_payload"):
                total += len(members[plane["state_payload"]])
    return total


def non_payload_bytes(blob: bytes) -> int:
    """Output size minus codec data (see the module docstring)."""
    return len(blob) - codec_data_bytes(blob)


def compress_checked(
    t: torch.Tensor,
    framing: Framing,
    layout: ContainerLayout = ContainerLayout.COMPACT,
) -> bytes:
    """Compress ``t`` and assert a bit-exact round trip with shape and dtype.

    ``layout`` is the layout of a full container (``Framing.CONTAINER`` or the
    compact-frame fallback).
    """
    blob = Compressor(
        CompressionConfig(
            input_format=Format.TORCH, framing=framing, container_layout=layout
        )
    ).compress(t)
    back = Decompressor().decompress(blob)
    assert isinstance(back, torch.Tensor)
    assert back.dtype == t.dtype, (back.dtype, t.dtype)
    assert back.shape == t.shape, (back.shape, t.shape)
    assert torch.equal(back.view(torch.uint8), t.view(torch.uint8))
    return blob


def measure(
    t: torch.Tensor,
    framing: Framing,
    dtype: str,
    kind: str,
    layout: ContainerLayout = ContainerLayout.COMPACT,
) -> Row:
    """Compress ``t`` once; ``Row.framing`` names the format written."""
    raw = raw_bytes(t)
    blob = compress_checked(t, framing, layout)
    z = zstandard.ZstdCompressor(level=19).compress(raw)
    return Row(
        framing=container_kind(blob) or "?",
        dtype=dtype,
        kind=kind,
        raw=len(raw),
        out=len(blob),
        non_payload=non_payload_bytes(blob),
        zstd19=len(z),
    )


def sweep(
    formats: tuple[tuple[Framing, ContainerLayout], ...] = FORMATS,
) -> list[Row]:
    rows = []
    for framing, layout in formats:
        for dname, dt in DTYPES:
            for nbytes in SIZES:
                for kind in KINDS:
                    t = make_tensor(dt, kind, nbytes)
                    rows.append(measure(t, framing, dname, kind, layout))
    return rows


def multi_tensors(
    n: int, dtype: str, shape_of: ShapeFn, seed: int = 7
) -> list[tuple[str, bytes, str, tuple[int, ...]]]:
    """``n`` random-normal tensors named like transformer bias vectors."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        shape = shape_of(i)
        values = torch.from_numpy(rng.standard_normal(shape, dtype=np.float32))
        t = values.to(torch.bfloat16 if dtype == "bfloat16" else torch.float32)
        out.append((f"model.layers.{i:03d}.attn.q.bias", raw_bytes(t), dtype, shape))
    return out


def measure_multi(
    case: str,
    dtype: str,
    shape_of: ShapeFn,
    n: int,
    layout: ContainerLayout,
    hashes: bool = True,
) -> MultiRow:
    """Write ``n`` tensors into one container and check every tensor."""
    tensors = multi_tensors(n, dtype, shape_of)
    blob = compress_ptwm_blob(
        tensors,
        emit_payload_hash=hashes,
        emit_plane_crc=hashes,
        container_layout=layout,
    )
    handle = _rust.ContainerHandle(blob)
    assert handle.names() == [name for name, *_ in tensors]
    for name, raw, _dtype, _shape in tensors:
        assert handle.decode(name) == raw, name
    return MultiRow(
        layout=container_kind(blob) or "?",
        case=case,
        hashes=hashes,
        n=n,
        out=len(blob),
        non_payload=non_payload_bytes(blob),
    )


def multi_sweep(counts: tuple[int, ...] = MULTI_COUNTS) -> list[MultiRow]:
    rows = []
    for case, dtype, shape_of in MULTI_CASES:
        for hashes in (True, False):
            for layout in (ContainerLayout.COMPACT, ContainerLayout.LEGACY):
                for n in counts:
                    rows.append(measure_multi(case, dtype, shape_of, n, layout, hashes))
    return rows


def main() -> None:
    header = (
        f"{'format':<7} {'dtype':<5} {'data':<9} {'raw B':>9} {'out B':>9} "
        f"{'non-pay B':>9} {'ratio':>8} {'zstd19 B':>9} {'zstd ratio':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in sweep():
        print(
            f"{r.framing:<7} {r.dtype:<5} {r.kind:<9} {r.raw:>9,} {r.out:>9,} "
            f"{r.non_payload:>9,} {r.ratio:>8.3f} {r.zstd19:>9,} {r.zstd_ratio:>10.2f}"
        )

    print(
        f"\n{'format':<7} {'rows':>5} {'raw B':>9} {'out B':>7} {'ratio':>7} {'zstd19 B':>9}"
    )
    for framing, layout in FORMATS:
        for rows in (1, 8, 64):
            r = measure(tiled_rows(rows), framing, "bf16", "tiled", layout)
            print(
                f"{r.framing:<7} {rows:>5} {r.raw:>9,} {r.out:>7,} "
                f"{r.ratio:>7.1f} {r.zstd19:>9,}"
            )

    header = (
        f"\n{'layout':<6} {'case':<14} {'hash+crc':<8} {'tensors':>7} "
        f"{'out B':>9} {'non-pay B':>9} {'B/tensor':>8}"
    )
    print(header)
    print("-" * (len(header) - 1))
    for m in multi_sweep():
        print(
            f"{m.layout:<6} {m.case:<14} {'on' if m.hashes else 'off':<8} {m.n:>7} "
            f"{m.out:>9,} {m.non_payload:>9,} {m.per_tensor:>8.1f}"
        )


if __name__ == "__main__":
    main()
