# ruff: noqa: T201
"""Fixed per-output overhead of single-tensor ``Compressor.compress`` output.

Sweeps bf16 and fp32 tensors from 64 B to 1 MB, constant and random-normal,
and reports for each framing (compact frame and full container): output
size, non-payload bytes, ratio (raw / output), and zstd-19 on the same raw
bytes. Every output is decoded and checked bit-exact, including shape and
dtype. Needs no shards.

    python -m benchmarks.container_overhead

"Non-payload" is the output size minus the codec payload. For the compact
frame the payload includes inline codec state; for the container it is the
sum of the plane payload lengths. A stored compact frame counts the raw bytes
as payload.

``tests/ptwm/benchmarks/test_container_overhead.py`` runs the same sweep in
CI and asserts the size limits, so a regression in the fixed overhead fails
the build.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import zstandard
from ptwm import _rust
from ptwm._config import CompressionConfig, Format, Framing
from ptwm.core import Compressor, Decompressor

SIZES = (64, 256, 1024, 4096, 16384, 65536, 1 << 20)
DTYPES = (("bf16", torch.bfloat16), ("fp32", torch.float32))
KINDS = ("constant", "normal")


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


def non_payload_bytes(blob: bytes) -> int:
    """Output size minus codec payload (see the module docstring)."""
    if blob[:5] == b"\x89PTWS":
        return len(blob) - _rust.decode_compact(blob)[6]
    payload = sum(s[5] for s in _rust.list_plane_summaries(blob))
    return len(blob) - payload


def compress_checked(t: torch.Tensor, framing: Framing) -> bytes:
    """Compress ``t`` and assert a bit-exact round trip with shape and dtype."""
    blob = Compressor(
        CompressionConfig(input_format=Format.TORCH, framing=framing)
    ).compress(t)
    back = Decompressor().decompress(blob)
    assert isinstance(back, torch.Tensor)
    assert back.dtype == t.dtype, (back.dtype, t.dtype)
    assert back.shape == t.shape, (back.shape, t.shape)
    assert torch.equal(back.view(torch.uint8), t.view(torch.uint8))
    return blob


def measure(t: torch.Tensor, framing: Framing, dtype: str, kind: str) -> Row:
    raw = raw_bytes(t)
    blob = compress_checked(t, framing)
    z = zstandard.ZstdCompressor(level=19).compress(raw)
    return Row(
        framing=framing.name.lower(),
        dtype=dtype,
        kind=kind,
        raw=len(raw),
        out=len(blob),
        non_payload=non_payload_bytes(blob),
        zstd19=len(z),
    )


def sweep(
    framings: tuple[Framing, ...] = (Framing.COMPACT, Framing.CONTAINER),
) -> list[Row]:
    rows = []
    for framing in framings:
        for dname, dt in DTYPES:
            for nbytes in SIZES:
                for kind in KINDS:
                    t = make_tensor(dt, kind, nbytes)
                    rows.append(measure(t, framing, dname, kind))
    return rows


def main() -> None:
    header = (
        f"{'framing':<10} {'dtype':<5} {'data':<9} {'raw B':>9} {'out B':>9} "
        f"{'non-pay B':>9} {'ratio':>8} {'zstd19 B':>9} {'zstd ratio':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in sweep():
        print(
            f"{r.framing:<10} {r.dtype:<5} {r.kind:<9} {r.raw:>9,} {r.out:>9,} "
            f"{r.non_payload:>9,} {r.ratio:>8.3f} {r.zstd19:>9,} {r.zstd_ratio:>10.2f}"
        )

    print(
        f"\n{'framing':<10} {'rows':>5} {'raw B':>9} {'out B':>7} {'ratio':>7} {'zstd19 B':>9}"
    )
    for framing in (Framing.COMPACT, Framing.CONTAINER):
        for rows in (1, 8, 64):
            r = measure(tiled_rows(rows), framing, "bf16", "tiled")
            print(
                f"{r.framing:<10} {rows:>5} {r.raw:>9,} {r.out:>7,} "
                f"{r.ratio:>7.1f} {r.zstd19:>9,}"
            )


if __name__ == "__main__":
    main()
