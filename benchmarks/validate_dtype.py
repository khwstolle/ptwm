# ruff: noqa: T201
"""Dtype-breadth losslessness + ratio validation.

For each float dtype, compress a structured (clustered-exponent) tensor through
the *production* path (Method.AUTO -> chains_for(dtype) + full codec menu) with
streaming chunks (so the parallel chunk decode is exercised), decompress, and
verify a bit-exact roundtrip. Reports ratio per dtype. Needs no shards.

    python -m benchmarks.validate_dtype
"""

from __future__ import annotations

import torch
from ptwm import (
    CompressionConfig,
    Compressor,
    DecompressionConfig,
    Decompressor,
)
from ptwm._config import Format, Method

from benchmarks._common import raw_bytes

DTYPES = [
    ("bfloat16", torch.bfloat16),
    ("float16", torch.float16),
    ("float32", torch.float32),
    ("float8_e4m3fn", torch.float8_e4m3fn),
    ("float8_e5m2", torch.float8_e5m2),
]


def main() -> None:
    torch.manual_seed(0)
    print(f"\n{'dtype':<16s} {'MB':>7} {'ratio':>8} {'lossless':>9}")
    print("-" * 48)
    all_ok = True
    for name, dt in DTYPES:
        # Structured weights: small, clustered-exponent values (like trained
        # weights) so the exponent plane actually compresses.
        base = (torch.randn(4096, 4096) * 0.02).to(dt)
        orig = raw_bytes(base)
        ccfg = CompressionConfig(
            method=Method.AUTO,
            input_format=Format.TORCH,
            is_streaming=True,
            streaming_chunk=512 * 1024,
        )
        try:
            comp = Compressor(ccfg).compress(base)
            back = Decompressor(DecompressionConfig()).decompress(comp)
            if isinstance(back, torch.Tensor):
                back_bytes = raw_bytes(back)
            elif isinstance(back, bytes):
                back_bytes = back
            else:  # numpy array
                back_bytes = back.tobytes()
            lossless = back_bytes == orig
            ratio = len(comp) / len(orig)
            print(f"{name:<16s} {len(orig) / 1e6:7.1f} {ratio:8.4f} {lossless!s:>9}")
            if not lossless:
                all_ok = False
                print(
                    f"  !! LOSSLESS FAILED for {name} "
                    f"(orig {len(orig)} vs back {len(back_bytes)})"
                )
        except Exception as e:  # noqa: BLE001
            all_ok = False
            print(f"{name:<16s}  ERROR: {type(e).__name__}: {e}")
    print("-" * 48)
    print("ALL LOSSLESS" if all_ok else "!!! SOME FAILURES")


if __name__ == "__main__":
    main()
