# ruff: noqa: T201
"""Decode/load throughput: PTWM decode_model (parallel).

Compress the model once (PTWM forced-rANS + 512K chunks), then time
decompression of the whole model. Verifies a lossless roundtrip before
timing. Run at ``RAYON_NUM_THREADS=1`` and ``=ncores`` to read scaling.

    RAYON_NUM_THREADS=N python -m benchmarks.decode_throughput SHARD.safetensors
"""

from __future__ import annotations

import os
import sys

from benchmarks._common import (
    build_records,
    compress_records,
    load_bf16_tensors,
    timeit,
)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: decode_throughput SHARD.safetensors", file=sys.stderr)
        sys.exit(1)

    from ptwm import _rust

    ncores = os.cpu_count() or 1
    rayon = os.environ.get("RAYON_NUM_THREADS", "(unset)")
    tensors = load_bf16_tensors(sys.argv[1])
    total = sum(len(r) for _, r, _ in tensors)
    orig = {name: raw for name, raw, _ in tensors}
    print(
        f"\nLoaded {len(tensors)} BF16 tensors, {total / 1e9:.2f} GB  "
        f"ncores={ncores}  RAYON={rayon}"
    )

    chains, records = build_records(tensors)
    blob = compress_records(records, chains, method_hint=4, chunk=512 * 1024)
    print(f"PTWM blob: {len(blob) / 1e9:.2f} GB  ratio={len(blob) / total:.4f}\n")

    print(f"{'Config':<34s} {'Raw':>7}  {'Median':>8}  {'MB/s'}")
    print("-" * 64)

    # PTWM decode_model: verify lossless, then time.
    decoded = {name: bytes(b) for name, b in _rust.decode_model(blob)}
    ok = len(decoded) == len(orig) and all(decoded[n] == orig[n] for n in orig)
    print(f"  PTWM decode_model lossless roundtrip: {ok}")
    if not ok:
        msg = "PTWM decode roundtrip FAILED"
        raise SystemExit(msg)
    timeit("PTWM decode_model (parallel)", lambda: _rust.decode_model(blob), total)


if __name__ == "__main__":
    main()
