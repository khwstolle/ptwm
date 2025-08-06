# ruff: noqa: T201
"""Multi-threaded encode throughput at matched thread counts.

PTWM's thread count is ``RAYON_NUM_THREADS`` (read once at pool init), so this
measures PTWM at whatever the process launched with and reports that value.

Run twice (``RAYON_NUM_THREADS=1`` and ``=ncores``) for PTWM single- vs
multi-thread.

    RAYON_NUM_THREADS=N python -m benchmarks.encode_threading SHARD.safetensors
"""

from __future__ import annotations

import os
import sys

from benchmarks._common import (
    bench,
    build_chain,
    compress_one,
    load_first_bf16_tensor,
)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: encode_threading SHARD.safetensors", file=sys.stderr)
        sys.exit(1)

    ncores = os.cpu_count() or 1
    rayon = os.environ.get("RAYON_NUM_THREADS", "(unset → all cores)")
    raw, shape = load_first_bf16_tensor(sys.argv[1])
    print(f"\nRaw: {len(raw) / 1e6:.1f} MB  shape={shape}")
    print(f"ncores={ncores}  RAYON_NUM_THREADS={rayon}\n")

    from ptwm.codecs import CodecId

    chain_reorder = build_chain(shape, True, True)
    huf_only = [int(CodecId.Huffman)]  # leanest forced path, no trial-encode

    print(f"{'Config':<46s} {'Raw':>7}  {'Median':>8}  {'MB/s':>8}  {'Ratio'}")
    print("-" * 84)

    bench(
        f"PTWM production default menu (RAYON={rayon})",
        lambda: compress_one(raw, shape, chain_reorder, None),
        raw,
    )
    bench(
        f"PTWM lean reorder+split+Huffman (RAYON={rayon})",
        lambda: compress_one(raw, shape, chain_reorder, huf_only),
        raw,
    )


if __name__ == "__main__":
    main()
