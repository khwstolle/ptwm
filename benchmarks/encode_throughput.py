# ruff: noqa: T201
"""Single-tensor encode-throughput breakdown across PTWM configurations.

All configurations operate on the same in-memory BF16 tensor (no I/O during
timing). Single-threaded: ``RAYON_NUM_THREADS=1``.

  1. ptwm_bytesplit_huf ByteSplit-only + Huffman (shuffle+huffman baseline)
  2. ptwm_reorder_huf   BitReorder+ByteSplit + Huffman (our transforms)
  3. ptwm_production    BitReorder+ByteSplit + default AC menu (full system)

Rows 1 vs 2 isolate the bit-reorder transform cost; 2 vs 3 Huffman vs AC.

    RAYON_NUM_THREADS=1 python -m benchmarks.encode_throughput SHARD.safetensors
"""

from __future__ import annotations

import sys

from benchmarks._common import (
    bench,
    build_chain,
    compress_one,
    load_first_bf16_tensor,
)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: encode_throughput SHARD.safetensors", file=sys.stderr)
        sys.exit(1)

    raw, shape = load_first_bf16_tensor(sys.argv[1])
    print(f"\nRaw tensor: {len(raw) / 1e6:.1f} MB  shape={shape}\n")

    from ptwm.codecs import CodecId

    bytesplit_chain = build_chain(shape, False, True)
    reorder_chain = build_chain(shape, True, True)
    huffman_menu = [int(CodecId.Identity), int(CodecId.Huffman)]

    print(f"{'Config':<46s} {'Raw':>7}  {'Median':>8}  {'MB/s':>8}  {'Ratio'}")
    print("-" * 84)

    bench(
        "ptwm_bytesplit_huffman (shuffle+huffman)",
        lambda: compress_one(raw, shape, bytesplit_chain, huffman_menu),
        raw,
    )
    bench(
        "ptwm_reorder_huffman",
        lambda: compress_one(raw, shape, reorder_chain, huffman_menu),
        raw,
    )
    bench(
        "ptwm_production (default AC menu)",
        lambda: compress_one(raw, shape, reorder_chain, None),
        raw,
    )


if __name__ == "__main__":
    main()
