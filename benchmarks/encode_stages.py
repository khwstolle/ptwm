# ruff: noqa: T201
"""Stage-by-stage profiling of the PTWM encode pipeline.

Identifies where time is spent by timing each stage independently on the same
in-memory tensor. Run single-threaded: ``RAYON_NUM_THREADS=1``.

Stages: framework overhead (Identity), transform cost (BitReorder+ByteSplit),
codec cost (Huffman on raw bytes), trial-encode cost (with/without Identity in
the menu), and the production arithmetic-coding menu. A cProfile deep-dive on
the two most interesting cases follows.

    RAYON_NUM_THREADS=1 python -m benchmarks.encode_stages SHARD.safetensors
"""

from __future__ import annotations

import cProfile
import io
import pstats
import sys

from benchmarks._common import (
    bench,
    build_chain,
    compress_one,
    load_first_bf16_tensor,
)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: encode_stages SHARD.safetensors", file=sys.stderr)
        sys.exit(1)

    raw, shape = load_first_bf16_tensor(sys.argv[1])
    print(f"\nRaw: {len(raw) / 1e6:.1f} MB  shape={shape}\n")

    from ptwm.codecs import CodecId

    chain_passthrough = build_chain(shape, False, False)
    chain_bytesplit = build_chain(shape, False, True)
    chain_reorder = build_chain(shape, True, True)

    id_menu = [int(CodecId.Identity)]
    huf_menu = [int(CodecId.Identity), int(CodecId.Huffman)]
    huf_only = [int(CodecId.Huffman)]  # no Identity -> no trial-encode comparison

    print(f"{'Stage':<46s} {'Raw':>7}  {'Median':>8}  {'MB/s':>8}  {'Ratio'}")
    print("-" * 84)

    bench(
        "1. PTWM passthrough + Identity (framework only)",
        lambda: compress_one(raw, shape, chain_passthrough, id_menu),
        raw,
    )
    bench(
        "2. PTWM BitReorder+ByteSplit + Identity (transform cost)",
        lambda: compress_one(raw, shape, chain_reorder, id_menu),
        raw,
    )
    bench(
        "3. PTWM passthrough + Huffman (codec on raw bytes)",
        lambda: compress_one(raw, shape, chain_passthrough, huf_menu),
        raw,
    )
    bench(
        "4. PTWM passthrough + Huffman (no Identity in menu)",
        lambda: compress_one(raw, shape, chain_passthrough, huf_only),
        raw,
    )
    bench(
        "5. PTWM ByteSplit + Huffman (shuffle+huffman, with Identity)",
        lambda: compress_one(raw, shape, chain_bytesplit, huf_menu),
        raw,
    )
    bench(
        "6. PTWM BitReorder+ByteSplit + Huffman (with Identity)",
        lambda: compress_one(raw, shape, chain_reorder, huf_menu),
        raw,
    )
    bench(
        "7. PTWM BitReorder+ByteSplit + Huffman (no Identity)",
        lambda: compress_one(raw, shape, chain_reorder, huf_only),
        raw,
    )
    bench(
        "8. PTWM production (default AC menu)",
        lambda: compress_one(raw, shape, chain_reorder, None),
        raw,
    )

    print("\n=== cProfile: PTWM passthrough + Huffman (stage 3) ===")
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(3):
        compress_one(raw, shape, chain_passthrough, huf_menu)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(15)
    print(s.getvalue())

    print("=== cProfile: PTWM BitReorder+ByteSplit + Huffman (stage 6) ===")
    pr2 = cProfile.Profile()
    pr2.enable()
    for _ in range(3):
        compress_one(raw, shape, chain_reorder, huf_menu)
    pr2.disable()
    s2 = io.StringIO()
    pstats.Stats(pr2, stream=s2).sort_stats("cumulative").print_stats(15)
    print(s2.getvalue())


if __name__ == "__main__":
    main()
