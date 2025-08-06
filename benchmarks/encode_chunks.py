# ruff: noqa: T201
"""Chunk-parallel scaling: does enabling ``streaming_chunk`` let PTWM scale?

Measures forced-rANS and forced-Huffman over a chunk-size sweep, reporting BOTH
MB/s and ratio at the process's ``RAYON_NUM_THREADS``. Run twice (1 and ncores)
to read scaling. Smaller chunks scale better but hurt ratio (per-chunk tables) —
both are printed so the speed/ratio tradeoff is visible.

    RAYON_NUM_THREADS=N python -m benchmarks.encode_chunks SHARD.safetensors
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
        print("usage: encode_chunks SHARD.safetensors", file=sys.stderr)
        sys.exit(1)

    ncores = os.cpu_count() or 1
    rayon = os.environ.get("RAYON_NUM_THREADS", "(unset)")
    raw, shape = load_first_bf16_tensor(sys.argv[1])
    print(f"\nRaw: {len(raw) / 1e6:.1f} MB  ncores={ncores}  RAYON={rayon}\n")

    from ptwm.codecs import CodecId

    chain = build_chain(shape, True, True)  # bit-reorder + byte-split
    huf = [int(CodecId.Huffman)]
    mb = 1024 * 1024
    chunks = [None, 8 * mb, 2 * mb, 512 * 1024, 128 * 1024, 32 * 1024]

    print(f"{'Config':<46s} {'Raw':>7}  {'Median':>8}  {'MB/s':>8}  {'Ratio'}")
    print("-" * 84)

    # Forced rANS (method_hint=4) — best fast-codec ratio.
    for ch in chunks:
        lbl = "None" if ch is None else f"{ch // 1024}K"
        bench(
            f"rANS  chunk={lbl}",
            lambda ch=ch: compress_one(
                raw, shape, chain, None, method_hint=4, chunk=ch
            ),
            raw,
            trials=3,
        )

    # Forced Huffman (method_hint=3 + 1-item allow list). NOTE: unlike the rANS
    # rows above, method_hint=3 doesn't set forced_codec in the Rust binding
    # (only 2/4/5 do), so these rows never hit the compressor's skip_trial_encode
    # fast path — each pays for an extra whole-plane trial-encode that the rANS
    # rows skip. Read within-codec chunk scaling here, not cross-codec MB/s.
    for ch in chunks:
        lbl = "None" if ch is None else f"{ch // 1024}K"
        bench(
            f"Huffman  chunk={lbl}",
            lambda ch=ch: compress_one(raw, shape, chain, huf, method_hint=3, chunk=ch),
            raw,
            trials=3,
        )


if __name__ == "__main__":
    main()
