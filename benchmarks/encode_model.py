# ruff: noqa: T201
"""Full-model encode throughput: PTWM inter-tensor parallel encode.

The single-tensor bench is PTWM's worst case (framework floor, no inter-tensor
parallelism). Real models have hundreds of tensors. PTWM compresses them in ONE
``compress_model`` call (par_iter across tensors), using all cores — this
measures aggregate MB/s and overall ratio for a realistic workload.

    python -m benchmarks.encode_model SHARD.safetensors
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
        print("usage: encode_model SHARD.safetensors", file=sys.stderr)
        sys.exit(1)

    ncores = os.cpu_count() or 1
    rayon = os.environ.get("RAYON_NUM_THREADS", "(unset)")
    tensors = load_bf16_tensors(sys.argv[1])
    total = sum(len(r) for _, r, _ in tensors)
    print(
        f"\nLoaded {len(tensors)} BF16 tensors, {total / 1e9:.2f} GB  "
        f"ncores={ncores}  RAYON={rayon}\n"
    )

    from ptwm.codecs import CodecId

    chains, records = build_records(tensors)

    print(f"{'Config':<34s} {'Raw':>7}  {'Time':>8}  {'MB/s':>8}  {'Ratio'}")
    print("-" * 80)

    # PTWM: all tensors in one call (par_iter across tensors).
    timeit(
        "PTWM rANS forced (inter only)",
        lambda: compress_records(records, chains, method_hint=4),
        total,
        size_fn=len,  # type: ignore[arg-type]
    )
    timeit(
        "PTWM Huffman forced (inter only)",
        lambda: compress_records(
            records, chains, method_hint=3, menu=[int(CodecId.Huffman)]
        ),
        total,
        size_fn=len,  # type: ignore[arg-type]
    )
    # Combined inter+intra: forced codec + chunk triggers the skip path, so big
    # tensors chunk-encode across cores instead of one serial selection pass.
    mb = 1024 * 1024
    for chunk in (2 * mb, 512 * 1024):
        timeit(
            f"PTWM rANS inter+intra chunk={chunk // 1024}K",
            lambda chunk=chunk: compress_records(
                records, chains, method_hint=4, chunk=chunk
            ),
            total,
            size_fn=len,  # type: ignore[arg-type]
        )


if __name__ == "__main__":
    main()
