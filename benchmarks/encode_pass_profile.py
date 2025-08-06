# ruff: noqa: T201
"""One-shot per-pass profile of compress_model on a full model (rANS inter+intra).

Run with ``PTWM_PROFILE=1`` and ``RAYON_NUM_THREADS=32``. The Rust side emits
per-pass timings to stderr; the second block (the timed run, after warm-up) is
the one to read.

    PTWM_PROFILE=1 RAYON_NUM_THREADS=32 \
        python -m benchmarks.encode_pass_profile SHARD.safetensors
"""

from __future__ import annotations

import sys
import time

from benchmarks._common import build_records, compress_records, load_bf16_tensors


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: encode_pass_profile SHARD.safetensors", file=sys.stderr)
        sys.exit(1)

    tensors = load_bf16_tensors(sys.argv[1])
    total = sum(len(r) for _, r, _ in tensors)
    chains, records = build_records(tensors)

    def run() -> bytes:
        return compress_records(records, chains, method_hint=4, chunk=512 * 1024)

    print("=== WARM RUN ===", file=sys.stderr)
    run()
    print("=== TIMED RUN ===", file=sys.stderr)
    t = time.perf_counter()
    out = run()
    dt = time.perf_counter() - t
    print(
        f"=== total {dt * 1e3:.0f}ms  {total / 1e6 / dt:.0f} MB/s  "
        f"ratio={len(out) / total:.4f} ===",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
