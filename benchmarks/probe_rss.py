# ruff: noqa: T201
"""Clean peak-RSS probe — one operation per process, no resident dicts.

Each mode does exactly one thing and reports getrusage peak RSS, so the number
reflects the library, not the test harness. The shard size is the reference: a
healthy compress peaks near shard+compressed; a healthy decode near the decoded
model size. Multiples of that point at parallel-buffer amplification.

  enc <fast|prod>  compress_safetensors_file streaming the Qwen shard from disk
  dec              read_sharded_ptwm on a pre-compressed dir (RSS_DIR env)

Shard path resolves under ``PTWM_HF_CACHE`` (defaults to the standard HF
hub cache location).

    python -m benchmarks.probe_rss enc fast
    RSS_DIR=/path/to/out python -m benchmarks.probe_rss dec
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from ptwm.integrations import compress_safetensors_file
from ptwm.sharding import read_sharded_ptwm

from benchmarks._common import QWEN_72B_PATH, peak_rss_gb


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: probe_rss <enc <fast|prod>|dec>", file=sys.stderr)
        sys.exit(1)

    mode = sys.argv[1]
    rayon = os.environ.get("RAYON_NUM_THREADS", "(all)")
    if mode == "enc":
        if len(sys.argv) < 3:
            print("usage: probe_rss enc <fast|prod>", file=sys.stderr)
            sys.exit(1)
        shard_gb = Path(QWEN_72B_PATH).stat().st_size / 1e9
        fast = sys.argv[2] == "fast"
        out = Path(tempfile.mkdtemp())
        t = time.perf_counter()
        compress_safetensors_file(QWEN_72B_PATH, out, mode="a", fast=fast)
        dt = time.perf_counter() - t
        comp = sum(s.stat().st_size for s in out.glob("*.ptwm")) / 1e9
        peak = peak_rss_gb()
        print(
            f"enc {'fast' if fast else 'prod':<4s} RAYON={rayon:<5s} "
            f"shard={shard_gb:.2f}GB comp={comp:.2f}GB peakRSS={peak:.2f}GB "
            f"ratioPeak={peak / shard_gb:.1f}x t={dt:.1f}s"
        )
        shutil.rmtree(out)
    elif mode == "dec":
        rss_dir = os.environ["RSS_DIR"]
        t = time.perf_counter()
        state = read_sharded_ptwm(rss_dir)
        dt = time.perf_counter() - t
        decoded_gb = sum(len(v) for v in state.values()) / 1e9
        peak = peak_rss_gb()
        print(
            f"dec      RAYON={rayon:<5s} decoded={decoded_gb:.2f}GB "
            f"peakRSS={peak:.2f}GB ratioPeak={peak / decoded_gb:.1f}x t={dt:.1f}s"
        )
    else:
        print(
            f"usage: probe_rss <enc <fast|prod>|dec>  (unknown mode {mode!r})",
            file=sys.stderr,
        )
        sys.exit(1)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
