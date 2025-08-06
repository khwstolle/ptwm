# ruff: noqa: T201
"""Fast-encode validation — fast=True vs production on real model shards.

For each shard: compress via the file integration twice (production full-menu,
then fast=True forced-rANS+chunked), measure ratio + encode speed for each, and
verify a bit-exact roundtrip of the fast-mode output. The fast path must stay
lossless on every dtype; the headline is the encode speedup at parity ratio on
pure-float models.

Shard paths resolve under ``PTWM_HF_CACHE``; output dir under ``PTWM_BENCH_OUT``
(default ``/tmp/ptwm_fast_bench_out``).

    python -m benchmarks.encode_fast
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

from ptwm.classify._heuristic import HeuristicClassifier
from ptwm.integrations import compress_safetensors_file
from ptwm.sharding import read_sharded_ptwm
from safetensors import safe_open

from benchmarks._common import GPT_OSS_20B_PATH, MISTRAL_7B_PATH, raw_bytes

OUT_ROOT = Path(os.environ.get("PTWM_BENCH_OUT", "/tmp/ptwm_fast_bench_out"))

MODELS = [
    (MISTRAL_7B_PATH, "Mistral-7B BF16", None),
    (GPT_OSS_20B_PATH, "gpt-oss-20b MXFP4", HeuristicClassifier()),
]


def encode(shard: str, label: str, classifier, fast: bool) -> tuple[float, float, Path]:
    out = OUT_ROOT / (label.replace(" ", "_") + ("_fast" if fast else "_prod"))
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    sz = Path(shard).stat().st_size
    t0 = time.perf_counter()
    compress_safetensors_file(shard, out, mode="a", classifier=classifier, fast=fast)
    dt = time.perf_counter() - t0
    comp = sum(s.stat().st_size for s in out.glob("*.ptwm"))
    return comp / sz, sz / 1e6 / dt, out


def verify(shard: str, out: Path) -> tuple[bool, int]:
    decoded = read_sharded_ptwm(out)
    ok, nfail = True, 0
    with safe_open(shard, framework="pt") as f:
        for name in f.keys():  # noqa: SIM118
            if decoded.get(name) != raw_bytes(f.get_tensor(name)):
                ok = False
                nfail += 1
    return ok, nfail


def bench(shard: str, label: str, classifier) -> None:
    if not Path(shard).exists():
        print(f"{label:<18s}  MISSING shard")
        return
    sz = Path(shard).stat().st_size
    pr_out = f_out = None
    try:
        pr_ratio, pr_mbs, pr_out = encode(shard, label, classifier, fast=False)
        f_ratio, f_mbs, f_out = encode(shard, label, classifier, fast=True)
        lossless, nfail = verify(shard, f_out)
        extra = f"  ({nfail} mismatched)" if nfail else ""
        print(f"{label:<18s} {sz / 1e9:4.1f}GB")
        print(f"  PROD  ratio={pr_ratio:.4f}  enc={pr_mbs:7.1f} MB/s")
        print(
            f"  FAST  ratio={f_ratio:.4f}  enc={f_mbs:7.1f} MB/s  "
            f"speedup={f_mbs / pr_mbs:4.1f}x  lossless={lossless}{extra}"
        )
        sys.stdout.flush()
    finally:
        if pr_out is not None and pr_out.exists():
            shutil.rmtree(pr_out)
        if f_out is not None and f_out.exists():
            shutil.rmtree(f_out)


def main() -> None:
    print(
        f"\nncores={os.cpu_count()}  RAYON={os.environ.get('RAYON_NUM_THREADS', '(unset)')}\n"
    )
    for shard, label, clf in MODELS:
        try:
            bench(shard, label, clf)
        except Exception as e:  # noqa: BLE001
            print(f"{label:<18s}  ERROR: {type(e).__name__}: {e}")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
