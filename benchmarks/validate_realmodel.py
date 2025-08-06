# ruff: noqa: T201
"""Real-model ratio / speed / losslessness validation across dtypes.

For each real model shard: compress via the file integration (production chains
+ classifier), measure ratio and encode speed, decode each .ptwm shard via the
parallel decode_model, measure decode speed, and verify a bit-exact roundtrip
against the original safetensors.

Shard paths are resolved under ``PTWM_HF_CACHE`` (defaults to the standard
HF hub cache location). The output directory is ``PTWM_BENCH_OUT`` (default
``/tmp/ptwm_bench_out``).

    python -m benchmarks.validate_realmodel
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

from ptwm import _rust
from ptwm.classify._heuristic import HeuristicClassifier
from ptwm.integrations import compress_safetensors_file
from safetensors import safe_open

from benchmarks._common import (
    GPT_OSS_20B_PATH,
    MISTRAL_7B_PATH,
    NEMOTRON_FP8_PATH,
    QWEN_72B_PATH,
    raw_bytes,
)

OUT_ROOT = Path(os.environ.get("PTWM_BENCH_OUT", "/tmp/ptwm_bench_out"))

MODELS = [
    (MISTRAL_7B_PATH, "Mistral-7B BF16", None),
    (GPT_OSS_20B_PATH, "gpt-oss-20b MXFP4", HeuristicClassifier()),
    (NEMOTRON_FP8_PATH, "Nemotron FP8", HeuristicClassifier()),
    (QWEN_72B_PATH, "Qwen2.5-72B BF16", None),
]


def bench(shard: str, label: str, classifier) -> None:
    if not Path(shard).exists():
        print(f"{label:<18s}  MISSING shard")
        return
    shard_size = Path(shard).stat().st_size
    out = OUT_ROOT / label.replace(" ", "_")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    try:
        t0 = time.perf_counter()
        compress_safetensors_file(shard, out, mode="a", classifier=classifier)
        enc_t = time.perf_counter() - t0
        shards = sorted(out.glob("*.ptwm"))
        comp_size = sum(s.stat().st_size for s in shards)
        ratio = comp_size / shard_size

        decoded: dict[str, bytes] = {}
        t1 = time.perf_counter()
        for s in shards:
            blob = s.read_bytes()
            for name, b in _rust.decode_model(blob):
                decoded[name] = bytes(b)
        dec_t = time.perf_counter() - t1

        lossless = True
        nfail = 0
        with safe_open(shard, framework="pt") as f:
            for name in f.keys():  # noqa: SIM118
                if decoded.get(name) != raw_bytes(f.get_tensor(name)):
                    lossless = False
                    nfail += 1
        extra = f"  ({nfail} mismatched)" if nfail else ""
        print(
            f"{label:<18s} {shard_size / 1e9:5.1f}GB  ratio={ratio:.4f}  "
            f"enc={shard_size / 1e6 / enc_t:6.0f} MB/s  "
            f"dec={shard_size / 1e6 / dec_t:6.0f} MB/s  lossless={lossless}{extra}"
        )
    finally:
        if out.exists():
            shutil.rmtree(out)


def main() -> None:
    ncores = os.cpu_count()
    print(
        f"\nncores={ncores}  RAYON={os.environ.get('RAYON_NUM_THREADS', '(unset)')}\n"
    )
    print(
        f"{'model':<18s} {'size':>6}  {'ratio':>11}  {'encode':>11}  "
        f"{'decode':>11}  lossless"
    )
    print("-" * 86)
    for shard, label, clf in MODELS:
        try:
            bench(shard, label, clf)
        except Exception as e:  # noqa: BLE001
            print(f"{label:<18s}  ERROR: {type(e).__name__}: {e}")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
