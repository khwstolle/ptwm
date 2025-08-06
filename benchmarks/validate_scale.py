# ruff: noqa: T201
"""Scale, shape, and edge-case validation.

Modes (run as separate processes so ru_maxrss / RAYON settings are isolated):
  small  many tiny tensors (framework-overhead regime) — needs no shards
  large  a 72B-class BF16 shard (memory + inter-tensor scaling); honors
         RAYON_NUM_THREADS for the scaling sweep — needs the Qwen shard
  edge   single-tensor files, non-float dtypes, odd/degenerate shapes — no shards

Every mode verifies a bit-exact roundtrip. Peak RSS is read from
getrusage(RUSAGE_SELF).ru_maxrss at the end of the process.

    python -m benchmarks.validate_scale {small|large|edge}
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch
from ptwm import (
    CompressionConfig,
    Compressor,
    DecompressionConfig,
    Decompressor,
)
from ptwm._config import Format, Method
from ptwm.integrations import compress_safetensors_file
from ptwm.sharding import read_sharded_ptwm
from safetensors import safe_open
from safetensors.torch import save_file

from benchmarks._common import QWEN_72B_PATH, peak_rss_gb, raw_bytes


def file_roundtrip(state: dict[str, torch.Tensor], fast: bool, tag: str) -> None:
    d = Path(tempfile.mkdtemp())
    try:
        src = d / "m.safetensors"
        save_file(state, str(src))
        sz = src.stat().st_size
        out = d / "out"
        out.mkdir()
        t0 = time.perf_counter()
        compress_safetensors_file(src, out, mode="a", fast=fast)
        enc = time.perf_counter() - t0
        comp = sum(s.stat().st_size for s in out.glob("*.ptwm"))
        t1 = time.perf_counter()
        decoded = read_sharded_ptwm(out)
        dec = time.perf_counter() - t1
        ok = all(decoded.get(n) == raw_bytes(t) for n, t in state.items())
        print(
            f"  {tag:<26s} n={len(state):>5} {sz / 1e6:7.1f}MB ratio={comp / sz:.4f} "
            f"enc={sz / 1e6 / enc:7.1f} dec={sz / 1e6 / dec:7.1f} MB/s lossless={ok}"
        )
    finally:
        shutil.rmtree(d)


def shard_roundtrip(src: str, fast: bool, tag: str) -> None:
    """Like ``file_roundtrip``, but compresses ``src`` directly (no resident dict)."""
    sz = Path(src).stat().st_size
    d = Path(tempfile.mkdtemp())
    try:
        out = d / "out"
        out.mkdir()
        t0 = time.perf_counter()
        compress_safetensors_file(src, out, mode="a", fast=fast)
        enc = time.perf_counter() - t0
        comp = sum(s.stat().st_size for s in out.glob("*.ptwm"))
        t1 = time.perf_counter()
        decoded = read_sharded_ptwm(out)
        dec = time.perf_counter() - t1
        with safe_open(src, framework="pt") as f:
            names = list(f.keys())  # noqa: SIM118
            ok = all(decoded.get(n) == raw_bytes(f.get_tensor(n)) for n in names)
        print(
            f"  {tag:<26s} n={len(names):>5} {sz / 1e6:7.1f}MB ratio={comp / sz:.4f} "
            f"enc={sz / 1e6 / enc:7.1f} dec={sz / 1e6 / dec:7.1f} MB/s lossless={ok}"
        )
    finally:
        shutil.rmtree(d)


def run_small() -> None:
    print("[small] many tiny tensors (framework-overhead regime)")
    torch.manual_seed(0)
    state = {
        f"blk.{i}.w": (torch.randn(64, 64) * 0.02).to(torch.bfloat16)
        for i in range(2000)
    }
    file_roundtrip(state, fast=False, tag="2000x64x64 bf16 PROD")
    file_roundtrip(state, fast=True, tag="2000x64x64 bf16 FAST")
    print(f"  peak RSS = {peak_rss_gb():.2f} GB")


def run_large() -> None:
    rayon = os.environ.get("RAYON_NUM_THREADS", "(all)")
    print(f"[large] Qwen2.5-72B shard  RAYON_NUM_THREADS={rayon}")
    if not Path(QWEN_72B_PATH).exists():
        print("  MISSING shard")
        return
    shard_roundtrip(QWEN_72B_PATH, fast=False, tag="72B-shard BF16 PROD")
    shard_roundtrip(QWEN_72B_PATH, fast=True, tag="72B-shard BF16 FAST")
    print(f"  peak RSS = {peak_rss_gb():.2f} GB")


def _core_roundtrip(t: torch.Tensor, tag: str) -> None:
    dtype = str(t.dtype).replace("torch.", "")
    try:
        cfg = CompressionConfig(method=Method.AUTO, input_format=Format.TORCH)
        comp = Compressor(cfg).compress(t)
        back = Decompressor(DecompressionConfig()).decompress(comp)
        back_b = (
            raw_bytes(back)
            if isinstance(back, torch.Tensor)
            else (back if isinstance(back, bytes) else back.tobytes())
        )
        ok = back_b == raw_bytes(t)
        print(f"  {tag:<34s} shape={tuple(t.shape)} dtype={dtype:<14s} lossless={ok}")
    except Exception as e:  # noqa: BLE001
        print(
            f"  {tag:<34s} shape={tuple(t.shape)} dtype={dtype:<14s} "
            f"ERROR {type(e).__name__}: {e}"
        )


def run_edge() -> None:
    print("[edge] single-tensor files, non-float dtypes, odd/degenerate shapes")
    torch.manual_seed(0)
    file_roundtrip(
        {"only.weight": (torch.randn(512, 512) * 0.02).to(torch.bfloat16)},
        fast=True,
        tag="single-tensor file FAST",
    )
    _core_roundtrip(
        torch.randint(-(2**30), 2**30, (256, 256), dtype=torch.int32), "non-float int32"
    )
    _core_roundtrip(
        torch.randint(-128, 127, (256, 256), dtype=torch.int8), "non-float int8"
    )
    _core_roundtrip(
        torch.randint(0, 255, (256, 256), dtype=torch.uint8), "non-float uint8"
    )
    _core_roundtrip(
        torch.randint(-(2**60), 2**60, (128, 128), dtype=torch.int64), "non-float int64"
    )
    _core_roundtrip((torch.randn(1) * 0.02).to(torch.bfloat16), "odd shape (1,)")
    _core_roundtrip(
        (torch.randn(3, 7, 11) * 0.02).to(torch.bfloat16), "odd shape (3,7,11)"
    )
    _core_roundtrip(
        (torch.randn(1, 1, 1, 1) * 0.02).to(torch.float16), "odd shape (1,1,1,1)"
    )
    _core_roundtrip(torch.zeros(0, dtype=torch.bfloat16), "empty shape (0,)")
    _core_roundtrip(torch.zeros(0, 16, dtype=torch.float16), "empty shape (0,16)")
    print(f"  peak RSS = {peak_rss_gb():.2f} GB")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "small"
    print(f"\n=== scale-validate mode={mode}  ncores={os.cpu_count()} ===")
    {"small": run_small, "large": run_large, "edge": run_edge}[mode]()
    sys.stdout.flush()


if __name__ == "__main__":
    main()
