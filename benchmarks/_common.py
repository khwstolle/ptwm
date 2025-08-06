# ruff: noqa: T201
"""Shared helpers for the PTWM benchmark harness.

These benchmarks are *manual* harnesses (not part of the pytest suite): they
read real model shards and are tuned via environment variables. Everything
that more than one script needs lives here so the individual benchmarks stay
thin.

Environment:
  RAYON_NUM_THREADS  PTWM's Rust thread count (read once at pool init). Set to 1
                     for single-thread numbers, to the core count for scaling.
  PTWM_PROFILE=1     emit per-pass encode timings from the Rust side to stderr.
  PTWM_HF_CACHE      root of the HuggingFace cache holding the real shards;
                     defaults to the standard HF hub cache location.
"""

from __future__ import annotations

import json
import os
import resource
import struct
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

# Root of the HuggingFace shard cache (standard hub layout; override locally).
HF_CACHE = os.environ.get(
    "PTWM_HF_CACHE", str(Path("~/.cache/huggingface/hub").expanduser())
)

# Real model shard paths shared across the harness (standard HF hub cache layout).
QWEN_72B_PATH = (
    f"{HF_CACHE}/models--Qwen--Qwen2.5-72B-Instruct/snapshots/"
    "495f39366efef23836d0cfae4fbe635880d2be31/model-00015-of-00037.safetensors"
)
MISTRAL_7B_PATH = (
    f"{HF_CACHE}/models--mistralai--Mistral-7B-v0.1/snapshots/"
    "26bca36bde8333b5d7f72e9ed20ccda6a618af24/model-00002-of-00002.safetensors"
)
GPT_OSS_20B_PATH = (
    f"{HF_CACHE}/models--openai--gpt-oss-20b/snapshots/"
    "6cee5e81ee83917806bbde320786a8fb61efebee/model-00000-of-00002.safetensors"
)
NEMOTRON_FP8_PATH = (
    f"{HF_CACHE}/models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-FP8/snapshots/"
    "9f80cb76c26738e29c4d4d7a30fe882f938a25a6/model-00001-of-00026.safetensors"
)

# BF16 dtype code (DType.BFLOAT16.code) and BYTE input format used by the
# single-tensor _rust.compress_model record tuples below.
_BF16_CODE = 6
_FMT_BYTE = 0


# --------------------------------------------------------------------------- #
# Loading raw tensor bytes straight out of a .safetensors shard.
# --------------------------------------------------------------------------- #
def load_first_bf16_tensor(
    path: str, min_bytes: int = 50_000_000
) -> tuple[bytes, list[int]]:
    """Return ``(raw_bytes, shape)`` of the first BF16 tensor >= ``min_bytes``."""
    with Path(path).open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data_start = 8 + n
        for name, meta in sorted(header.items()):
            if name == "__metadata__" or meta.get("dtype") != "BF16":
                continue
            a, b = meta["data_offsets"]
            nb = b - a
            if nb < min_bytes:
                continue
            f.seek(data_start + a)
            raw = f.read(nb)
            print(f"Using tensor: {name}  shape={meta['shape']}  {nb / 1e6:.1f} MB")
            return raw, meta["shape"]
    msg = f"No BF16 tensor >= {min_bytes / 1e6:.0f} MB found in {path}"
    raise RuntimeError(msg)


def load_bf16_tensors(
    path: str, cap_gb: float = 3.0
) -> list[tuple[str, bytes, list[int]]]:
    """Return ``(name, raw_bytes, shape)`` for BF16 tensors up to ``cap_gb``."""
    out: list[tuple[str, bytes, list[int]]] = []
    total = 0
    cap = cap_gb * 1e9
    with Path(path).open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data_start = 8 + n
        for name, meta in sorted(header.items()):
            if name == "__metadata__" or meta.get("dtype") != "BF16":
                continue
            a, b = meta["data_offsets"]
            nb = b - a
            if total + nb > cap:
                continue
            f.seek(data_start + a)
            out.append((name, f.read(nb), meta["shape"]))
            total += nb
    if not out:
        msg = f"No BF16 tensors found in {path}"
        raise RuntimeError(msg)
    return out


# --------------------------------------------------------------------------- #
# Building preprocessing chains for the single-tensor _rust path.
# --------------------------------------------------------------------------- #
def build_chain(shape: list[int], use_bitreorder: bool, use_bytesplit: bool) -> bytes:
    """Serialize a BF16 preprocessing chain.

    Three configurations, isolating each transform's contribution:

    * ``(False, False)`` — Source -> BytePassthrough -> Raw terminal (no transform)
    * ``(False, True)``  — Source -> ByteSplit(2) (shuffle-only baseline)
    * ``(True, True)``   — Source -> BitReorderIeee16 -> ByteSplit(2) (production)
    """
    import ptwm.preprocessing._chains as ch
    from ptwm.utils import DType as _DT

    dtype_code = _DT.BFLOAT16.code
    if not use_bitreorder and not use_bytesplit:
        return ch.Chain(
            nodes=[
                ch.ChainNode(
                    op_id=ch._Op.Source,
                    params=ch._source_params(shape, ch._CHAIN_DTYPE[dtype_code]),
                ),
                ch.ChainNode(op_id=ch._Op.BytePassthrough),
            ],
            edges=[ch._simple_edge(0, 1)],
            terminals=[ch.TerminalRef(1, 0, ch._role_raw())],
        ).to_bytes()
    if not use_bitreorder and use_bytesplit:
        return ch.Chain(
            nodes=[
                ch.ChainNode(
                    op_id=ch._Op.Source,
                    params=ch._source_params(shape, ch._CHAIN_DTYPE[dtype_code]),
                ),
                ch.ChainNode(op_id=ch._Op.ByteSplit, params=bytes([2])),
            ],
            edges=[ch._simple_edge(0, 1)],
            terminals=[
                ch.TerminalRef(1, 0, ch._role_mantissa_byte(0, 2)),
                ch.TerminalRef(1, 1, ch._role_mantissa_byte(1, 2)),
            ],
        ).to_bytes()
    return ch.CHAIN_BF16_SPLIT(shape).to_bytes()


def build_records(
    tensors: list[tuple[str, bytes, list[int]]],
) -> tuple[list[list[bytes]], list[tuple]]:
    """Build ``(chains, records)`` for a full-model ``compress_model`` call.

    Every tensor gets the production BitReorder+ByteSplit chain and a BF16
    record tuple, matching what ``compress_one`` does for a single tensor.
    """
    chains = [[build_chain(shape, True, True)] for _, _, shape in tensors]
    records = [
        (name, _BF16_CODE, _FMT_BYTE, raw, shape, "bfloat16", None)
        for name, raw, shape in tensors
    ]
    return chains, records


# --------------------------------------------------------------------------- #
# Driving the Rust compressor.
# --------------------------------------------------------------------------- #
def compress_records(
    records: list[tuple],
    chains: list[list[bytes]],
    *,
    method_hint: int = 3,
    menu: list[int] | None = None,
    chunk: int | None = None,
) -> bytes:
    """Compress a batch of records in one ``compress_model`` call."""
    from ptwm import _rust

    return _rust.compress_model(
        records=records,
        chains_per_tensor=chains,
        emit_payload_hash=False,
        emit_plane_crc=False,
        method_hint=method_hint,
        streaming_chunk=chunk,
        codec_menu=menu,
    )


def compress_one(
    raw: bytes,
    shape: list[int],
    chain_bytes: bytes,
    menu: list[int] | None,
    *,
    method_hint: int = 3,
    chunk: int | None = None,
) -> bytes:
    """Compress a single BF16 tensor through the Rust path."""
    record = ("t", _BF16_CODE, _FMT_BYTE, raw, shape, "bfloat16", None)
    return compress_records(
        [record], [[chain_bytes]], method_hint=method_hint, menu=menu, chunk=chunk
    )


# --------------------------------------------------------------------------- #
# Timing.
# --------------------------------------------------------------------------- #
def bench(label: str, fn: Callable[[], bytes], raw: bytes, trials: int = 5) -> float:
    """Warm once, time ``trials`` runs, print median MB/s + ratio. Returns median."""
    fn()  # warm-up
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    median = sorted(times)[len(times) // 2]
    mb_s = len(raw) / 1e6 / median
    ratio = len(out) / len(raw)
    print(
        f"  {label:<46s} {len(raw) / 1e6:6.0f}MB  {median * 1e3:7.0f}ms  "
        f"{mb_s:7.0f} MB/s  ratio={ratio:.4f}"
    )
    return median


def timeit(
    label: str,
    fn: Callable[[], object],
    total_bytes: int,
    *,
    trials: int = 3,
    size_fn: Callable[[object], int] | None = None,
) -> None:
    """Warm once, time ``trials`` runs, print median MB/s (and ratio if size_fn)."""
    fn()  # warm-up
    times = []
    out: object = None
    for _ in range(trials):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    dt = sorted(times)[len(times) // 2]
    tail = ""
    if size_fn is not None:
        tail = f"  ratio={size_fn(out) / total_bytes:.4f}"
    print(
        f"  {label:<34s} {total_bytes / 1e9:.2f}GB  {dt * 1e3:8.0f}ms  "
        f"{total_bytes / 1e6 / dt:8.0f} MB/s{tail}"
    )


# --------------------------------------------------------------------------- #
# Misc.
# --------------------------------------------------------------------------- #
def peak_rss_gb() -> float:
    """Process peak RSS in GB (getrusage ru_maxrss is KiB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def raw_bytes(t: torch.Tensor) -> bytes:
    """Flatten a tensor to its raw little-endian bytes."""
    import torch

    return t.flatten().contiguous().view(torch.uint8).numpy().tobytes()
