"""Top-level driver: safetensors file → sharded ``.ptwm`` directory."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from safetensors import safe_open

from ..classify import (
    AuditLog,
    ClassifierChain,
    TensorClassifier,
    TensorRole,
)
from ..preprocessing._cache import (
    CacheEntry,
    load_cached_builders,
    save_cached_entries,
)
from ..preprocessing._chains import (
    CHAIN_BYTE_PASSTHROUGH_VALUE,
    ClassifierRole,
    chains_for,
)
from ..preprocessing._explorer import ExploreOptions, explore_chains
from ..sharding import (
    PtwmIndex,
    plan_shards,
    shard_filename,
    write_ptwm_shard,
)
from ..utils import DType

_log = logging.getLogger("ptwm.integrations.compress_safetensors")

__all__ = ["compress_safetensors_file"]

# Defensive import in case write_mode_b_shell is unavailable.
try:
    from ._safetensors import write_mode_b_shell as _write_mode_b_shell
except ImportError:
    _write_mode_b_shell = None  # type: ignore[assignment]


_DEFAULT_MAX_SHARD_SIZE = 5 * 1024**3


_BYTE_VIEW_DTYPES = {1: "uint8", 2: "uint16", 4: "uint32", 8: "uint64"}

_ROLE_MAP: dict[TensorRole, ClassifierRole] = {
    TensorRole.STANDARD: ClassifierRole.STANDARD,
    TensorRole.SCALE_BLOCK: ClassifierRole.SCALE_BLOCK,
    TensorRole.SCALE_GLOBAL: ClassifierRole.SCALE_GLOBAL,
    TensorRole.PACKED_VALUES: ClassifierRole.PACKED_VALUES,
}


def _read_safetensors(
    src: Path,
) -> tuple[list[tuple[str, bytes, Any, tuple[int, ...]]], int]:
    """Return (tensors, total_uncompressed_size).

    Uses ``framework="pt"`` so bf16 (which numpy cannot materialise) loads
    correctly. Each tensor is viewed as the equivalent unsigned-integer type
    of the same byte width to extract raw bytes; the original torch dtype is
    preserved in the returned tuple so downstream ``DType.from_dtype``
    recovers the correct spec_id.
    """
    tensors: list[tuple[str, bytes, Any, tuple[int, ...]]] = []
    total = 0
    with safe_open(str(src), framework="pt") as f:
        for name in f.keys():  # noqa: SIM118  (safe_open has no __iter__)
            t = f.get_tensor(name).contiguous()
            shape = tuple(t.shape)
            dtype = t.dtype
            byte_count = t.numel() * t.element_size()
            if byte_count == 0:
                raw = b""
            else:
                width = t.element_size()
                view_dtype_name = _BYTE_VIEW_DTYPES.get(width)
                if view_dtype_name is None:
                    msg = (
                        f"unsupported element_size={width} for tensor "
                        f"{name!r} dtype={dtype}"
                    )
                    raise ValueError(msg)
                target = getattr(torch, view_dtype_name)
                flat = t.flatten()
                if width == 1:
                    raw = bytes(flat.view(torch.uint8).cpu().numpy())
                else:
                    raw = bytes(flat.view(target).cpu().numpy().tobytes())
            tensors.append((name, raw, dtype, shape))
            total += byte_count
    return tensors, total


def _chain_bytes_for(
    dtype_code: int,
    role: TensorRole,
    shape: tuple[int, ...],
    explore_options: ExploreOptions | None = None,
    *,
    builder_cache: dict[tuple[int, TensorRole], list] | None = None,
    warned_passthrough: set[tuple[int, TensorRole]] | None = None,
    use_user_cache: bool = True,
    new_templates: dict[tuple[int, TensorRole], list[CacheEntry]] | None = None,
) -> list[bytes]:
    """Return candidate chain wire-byte blobs for a tensor.

    The candidate set is, in order: production chains for the
    ``(dtype_code, role)`` pair, builders loaded from the user-local cache
    (when ``use_user_cache`` is set), and — when *explore_options* is
    provided — fresh candidates from the explorer.

    When ``new_templates`` is supplied, freshly discovered explorer templates
    land in the mapping so the caller can persist them to the user cache
    after compression succeeds.
    """
    classifier_role = _ROLE_MAP[role]
    cache_key = (dtype_code, role)

    if builder_cache is not None and cache_key in builder_cache:
        builders = builder_cache[cache_key]
    else:
        builders = list(chains_for(dtype_code, classifier_role))
        if not builders:
            if warned_passthrough is not None and cache_key not in warned_passthrough:
                _log.warning(
                    "No production chain registered for dtype_code=0x%04X "
                    "role=%s; falling back to BytePassthrough (no compression). "
                    "Consider registering a chain or supplying a classifier rule.",
                    dtype_code,
                    role.name,
                )
                warned_passthrough.add(cache_key)
            builders = [CHAIN_BYTE_PASSTHROUGH_VALUE]

        if use_user_cache:
            builders = list(builders) + load_cached_builders(
                dtype_code, classifier_role
            )

        if explore_options is not None:
            discovered = []
            for b in explore_chains(dtype_code, classifier_role, explore_options):
                discovered.append(b)
                template = getattr(b, "template", None)
                # Templates only attach to explorer-yielded callables; harvest
                # them so the driver can persist them after a successful
                # compress.
                if template is not None and new_templates is not None:
                    new_templates.setdefault(cache_key, []).append(template)
            builders = list(builders) + discovered

        if builder_cache is not None:
            builder_cache[cache_key] = builders

    # Builders may decline a shape by returning None (e.g. spherical-normalize
    # only accepts 2D tensors); skip those.
    return [c.to_bytes() for b in builders if (c := b(list(shape))) is not None]


def compress_safetensors_file(
    src_path: str | Path,
    out_dir: str | Path,
    *,
    mode: Literal["a", "b"] = "a",
    classifier: TensorClassifier | ClassifierChain | None = None,
    max_shard_size: int = _DEFAULT_MAX_SHARD_SIZE,
    method_hint: int = 3,
    explore_options: ExploreOptions | None = None,
    use_user_cache: bool = True,
    fast: bool = False,
    fast_chunk: int = 512 * 1024,
) -> AuditLog:
    """Compress a safetensors file into a sharded ``.ptwm`` directory.

    Parameters
    ----------
    src_path:
        Path to the source ``.safetensors`` file.
    out_dir:
        Output directory. Created if it does not exist.
    mode:
        ``"a"`` — native ``.ptwm`` shards (default).
        ``"b"`` — ``.safetensors`` shell containing a ``.ptwm`` blob
        (requires ``write_mode_b_shell``).
    classifier:
        Optional classifier or chain. ``None`` → empty chain → every
        tensor classified ``STANDARD``.
    max_shard_size:
        Maximum uncompressed byte budget per shard (default 5 GiB).
    method_hint:
        Method hint forwarded to ``write_ptwm_shard``
        (1=HUFFMAN, 2=ZSTD, 3=MICROSCALE, 4=RANS, 5=IDENTITY).

    Returns
    -------
    AuditLog
        Per-tensor classification audit log.
    """
    src = Path(src_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tensors, total_size = _read_safetensors(src)
    archive_keys = tuple(name for name, _, _, _ in tensors)

    chain: ClassifierChain
    if classifier is None:
        chain = ClassifierChain([])
    elif isinstance(classifier, ClassifierChain):
        chain = classifier
    else:
        chain = ClassifierChain([classifier])

    audit = AuditLog()
    classified: list[tuple[str, bytes, np.dtype, tuple[int, ...], TensorRole]] = []
    for name, raw, dtype, shape in tensors:
        canonical = DType.from_dtype(dtype).dtype_str
        c = chain.classify(name, canonical, shape, archive_keys)
        audit.record(name, c)
        classified.append((name, raw, dtype, shape, c.role))

    # Shard planning uses uncompressed sizes as a proxy.
    sized = [(name, len(raw)) for name, raw, _, _, _ in classified]
    shards = plan_shards(sized, max_shard_size=max_shard_size)
    n_shards = len(shards)

    # Fast path: force rANS as the only per-plane codec (method_hint=4) +
    # chunked output, and use the dtype-default STANDARD chains rather than the
    # classifier's role-aware set. Forcing rANS skips the Huffman/Zstd/Identity
    # trial menu per plane while the transform-chain search still runs, so the
    # ratio is preserved (rANS wins the exponent plane anyway). For dtypes whose
    # STANDARD table has a single chain (FP8, FP4) this additionally engages the
    # skip+chunk+fuse fast path. Measured on a 32-core node: ~4x throughput on
    # pure-float at parity ratio (Mistral-7B BF16: 36.7 -> 148.1 MB/s, ratio
    # 0.6609 -> 0.6608) and ~8.5x on quant (gpt-oss-20b MXFP4: 55.7 -> 471.5
    # MB/s). Quant scale planes stay lossless but lose their Order1ScaleAC ratio
    # edge (+0.6% on MXFP4). Always lossless.
    eff_method = 4 if fast else method_hint
    eff_stream = fast_chunk if fast else None
    build_chains = not fast

    by_name = {
        name: (raw, dtype, shape, role) for name, raw, dtype, shape, role in classified
    }

    suffix = "ptwm" if mode == "a" else "safetensors"

    # Track discovered-chain reporting per (dtype_code, role) pair.
    _reported: set[tuple[int, TensorRole]] = set()
    # Cache builder lists per (dtype, role) to avoid re-running
    # explore_chains for every tensor of the same role; one-time passthrough
    # warning set.
    _builder_cache: dict[tuple[int, TensorRole], list] = {}
    _warned_passthrough: set[tuple[int, TensorRole]] = set()
    # Harvest discovered structural templates and write them to the user
    # cache only after the whole compress completes successfully — partial
    # writes from a failing run would poison the cache.
    _new_templates: dict[tuple[int, TensorRole], list[CacheEntry]] = {}

    def _build_chains_and_record(
        name: str,
        dtype_in: Any,
        shape: tuple[int, ...],
        role: TensorRole,
    ) -> list[bytes]:
        dt_enum = DType.from_dtype(dtype_in)
        dtype_code = int(dt_enum.code)
        candidate_bytes = _chain_bytes_for(
            dtype_code,
            role,
            shape,
            explore_options,
            builder_cache=_builder_cache,
            warned_passthrough=_warned_passthrough,
            use_user_cache=use_user_cache,
            new_templates=_new_templates,
        )
        if explore_options is not None and (dtype_code, role) not in _reported:
            _reported.add((dtype_code, role))
            # Record the first explored candidate for this (dtype, role) pair.
            prod_count = len(chains_for(dtype_code, _ROLE_MAP[role])) or 1
            if len(candidate_bytes) > prod_count:
                audit.record_discovered(
                    chain_bytes=candidate_bytes[prod_count],
                    dtype_code=dtype_code,
                    role=role,
                    n_candidates_tried=len(candidate_bytes) - prod_count,
                    sample_tensor_name=name,
                )
        return candidate_bytes

    if n_shards == 1 and mode == "a":
        # Single-file output: model.ptwm, no index needed.
        shard_path = out / "model.ptwm"
        shard_tensors = [
            (name, by_name[name][0], by_name[name][1], by_name[name][2])
            for name, _ in shards[0]
        ]
        shard_chains = (
            [
                _build_chains_and_record(
                    name, by_name[name][1], by_name[name][2], by_name[name][3]
                )
                for name, _ in shards[0]
            ]
            if build_chains
            else None
        )
        write_ptwm_shard(
            shard_path,
            shard_tensors,
            chains_per_tensor=shard_chains,
            method_hint=eff_method,
            streaming_chunk=eff_stream,
        )
        _persist_discovered(_new_templates, use_user_cache)
        return audit

    weight_map: dict[str, str] = {}
    compressed_sizes: list[int] = []

    for shard_idx, shard in enumerate(shards, start=1):
        shard_name = shard_filename(shard_idx, n_shards, suffix=suffix)
        shard_tensors = [
            (name, by_name[name][0], by_name[name][1], by_name[name][2])
            for name, _ in shard
        ]
        shard_chains = (
            [
                _build_chains_and_record(
                    name, by_name[name][1], by_name[name][2], by_name[name][3]
                )
                for name, _ in shard
            ]
            if build_chains
            else None
        )
        if mode == "a":
            write_ptwm_shard(
                out / shard_name,
                shard_tensors,
                chains_per_tensor=shard_chains,
                method_hint=eff_method,
                streaming_chunk=eff_stream,
            )
        else:
            if _write_mode_b_shell is None:
                msg = (
                    "Mode B requires write_mode_b_shell, which is unavailable "
                    "in this install"
                )
                raise NotImplementedError(msg)
            tmp_ptwm = out / (shard_name + ".tmp.ptwm")
            write_ptwm_shard(
                tmp_ptwm,
                shard_tensors,
                chains_per_tensor=shard_chains,
                method_hint=eff_method,
                streaming_chunk=eff_stream,
            )
            ptwm_bytes = tmp_ptwm.read_bytes()
            tmp_ptwm.unlink()
            _write_mode_b_shell(out / shard_name, ptwm_bytes)

        for name, _ in shard:
            weight_map[name] = shard_name
        compressed_sizes.append((out / shard_name).stat().st_size)

    if mode == "a":
        index_path = out / "model.ptwm.index.json"
    else:
        index_path = out / "model.safetensors.index.json"
    PtwmIndex(
        total_size=total_size,
        compressed_total_size=sum(compressed_sizes),
        weight_map=weight_map,
    ).write(index_path)
    _persist_discovered(_new_templates, use_user_cache)
    return audit


def _persist_discovered(
    discovered: dict[tuple[int, TensorRole], list[CacheEntry]],
    use_user_cache: bool,
) -> None:
    # Cache write happens after all shards land on disk so a crash mid-run
    # cannot leave the cache holding templates whose .ptwm output never
    # materialised.
    if not use_user_cache or not discovered:
        return
    for (dtype_code, role), entries in discovered.items():
        save_cached_entries(dtype_code, _ROLE_MAP[role], entries)
