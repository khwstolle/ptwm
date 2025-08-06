"""HuggingFace Transformers integration for loading ``.ptwm`` checkpoints.

The integration exposes two small surfaces:

* :func:`patch_transformers` installs one patch on
  :func:`transformers.modeling_utils.load_state_dict` so that any path ending
  in ``.ptwm`` decompresses transparently before handing off to transformers'
  native loader. The original loader handles every file format transformers
  already knows about (safetensors, pytorch-bin, zip, …), so this integration
  does not replicate that dispatch logic.

  It also installs a best-effort patch on
  :func:`transformers.modeling_utils._get_resolved_checkpoint_files` so that
  :meth:`~transformers.PreTrainedModel.from_pretrained` discovers
  ``model.ptwm.index.json`` + ``.ptwm`` shards stored in a local directory.

* :func:`materialize_hf_cache` walks an HF cache snapshot directory and
  decompresses every ``.ptwm`` file into its native name. Run it once to
  turn a ``.safetensors.ptwm`` snapshot into a plain ``.safetensors``
  snapshot that :func:`~transformers.PreTrainedModel.from_pretrained`
  consumes with no patches.
"""

from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any

from packaging.version import Version

from .. import _rust
from ..core import Decompressor

try:
    import transformers  # noqa: F401  # imported for presence check
    from transformers import modeling_utils
except ImportError:
    modeling_utils = None  # type: ignore[assignment]

# Minimum supported transformers release. Older versions pass a different
# argument list to ``load_state_dict`` and are not worth maintaining against.
_MIN_TRANSFORMERS_VERSION = Version("4.45.2")


def _transformers_version() -> Version:
    """Installed transformers version, parsed via :mod:`packaging`."""
    try:
        return Version(_pkg_version("transformers"))
    except PackageNotFoundError as e:
        msg = "HuggingFace Transformers is not installed"
        raise ImportError(msg) from e


_COMPRESSED_SUFFIXES = (".ptwm",)


def _decompress_alongside(path: Path) -> Path:
    """Decompress ``path`` to the same directory under its native name.

    ``model.safetensors.ptwm`` becomes ``model.safetensors``. Returns the
    decompressed path. Idempotent: a pre-existing output is returned without
    re-decompressing.

    Supports BYTE-mode blobs only: the output is the decompressed byte stream
    verbatim, which is what HF cache consumers (safetensors, pytorch-bin, …)
    expect. ``materialize_hf_cache`` is meaningful only for BYTE-format
    snapshots, so the Decompressor's container handling must yield raw bytes.
    """
    suffix_match = next(
        (s for s in _COMPRESSED_SUFFIXES if path.name.endswith(s)), None
    )
    if suffix_match is None:
        msg = f"{path} does not have a .ptwm suffix"
        raise ValueError(msg)
    out_path = path.with_name(path.name.removesuffix(suffix_match))
    if out_path.exists():
        return out_path

    compressed = path.read_bytes()
    decompressed = Decompressor().decompress(compressed)
    if not isinstance(decompressed, bytes | bytearray | memoryview):
        msg = (
            f"{path} was compressed with a non-BYTE input_format; "
            "materialize_hf_cache only supports BYTE-format blobs"
        )
        raise ValueError(msg)
    out_path.write_bytes(bytes(decompressed))
    return out_path


def _is_multi_tensor(blob: bytes) -> bool:
    """Return True if ``blob`` is a multi-tensor v3 ``.ptwm`` container.

    A multi-tensor v3 container has a non-empty tensor name list, and none of
    the names is ``"_"`` (the BYTE-format sentinel for single-tensor blobs).
    """
    try:
        names = _rust.list_tensor_names(blob)
    except Exception:  # noqa: BLE001
        return False
    return bool(names) and "_" not in names


def _load_multi_tensor_ptwm(path: Path) -> dict[str, Any]:
    """Decompress a multi-tensor v3 ``.ptwm`` container to a state dict.

    Reads ``path``, lists every tensor name via the embedded index, decodes
    each tensor's raw bytes and shape metadata, and reconstructs the original
    :class:`torch.Tensor` objects.

    BF16 round-trip: the Rust layer stores BF16 data as raw ``uint16`` bytes;
    recovery uses the shared :func:`ptwm.utils.decode_ptwm_tensor` helper.
    """
    from ..utils import decode_ptwm_tensor

    blob = path.read_bytes()
    names = _rust.list_tensor_names(blob)
    return {name: decode_ptwm_tensor(blob, name) for name in names}


def materialize_hf_cache(
    snapshot_dir: str | os.PathLike[str],
    *,
    remove_source: bool = False,
) -> list[Path]:
    """Decompress every ``.ptwm`` file under ``snapshot_dir``.

    Recursively walks the directory, decompressing each ``*.ptwm`` file into
    its sibling without the suffix (``model.safetensors.ptwm`` →
    ``model.safetensors``). Skips files whose decompressed form already
    exists, so repeat runs are cheap.

    Parameters
    ----------
    snapshot_dir:
        Any directory; typically an HF cache snapshot
        (``~/.cache/huggingface/hub/models--<org>--<name>/snapshots/<rev>``).
    remove_source:
        If ``True``, delete each ``.ptwm`` file once its decompressed sibling
        is on disk. Defaults to ``False`` (keeps both).

    Returns
    -------
    list of :class:`pathlib.Path`
        Paths of the decompressed files, in traversal order.
    """
    snapshot = Path(snapshot_dir)
    if not snapshot.is_dir():
        msg = f"{snapshot_dir!s} is not a directory"
        raise NotADirectoryError(msg)
    decompressed: list[Path] = []
    for suffix in _COMPRESSED_SUFFIXES:
        for src_path in snapshot.rglob(f"*{suffix}"):
            out_path = _decompress_alongside(src_path)
            if remove_source:
                src_path.unlink()
            decompressed.append(out_path)
    return decompressed


def discover_ptwm_shards(model_dir: str | os.PathLike[str]) -> list[Path]:
    """Return the list of ``.ptwm`` shards in ``model_dir`` (or empty list).

    Uses ``model.ptwm.index.json`` if present; otherwise looks for a single
    ``model.ptwm``.

    Parameters
    ----------
    model_dir:
        Directory to search for ``.ptwm`` shards.

    Returns
    -------
    list of :class:`pathlib.Path`
        Shard paths in sorted order, or an empty list if none are found.
    """
    root = Path(model_dir)
    index_path = root / "model.ptwm.index.json"
    if index_path.exists():
        from ..sharding import PtwmIndex

        idx = PtwmIndex.read(index_path)
        return [root / shard for shard in sorted(set(idx.weight_map.values()))]
    single = root / "model.ptwm"
    if single.exists():
        return [single]
    return []


def _patch_resolution(mu: Any) -> None:
    """Best-effort: extend transformers' file discovery to recognise .ptwm.

    Patches :func:`transformers.modeling_utils._get_resolved_checkpoint_files`
    so that a local directory containing ``model.ptwm.index.json`` or
    ``model.ptwm`` returns shard paths instead of an empty list.

    If the target function is absent in the installed transformers version,
    emits a warning and falls back to :func:`materialize_hf_cache`.
    """
    target_name = "_get_resolved_checkpoint_files"
    target = getattr(mu, target_name, None)
    if target is None:
        import warnings

        warnings.warn(
            f"patch_transformers: transformers.modeling_utils.{target_name} "
            "not found; .ptwm discovery via from_pretrained is disabled. "
            "Use materialize_hf_cache() as a fallback.",
            stacklevel=2,
        )
        return

    def patched(*args: Any, **kwargs: Any) -> Any:
        # Check for .ptwm shards *before* calling the original to short-
        # circuit the OSError that transformers raises when it finds neither
        # model.safetensors nor pytorch_model.bin. The call uses keyword
        # args; positional arg[0] is the fallback.
        try:
            candidates: list[Any] = list(args)
            pmnop = kwargs.get("pretrained_model_name_or_path")
            if pmnop is not None:
                candidates.insert(0, pmnop)
            for cand_raw in candidates:
                if isinstance(cand_raw, str | Path):
                    cand = Path(cand_raw)
                    if cand.is_dir():
                        shards = discover_ptwm_shards(cand)
                        if shards:
                            return [str(p) for p in shards], None
        except Exception:  # noqa: BLE001
            pass
        return target(*args, **kwargs)

    setattr(mu, target_name, patched)


def patch_transformers() -> None:
    """Install the ``.ptwm``-aware ``load_state_dict`` replacement.

    Patches :func:`transformers.modeling_utils.load_state_dict` so that any
    transformers path funnelling through it — including
    :meth:`PreTrainedModel.from_pretrained` with an explicit local ``.ptwm``
    path — decompresses transparently.

    Also installs a best-effort patch on
    :func:`transformers.modeling_utils._get_resolved_checkpoint_files` so
    ``from_pretrained`` discovers ``model.ptwm.index.json`` + ``.ptwm`` shards
    in local directories. If the patch target is absent in the installed
    transformers version, emits a warning and skips the patch
    (:func:`materialize_hf_cache` still works as a fallback).

    For HF cache snapshots that store ``*.safetensors.ptwm`` instead of
    ``*.safetensors``, run :func:`materialize_hf_cache` on the snapshot
    directory once; transformers' own file discovery in ``from_pretrained``
    then finds the decompressed files by their native names with no further
    patching.
    """
    if modeling_utils is None:
        msg = "HuggingFace Transformers is not installed"
        raise ImportError(msg)

    installed = _transformers_version()
    if installed < _MIN_TRANSFORMERS_VERSION:
        msg = (
            f"patch_transformers requires transformers >= {_MIN_TRANSFORMERS_VERSION}, "
            f"installed version is {installed}"
        )
        raise RuntimeError(msg)

    original_load_state_dict = modeling_utils.load_state_dict

    def custom_load_state_dict(
        checkpoint_file: str | os.PathLike[str],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        path = Path(checkpoint_file)
        if any(path.name.endswith(s) for s in _COMPRESSED_SUFFIXES):
            blob = path.read_bytes()
            if _is_multi_tensor(blob):
                # Multi-tensor v3 .ptwm: decode via tensor index and return
                # the state dict directly; the original load_state_dict does
                # not understand this format.
                return _load_multi_tensor_ptwm(path)
            # Single-tensor BYTE-format blob (legacy path): decompress to a
            # sibling file and hand off to the original loader.
            path = _decompress_alongside(path)
        return original_load_state_dict(str(path), *args, **kwargs)

    modeling_utils.load_state_dict = custom_load_state_dict
    _patch_resolution(modeling_utils)


__all__ = [
    "_load_multi_tensor_ptwm",
    "discover_ptwm_shards",
    "materialize_hf_cache",
    "patch_transformers",
]
