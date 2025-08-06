"""Shared PTWM-blob → tensor decode helpers.

A single home for the raw-bytes → :class:`torch.Tensor` reconstruction logic
that was previously duplicated across the safetensors integration, the
HuggingFace loader, and :class:`~ptwm.random_access.TensorIndex` — including
the BF16-via-``uint16`` reinterpret path (NumPy has no native BF16) and the
FP8 / packed-FP4 paths that ``torch.frombuffer`` cannot ingest directly.
"""

from __future__ import annotations

import warnings

import torch

from .. import _rust
from ._torch import DType

__all__ = ["raw_to_tensor", "decode_ptwm_tensor"]


def raw_to_tensor(
    raw: bytes,
    shape: list[int] | tuple[int, ...] | None,
    dtype_name: str,
) -> torch.Tensor:
    """Reinterpret ``raw`` bytes as a tensor of ``dtype_name`` and ``shape``.

    Recovers the original torch dtype from the canonical ``dtype_name``. Dtypes
    that ``torch.frombuffer`` cannot ingest directly (BF16, FP8, packed FP4) are
    materialised through an unsigned-integer view of matching byte width. Falls
    back to a flat ``uint8`` view (with a warning) when the dtype is unknown.
    """
    # ``raw`` is a freshly-decompressed, read-only buffer that nothing else
    # aliases; pass it to ``frombuffer`` directly instead of taking a mutable
    # ``bytearray`` copy of the (potentially multi-GiB) tensor.
    buf = raw
    try:
        target = DType.from_dtype(dtype_name.lower()).torch_dtype
    except Exception:  # noqa: BLE001
        target = None

    fell_back = False
    if target is None:
        warnings.warn(
            f"weights: unrecognised dtype {dtype_name!r}; returning flat uint8 tensor",
            stacklevel=2,
        )
        tensor = torch.frombuffer(buf, dtype=torch.uint8)
        fell_back = True
    elif target == torch.bfloat16:
        tensor = torch.frombuffer(buf, dtype=torch.uint16).view(torch.bfloat16)
    else:
        try:
            tensor = torch.frombuffer(buf, dtype=target)
        except (TypeError, ValueError, RuntimeError):
            # FP8 / packed-FP4 and similar single-byte dtypes are not accepted
            # by frombuffer; reinterpret a uint8 view of equal byte width.
            tensor = torch.frombuffer(buf, dtype=torch.uint8)
            if torch.empty(0, dtype=target).element_size() == 1:
                tensor = tensor.view(target)
            else:
                fell_back = True

    if not shape:
        return tensor.reshape(())
    try:
        return tensor.reshape(shape)
    except RuntimeError:
        # A sub-byte / unrepresentable dtype recovered as a flat uint8 view
        # cannot honour the logical element shape; return the flat tensor.
        if fell_back:
            return tensor
        raise


def decode_ptwm_tensor(blob: bytes, name: str) -> torch.Tensor:
    """Decode a single named tensor from a ``.ptwm`` container blob.

    Falls back to a flat ``uint8`` tensor (with a warning) when the container
    has no shape metadata for ``name``.
    """
    raw = bytes(_rust.decode_tensor(blob, name))
    info = _rust.decode_tensor_shape(blob, name)
    if info is None:
        warnings.warn(
            f"weights: no shape metadata for tensor {name!r}; "
            "returning flat uint8 tensor",
            stacklevel=2,
        )
        return torch.frombuffer(raw, dtype=torch.uint8)
    shape, dtype_name = info
    return raw_to_tensor(raw, shape, dtype_name)
