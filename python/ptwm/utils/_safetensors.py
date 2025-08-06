"""Utils for handling safetensors files."""

import json
from typing import TypedDict

import torch

METADATA_KEY = "ptwm_compressed_vectors"


COMPRESSION_METHOD = "HUFFMAN"
COMPRESSED_DTYPE = torch.uint8


class CompressedTensorInfo(TypedDict):
    """
    Metadata saved for a compress tensor.

    Attributes
    ----------
        dtype (str): The dtype of the underlying uncompressed tensor.
        shape (str): The shape of the underlying uncompressed tensor.
    """

    dtype: str
    shape: str


def build_compressed_tensor_info(
    uncompressed_tensor: torch.tensor,
) -> CompressedTensorInfo:
    """Return metadata to be saved for the respective compressed tensor."""
    dtype = str(uncompressed_tensor.dtype)
    dtype = dtype.removeprefix("torch.")

    return CompressedTensorInfo(dtype=dtype, shape=str(list(uncompressed_tensor.shape)))


def set_compressed_tensors_metadata(
    compressed_tensor_infos: dict[str, CompressedTensorInfo], metadata: dict[str, str]
) -> None:
    """Set file-level metadata on all compressed tensors."""
    if metadata:
        metadata[METADATA_KEY] = json.dumps(compressed_tensor_infos)


def get_compressed_tensors_metadata(
    metadata: dict[str, str],
) -> dict[str, CompressedTensorInfo]:
    """Retrieve file-level metadata on all compressed tensors."""
    if metadata:
        return json.loads(metadata.get(METADATA_KEY) or {})
    return {}
