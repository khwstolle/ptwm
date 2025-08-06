"""Tests for :mod:`ptwm.integrations._hf` that don't require transformers.

:func:`materialize_hf_cache` is pure Python (only needs the compressor) and
is testable in isolation. The ``patch_transformers`` monkey-patch requires a real
transformers install to exercise end-to-end; that's covered by the
integration-marked tests in :mod:`tests.ptwm.integrations.test_safetensors`
and manual smoke tests.
"""

from __future__ import annotations

import pytest
from ptwm import (
    CompressionConfig,
    Compressor,
    Format,
    materialize_hf_cache,
)


@pytest.fixture
def snapshot_with_ptwm(tmp_path):
    """Fake HF cache snapshot with two compressed weight files."""
    snapshot = tmp_path / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    compressor = Compressor(CompressionConfig(input_format=Format.BYTE))
    (snapshot / "model.safetensors.ptwm").write_bytes(
        compressor.compress(b"\x00" * 1024)
    )
    (snapshot / "config.json").write_text('{"model_type": "test"}')
    subdir = snapshot / "extra"
    subdir.mkdir()
    (subdir / "tokenizer.bin.ptwm").write_bytes(compressor.compress(b"\xff" * 2048))
    return snapshot


def test_materialize_decompresses_every_ptwm_file(snapshot_with_ptwm):
    result = materialize_hf_cache(snapshot_with_ptwm)
    names = {p.name for p in result}
    assert names == {"model.safetensors", "tokenizer.bin"}
    for path in result:
        assert path.exists()
        assert path.is_file()


def test_materialize_is_idempotent(snapshot_with_ptwm):
    first = materialize_hf_cache(snapshot_with_ptwm)
    second = materialize_hf_cache(snapshot_with_ptwm)
    # Same set of decompressed paths on the second run.
    assert {str(p) for p in first} == {str(p) for p in second}


def test_materialize_leaves_ptwm_by_default(snapshot_with_ptwm):
    materialize_hf_cache(snapshot_with_ptwm)
    ptwm_files = list(snapshot_with_ptwm.rglob("*.ptwm"))
    assert len(ptwm_files) == 2


def test_materialize_remove_source_deletes_ptwm(snapshot_with_ptwm):
    materialize_hf_cache(snapshot_with_ptwm, remove_source=True)
    ptwm_files = list(snapshot_with_ptwm.rglob("*.ptwm"))
    assert ptwm_files == []
    # Decompressed files still exist.
    assert (snapshot_with_ptwm / "model.safetensors").exists()
    assert (snapshot_with_ptwm / "extra" / "tokenizer.bin").exists()


def test_materialize_errors_on_missing_directory(tmp_path):
    with pytest.raises(NotADirectoryError):
        materialize_hf_cache(tmp_path / "does-not-exist")


def test_materialize_leaves_non_ptwm_files_alone(snapshot_with_ptwm):
    materialize_hf_cache(snapshot_with_ptwm)
    # config.json was never .ptwm, must survive untouched.
    assert (snapshot_with_ptwm / "config.json").read_text() == '{"model_type": "test"}'


def test_materialize_rejects_non_byte_format_ptwm(tmp_path):
    """A .ptwm compressed with TORCH format must be rejected, not crashed on."""
    import torch

    snapshot = tmp_path / "snapshots" / "torchfmt"
    snapshot.mkdir(parents=True)
    tensor = torch.zeros(16, dtype=torch.float32)
    compressor = Compressor(CompressionConfig(input_format=Format.TORCH))
    (snapshot / "model.safetensors.ptwm").write_bytes(compressor.compress(tensor))

    with pytest.raises(ValueError, match="input_format"):
        materialize_hf_cache(snapshot)


def test_materialize_rejects_non_byte_format_ptwm_numpy(tmp_path):
    """A .ptwm compressed with NUMPY format must be rejected identically."""
    import numpy as np

    snapshot = tmp_path / "snapshots" / "numpyfmt"
    snapshot.mkdir(parents=True)
    arr = np.zeros(16, dtype=np.float32)
    compressor = Compressor(CompressionConfig(input_format=Format.NUMPY))
    (snapshot / "ptwm.bin.ptwm").write_bytes(compressor.compress(arr))

    with pytest.raises(ValueError, match="input_format"):
        materialize_hf_cache(snapshot)
