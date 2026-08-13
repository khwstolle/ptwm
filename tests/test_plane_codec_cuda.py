"""Test for the CUDA plane-codec PyO3 bridge.

`plane_codec_encode_cuda`/`plane_codec_decode_cuda` resolve a selector and a
codec id, then dispatch through `PlaneCodecCudaRouter`. No installed
`plane_codec` CUDA contribution exists yet, so with an empty installed
list, resolving even a real builtin canonical id (such as "identity")
still fails at the router lookup (no native contribution is registered for
it) before any device pointer or stream is ever touched. That makes this
assertion reachable without CUDA, without torch, and without any native
extension build: it exercises the router-miss branch of the bridge using
the buffer-protocol fallback (plain `bytes`/`bytearray`) for both `src`
and `dst`.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_extensions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Hermetic: keep this test from ever picking up a real installed
    # extension on the host running the suite.
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    return tmp_path


def test_unregistered_codec_raises() -> None:
    from ptwm._rust.plane_codec_cuda import plane_codec_decode_cuda

    with pytest.raises(ValueError, match="no plane_codec CUDA contribution registered"):
        plane_codec_decode_cuda("identity", "identity", b"", b"", bytearray(16), 0)


def test_encode_unregistered_codec_raises() -> None:
    from ptwm._rust.plane_codec_cuda import plane_codec_encode_cuda

    with pytest.raises(ValueError, match="no plane_codec CUDA contribution registered"):
        plane_codec_encode_cuda("identity", "identity", b"", b"", bytearray(16), 0)
