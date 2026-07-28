"""End-to-end test for hardware_backend dispatch through the real native
build path.

Installs the ref_hardware_backend native (CPU-passthrough) reference
contribution as a local bundle, then round-trips
hardware_backend_dispatch_decode_cuda through the full stack: Python ->
PyO3 -> HardwareBackendRouter -> NativeExtension -> the actual reference
module (not a hand-rolled stand-in).

Requires the reference module to already be built:
    cd extensions/ref_hardware_backend/rust
    cargo build --release

Both calls pass an explicit `policy`: `hardware_class = "cpu"` is a
capability the router denies by default (HardwareBackendRouter::new's
HostPolicy has an empty available_hardware list), so a ResolvedPolicy
that grants "cpu" is required for resolution to succeed at all. The
policy can only be produced by writing a policy file to disk and
resolving it; there is no in-memory constructor.

`hardware_backend_dispatch_decode_cuda`'s compressed/out arguments are
ordinarily `torch.Tensor` objects read via `__dlpack__()`. This test
passes plain `bytes`/`bytearray` instead, which fall back to the
buffer-protocol path added to that function for exactly this case: it
lets this CPU-only smoke test exercise the real dispatch path (router,
native ABI, dlopen, FFI call) without requiring `torch` or constructing
a fake CUDA-shaped DLPack tensor. Real CUDA tensors still go through
`__dlpack__` normally.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

REF_DIR = Path(__file__).parents[3] / "extensions" / "ref_hardware_backend" / "rust"
REF_SO_CANDIDATES = list((REF_DIR / "target" / "release").glob("*.so")) + list(
    (REF_DIR / "target" / "release").glob("*.dylib")
)
CANONICAL_ID = "blake3:" + "00" * 32  # matches ref_hardware_backend/rust/manifest.toml's id


@pytest.fixture
def isolated_extensions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    return tmp_path


def _cpu_policy(tmp_path: Path):
    from ptwm._rust.policy import PolicyFile

    pol = tmp_path / "policy.toml"
    pol.write_text('[capabilities]\navailable_hardware = ["cpu"]\n', encoding="utf-8")
    return PolicyFile.load(str(pol)).resolve([])


@pytest.mark.interop
@pytest.mark.skipif(not REF_SO_CANDIDATES, reason="ref_hardware_backend native .so not built")
def test_hardware_backend_cpu_passthrough_round_trip(isolated_extensions: Path) -> None:
    from ptwm._rust.ext import ext_install
    from ptwm._rust.hardware import (
        hardware_backend_cuda_stream_handle,
        hardware_backend_dispatch_decode_cuda,
    )

    src = isolated_extensions / "src"
    src.mkdir(parents=True, exist_ok=True)
    shutil.copy(REF_DIR / "manifest.toml", src / "manifest.toml")
    shutil.copy(REF_SO_CANDIDATES[0], src / f"ref-hardware-backend{REF_SO_CANDIDATES[0].suffix}")

    bundle_dir = ext_install(str(src))
    assert bundle_dir

    policy = _cpu_policy(isolated_extensions)

    stream = hardware_backend_cuda_stream_handle(CANONICAL_ID, 0, policy=policy)
    assert stream != 0

    # CPU-passthrough scaffold reinterprets its "device" pointers as host
    # pointers, so this test uses plain bytes/bytearray rather than a real
    # CUDA tensor, via hardware_backend_dispatch_decode_cuda's
    # buffer-protocol fallback for objects without __dlpack__. Real-GPU
    # coverage is a separate gpu-marked test (Task 30).
    payload = b"hardware backend dispatch smoke test payload"
    out = bytearray(len(payload))
    n = hardware_backend_dispatch_decode_cuda(
        CANONICAL_ID, CANONICAL_ID, b"", payload, out, 0, policy=policy
    )
    assert n == len(payload)
    assert bytes(out) == payload


@pytest.mark.interop
@pytest.mark.skipif(not REF_SO_CANDIDATES, reason="ref_hardware_backend native .so not built")
def test_hardware_backend_unknown_canonical_id_raises(isolated_extensions: Path) -> None:
    from ptwm._rust.hardware import hardware_backend_cuda_stream_handle

    # No policy needed: an unknown canonical id is rejected by
    # HardwareBackendRouter::resolve's install lookup before the
    # capability check runs, so the router's default-deny HostPolicy
    # never enters into it.
    with pytest.raises(ValueError, match="no hardware_backend registered"):
        hardware_backend_cuda_stream_handle("blake3:" + "ff" * 32, 0)
