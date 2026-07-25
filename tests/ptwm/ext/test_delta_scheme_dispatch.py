"""End-to-end test for delta_scheme dispatch through the Phase 1 plumbing.

Installs the real `ref_delta_scheme` WASM reference contribution
(built from `extensions/ref_delta_scheme/rust`) as a local bundle, then
round-trips `delta_scheme_encode`/`delta_scheme_decode` through the full
stack: Python -> PyO3 -> DeltaSchemeRouter -> WasmExtension -> the actual
reference module (not a hand-rolled WAT stand-in).

Requires the reference module to already be built:
    cd extensions/ref_delta_scheme/rust
    cargo build --target wasm32-wasip1 --release
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

REF_WASM = (
    Path(__file__).parents[3]
    / "extensions"
    / "ref_delta_scheme"
    / "rust"
    / "target"
    / "wasm32-wasip1"
    / "release"
    / "ref_delta_scheme.wasm"
)

CANONICAL_ID = "blake3:" + "00" * 32


@pytest.fixture
def isolated_extensions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    return tmp_path


@pytest.mark.skipif(not REF_WASM.exists(), reason="ref_delta_scheme.wasm not built")
def test_delta_scheme_round_trip_through_real_wasm_reference(
    isolated_extensions: Path,
) -> None:
    from ptwm._rust.delta_scheme import delta_scheme_decode, delta_scheme_encode
    from ptwm._rust.ext import ext_install

    src = isolated_extensions / "src"
    src.mkdir(parents=True, exist_ok=True)
    (src / "manifest.toml").write_text(
        f"""\
[bundle]
name = "ref-delta-scheme"
version = "0.1.0"
author_pubkey = "ed25519:{"00" * 32}"
description = "test install of the delta_scheme reference contribution"

[[contributions]]
id = "{CANONICAL_ID}"
label = "io.ptwm.ref.delta_scheme"
kind = "delta_scheme"
abi_version = 1
lifecycle = "thread"
flavors = ["wasm"]
capabilities = {{ determinism = true, hardware_class = "cpu" }}
""",
        encoding="utf-8",
    )
    shutil.copy(REF_WASM, src / "ref_delta_scheme.wasm")

    bundle_dir = ext_install(str(src))
    assert bundle_dir

    base = b"base tensor bytes (ignored by the passthrough reference)"
    target = b"the real target tensor payload"

    delta = delta_scheme_encode(CANONICAL_ID, base, target, len(target) + 64)
    assert delta == target  # reference encode copies target unchanged

    recon = delta_scheme_decode(CANONICAL_ID, base, delta, len(target))
    assert recon == target  # reference decode copies delta unchanged


@pytest.mark.skipif(not REF_WASM.exists(), reason="ref_delta_scheme.wasm not built")
def test_delta_scheme_unknown_canonical_id_raises(isolated_extensions: Path) -> None:
    from ptwm._rust.delta_scheme import delta_scheme_encode

    with pytest.raises(ValueError, match="no delta_scheme registered"):
        delta_scheme_encode("blake3:" + "ff" * 32, b"base", b"target", 64)
