"""Pin the chain-blob fixture the Rust descriptor tests read.

``crates/ptwm-core/tests/fixtures/production_chains.bin`` holds every
production chain builder's output (and the explorer's first candidates) for
a set of shapes. The Rust test
``chain::runtime::tests::symbolic_descriptors_match_forward_on_production_chains``
checks that the symbolic descriptor walk equals the forward pass on each of
them. This test keeps the fixture in step with the Python builders.

Regenerate after changing a builder with::

    PTWM_REGEN_FIXTURES=1 uv run --no-sync pytest \
        tests/ptwm/preprocessing/test_production_chain_fixture.py

Format: repeated records of ``u16 name_len | name (UTF-8) | u32 blob_len |
blob`` (little-endian), where ``blob`` is ``Chain.to_bytes()``.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

from ptwm.preprocessing import _chains
from ptwm.preprocessing._chains import PRODUCTION_CHAINS, ClassifierRole
from ptwm.preprocessing._explorer import ExploreOptions, explore_chains

FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "crates"
    / "ptwm-core"
    / "tests"
    / "fixtures"
    / "production_chains.bin"
)

# Shapes in elements of each builder's Source dtype. Small enough for the
# Rust forward pass to stay fast; they cover 0-d, empty, odd, 1-D, 2-D
# (spherical / alpha-stable only accept 2-D) and 3-D tensors.
SHAPES: list[list[int]] = [
    [],
    [0],
    [7],
    [1000],
    [64, 32],
    [7, 33],
    [3, 5, 8],
]


def _builders() -> list[tuple[str, _chains.ChainBuilder]]:
    named = [
        (name, getattr(_chains, name))
        for name in sorted(_chains.__all__)
        if name.startswith("CHAIN_")
    ]
    seen = {id(b) for _, b in named}
    for (dtype_code, role), builders in sorted(PRODUCTION_CHAINS.items()):
        for i, b in enumerate(builders):
            if id(b) not in seen:
                seen.add(id(b))
                named.append((f"production[{dtype_code},{role.name}][{i}]", b))
    opts = ExploreOptions(max_candidates_per_dtype=8, time_budget_ms=2000)
    for dtype_code in (1, 4, 6, 13, 29, 30, 31):
        for i, b in enumerate(
            explore_chains(dtype_code, ClassifierRole.STANDARD, opts)
        ):
            named.append((f"explore[{dtype_code}][{i}]", b))
    return named


def _fixture_bytes() -> bytes:
    out = bytearray()
    for name, builder in _builders():
        for shape in SHAPES:
            chain = builder(list(shape))
            if chain is None:
                continue
            label = f"{name}{shape}".encode()
            blob = chain.to_bytes()
            out += struct.pack("<H", len(label)) + label
            out += struct.pack("<I", len(blob)) + blob
    return bytes(out)


def test_production_chain_fixture_is_current() -> None:
    data = _fixture_bytes()
    if os.environ.get("PTWM_REGEN_FIXTURES"):
        FIXTURE.write_bytes(data)
    assert FIXTURE.read_bytes() == data, (
        "production chain fixture is stale; regenerate it (see module docstring)"
    )
