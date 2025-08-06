"""PPG chain definitions and production lookup table.

A *chain* is a typed DAG whose nodes are preprocessing ops (Source,
BitReorder*, ByteSplit, NibbleSplit, BytePassthrough, …) and whose edges
carry :class:`Role`-tagged ``PlaneDescriptor`` values. The Rust runtime
executes chains during compression (forward pass) and decompression
(inverse pass).

This module provides:

* :class:`Chain`, :class:`ChainNode`, :class:`ChainEdge`, :class:`TerminalRef` —
  Python mirrors of the Rust structs; ``Chain.to_bytes()`` serialises to the
  wire format accepted by ``compress_model``.
* :class:`ClassifierRole` — tensor role enum for the production lookup table.
* :data:`PRODUCTION_CHAINS` — static ``(dtype_code, ClassifierRole)`` →
  ``list[ChainBuilder]`` table
* :func:`chains_for` — thin wrapper over :data:`PRODUCTION_CHAINS`.

Chain builders are callables ``(shape: list[int]) -> Chain``. The compressor
driver calls them per-tensor so each chain instance carries the tensor's
concrete shape in its Source node params, which the decoder's placeholder-
forward descriptor recovery path requires.
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum

from ..utils import DType as _DT

# Canonical extension-table identifier for chain discovery.
__ptwm_canonical_id__ = "io.ptwm.builtin.production_chains"

# ---------------------------------------------------------------------------
# Op IDs (wire values from transforms/op.rs)
# ---------------------------------------------------------------------------


class _Op(IntEnum):
    Source = 0x0000
    Terminal = 0x0001
    BitReorderIeee16 = 0x0010
    BitReorderIeee32 = 0x0011
    BitReorderFp8E4M3 = 0x0012
    BitReorderFp8E5M2 = 0x0013
    ByteSplit = 0x0020
    NibbleSplit = 0x0021
    BytePassthrough = 0x0030
    MxFp4Deinterleave = 0x0040
    BlockMicroscalingRepack = 0x0041
    XorDelta = 0x0050
    FloatDelta = 0x0051
    IntDelta = 0x0052
    PredictorXor = 0x0053
    IndexBitwidthPack = 0x0060
    EntropyEstimate = 0x0061
    Concat = 0x0062
    Reshape = 0x0063
    MantissaZeroStrip = 0x0064
    BurrowsWheeler = 0x0070
    MoveToFront = 0x0071
    SphericalNormalize = 0x0080
    AlphaStableNormalize = 0x0081


# ---------------------------------------------------------------------------
# Internal chain dtype codes
#
# The chain's Source node params use a *different* dtype encoding than the
# Python DType codes stored in the tensor record. These internal codes feed
# source.rs::bytes_per_element and element_width_for; build chain wire
# bytes with these, not the DType codes.
# ---------------------------------------------------------------------------

_CHAIN_DTYPE: dict[int, int] = {
    1: 0x0003,  # Float32
    2: 0x0003,  # Float (alias for Float32)
    4: 0x0002,  # Float16
    5: 0x0002,  # Half (alias for Float16)
    6: 0x000F,  # BFloat16
    13: 0x0006,  # Uint8
    15: 0x000A,  # Uint32
    17: 0x0005,  # Int8
    18: 0x0007,  # Int16
    19: 0x0007,  # Short (alias for Int16)
    20: 0x0009,  # Int32
    21: 0x0009,  # Int (alias for Int32)
    22: 0x000B,  # Int64
    23: 0x000B,  # Long (alias for Int64)
    29: 0x0010,  # Float8E4M3FN
    30: 0x0011,  # Float8E5M2
    31: 0x001F,  # Float4E2M1FNx2 (packed fp4, 1 byte/elem in Rust fallthrough)
}


# ---------------------------------------------------------------------------
# Role wire helpers (from types/role.rs)
# ---------------------------------------------------------------------------


def _role_scale(scale_fmt: int) -> bytes:
    """Scale { format }.  scale_fmt: E8M0=0, E4M3=1, E5M2=2, F32=3, F16=4."""
    return bytes([0x00, scale_fmt])


def _role_global_scale(scale_fmt: int) -> bytes:
    """GlobalScale { format }."""
    return bytes([0x07, scale_fmt])


def _role_nibble(kind: int) -> bytes:
    """Nibble { kind }.  kind: Exponent=0, SignMantissa=1, Value=2."""
    return bytes([0x04, kind])


def _role_mantissa_byte(index: int, of: int) -> bytes:
    return bytes([0x02, index, of])


def _role_exponent_byte() -> bytes:
    return bytes([0x03])


def _role_value_fp4() -> bytes:
    """Value { Fp4E2m1 }."""
    return bytes([0x01, 0x00])


def _role_value_intn(bits: int) -> bytes:
    """Value { IntN { bits } }."""
    return bytes([0x01, 0x01, bits])


def _role_raw() -> bytes:
    """Raw (untyped) plane."""
    return bytes([0x08])


# ---------------------------------------------------------------------------
# Chain dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ChainNode:
    op_id: int
    params: bytes = field(default=b"")


@dataclass
class ChainEdge:
    src_node: int
    src_output_idx: int
    dst_node: int
    dst_input_idx: int
    role_override: bytes | None = None  # serialised Role bytes, or None
    vendor_bytes: bytes = field(default=b"")


@dataclass
class TerminalRef:
    node_idx: int
    output_idx: int
    role: bytes  # serialised Role bytes


@dataclass
class Chain:
    nodes: list[ChainNode]
    edges: list[ChainEdge]
    terminals: list[TerminalRef]

    def to_bytes(self) -> bytes:
        """Serialise to the wire format consumed by ``compress_model``.

        The wire format is a self-contained blob consisting
        of a local extension table followed by table-indexed chain bytes.
        The conversion from legacy op-id bytes to the new format is done
        via the Rust bridge to ensure consistency with the Rust canonical-id
        implementation.
        """
        from .._core import (  # noqa: PLC0415
            chain_blob_from_legacy,  # type: ignore[import]
        )

        legacy_bytes = _write_chain(self)
        return bytes(chain_blob_from_legacy(legacy_bytes))


# ---------------------------------------------------------------------------
# Source node params builder
# ---------------------------------------------------------------------------


def _source_params(shape: list[int], internal_dtype: int) -> bytes:
    """Encode Source node params: dim_count(u8) + dims([u32 LE]) + dtype(u16 LE)."""
    out = bytearray([len(shape)])
    for d in shape:
        out += struct.pack("<I", d)
    out += struct.pack("<H", internal_dtype)
    return bytes(out)


# ---------------------------------------------------------------------------
# Wire serialisation (matches write_chain in chain/wire.rs)
# ---------------------------------------------------------------------------


def _write_chain(chain: Chain) -> bytes:
    nodes_parts = []
    for node in chain.nodes:
        nodes_parts.append(struct.pack("<HB", node.op_id, len(node.params)))
        nodes_parts.append(node.params)
    nodes_buf = b"".join(nodes_parts)

    edges_parts = []
    for edge in chain.edges:
        edges_parts.append(
            bytes(
                [
                    edge.src_node,
                    edge.src_output_idx,
                    edge.dst_node,
                    edge.dst_input_idx,
                ]
            )
        )
        if edge.role_override is not None:
            edges_parts.append(b"\x01")
            edges_parts.append(edge.role_override)
        else:
            edges_parts.append(b"\x00")
        edges_parts.append(struct.pack("<H", len(edge.vendor_bytes)))
        edges_parts.append(edge.vendor_bytes)
    edges_buf = b"".join(edges_parts)

    terminals_parts = []
    for term in chain.terminals:
        terminals_parts.append(bytes([term.node_idx, term.output_idx]))
        terminals_parts.append(term.role)
    terminals_buf = b"".join(terminals_parts)

    header = struct.pack(
        "<BBBHHH",
        len(chain.nodes),
        len(chain.edges),
        len(chain.terminals),
        len(nodes_buf),
        len(edges_buf),
        len(terminals_buf),
    )
    return b"".join([header, nodes_buf, edges_buf, terminals_buf])


# ---------------------------------------------------------------------------
# ClassifierRole
# ---------------------------------------------------------------------------


class ClassifierRole(IntEnum):
    """Tensor role used as the second key in :data:`PRODUCTION_CHAINS`."""

    STANDARD = 0
    SCALE_BLOCK = 1
    SCALE_GLOBAL = 2
    PACKED_VALUES = 3


# ---------------------------------------------------------------------------
# Chain builders
# ---------------------------------------------------------------------------

#: Type alias for a chain builder callable.
ChainBuilder = Callable[[list[int]], Chain | None]


def _simple_edge(src: int, dst: int) -> ChainEdge:
    return ChainEdge(
        src_node=src,
        src_output_idx=0,
        dst_node=dst,
        dst_input_idx=0,
    )


def _bf16_split_chain(shape: list[int]) -> Chain:
    """BF16 / FP16: Source → BitReorderIeee16 → ByteSplit(n=2) → 2 terminals."""
    return Chain(
        nodes=[
            ChainNode(
                op_id=_Op.Source,
                params=_source_params(shape, _CHAIN_DTYPE[_DT.BFLOAT16.code]),
            ),
            ChainNode(op_id=_Op.BitReorderIeee16),
            ChainNode(op_id=_Op.ByteSplit, params=bytes([2])),
        ],
        edges=[_simple_edge(0, 1), _simple_edge(1, 2)],
        terminals=[
            TerminalRef(2, 0, _role_mantissa_byte(0, 2)),
            TerminalRef(2, 1, _role_exponent_byte()),
        ],
    )


def _fp32_split_chain(shape: list[int]) -> Chain:
    """FP32: Source → BitReorderIeee32 → ByteSplit(n=4) → 4 terminals."""
    return Chain(
        nodes=[
            ChainNode(
                op_id=_Op.Source,
                params=_source_params(shape, _CHAIN_DTYPE[_DT.FLOAT32.code]),
            ),
            ChainNode(op_id=_Op.BitReorderIeee32),
            ChainNode(op_id=_Op.ByteSplit, params=bytes([4])),
        ],
        edges=[_simple_edge(0, 1), _simple_edge(1, 2)],
        terminals=[
            TerminalRef(2, 0, _role_mantissa_byte(0, 4)),
            TerminalRef(2, 1, _role_mantissa_byte(1, 4)),
            TerminalRef(2, 2, _role_mantissa_byte(2, 4)),
            TerminalRef(2, 3, _role_exponent_byte()),
        ],
    )


def _bf16_split_intdelta_exponent_chain(shape: list[int]) -> Chain:
    """BF16 / FP16 variant: split + IntDelta on the exponent terminal.

    Same as :func:`_bf16_split_chain` but inserts an IntDelta op between
    ByteSplit and the exponent terminal. Targets the position-local
    correlation in adjacent weights' exponents (attention heads, FFN
    rows) that the existing bit-reorder → byte-split → Huffman chain
    only partially captures. Trial-encode picks this chain when it
    beats the plain split chain on the per-tensor `total_bytes` signal.
    """
    return Chain(
        nodes=[
            ChainNode(
                op_id=_Op.Source,
                params=_source_params(shape, _CHAIN_DTYPE[_DT.BFLOAT16.code]),
            ),
            ChainNode(op_id=_Op.BitReorderIeee16),
            ChainNode(op_id=_Op.ByteSplit, params=bytes([2])),
            ChainNode(op_id=_Op.IntDelta),
        ],
        edges=[
            _simple_edge(0, 1),
            _simple_edge(1, 2),
            ChainEdge(src_node=2, src_output_idx=1, dst_node=3, dst_input_idx=0),
        ],
        terminals=[
            TerminalRef(2, 0, _role_mantissa_byte(0, 2)),
            TerminalRef(3, 0, _role_exponent_byte()),
        ],
    )


def _bf16_split_predictor_xor_exponent_chain(shape: list[int]) -> Chain:
    """BF16 / FP16 variant: split + PredictorXor on the exponent terminal.

    Like :func:`_bf16_split_intdelta_exponent_chain`, but uses the fcm
    hash-table predictor instead of a simple element-to-element integer
    delta. Targets streams whose adjacent-element predictability is
    higher-order than what `IntDelta` captures (e.g. exponents that
    repeat in a short cycle along an attention head). Trial-encode
    picks whichever variant produces the smaller payload per tensor.
    """
    return Chain(
        nodes=[
            ChainNode(
                op_id=_Op.Source,
                params=_source_params(shape, _CHAIN_DTYPE[_DT.BFLOAT16.code]),
            ),
            ChainNode(op_id=_Op.BitReorderIeee16),
            ChainNode(op_id=_Op.ByteSplit, params=bytes([2])),
            ChainNode(op_id=_Op.PredictorXor),
        ],
        edges=[
            _simple_edge(0, 1),
            _simple_edge(1, 2),
            ChainEdge(src_node=2, src_output_idx=1, dst_node=3, dst_input_idx=0),
        ],
        terminals=[
            TerminalRef(2, 0, _role_mantissa_byte(0, 2)),
            TerminalRef(3, 0, _role_exponent_byte()),
        ],
    )


def _bf16_microscale_repack_chain(shape: list[int]) -> Chain:
    """BF16: Source → BlockMicroscalingRepack(32) → split + E8M0 scale.

    Re-expresses BF16 as a per-block E8M0 scale plane plus a scale-relative
    element plane. The element plane then takes the standard
    bit-reorder + byte-split route, so its rebased exponent byte — now
    concentrated near zero — compresses harder, while the small per-block
    scale plane joins the entropy-coder menu as its own terminal. Lossless:
    the repack is a reversible block-relative exponent rebasing.
    """
    return Chain(
        nodes=[
            ChainNode(
                op_id=_Op.Source,
                params=_source_params(shape, _CHAIN_DTYPE[_DT.BFLOAT16.code]),
            ),
            ChainNode(op_id=_Op.BlockMicroscalingRepack, params=bytes([32])),
            ChainNode(op_id=_Op.BitReorderIeee16),
            ChainNode(op_id=_Op.ByteSplit, params=bytes([2])),
        ],
        edges=[
            _simple_edge(0, 1),
            _simple_edge(1, 2),  # Repack output 0 (element plane) → BitReorder
            _simple_edge(2, 3),
        ],
        terminals=[
            TerminalRef(3, 0, _role_mantissa_byte(0, 2)),
            TerminalRef(3, 1, _role_exponent_byte()),
            TerminalRef(1, 1, _role_scale(0)),  # Repack output 1 = E8M0 scale
        ],
    )


def _fp32_split_predictor_xor_exponent_chain(shape: list[int]) -> Chain:
    """FP32 variant: split + PredictorXor on the exponent terminal.

    See :func:`_bf16_split_predictor_xor_exponent_chain` for the rationale.
    """
    return Chain(
        nodes=[
            ChainNode(
                op_id=_Op.Source,
                params=_source_params(shape, _CHAIN_DTYPE[_DT.FLOAT32.code]),
            ),
            ChainNode(op_id=_Op.BitReorderIeee32),
            ChainNode(op_id=_Op.ByteSplit, params=bytes([4])),
            ChainNode(op_id=_Op.PredictorXor),
        ],
        edges=[
            _simple_edge(0, 1),
            _simple_edge(1, 2),
            ChainEdge(src_node=2, src_output_idx=3, dst_node=3, dst_input_idx=0),
        ],
        terminals=[
            TerminalRef(2, 0, _role_mantissa_byte(0, 4)),
            TerminalRef(2, 1, _role_mantissa_byte(1, 4)),
            TerminalRef(2, 2, _role_mantissa_byte(2, 4)),
            TerminalRef(3, 0, _role_exponent_byte()),
        ],
    )


def _make_bf16_split_mantissa_zero_strip_chain(k: int) -> ChainBuilder:
    """Build a BF16 / FP16 chain with MantissaZeroStrip(k) on the mantissa terminal.

    Right-shifts every mantissa byte by *k* bits before the entropy
    coder sees it. Targets the post-quantization case where a tensor
    was dequantized from INT(8-k) back into BF16 / FP16: every
    mantissa byte then has *k* trailing zero bits, so stripping them
    shrinks the alphabet the entropy coder works with by a factor of
    ``2**k`` on GGUF / GPTQ-style checkpoints.

    Dispatcher behaviour: trial-encode tries each ``k`` variant; the
    forward pass of :class:`crate::transforms::MantissaZeroStrip`
    short-circuits on the first byte that lacks ``k`` trailing
    zeros, so the variant is essentially free on non-quantized
    inputs. The variant that captures the largest valid ``k`` wins
    on byte count.
    """

    def _build(shape: list[int]) -> Chain:
        return Chain(
            nodes=[
                ChainNode(
                    op_id=_Op.Source,
                    params=_source_params(shape, _CHAIN_DTYPE[_DT.BFLOAT16.code]),
                ),
                ChainNode(op_id=_Op.BitReorderIeee16),
                ChainNode(op_id=_Op.ByteSplit, params=bytes([2])),
                ChainNode(
                    op_id=_Op.MantissaZeroStrip,
                    params=bytes([k]),
                ),
            ],
            edges=[
                _simple_edge(0, 1),
                _simple_edge(1, 2),
                # ByteSplit out[0] = mantissa plane → MantissaZeroStrip input.
                ChainEdge(src_node=2, src_output_idx=0, dst_node=3, dst_input_idx=0),
            ],
            terminals=[
                # MantissaZeroStrip output → mantissa terminal (descriptor
                # is preserved; only byte values change).
                TerminalRef(3, 0, _role_mantissa_byte(0, 2)),
                # ByteSplit out[1] = exponent plane → exponent terminal,
                # unchanged from the baseline split chain.
                TerminalRef(2, 1, _role_exponent_byte()),
            ],
        )

    return _build


def _make_fp32_split_mantissa_zero_strip_chain(k: int) -> ChainBuilder:
    """Build an FP32 chain with MantissaZeroStrip(k) on the low mantissa byte.

    FP32 has three mantissa bytes after ByteSplit (output_idx 0, 1,
    2) and one exponent byte (output_idx 3). The low mantissa byte
    (index 0) is the one quantization rounding always clears first,
    so this variant slots :class:`crate::transforms::MantissaZeroStrip`
    onto that plane only. A more aggressive variant could strip the
    middle mantissa byte too, but the gain diminishes quickly because
    FP32 has 23 mantissa bits and only the lowest few are
    quantization-zeroed in practice.
    """

    def _build(shape: list[int]) -> Chain:
        return Chain(
            nodes=[
                ChainNode(
                    op_id=_Op.Source,
                    params=_source_params(shape, _CHAIN_DTYPE[_DT.FLOAT32.code]),
                ),
                ChainNode(op_id=_Op.BitReorderIeee32),
                ChainNode(op_id=_Op.ByteSplit, params=bytes([4])),
                ChainNode(
                    op_id=_Op.MantissaZeroStrip,
                    params=bytes([k]),
                ),
            ],
            edges=[
                _simple_edge(0, 1),
                _simple_edge(1, 2),
                # ByteSplit out[0] = low mantissa byte → MantissaZeroStrip input.
                ChainEdge(src_node=2, src_output_idx=0, dst_node=3, dst_input_idx=0),
            ],
            terminals=[
                TerminalRef(3, 0, _role_mantissa_byte(0, 4)),
                TerminalRef(2, 1, _role_mantissa_byte(1, 4)),
                TerminalRef(2, 2, _role_mantissa_byte(2, 4)),
                TerminalRef(2, 3, _role_exponent_byte()),
            ],
        )

    return _build


def _fp32_split_intdelta_exponent_chain(shape: list[int]) -> Chain:
    """FP32 variant: split + IntDelta on the exponent terminal.

    See :func:`_bf16_split_intdelta_exponent_chain` for the rationale.
    """
    return Chain(
        nodes=[
            ChainNode(
                op_id=_Op.Source,
                params=_source_params(shape, _CHAIN_DTYPE[_DT.FLOAT32.code]),
            ),
            ChainNode(op_id=_Op.BitReorderIeee32),
            ChainNode(op_id=_Op.ByteSplit, params=bytes([4])),
            ChainNode(op_id=_Op.IntDelta),
        ],
        edges=[
            _simple_edge(0, 1),
            _simple_edge(1, 2),
            ChainEdge(src_node=2, src_output_idx=3, dst_node=3, dst_input_idx=0),
        ],
        terminals=[
            TerminalRef(2, 0, _role_mantissa_byte(0, 4)),
            TerminalRef(2, 1, _role_mantissa_byte(1, 4)),
            TerminalRef(2, 2, _role_mantissa_byte(2, 4)),
            TerminalRef(3, 0, _role_exponent_byte()),
        ],
    )


def _fp8e4m3_nibble_chain(shape: list[int]) -> Chain:
    """FP8 E4M3FN: Source → BitReorderFp8E4M3 → NibbleSplit → 2 terminals."""
    return Chain(
        nodes=[
            ChainNode(
                op_id=_Op.Source,
                params=_source_params(shape, _CHAIN_DTYPE[_DT.FLOAT8_E4M3FN.code]),
            ),
            ChainNode(op_id=_Op.BitReorderFp8E4M3),
            ChainNode(op_id=_Op.NibbleSplit),
        ],
        edges=[_simple_edge(0, 1), _simple_edge(1, 2)],
        terminals=[
            TerminalRef(2, 0, _role_nibble(0)),  # Exponent
            TerminalRef(2, 1, _role_nibble(1)),  # SignMantissa
        ],
    )


def _fp8e5m2_nibble_chain(shape: list[int]) -> Chain:
    """FP8 E5M2: Source → BitReorderFp8E5M2 → NibbleSplit → 2 terminals."""
    return Chain(
        nodes=[
            ChainNode(
                op_id=_Op.Source,
                params=_source_params(shape, _CHAIN_DTYPE[_DT.FLOAT8_E5M2.code]),
            ),
            ChainNode(op_id=_Op.BitReorderFp8E5M2),
            ChainNode(op_id=_Op.NibbleSplit),
        ],
        edges=[_simple_edge(0, 1), _simple_edge(1, 2)],
        terminals=[
            TerminalRef(2, 0, _role_nibble(0)),  # Exponent
            TerminalRef(2, 1, _role_nibble(1)),  # SignMantissa
        ],
    )


def _fp8e4m3_scale_passthrough_chain(shape: list[int]) -> Chain:
    """FP8 E4M3FN scale block: Source → BytePassthrough → Scale{E4M3} terminal.

    The Scale terminal role enables Order1ScaleAC when the plane layout is Rows.
    """
    return Chain(
        nodes=[
            ChainNode(
                op_id=_Op.Source,
                params=_source_params(shape, _CHAIN_DTYPE[_DT.FLOAT8_E4M3FN.code]),
            ),
            ChainNode(op_id=_Op.BytePassthrough),
        ],
        edges=[_simple_edge(0, 1)],
        terminals=[TerminalRef(1, 0, _role_scale(1))],  # E4M3 = 1
    )


def _u8_scale_passthrough_chain(shape: list[int]) -> Chain:
    """Uint8 scale block (E8M0 MXFP4): Source → BytePassthrough → Scale{E8M0}."""
    return Chain(
        nodes=[
            ChainNode(
                op_id=_Op.Source,
                params=_source_params(shape, _CHAIN_DTYPE[_DT.UINT8.code]),
            ),
            ChainNode(op_id=_Op.BytePassthrough),
        ],
        edges=[_simple_edge(0, 1)],
        terminals=[TerminalRef(1, 0, _role_scale(0))],  # E8M0 = 0
    )


def _fp32_global_scale_chain(shape: list[int]) -> Chain:
    """FP32 global scale (NVFP4 input_scale): Source → BytePassthrough → GlobalScale."""
    return Chain(
        nodes=[
            ChainNode(
                op_id=_Op.Source,
                params=_source_params(shape, _CHAIN_DTYPE[_DT.FLOAT32.code]),
            ),
            ChainNode(op_id=_Op.BytePassthrough),
        ],
        edges=[_simple_edge(0, 1)],
        terminals=[TerminalRef(1, 0, _role_global_scale(3))],  # F32 = 3
    )


def _fp4_passthrough_chain(shape: list[int]) -> Chain:
    """Packed FP4 values: Source → BytePassthrough → Value{Fp4E2m1}.

    The Value{Fp4E2m1} terminal role enables PerGroupCodebook when the plane
    has ``is_nibble_packed = True``.
    """
    return Chain(
        nodes=[
            ChainNode(
                op_id=_Op.Source,
                params=_source_params(shape, _CHAIN_DTYPE[_DT.FLOAT4_E2M1FN_X2.code]),
            ),
            ChainNode(op_id=_Op.BytePassthrough),
        ],
        edges=[_simple_edge(0, 1)],
        terminals=[TerminalRef(1, 0, _role_value_fp4())],
    )


def _u8_byte_passthrough_value_chain(shape: list[int]) -> Chain:
    """Uint8 standard: Source → BytePassthrough → Value{IntN{bits=8}}."""
    return Chain(
        nodes=[
            ChainNode(
                op_id=_Op.Source,
                params=_source_params(shape, _CHAIN_DTYPE[_DT.UINT8.code]),
            ),
            ChainNode(op_id=_Op.BytePassthrough),
        ],
        edges=[_simple_edge(0, 1)],
        terminals=[TerminalRef(1, 0, _role_value_intn(8))],
    )


# ---------------------------------------------------------------------------
# Spherical-normalize chains
#
# Per-row (radius, direction) reparameterization with an exact XOR residual.
# The op is lossless by construction: forward and inverse recompute the same
# prediction from the stored radius/direction and XOR the residual back, so
# the chain round-trips bit-exactly regardless of float-op accuracy. Routed
# as a self-selecting candidate for 2D STANDARD float tensors; the dispatcher
# keeps it only where its three planes (radius / direction / residual)
# compress below the plain split chain.
# ---------------------------------------------------------------------------


def _make_spherical_normalize_chain(
    chain_dtype_code: int, sph_dtype: int
) -> ChainBuilder:
    """Build a spherical-normalize chain builder for a given dtype.

    ``chain_dtype_code`` is the Python ``DType`` code used to look up the
    Source node's internal dtype; ``sph_dtype`` is the op's dtype byte
    (0=FP32, 1=BF16, 2=FP16).

    The chain is: ``Source → SphericalNormalize(mode=0, precision=0, dtype)``
    with three terminals — radius (f32/row), direction (bf16), and the XOR
    residual. The Source node tags a 2D (or higher) tensor with a ``Rows``
    layout whose ``row_len`` is the last dim, which the op reads as the
    embedding dimension ``d``.
    """
    internal_dtype = _CHAIN_DTYPE[chain_dtype_code]

    def _builder(shape: list[int]) -> Chain | None:
        # Self-selecting gate: only 2D row-structured tensors have a
        # meaningful (rows, embedding-dim) factorization to normalize.
        # Returning None drops this candidate so the op never runs on 1D
        # (bias/norm) or higher-rank tensors, bounding trial-encode cost.
        if len(shape) != 2:
            return None
        return Chain(
            nodes=[
                ChainNode(
                    op_id=_Op.Source,
                    params=_source_params(shape, internal_dtype),
                ),
                ChainNode(
                    op_id=_Op.SphericalNormalize,
                    params=bytes([0, 0, sph_dtype]),  # mode 0, precision 0
                ),
            ],
            edges=[_simple_edge(0, 1)],
            terminals=[
                TerminalRef(1, 0, _role_raw()),  # radius
                TerminalRef(1, 1, _role_raw()),  # direction
                TerminalRef(1, 2, _role_raw()),  # XOR residual
            ],
        )

    return _builder


# ---------------------------------------------------------------------------
# alpha_stable_normalize: per-row robust affine normalization
# ``(v − δ)/γ`` (δ = median, γ = MAD) with an exact XOR residual, so the chain
# round-trips bit-exactly regardless of float-op accuracy. Routed as a
# self-selecting candidate for 2D STANDARD float tensors; the dispatcher keeps
# it only where its three planes (scale / normalized / residual) compress below
# the plain split chain.
# ---------------------------------------------------------------------------


def _make_alpha_stable_normalize_chain(
    chain_dtype_code: int, asn_dtype: int
) -> ChainBuilder:
    """Build an alpha-stable-normalize chain builder for a given dtype.

    ``chain_dtype_code`` is the Python ``DType`` code used to look up the
    Source node's internal dtype; ``asn_dtype`` is the op's dtype byte
    (0=FP32, 1=BF16, 2=FP16).

    The chain is: ``Source → AlphaStableNormalize(precision=0, dtype)`` with
    three terminals — scale (γ,δ f32 pair/row), normalized (bf16), and the XOR
    residual. The builder accepts only 2D tensors (returns ``None`` otherwise);
    the Source node tags the tensor with a ``Rows`` layout whose ``row_len`` is
    the last dim, which the op reads as the row width ``d``.
    """
    internal_dtype = _CHAIN_DTYPE[chain_dtype_code]

    def _builder(shape: list[int]) -> Chain | None:
        # Self-selecting gate: only 2D row-structured tensors have a
        # meaningful (rows, embedding-dim) factorization to normalize.
        # Returning None drops this candidate so the op never runs on 1D
        # (bias/norm) or higher-rank tensors, bounding trial-encode cost.
        if len(shape) != 2:
            return None
        return Chain(
            nodes=[
                ChainNode(
                    op_id=_Op.Source,
                    params=_source_params(shape, internal_dtype),
                ),
                ChainNode(
                    op_id=_Op.AlphaStableNormalize,
                    params=bytes([0, asn_dtype]),  # precision 0
                ),
            ],
            edges=[_simple_edge(0, 1)],
            terminals=[
                TerminalRef(1, 0, _role_raw()),  # scale (γ, δ)
                TerminalRef(1, 1, _role_raw()),  # normalized
                TerminalRef(1, 2, _role_raw()),  # XOR residual
            ],
        )

    return _builder


# ---------------------------------------------------------------------------
# huff_llm_5bit: raw 16-bit-word passthrough feeding the field-separated
# Huffman codec. No transform — the codec competes directly on the un-split
# 16-bit word plane (ElementWidth.Word2, Role::Raw). Smallest-wins selection
# means it only displaces the byte-split chains where field-separated Huffman
# is smaller.
# ---------------------------------------------------------------------------


def _make_huff_llm_raw_chain(dtype_code: int) -> ChainBuilder:
    """Source → raw 16-bit-word terminal, for the huff_llm_5bit codec.

    No transform: the codec competes directly on the un-split 16-bit word plane
    (ElementWidth.Word2, Role::Raw). Smallest-wins selection means it only
    displaces the byte-split chains where field-separated Huffman is smaller.
    """
    internal_dtype = _CHAIN_DTYPE[dtype_code]

    def _builder(shape: list[int]) -> Chain | None:
        return Chain(
            nodes=[
                ChainNode(
                    op_id=_Op.Source,
                    params=_source_params(shape, internal_dtype),
                )
            ],
            edges=[],
            terminals=[TerminalRef(0, 0, _role_raw())],
        )

    return _builder


# ---------------------------------------------------------------------------
# Named chain builder constants
# ---------------------------------------------------------------------------

CHAIN_BF16_SPLIT: ChainBuilder = _bf16_split_chain
"""BF16 / FP16 standard: BitReorderIeee16 + ByteSplit(n=2)."""

CHAIN_BF16_SPLIT_INTDELTA: ChainBuilder = _bf16_split_intdelta_exponent_chain
"""BF16 / FP16 variant: split + IntDelta on the exponent terminal."""

CHAIN_BF16_SPLIT_PREDICTOR_XOR: ChainBuilder = _bf16_split_predictor_xor_exponent_chain
"""BF16 / FP16 variant: split + fcm PredictorXor on the exponent terminal."""

CHAIN_BF16_MICROSCALE_REPACK: ChainBuilder = _bf16_microscale_repack_chain
"""BF16 variant: per-block E8M0 microscaling repack + standard split."""

CHAIN_FP32_SPLIT: ChainBuilder = _fp32_split_chain
"""FP32 standard: BitReorderIeee32 + ByteSplit(n=4)."""

CHAIN_FP32_SPLIT_INTDELTA: ChainBuilder = _fp32_split_intdelta_exponent_chain
"""FP32 variant: split + IntDelta on the exponent terminal."""

CHAIN_FP32_SPLIT_PREDICTOR_XOR: ChainBuilder = _fp32_split_predictor_xor_exponent_chain
"""FP32 variant: split + fcm PredictorXor on the exponent terminal."""

# MantissaZeroStrip variants — one per quantization width. k corresponds to
# the post-dequantization trailing-zero count: INT4 → k=4, INT5 → k=3,
# INT6 → k=2, INT7 → k=1. The dispatcher trial-encodes each variant; the
# largest valid k wins on byte count, and variants whose k exceeds the
# tensor's actual trailing-zero count fail at MantissaZeroStrip.forward()
# without allocating output bytes.
CHAIN_BF16_SPLIT_MZS_K1: ChainBuilder = _make_bf16_split_mantissa_zero_strip_chain(1)
"""BF16 / FP16 variant: split + MantissaZeroStrip(k=1) on the mantissa terminal."""

CHAIN_BF16_SPLIT_MZS_K2: ChainBuilder = _make_bf16_split_mantissa_zero_strip_chain(2)
"""BF16 / FP16 variant: split + MantissaZeroStrip(k=2) on the mantissa terminal."""

CHAIN_BF16_SPLIT_MZS_K3: ChainBuilder = _make_bf16_split_mantissa_zero_strip_chain(3)
"""BF16 / FP16 variant: split + MantissaZeroStrip(k=3) on the mantissa terminal."""

CHAIN_BF16_SPLIT_MZS_K4: ChainBuilder = _make_bf16_split_mantissa_zero_strip_chain(4)
"""BF16 / FP16 variant: split + MantissaZeroStrip(k=4) on the mantissa terminal.

Primary use case: tensors dequantized from INT4 back to BF16 / FP16."""

CHAIN_FP32_SPLIT_MZS_K1: ChainBuilder = _make_fp32_split_mantissa_zero_strip_chain(1)
"""FP32 variant: split + MantissaZeroStrip(k=1) on the low mantissa byte."""

CHAIN_FP32_SPLIT_MZS_K2: ChainBuilder = _make_fp32_split_mantissa_zero_strip_chain(2)
"""FP32 variant: split + MantissaZeroStrip(k=2) on the low mantissa byte."""

CHAIN_FP32_SPLIT_MZS_K3: ChainBuilder = _make_fp32_split_mantissa_zero_strip_chain(3)
"""FP32 variant: split + MantissaZeroStrip(k=3) on the low mantissa byte."""

CHAIN_FP32_SPLIT_MZS_K4: ChainBuilder = _make_fp32_split_mantissa_zero_strip_chain(4)
"""FP32 variant: split + MantissaZeroStrip(k=4) on the low mantissa byte."""

CHAIN_FP8E4M3_NIBBLE: ChainBuilder = _fp8e4m3_nibble_chain
"""FP8 E4M3FN standard: BitReorderFp8E4M3 + NibbleSplit."""

CHAIN_FP8E5M2_NIBBLE: ChainBuilder = _fp8e5m2_nibble_chain
"""FP8 E5M2 standard: BitReorderFp8E5M2 + NibbleSplit."""

CHAIN_F8E4M3_SCALE_PASSTHROUGH: ChainBuilder = _fp8e4m3_scale_passthrough_chain
"""FP8 E4M3FN scale-block passthrough (enables O1SAC on block-scale tensors)."""

# Alias used by both the SCALE_BLOCK and STANDARD spec table entries.
CHAIN_F8E4M3_NIBBLE: ChainBuilder = _fp8e4m3_nibble_chain

CHAIN_U8_SCALE_PASSTHROUGH: ChainBuilder = _u8_scale_passthrough_chain
"""Uint8 scale-block passthrough (MXFP4 E8M0 scales)."""

CHAIN_F32_GLOBAL_SCALE: ChainBuilder = _fp32_global_scale_chain
"""FP32 global-scale passthrough (NVFP4 input_scale tensors)."""

CHAIN_FP4_NIBBLE_PASSTHROUGH: ChainBuilder = _fp4_passthrough_chain
"""Packed FP4 values passthrough (enables PerGroupCodebook)."""

CHAIN_BYTE_PASSTHROUGH_VALUE: ChainBuilder = _u8_byte_passthrough_value_chain
"""Uint8 standard passthrough."""

CHAIN_FP32_SPHERICAL_NORMALIZE: ChainBuilder = _make_spherical_normalize_chain(
    _DT.FLOAT32.code, 0
)
"""FP32 variant: per-row (radius, direction) reparameterization + XOR residual."""

CHAIN_BF16_SPHERICAL_NORMALIZE: ChainBuilder = _make_spherical_normalize_chain(
    _DT.BFLOAT16.code, 1
)
"""BF16 variant: per-row (radius, direction) reparameterization + XOR residual."""

CHAIN_FP16_SPHERICAL_NORMALIZE: ChainBuilder = _make_spherical_normalize_chain(
    _DT.FLOAT16.code, 2
)
"""FP16 variant: per-row (radius, direction) reparameterization + XOR residual."""

CHAIN_FP32_ALPHA_STABLE_NORMALIZE: ChainBuilder = _make_alpha_stable_normalize_chain(
    _DT.FLOAT32.code, 0
)
"""FP32 variant: per-row robust affine normalization + XOR residual."""

CHAIN_BF16_ALPHA_STABLE_NORMALIZE: ChainBuilder = _make_alpha_stable_normalize_chain(
    _DT.BFLOAT16.code, 1
)
"""BF16 variant: per-row robust affine normalization + XOR residual."""

CHAIN_FP16_ALPHA_STABLE_NORMALIZE: ChainBuilder = _make_alpha_stable_normalize_chain(
    _DT.FLOAT16.code, 2
)
"""FP16 variant: per-row robust affine normalization + XOR residual."""

CHAIN_BF16_HUFF_LLM_RAW: ChainBuilder = _make_huff_llm_raw_chain(_DT.BFLOAT16.code)
"""BF16 raw-word chain feeding the huff_llm_5bit codec."""

CHAIN_FP16_HUFF_LLM_RAW: ChainBuilder = _make_huff_llm_raw_chain(_DT.FLOAT16.code)
"""FP16 raw-word chain feeding the huff_llm_5bit codec."""


# ---------------------------------------------------------------------------
# Production chains table
# ---------------------------------------------------------------------------

PRODUCTION_CHAINS: dict[tuple[int, ClassifierRole], list[ChainBuilder]] = {
    # FP8 E4M3FN weight_scale (NVFP4): passthrough (O1SAC path) + nibble-split.
    (_DT.FLOAT8_E4M3FN.code, ClassifierRole.SCALE_BLOCK): [
        CHAIN_F8E4M3_SCALE_PASSTHROUGH,
        CHAIN_F8E4M3_NIBBLE,
    ],
    # Uint8 scales (MXFP4 E8M0): passthrough only.
    (_DT.UINT8.code, ClassifierRole.SCALE_BLOCK): [CHAIN_U8_SCALE_PASSTHROUGH],
    # FP32 global scales (NVFP4 input_scale / weight_scale_2): GlobalScale terminal.
    (_DT.FLOAT32.code, ClassifierRole.SCALE_GLOBAL): [CHAIN_F32_GLOBAL_SCALE],
    # Packed FP4 values: passthrough; is_nibble_packed enables PerGroupCodebook.
    (_DT.FLOAT4_E2M1FN_X2.code, ClassifierRole.PACKED_VALUES): [
        CHAIN_FP4_NIBBLE_PASSTHROUGH,
    ],
    # Standard chains for common weight dtypes. The IntDelta and
    # PredictorXor variants compete with the plain split chain via
    # trial-encode; the dispatcher only commits to a variant when it
    # beats the baseline on byte count.
    (_DT.BFLOAT16.code, ClassifierRole.STANDARD): [
        CHAIN_BF16_SPLIT,
        CHAIN_BF16_SPLIT_INTDELTA,
        CHAIN_BF16_SPLIT_PREDICTOR_XOR,
        CHAIN_BF16_MICROSCALE_REPACK,
        CHAIN_BF16_SPLIT_MZS_K1,
        CHAIN_BF16_SPLIT_MZS_K2,
        CHAIN_BF16_SPLIT_MZS_K3,
        CHAIN_BF16_SPLIT_MZS_K4,
        CHAIN_BF16_SPHERICAL_NORMALIZE,
        CHAIN_BF16_ALPHA_STABLE_NORMALIZE,
        CHAIN_BF16_HUFF_LLM_RAW,
    ],
    (_DT.FLOAT16.code, ClassifierRole.STANDARD): [
        CHAIN_BF16_SPLIT,
        CHAIN_BF16_SPLIT_INTDELTA,
        CHAIN_BF16_SPLIT_PREDICTOR_XOR,
        CHAIN_BF16_SPLIT_MZS_K1,
        CHAIN_BF16_SPLIT_MZS_K2,
        CHAIN_BF16_SPLIT_MZS_K3,
        CHAIN_BF16_SPLIT_MZS_K4,
        CHAIN_FP16_SPHERICAL_NORMALIZE,
        CHAIN_FP16_ALPHA_STABLE_NORMALIZE,
        CHAIN_FP16_HUFF_LLM_RAW,
    ],
    (_DT.FLOAT32.code, ClassifierRole.STANDARD): [
        CHAIN_FP32_SPLIT,
        CHAIN_FP32_SPLIT_INTDELTA,
        CHAIN_FP32_SPLIT_PREDICTOR_XOR,
        CHAIN_FP32_SPLIT_MZS_K1,
        CHAIN_FP32_SPLIT_MZS_K2,
        CHAIN_FP32_SPLIT_MZS_K3,
        CHAIN_FP32_SPLIT_MZS_K4,
        CHAIN_FP32_SPHERICAL_NORMALIZE,
        CHAIN_FP32_ALPHA_STABLE_NORMALIZE,
    ],
    (_DT.FLOAT8_E4M3FN.code, ClassifierRole.STANDARD): [CHAIN_FP8E4M3_NIBBLE],
    (_DT.FLOAT8_E5M2.code, ClassifierRole.STANDARD): [CHAIN_FP8E5M2_NIBBLE],
    (_DT.UINT8.code, ClassifierRole.STANDARD): [CHAIN_BYTE_PASSTHROUGH_VALUE],
}


def chains_for(dtype_code: int, role: ClassifierRole) -> list[ChainBuilder]:
    """Return the chain builders for ``(dtype_code, role)``.

    Returns an empty list when the combination has no production entry.
    """
    return PRODUCTION_CHAINS.get((dtype_code, role), [])


__all__ = [
    "CHAIN_BF16_ALPHA_STABLE_NORMALIZE",
    "CHAIN_BF16_MICROSCALE_REPACK",
    "CHAIN_BF16_SPLIT",
    "CHAIN_BF16_SPLIT_INTDELTA",
    "CHAIN_BF16_SPLIT_MZS_K1",
    "CHAIN_BF16_SPLIT_MZS_K2",
    "CHAIN_BF16_SPLIT_MZS_K3",
    "CHAIN_BF16_SPLIT_MZS_K4",
    "CHAIN_BF16_SPHERICAL_NORMALIZE",
    "CHAIN_BF16_SPLIT_PREDICTOR_XOR",
    "CHAIN_BF16_HUFF_LLM_RAW",
    "CHAIN_BYTE_PASSTHROUGH_VALUE",
    "CHAIN_F32_GLOBAL_SCALE",
    "CHAIN_F8E4M3_NIBBLE",
    "CHAIN_F8E4M3_SCALE_PASSTHROUGH",
    "CHAIN_FP32_SPLIT",
    "CHAIN_FP32_SPLIT_INTDELTA",
    "CHAIN_FP32_SPLIT_MZS_K1",
    "CHAIN_FP32_SPLIT_MZS_K2",
    "CHAIN_FP32_SPLIT_MZS_K3",
    "CHAIN_FP32_SPLIT_MZS_K4",
    "CHAIN_FP32_SPLIT_PREDICTOR_XOR",
    "CHAIN_FP32_ALPHA_STABLE_NORMALIZE",
    "CHAIN_FP32_SPHERICAL_NORMALIZE",
    "CHAIN_FP16_ALPHA_STABLE_NORMALIZE",
    "CHAIN_FP16_SPHERICAL_NORMALIZE",
    "CHAIN_FP16_HUFF_LLM_RAW",
    "CHAIN_FP4_NIBBLE_PASSTHROUGH",
    "CHAIN_FP8E4M3_NIBBLE",
    "CHAIN_FP8E5M2_NIBBLE",
    "CHAIN_U8_SCALE_PASSTHROUGH",
    "Chain",
    "ChainBuilder",
    "ChainEdge",
    "ChainNode",
    "ClassifierRole",
    "PRODUCTION_CHAINS",
    "TerminalRef",
    "chains_for",
]
