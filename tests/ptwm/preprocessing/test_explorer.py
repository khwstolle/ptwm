"""Tests for the generative chain explorer."""

from __future__ import annotations

from ptwm.preprocessing._chains import (
    Chain,
    ChainEdge,
    ChainNode,
    ClassifierRole,
    TerminalRef,
    _Op,
    _role_nibble,
    _role_value_intn,
    _source_params,
)
from ptwm.preprocessing._explorer import (
    ExploreOptions,
    _build_linear_chain,
    _extract_source_dtype,
    explore_chains,
    is_legal_chain,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DTYPE_BF16 = 0x000F  # internal code
_DTYPE_FP32 = 0x0003
_DTYPE_FP8_E4M3 = 0x0010
_DTYPE_FP8_E5M2 = 0x0011
_DTYPE_UINT8 = 0x0006

_SHAPE = [4, 4]


def _minimal_opts() -> ExploreOptions:
    return ExploreOptions(
        max_nodes=8,
        max_depth=5,
        max_candidates_per_dtype=32,
        allow_cross_tensor=False,
        allow_cross_plane=False,
        time_budget_ms=5000,
    )


def _passthrough_chain(shape: list[int] = _SHAPE) -> Chain:
    """Minimal legal chain: Source → BytePassthrough → terminal."""
    return _build_linear_chain(_DTYPE_UINT8, shape, (int(_Op.BytePassthrough),))


def _nibble_chain(internal_dtype: int = _DTYPE_FP8_E4M3) -> Chain:
    return _build_linear_chain(internal_dtype, _SHAPE, (int(_Op.NibbleSplit),))


def _byte_split_chain(n: int = 2) -> Chain:
    return _build_linear_chain(
        _DTYPE_BF16, _SHAPE, (int(_Op.ByteSplit),), byte_split_planes=n
    )


# ---------------------------------------------------------------------------
# Rule 1: max_nodes
# ---------------------------------------------------------------------------


def test_rule1_too_many_nodes_rejected() -> None:
    """Chain with > max_nodes nodes is rejected."""
    opts = ExploreOptions(max_nodes=2)
    # Build a chain that has 3 nodes: Source → BytePassthrough → BytePassthrough
    nodes = [
        ChainNode(op_id=int(_Op.Source), params=_source_params(_SHAPE, _DTYPE_UINT8)),
        ChainNode(op_id=int(_Op.BytePassthrough), params=b""),
        ChainNode(op_id=int(_Op.BytePassthrough), params=b""),
    ]
    edges = [
        ChainEdge(0, 0, 1, 0),
        ChainEdge(1, 0, 2, 0),
    ]
    terminals = [TerminalRef(node_idx=2, output_idx=0, role=_role_value_intn(8))]
    chain = Chain(nodes=nodes, edges=edges, terminals=terminals)
    assert not is_legal_chain(chain, opts)


# ---------------------------------------------------------------------------
# Rule 2: max_depth
# ---------------------------------------------------------------------------


def test_rule2_too_deep_rejected() -> None:
    """Chain exceeding max_depth is rejected."""
    opts = ExploreOptions(max_depth=1)
    # 3-node chain has depth 2 (Source=0, PT=1, PT=2 → depth from source = 2)
    nodes = [
        ChainNode(op_id=int(_Op.Source), params=_source_params(_SHAPE, _DTYPE_UINT8)),
        ChainNode(op_id=int(_Op.BytePassthrough), params=b""),
        ChainNode(op_id=int(_Op.BytePassthrough), params=b""),
    ]
    edges = [ChainEdge(0, 0, 1, 0), ChainEdge(1, 0, 2, 0)]
    terminals = [TerminalRef(node_idx=2, output_idx=0, role=_role_value_intn(8))]
    chain = Chain(nodes=nodes, edges=edges, terminals=terminals)
    assert not is_legal_chain(chain, opts)


# ---------------------------------------------------------------------------
# Rule 3: two consecutive BitReorder* ops
# ---------------------------------------------------------------------------


def test_rule3_consecutive_bit_reorder_rejected() -> None:
    """Two consecutive BitReorder ops are rejected."""
    nodes = [
        ChainNode(op_id=int(_Op.Source), params=_source_params(_SHAPE, _DTYPE_BF16)),
        ChainNode(op_id=int(_Op.BitReorderIeee16), params=b""),
        ChainNode(op_id=int(_Op.BitReorderIeee16), params=b""),
    ]
    edges = [ChainEdge(0, 0, 1, 0), ChainEdge(1, 0, 2, 0)]
    terminals = [TerminalRef(node_idx=2, output_idx=0, role=_role_value_intn(8))]
    chain = Chain(nodes=nodes, edges=edges, terminals=terminals)
    assert not is_legal_chain(chain, _minimal_opts())


# ---------------------------------------------------------------------------
# Rule 4: ByteSplit → ByteSplit on same plane
# ---------------------------------------------------------------------------


def test_rule4_consecutive_byte_split_rejected() -> None:
    """ByteSplit → ByteSplit is rejected."""
    nodes = [
        ChainNode(op_id=int(_Op.Source), params=_source_params(_SHAPE, _DTYPE_UINT8)),
        ChainNode(op_id=int(_Op.ByteSplit), params=bytes([2])),
        ChainNode(op_id=int(_Op.ByteSplit), params=bytes([2])),
    ]
    edges = [ChainEdge(0, 0, 1, 0), ChainEdge(1, 0, 2, 0)]
    terminals = [TerminalRef(node_idx=2, output_idx=0, role=_role_value_intn(8))]
    chain = Chain(nodes=nodes, edges=edges, terminals=terminals)
    assert not is_legal_chain(chain, _minimal_opts())


# ---------------------------------------------------------------------------
# Rule 5: NibbleSplit → NibbleSplit
# ---------------------------------------------------------------------------


def test_rule5_consecutive_nibble_split_rejected() -> None:
    """NibbleSplit → NibbleSplit is rejected."""
    nodes = [
        ChainNode(op_id=int(_Op.Source), params=_source_params(_SHAPE, _DTYPE_UINT8)),
        ChainNode(op_id=int(_Op.NibbleSplit), params=b""),
        ChainNode(op_id=int(_Op.NibbleSplit), params=b""),
    ]
    edges = [ChainEdge(0, 0, 1, 0), ChainEdge(1, 0, 2, 0)]
    terminals = [TerminalRef(node_idx=2, output_idx=0, role=_role_nibble(0))]
    chain = Chain(nodes=nodes, edges=edges, terminals=terminals)
    assert not is_legal_chain(chain, _minimal_opts())


# ---------------------------------------------------------------------------
# Rule 6: BitReorder dtype compatibility
# ---------------------------------------------------------------------------


def test_rule6_bit_reorder_wrong_dtype_rejected() -> None:
    """BitReorderIeee16 on a non-fp16/bf16 dtype is rejected."""
    nodes = [
        ChainNode(op_id=int(_Op.Source), params=_source_params(_SHAPE, _DTYPE_UINT8)),
        ChainNode(op_id=int(_Op.BitReorderIeee16), params=b""),
        ChainNode(op_id=int(_Op.BytePassthrough), params=b""),
    ]
    edges = [ChainEdge(0, 0, 1, 0), ChainEdge(1, 0, 2, 0)]
    terminals = [TerminalRef(node_idx=2, output_idx=0, role=_role_value_intn(8))]
    chain = Chain(nodes=nodes, edges=edges, terminals=terminals)
    assert not is_legal_chain(chain, _minimal_opts())


def test_rule6_bit_reorder32_on_fp16_rejected() -> None:
    """BitReorderIeee32 on a fp16 dtype is rejected."""
    nodes = [
        ChainNode(op_id=int(_Op.Source), params=_source_params(_SHAPE, _DTYPE_BF16)),
        ChainNode(op_id=int(_Op.BitReorderIeee32), params=b""),
        ChainNode(op_id=int(_Op.BytePassthrough), params=b""),
    ]
    edges = [ChainEdge(0, 0, 1, 0), ChainEdge(1, 0, 2, 0)]
    terminals = [TerminalRef(node_idx=2, output_idx=0, role=_role_value_intn(8))]
    chain = Chain(nodes=nodes, edges=edges, terminals=terminals)
    assert not is_legal_chain(chain, _minimal_opts())


# ---------------------------------------------------------------------------
# Rule 7: MxFp4Deinterleave dtype compatibility
# ---------------------------------------------------------------------------


def test_rule7_mxfp4_on_non_fp4_rejected() -> None:
    """MxFp4Deinterleave on a non-fp4 dtype is rejected."""
    nodes = [
        ChainNode(op_id=int(_Op.Source), params=_source_params(_SHAPE, _DTYPE_UINT8)),
        ChainNode(op_id=int(_Op.MxFp4Deinterleave), params=b""),
        ChainNode(op_id=int(_Op.BytePassthrough), params=b""),
    ]
    edges = [ChainEdge(0, 0, 1, 0), ChainEdge(1, 0, 2, 0)]
    terminals = [TerminalRef(node_idx=2, output_idx=0, role=_role_value_intn(8))]
    chain = Chain(nodes=nodes, edges=edges, terminals=terminals)
    assert not is_legal_chain(chain, _minimal_opts())


# ---------------------------------------------------------------------------
# Rule 8: XorDelta / FloatDelta when allow_cross_tensor=False
# ---------------------------------------------------------------------------


def test_rule8_xor_delta_rejected_when_cross_tensor_off() -> None:
    """XorDelta edge is rejected when allow_cross_tensor=False."""
    opts = ExploreOptions(allow_cross_tensor=False)
    nodes = [
        ChainNode(op_id=int(_Op.Source), params=_source_params(_SHAPE, _DTYPE_UINT8)),
        ChainNode(op_id=int(_Op.XorDelta), params=b""),
        ChainNode(op_id=int(_Op.BytePassthrough), params=b""),
    ]
    edges = [ChainEdge(0, 0, 1, 0), ChainEdge(1, 0, 2, 0)]
    terminals = [TerminalRef(node_idx=2, output_idx=0, role=_role_value_intn(8))]
    chain = Chain(nodes=nodes, edges=edges, terminals=terminals)
    assert not is_legal_chain(chain, opts)


# ---------------------------------------------------------------------------
# Rule 9: Concat / Reshape when allow_cross_plane=False
# ---------------------------------------------------------------------------


def test_rule9_concat_rejected_when_cross_plane_off() -> None:
    """Concat edge is rejected when allow_cross_plane=False."""
    opts = ExploreOptions(allow_cross_plane=False)
    nodes = [
        ChainNode(op_id=int(_Op.Source), params=_source_params(_SHAPE, _DTYPE_UINT8)),
        ChainNode(op_id=int(_Op.Concat), params=b""),
        ChainNode(op_id=int(_Op.BytePassthrough), params=b""),
    ]
    edges = [ChainEdge(0, 0, 1, 0), ChainEdge(1, 0, 2, 0)]
    terminals = [TerminalRef(node_idx=2, output_idx=0, role=_role_value_intn(8))]
    chain = Chain(nodes=nodes, edges=edges, terminals=terminals)
    assert not is_legal_chain(chain, opts)


# ---------------------------------------------------------------------------
# Rule 10: zero terminals
# ---------------------------------------------------------------------------


def test_rule10_no_terminals_rejected() -> None:
    """Chain with no terminals is rejected."""
    chain = _passthrough_chain()
    chain_no_term = Chain(nodes=chain.nodes, edges=chain.edges, terminals=[])
    assert not is_legal_chain(chain_no_term, _minimal_opts())


# ---------------------------------------------------------------------------
# Valid chains accepted
# ---------------------------------------------------------------------------


def test_legal_passthrough_accepted() -> None:
    assert is_legal_chain(_passthrough_chain(), _minimal_opts())


def test_legal_nibble_split_accepted() -> None:
    assert is_legal_chain(_nibble_chain(_DTYPE_FP8_E4M3), _minimal_opts())


def test_legal_byte_split_accepted() -> None:
    assert is_legal_chain(_byte_split_chain(2), _minimal_opts())


def test_legal_bit_reorder_fp16_accepted() -> None:
    chain = _build_linear_chain(
        _DTYPE_BF16, _SHAPE, (int(_Op.BitReorderIeee16), int(_Op.BytePassthrough))
    )
    assert is_legal_chain(chain, _minimal_opts())


# ---------------------------------------------------------------------------
# _extract_source_dtype
# ---------------------------------------------------------------------------


def test_extract_source_dtype_bf16() -> None:
    chain = _passthrough_chain()
    # Change source to BF16
    chain.nodes[0] = ChainNode(
        op_id=int(_Op.Source), params=_source_params(_SHAPE, _DTYPE_BF16)
    )

    assert _extract_source_dtype(chain) == _DTYPE_BF16


def test_extract_source_dtype_empty_chain() -> None:
    assert _extract_source_dtype(Chain(nodes=[], edges=[], terminals=[])) is None


# ---------------------------------------------------------------------------
# explore_chains: basic enumeration
# ---------------------------------------------------------------------------


def test_explore_chains_yields_candidates_for_uint8() -> None:
    """Explorer yields at least one candidate for uint8."""
    opts = ExploreOptions(max_candidates_per_dtype=5, time_budget_ms=1000)
    results = list(explore_chains(13, ClassifierRole.STANDARD, opts))  # 13 = Uint8
    assert len(results) >= 1


def test_explore_chains_yields_candidates_for_bf16() -> None:
    """Explorer yields at least one candidate for bf16."""
    opts = ExploreOptions(max_candidates_per_dtype=5, time_budget_ms=1000)
    results = list(explore_chains(6, ClassifierRole.STANDARD, opts))  # 6 = BF16
    assert len(results) >= 1


def test_explore_chains_builders_are_callable() -> None:
    """Each yielded value is a callable that produces a Chain."""
    opts = ExploreOptions(max_candidates_per_dtype=3, time_budget_ms=500)
    for builder in explore_chains(6, ClassifierRole.STANDARD, opts):
        chain = builder([4, 4])
        assert isinstance(chain, Chain)
        assert chain.nodes
        assert chain.terminals


def test_explore_chains_respects_max_candidates() -> None:
    """Number of yielded candidates does not exceed max_candidates_per_dtype."""
    limit = 4
    opts = ExploreOptions(max_candidates_per_dtype=limit, time_budget_ms=5000)
    results = list(explore_chains(6, ClassifierRole.STANDARD, opts))
    assert len(results) <= limit


def test_explore_chains_all_legal() -> None:
    """All explored candidates pass is_legal_chain."""
    opts = ExploreOptions(max_candidates_per_dtype=10, time_budget_ms=2000)
    for builder in explore_chains(6, ClassifierRole.STANDARD, opts):
        chain = builder([4, 4])
        assert is_legal_chain(chain, opts), f"Illegal chain yielded: {chain}"


def test_explore_chains_wire_bytes_non_empty() -> None:
    """Each builder produces a chain that serialises to non-empty bytes."""
    opts = ExploreOptions(max_candidates_per_dtype=3, time_budget_ms=500)
    for builder in explore_chains(6, ClassifierRole.STANDARD, opts):
        assert builder([4, 4]).to_bytes()


def test_explore_chains_bit_reorder_present_for_bf16() -> None:
    """At least one BF16 candidate uses BitReorderIeee16."""
    opts = ExploreOptions(max_candidates_per_dtype=16, time_budget_ms=2000)
    has_bit_reorder = False
    for builder in explore_chains(6, ClassifierRole.STANDARD, opts):
        chain = builder([4, 4])
        if any(n.op_id == int(_Op.BitReorderIeee16) for n in chain.nodes):
            has_bit_reorder = True
            break
    assert has_bit_reorder, "Expected at least one BF16 chain with BitReorderIeee16"
