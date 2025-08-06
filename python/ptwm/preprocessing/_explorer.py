"""PPG generative chain explorer.

Enumerates legal chain compositions via a depth-bounded BFS over the op
catalogue, starting from a ``Source(dtype)`` node, and validates each
candidate against the 10 pruning rules before yielding it.

Public API:
* :class:`ExploreOptions` — search-bound configuration.
* :func:`is_legal_chain` — standalone rule checker.
* :func:`explore_chains` — BFS enumerator; yields :data:`ChainBuilder`
  callables ``(shape: list[int]) -> Chain``.
"""

from __future__ import annotations

import struct
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass

from ._chains import (
    _CHAIN_DTYPE,
    Chain,
    ChainBuilder,
    ChainEdge,
    ChainNode,
    ClassifierRole,
    TerminalRef,
    _Op,
    _role_nibble,
    _role_value_intn,
    _source_params,
)

__all__ = ["ExploreOptions", "explore_chains", "is_legal_chain"]

# ---------------------------------------------------------------------------
# Op classification sets
# ---------------------------------------------------------------------------

_BIT_REORDER_OPS: frozenset[int] = frozenset(
    {
        int(_Op.BitReorderIeee16),
        int(_Op.BitReorderIeee32),
        int(_Op.BitReorderFp8E4M3),
        int(_Op.BitReorderFp8E5M2),
    }
)
_CROSS_TENSOR_OPS: frozenset[int] = frozenset({int(_Op.XorDelta), int(_Op.FloatDelta)})
_CROSS_PLANE_OPS: frozenset[int] = frozenset({int(_Op.Concat), int(_Op.Reshape)})

# Internal dtype codes partitioned by BitReorder applicability (rule 6)
_FP16_INTERNAL: frozenset[int] = frozenset({0x0002, 0x000F})
_FP32_INTERNAL: frozenset[int] = frozenset({0x0003})
_FP8_E4M3_INTERNAL: frozenset[int] = frozenset({0x0010})
_FP8_E5M2_INTERNAL: frozenset[int] = frozenset({0x0011})
_FP4_INTERNAL: frozenset[int] = frozenset({0x001F})

# Map internal dtype → the applicable BitReorder op
_BIT_REORDER_BY_INTERNAL: dict[int, int] = {
    0x0002: int(_Op.BitReorderIeee16),
    0x000F: int(_Op.BitReorderIeee16),
    0x0003: int(_Op.BitReorderIeee32),
    0x0010: int(_Op.BitReorderFp8E4M3),
    0x0011: int(_Op.BitReorderFp8E5M2),
}


# ---------------------------------------------------------------------------
# ExploreOptions
# ---------------------------------------------------------------------------


@dataclass
class ExploreOptions:
    """Search-bound configuration for the generative explorer."""

    max_nodes: int = 8
    max_depth: int = 5
    max_candidates_per_dtype: int = 32
    allow_cross_tensor: bool = False
    allow_cross_plane: bool = False
    time_budget_ms: int = 5000


# ---------------------------------------------------------------------------
# Pruning rules
# ---------------------------------------------------------------------------


def is_legal_chain(chain: Chain, opts: ExploreOptions | None = None) -> bool:  # noqa: PLR0911
    """Return ``True`` iff *chain* passes all 10 pruning rules."""
    if opts is None:
        opts = ExploreOptions()

    # Rule 1: total node count
    if len(chain.nodes) > opts.max_nodes:
        return False

    # Rule 10: zero terminals
    if not chain.terminals:
        return False

    # Build successor adjacency for depth calculation
    succ: dict[int, list[int]] = {i: [] for i in range(len(chain.nodes))}
    for edge in chain.edges:
        succ[edge.src_node].append(edge.dst_node)

    # Rule 2: longest source-to-any-node path > max_depth
    depths: dict[int, int] = {0: 0}
    bfs_q: deque[int] = deque([0])
    while bfs_q:
        cur = bfs_q.popleft()
        for nxt in succ.get(cur, []):
            if nxt not in depths:
                depths[nxt] = depths[cur] + 1
                bfs_q.append(nxt)
    if max(depths.values(), default=0) > opts.max_depth:
        return False

    op_ids = [n.op_id for n in chain.nodes]

    for edge in chain.edges:
        src_op = op_ids[edge.src_node]
        dst_op = op_ids[edge.dst_node]

        # Rule 3: two consecutive BitReorder* ops
        if src_op in _BIT_REORDER_OPS and dst_op in _BIT_REORDER_OPS:
            return False

        # Rule 4: ByteSplit → ByteSplit on same plane
        if src_op == int(_Op.ByteSplit) and dst_op == int(_Op.ByteSplit):
            return False

        # Rule 5: NibbleSplit output fed into NibbleSplit (already nibble-packed)
        if src_op == int(_Op.NibbleSplit) and dst_op == int(_Op.NibbleSplit):
            return False

        # Rule 8: XorDelta / FloatDelta when allow_cross_tensor is False
        if not opts.allow_cross_tensor and dst_op in _CROSS_TENSOR_OPS:
            return False

        # Rule 9: Concat / Reshape when allow_cross_plane is False
        if not opts.allow_cross_plane and dst_op in _CROSS_PLANE_OPS:
            return False

    # Rules 6 & 7: BitReorder / MxFp4Deinterleave dtype compatibility
    source_dtype = _extract_source_dtype(chain)
    if source_dtype is not None:
        for node in chain.nodes:
            op = node.op_id
            if op == int(_Op.BitReorderIeee16) and source_dtype not in _FP16_INTERNAL:
                return False  # rule 6
            if op == int(_Op.BitReorderIeee32) and source_dtype not in _FP32_INTERNAL:
                return False  # rule 6
            if (
                op == int(_Op.BitReorderFp8E4M3)
                and source_dtype not in _FP8_E4M3_INTERNAL
            ):
                return False  # rule 6
            if (
                op == int(_Op.BitReorderFp8E5M2)
                and source_dtype not in _FP8_E5M2_INTERNAL
            ):
                return False  # rule 6
            if op == int(_Op.MxFp4Deinterleave) and source_dtype not in _FP4_INTERNAL:
                return False  # rule 7

    # Connectivity guard: every node output must be consumed by an edge or
    # a terminal. The Rust validator rejects unconsumed outputs at compress
    # time; mirroring the rule here keeps the BFS from yielding candidates
    # that downstream code would discard.
    consumed_outputs: set[tuple[int, int]] = set()
    for edge in chain.edges:
        consumed_outputs.add((edge.src_node, edge.src_output_idx))
    for term in chain.terminals:
        consumed_outputs.add((term.node_idx, term.output_idx))

    for node_idx, node in enumerate(chain.nodes):
        n_outputs = _expected_output_count(node)
        for out_idx in range(n_outputs):
            if (node_idx, out_idx) not in consumed_outputs:
                return False

    return True


def _expected_output_count(node: ChainNode) -> int:
    """Return the number of outputs the given op produces."""
    op = node.op_id
    if op == int(_Op.ByteSplit):
        # `params[0]` carries the plane count.
        return int(node.params[0]) if node.params else 1
    if op == int(_Op.NibbleSplit):
        return 2
    return 1


def _extract_source_dtype(chain: Chain) -> int | None:
    """Parse the internal dtype code from the Source node params."""
    if not chain.nodes:
        return None
    params = chain.nodes[0].params
    if len(params) < 1:
        return None
    dim_count = params[0]
    offset = 1 + 4 * dim_count
    if len(params) < offset + 2:
        return None
    return struct.unpack_from("<H", params, offset)[0]


# ---------------------------------------------------------------------------
# Linear chain builder (used by the enumerator)
# ---------------------------------------------------------------------------


def _build_linear_chain(
    source_dtype: int,
    shape: list[int],
    ops: tuple[int, ...],
    byte_split_planes: int = 1,
) -> Chain:
    """Build a linear Chain: Source → op₁ → … → opₙ → terminal(s)."""
    nodes: list[ChainNode] = [
        ChainNode(
            op_id=int(_Op.Source),
            params=_source_params(shape, source_dtype),
        )
    ]
    for op in ops:
        params = bytes([byte_split_planes]) if op == int(_Op.ByteSplit) else b""
        nodes.append(ChainNode(op_id=op, params=params))

    edges = [
        ChainEdge(
            src_node=i,
            src_output_idx=0,
            dst_node=i + 1,
            dst_input_idx=0,
        )
        for i in range(len(nodes) - 1)
    ]

    last_idx = len(nodes) - 1
    last_op = ops[-1] if ops else int(_Op.Source)

    if last_op == int(_Op.NibbleSplit):
        terminals: list[TerminalRef] = [
            TerminalRef(node_idx=last_idx, output_idx=0, role=_role_nibble(0)),
            TerminalRef(node_idx=last_idx, output_idx=1, role=_role_nibble(1)),
        ]
    elif last_op == int(_Op.ByteSplit):
        terminals = [
            TerminalRef(node_idx=last_idx, output_idx=i, role=_role_value_intn(8))
            for i in range(byte_split_planes)
        ]
    else:
        terminals = [
            TerminalRef(node_idx=last_idx, output_idx=0, role=_role_value_intn(8))
        ]

    return Chain(nodes=nodes, edges=edges, terminals=terminals)


# ---------------------------------------------------------------------------
# Generative BFS enumerator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _BfsState:
    ops: tuple[int, ...]
    had_bit_reorder: bool
    had_byte_split: bool
    had_nibble_split: bool
    byte_split_planes: int  # 1 unless ByteSplit was applied


def explore_chains(
    dtype_code: int,
    role: ClassifierRole,  # noqa: ARG001 — reserved for role-specialised pruning
    opts: ExploreOptions | None = None,
) -> Iterator[ChainBuilder]:
    """Enumerate legal chain builders for *(dtype_code, role)*.

    Each yielded value is a :data:`ChainBuilder` callable
    ``(shape: list[int]) -> Chain``. Yielded callables also carry a
    ``template`` attribute (an `~ptwm.preprocessing._cache.CacheEntry`) so the
    user-local chain cache can record the structural template without re-
    running the BFS.
    """
    if opts is None:
        opts = ExploreOptions()

    # Late import: _cache imports from _chains (this module's sibling) and
    # also reaches back into _build_linear_chain here — keep the import
    # local to avoid a circular load.
    from ._cache import CacheEntry  # noqa: PLC0415

    internal_dtype = _CHAIN_DTYPE.get(dtype_code, 0x0006)
    bit_reorder_op = _BIT_REORDER_BY_INTERNAL.get(internal_dtype)

    initial = _BfsState(
        ops=(),
        had_bit_reorder=False,
        had_byte_split=False,
        had_nibble_split=False,
        byte_split_planes=1,
    )

    bfs: deque[_BfsState] = deque([initial])
    count = 0
    deadline = time.monotonic() + opts.time_budget_ms / 1000.0
    seen: set[tuple[int | tuple[int, ...], ...]] = set()

    while bfs and count < opts.max_candidates_per_dtype:
        if time.monotonic() > deadline:
            break

        state = bfs.popleft()
        key = (state.ops, state.byte_split_planes)

        # Yield as a complete chain when ops is non-empty and the last op
        # is a splitting or passthrough op (not a pure reorder step).
        if state.ops and key not in seen:
            seen.add(key)
            last_op = state.ops[-1]
            if last_op not in _BIT_REORDER_OPS:
                _ops = state.ops
                _planes = state.byte_split_planes

                def _builder(
                    shape: list[int],
                    _dtype: int = internal_dtype,
                    _o: tuple[int, ...] = _ops,
                    _p: int = _planes,
                ) -> Chain:
                    return _build_linear_chain(_dtype, shape, _o, _p)

                _builder.template = CacheEntry(  # type: ignore[attr-defined]
                    internal_dtype=internal_dtype,
                    ops=_ops,
                    byte_split_planes=_planes,
                )

                test = _build_linear_chain(internal_dtype, [1], _ops, _planes)
                if is_legal_chain(test, opts):
                    count += 1
                    yield _builder

        # Stop expanding when depth or node limit is reached
        if len(state.ops) >= opts.max_depth or len(state.ops) + 2 > opts.max_nodes:
            continue

        # --- Enumerate successor ops ---

        # BitReorder: applicable only before any split, once per chain
        if (
            not state.had_bit_reorder
            and not state.had_byte_split
            and not state.had_nibble_split
            and bit_reorder_op is not None
        ):
            bfs.append(
                _BfsState(
                    ops=state.ops + (bit_reorder_op,),
                    had_bit_reorder=True,
                    had_byte_split=state.had_byte_split,
                    had_nibble_split=state.had_nibble_split,
                    byte_split_planes=state.byte_split_planes,
                )
            )

        # BytePassthrough: always eligible
        bfs.append(
            _BfsState(
                ops=state.ops + (int(_Op.BytePassthrough),),
                had_bit_reorder=state.had_bit_reorder,
                had_byte_split=state.had_byte_split,
                had_nibble_split=state.had_nibble_split,
                byte_split_planes=state.byte_split_planes,
            )
        )

        # ByteSplit(n): only once, and not after NibbleSplit
        if not state.had_byte_split and not state.had_nibble_split:
            for n_planes in (2, 4):
                bfs.append(
                    _BfsState(
                        ops=state.ops + (int(_Op.ByteSplit),),
                        had_bit_reorder=state.had_bit_reorder,
                        had_byte_split=True,
                        had_nibble_split=False,
                        byte_split_planes=n_planes,
                    )
                )

        # NibbleSplit: only once, and not after ByteSplit
        if not state.had_nibble_split and not state.had_byte_split:
            bfs.append(
                _BfsState(
                    ops=state.ops + (int(_Op.NibbleSplit),),
                    had_bit_reorder=state.had_bit_reorder,
                    had_byte_split=False,
                    had_nibble_split=True,
                    byte_split_planes=state.byte_split_planes,
                )
            )
