//! PPG DAG runtime — forward executor, inverse executor, fusion picker.

use std::collections::HashMap;
use std::sync::Arc;

use crate::chain::dispatch::op_from_id;
use crate::chain::graph::Chain;
use crate::error::PtwmCoreError;
use crate::transforms::fusion::{
    FusedFp8E4M3NibbleSplit, FusedFp8E5M2NibbleSplit, FusedIeee16ByteSplit2, FusedIeee32ByteSplit4,
};
use crate::transforms::op::{OpId, Plane};
use crate::types::descriptor::PlaneDescriptor;
use crate::types::role::Role;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a topological ordering of `chain.nodes` using Kahn's algorithm.
/// Returns node indices in dependency-first order.
///
/// Bounds-checks `edge.src_node` / `edge.dst_node` so a malformed wire-
/// derived chain that bypasses [`validate_chain`] cannot panic via
/// out-of-range indexing into `adj` / `in_degree`.
fn topo_sort(chain: &Chain) -> Result<Vec<usize>, PtwmCoreError> {
    let n = chain.nodes.len();
    let mut in_degree = vec![0usize; n];
    let mut adj: Vec<Vec<usize>> = vec![vec![]; n];

    for edge in &chain.edges {
        let src = edge.src_node as usize;
        let dst = edge.dst_node as usize;
        if src >= n || dst >= n {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "topo_sort: edge references out-of-range node (src={src}, dst={dst}, n={n})"
            )));
        }
        adj[src].push(dst);
        in_degree[dst] += 1;
    }

    let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
    let mut order = Vec::with_capacity(n);

    while let Some(u) = queue.pop() {
        order.push(u);
        for &v in &adj[u] {
            in_degree[v] -= 1;
            if in_degree[v] == 0 {
                queue.push(v);
            }
        }
    }

    if order.len() != n {
        return Err(PtwmCoreError::InvalidContainer(
            "runtime: cycle detected in chain (topological sort failed)".into(),
        ));
    }

    Ok(order)
}

/// Returns `true` when the op is a cross-tensor delta op that requires a dep plane.
fn is_delta_op(op_id: OpId) -> bool {
    matches!(op_id, OpId::XorDelta | OpId::FloatDelta)
}

/// Extract dep_idx from the raw params of a delta op.
fn delta_dep_idx(params: &[u8]) -> u8 {
    // For both XorDelta and FloatDelta, dep_idx is the first byte of params.
    params[0]
}

// ---------------------------------------------------------------------------
// Forward executor
// ---------------------------------------------------------------------------

/// Context supplied by the caller for a forward (encode) pass.
///
/// `source_bytes` is borrowed: the runtime owns the single allocation that
/// seeds node 0's `Plane`, sparing callers (e.g. multi-chain trial-encode)
/// from cloning the raw tensor for every candidate.
pub struct ForwardContext<'a> {
    /// Raw bytes of the source tensor.
    pub source_bytes: &'a [u8],
    /// Descriptor for the source tensor plane.
    pub source_descriptor: PlaneDescriptor,
    /// Reference tensor planes for cross-tensor delta ops, indexed by dep_idx.
    pub deps: &'a [Plane],
}

/// Output of a forward pass.
pub struct ForwardResult {
    /// One `(Plane, Role)` per entry in `chain.terminals`, in terminal order.
    pub terminal_planes: Vec<(Plane, Role)>,
}

/// Execute the chain in forward (encode) order.
///
/// Node 0 must be `Source`; its output wraps `ctx.source_bytes` in a
/// `Plane`. Each subsequent node comes from `op_from_id`, and the runtime
/// calls `forward` with the gathered input planes. Terminal refs collect
/// at the end in `chain.terminals` order.
pub fn forward_chain(chain: &Chain, ctx: &ForwardContext) -> Result<ForwardResult, PtwmCoreError> {
    if chain.nodes.is_empty() {
        return Err(PtwmCoreError::InvalidContainer(
            "forward_chain: chain has no nodes".into(),
        ));
    }
    if chain.nodes[0].op != OpId::Source {
        return Err(PtwmCoreError::InvalidContainer(
            "forward_chain: node[0] must be Source".into(),
        ));
    }

    let topo = topo_sort(chain)?;

    // Pre-compute fused pairs: map consumer_node → FusedExecution.
    // The standard forward loop skips producer nodes in a fused pair; the
    // fused path runs both transforms at the consumer node.
    let mut fused_consumers: HashMap<u8, FusedExecution> = HashMap::new();
    let mut fused_producers: std::collections::HashSet<u8> = std::collections::HashSet::new();
    for &node_idx in &topo {
        if node_idx == 0 {
            continue;
        }
        if let Some(fused) = try_fuse_at(chain, node_idx as u8) {
            fused_producers.insert(fused.producer_node);
            fused_consumers.insert(node_idx as u8, fused);
        }
    }

    // Working state: (node_idx, output_idx) → Plane
    let mut state: HashMap<(u8, u8), Plane> = HashMap::new();

    // Seed the Source node's output (node 0, output 0).
    state.insert(
        (0u8, 0u8),
        Plane {
            bytes: Arc::from(ctx.source_bytes),
            descriptor: ctx.source_descriptor.clone(),
        },
    );

    for &node_idx in &topo {
        if node_idx == 0 {
            // Source: already seeded above.
            continue;
        }

        // Skip producer nodes in fused pairs — the consumer handles them.
        if fused_producers.contains(&(node_idx as u8)) {
            continue;
        }

        // Check whether this node is a fusion consumer.
        if let Some(fused) = fused_consumers.remove(&(node_idx as u8)) {
            // The fused input is the plane feeding the PRODUCER node.
            let producer_input_edge = chain
                .edges
                .iter()
                .find(|e| e.dst_node == fused.producer_node)
                .ok_or_else(|| {
                    PtwmCoreError::InvalidContainer(format!(
                        "forward_chain: fused: cannot find input edge for producer node {}",
                        fused.producer_node
                    ))
                })?;

            let input_plane = state
                .get(&(
                    producer_input_edge.src_node,
                    producer_input_edge.src_output_idx,
                ))
                .ok_or_else(|| {
                    PtwmCoreError::InvalidContainer(format!(
                        "forward_chain: fused: missing input plane ({}, {})",
                        producer_input_edge.src_node, producer_input_edge.src_output_idx
                    ))
                })?
                .clone();

            let outputs = (fused.run_forward)(&input_plane)?;

            // Insert outputs at the consumer node's output slots.
            for (out_idx, plane) in outputs.into_iter().enumerate() {
                state.insert((fused.consumer_node, out_idx as u8), plane);
            }

            continue;
        }

        let node = &chain.nodes[node_idx];

        // Gather input planes: edges targeting this node, sorted by dst_input_idx.
        let mut input_edges: Vec<_> = chain
            .edges
            .iter()
            .filter(|e| e.dst_node as usize == node_idx)
            .collect();
        input_edges.sort_by_key(|e| e.dst_input_idx);

        let mut inputs: Vec<Plane> = Vec::with_capacity(input_edges.len() + 1);
        for edge in &input_edges {
            let key = (edge.src_node, edge.src_output_idx);
            let mut plane = state
                .get(&key)
                .ok_or_else(|| {
                    PtwmCoreError::InvalidContainer(format!(
                        "forward_chain: missing plane for node={}, output={}",
                        edge.src_node, edge.src_output_idx
                    ))
                })?
                .clone();
            if let Some(ref role) = edge.role_override {
                plane.descriptor.role = role.clone();
            }
            inputs.push(plane);
        }

        // Cross-tensor delta ops require the dep plane appended as inputs[1].
        if is_delta_op(node.op) && !node.params.is_empty() {
            let dep_idx = delta_dep_idx(&node.params) as usize;
            let dep_plane = ctx.deps.get(dep_idx).ok_or_else(|| {
                PtwmCoreError::InvalidContainer(format!(
                    "forward_chain: node[{node_idx}] dep_idx={dep_idx} out of range (deps.len={})",
                    ctx.deps.len()
                ))
            })?;
            inputs.push(dep_plane.clone());
        }

        let op = op_from_id(node.op, &node.params)?;
        let outputs = op.forward(&inputs)?;

        for (out_idx, plane) in outputs.into_iter().enumerate() {
            state.insert((node_idx as u8, out_idx as u8), plane);
        }
    }

    // Collect terminal planes in terminal order. Apply the terminal's
    // declared role to the plane's descriptor so downstream codec dispatch
    // (which keys on `descriptor.role`) sees the post-chain canonical role
    // rather than the role propagated up from `Source`. Without this
    // re-tag, e.g. a Source(FP8E4M3) → BytePassthrough → Terminal(Scale
    // {E4M3}) chain would still dispatch with `Role::Raw` and
    // Order1ScaleAC would never accept the descriptor.
    let mut terminal_planes = Vec::with_capacity(chain.terminals.len());
    for term in &chain.terminals {
        let key = (term.node_idx, term.output_idx);
        let mut plane = state
            .get(&key)
            .ok_or_else(|| {
                PtwmCoreError::InvalidContainer(format!(
                    "forward_chain: terminal references missing plane (node={}, output={})",
                    term.node_idx, term.output_idx
                ))
            })?
            .clone();
        plane.descriptor.role = term.role.clone();
        terminal_planes.push((plane, term.role.clone()));
    }

    Ok(ForwardResult { terminal_planes })
}

// ---------------------------------------------------------------------------
// Descriptor reconstruction (per-terminal, derived from chain structure)
// ---------------------------------------------------------------------------

/// Descriptor of the plane the chain's `Source` node emits.
///
/// Checks that the chain is non-empty and that node 0 is `Source`. The
/// length comes from the Source params (shape × dtype) with checked
/// arithmetic, so a corrupt shape returns an error instead of wrapping.
pub fn chain_source_descriptor(chain: &Chain) -> Result<PlaneDescriptor, PtwmCoreError> {
    let first = chain.nodes.first().ok_or_else(|| {
        PtwmCoreError::InvalidContainer("chain descriptors: chain has no nodes".into())
    })?;
    if first.op != OpId::Source {
        return Err(PtwmCoreError::InvalidContainer(
            "chain descriptors: node[0] must be Source".into(),
        ));
    }
    let mut descs = op_from_id(OpId::Source, &first.params)?.propagate_descriptors(&[])?;
    if descs.len() != 1 {
        return Err(PtwmCoreError::InvalidContainer(format!(
            "chain descriptors: Source op produced {} descriptors (expected 1)",
            descs.len()
        )));
    }
    Ok(descs.remove(0))
}

/// Largest plane, at any node of a chain, that a decoder accepts for a
/// tensor of `orig_size` raw bytes.
///
/// The factor 16 is the product of two cases that existing files can
/// combine:
///
/// - A Source that overstates the raw size by up to 8×. Only files written
///   by the Python byte-input path before the P4 fix do this, and their
///   chains are 1-D byte passthroughs or byte splits, whose planes are no
///   larger than the Source.
/// - Ops that expand a plane. MxFp4 deinterleave at most doubles it. The
///   spherical and alpha-stable scale planes can reach 4× their input (one
///   8-byte scale per row of a single 16-bit element), but their builders
///   accept only 2-D shapes, which the Python path always sizes exactly.
///
/// So no chain a writer emits multiplies the two, and 16× leaves 2× of
/// headroom over the largest case (8× overstated Source, or a 4× scale
/// plane). The constant 4096 covers block padding such as the
/// BurrowsWheeler index (4 bytes per 256 KiB block). Decoders use this one
/// helper so the bound cannot drift between layouts.
pub fn max_plane_len(orig_size: u64) -> u64 {
    orig_size.saturating_mul(16).saturating_add(4096)
}

/// Recover the [`PlaneDescriptor`] of every chain terminal in
/// `chain.terminals` order, without the tensor bytes and without allocating
/// any plane buffer.
///
/// Same as [`propagate_terminal_descriptors_bounded`] with no bound.
pub fn propagate_terminal_descriptors(
    chain: &Chain,
) -> Result<Vec<PlaneDescriptor>, PtwmCoreError> {
    propagate_terminal_descriptors_bounded(chain, u64::MAX)
}

/// Recover terminal descriptors symbolically and reject any node output
/// longer than `max_plane_len` bytes.
///
/// The walk calls `propagate_descriptors` on every node in topological
/// order. It mirrors [`forward_chain`]:
///
/// - inputs are gathered from the incoming edges sorted by `dst_input_idx`,
///   and an edge's `role_override` replaces the input's role;
/// - fused pairs (see [`try_fuse_at`]) need no special case: fusion applies
///   only to edges without a role override, and every fused kernel emits the
///   same descriptors as the unfused producer and consumer;
/// - each terminal's descriptor takes the terminal's role, as
///   `forward_chain` re-tags terminal planes.
///
/// No op's output size, layout or role depends on plane contents, so the
/// result equals the descriptors `forward_chain` produces on a tensor of
/// the Source's declared size. The one data-dependent field is the estimate
/// `EntropyEstimate` writes into `vendor_bytes`: this walk reports zeros
/// (its `propagate_descriptors` value). The earlier decoder computed it
/// from placeholder planes, which was not the encode-time value either, and
/// no codec or decoder reads it.
///
/// Cross-tensor delta ops (`XorDelta`, `FloatDelta`) need dependency planes
/// and return an error here.
///
/// The Source descriptor is checked against the bound first, so a Source
/// that declares gigabytes fails before any other node is visited.
pub fn propagate_terminal_descriptors_bounded(
    chain: &Chain,
    max_plane_len: u64,
) -> Result<Vec<PlaneDescriptor>, PtwmCoreError> {
    let source = chain_source_descriptor(chain)?;
    if source.length_bytes > max_plane_len {
        return Err(PtwmCoreError::InvalidContainer(format!(
            "chain descriptors: Source declares {} bytes, bound is {max_plane_len}",
            source.length_bytes
        )));
    }
    let topo = topo_sort(chain)?;
    let mut outs: Vec<Option<Vec<PlaneDescriptor>>> = vec![None; chain.nodes.len()];
    for &node_idx in &topo {
        if node_idx == 0 {
            outs[0] = Some(vec![source.clone()]);
            continue;
        }
        let node = &chain.nodes[node_idx];
        if is_delta_op(node.op) {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "chain descriptors: node[{node_idx}] ({:?}) is a cross-tensor delta op \
                 and needs dependency planes",
                node.op
            )));
        }
        let mut input_edges: Vec<_> = chain
            .edges
            .iter()
            .filter(|e| e.dst_node as usize == node_idx)
            .collect();
        input_edges.sort_by_key(|e| e.dst_input_idx);
        let mut inputs = Vec::with_capacity(input_edges.len());
        for edge in input_edges {
            let mut desc = outs
                .get(edge.src_node as usize)
                .and_then(|o| o.as_ref())
                .and_then(|o| o.get(edge.src_output_idx as usize))
                .cloned()
                .ok_or_else(|| {
                    PtwmCoreError::InvalidContainer(format!(
                        "chain descriptors: missing plane for node={}, output={}",
                        edge.src_node, edge.src_output_idx
                    ))
                })?;
            if let Some(ref role) = edge.role_override {
                desc.role = role.clone();
            }
            inputs.push(desc);
        }
        let produced = op_from_id(node.op, &node.params)?.propagate_descriptors(&inputs)?;
        if let Some(d) = produced.iter().find(|d| d.length_bytes > max_plane_len) {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "chain descriptors: node[{node_idx}] ({:?}) declares a {}-byte plane, \
                 bound is {max_plane_len}",
                node.op, d.length_bytes
            )));
        }
        outs[node_idx] = Some(produced);
    }
    chain
        .terminals
        .iter()
        .map(|term| {
            let mut desc = outs
                .get(term.node_idx as usize)
                .and_then(|o| o.as_ref())
                .and_then(|o| o.get(term.output_idx as usize))
                .cloned()
                .ok_or_else(|| {
                    PtwmCoreError::InvalidContainer(format!(
                        "chain descriptors: terminal references missing plane \
                         (node={}, output={})",
                        term.node_idx, term.output_idx
                    ))
                })?;
            desc.role = term.role.clone();
            Ok(desc)
        })
        .collect()
}

/// Recover terminal descriptors. Kept for callers of the earlier API; it is
/// [`propagate_terminal_descriptors`].
pub fn chain_terminal_descriptors(chain: &Chain) -> Result<Vec<PlaneDescriptor>, PtwmCoreError> {
    propagate_terminal_descriptors(chain)
}

// ---------------------------------------------------------------------------
// Inverse executor
// ---------------------------------------------------------------------------

/// Forward input descriptors of every node, indexed by node, computed
/// symbolically from the Source params as [`propagate_terminal_descriptors`]
/// does.
///
/// The walk does not fail: a node gets `None` when the Source or an
/// upstream node has no descriptor, when the node is a cross-tensor delta
/// op (its descriptors need dependency planes), or when its
/// `propagate_descriptors` rejects the inputs. An op that receives `None`
/// in [`Op::inverse_with_input_descs`] falls back to its plain inverse.
///
/// A `Some` descriptor can still disagree with the real planes: legacy
/// files carry a Source that overstates or understates the raw size, and
/// the non-strict decoder accepts them. An op that uses the descriptors
/// must fall back to its plain inverse on such a mismatch, so the walk
/// never makes a decodable chain fail.
///
/// [`Op::inverse_with_input_descs`]: crate::transforms::op::Op::inverse_with_input_descs
fn symbolic_node_inputs(chain: &Chain, topo: &[usize]) -> Vec<Option<Vec<PlaneDescriptor>>> {
    let n = chain.nodes.len();
    let mut ins: Vec<Option<Vec<PlaneDescriptor>>> = vec![None; n];
    let mut outs: Vec<Option<Vec<PlaneDescriptor>>> = vec![None; n];
    let Ok(source) = chain_source_descriptor(chain) else {
        return ins;
    };
    for &node_idx in topo {
        if node_idx == 0 {
            outs[0] = Some(vec![source.clone()]);
            continue;
        }
        let node = &chain.nodes[node_idx];
        if is_delta_op(node.op) {
            continue;
        }
        let mut input_edges: Vec<_> = chain
            .edges
            .iter()
            .filter(|e| e.dst_node as usize == node_idx)
            .collect();
        input_edges.sort_by_key(|e| e.dst_input_idx);
        let inputs: Option<Vec<PlaneDescriptor>> = input_edges
            .iter()
            .map(|edge| {
                let mut desc = outs
                    .get(edge.src_node as usize)?
                    .as_ref()?
                    .get(edge.src_output_idx as usize)?
                    .clone();
                if let Some(ref role) = edge.role_override {
                    desc.role = role.clone();
                }
                Some(desc)
            })
            .collect();
        let Some(inputs) = inputs else {
            continue;
        };
        // A node whose `propagate_descriptors` rejects its inputs gets no
        // input descriptors either, so its op takes the plain inverse.
        if let Ok(produced) =
            op_from_id(node.op, &node.params).and_then(|op| op.propagate_descriptors(&inputs))
        {
            outs[node_idx] = Some(produced);
            ins[node_idx] = Some(inputs);
        }
    }
    ins
}

/// Context supplied by the caller for an inverse (decode) pass.
pub struct InverseContext<'a> {
    /// Decoded terminal planes, in the same order as `chain.terminals`.
    pub terminal_planes: &'a [Plane],
    /// Reference tensor planes for cross-tensor delta ops.
    pub deps: &'a [Plane],
}

/// Execute the chain in inverse (decode) order, reconstructing the source plane.
///
/// The traversal walks nodes in reverse topological order. For each node,
/// `op.inverse` runs on the node's output planes (from the working state)
/// to reconstruct the input planes, then stores them under the
/// corresponding source edges.
///
/// After the traversal, the plane at `(0, 0)` is the reconstructed source.
pub fn inverse_chain(chain: &Chain, ctx: &InverseContext) -> Result<Plane, PtwmCoreError> {
    if chain.nodes.is_empty() {
        return Err(PtwmCoreError::InvalidContainer(
            "inverse_chain: chain has no nodes".into(),
        ));
    }
    if chain.nodes[0].op != OpId::Source {
        return Err(PtwmCoreError::InvalidContainer(
            "inverse_chain: node[0] must be Source".into(),
        ));
    }
    if ctx.terminal_planes.len() != chain.terminals.len() {
        return Err(PtwmCoreError::InvalidContainer(format!(
            "inverse_chain: terminal_planes.len()={} != chain.terminals.len()={}",
            ctx.terminal_planes.len(),
            chain.terminals.len()
        )));
    }

    let topo = topo_sort(chain)?;

    // Pre-compute fused pairs for the inverse pass. In reverse order, the
    // fused inverse reconstructs the producer's input directly when the
    // traversal hits the consumer node.
    let mut fused_consumers_inv: HashMap<u8, FusedExecution> = HashMap::new();
    let mut fused_producers_inv: std::collections::HashSet<u8> = std::collections::HashSet::new();
    for &node_idx in &topo {
        if node_idx == 0 {
            continue;
        }
        if let Some(fused) = try_fuse_at(chain, node_idx as u8) {
            fused_producers_inv.insert(fused.producer_node);
            fused_consumers_inv.insert(node_idx as u8, fused);
        }
    }

    // Forward input descriptors per node, for ops whose outputs do not fix
    // the input size (`IndexBitwidthPack` pads its last byte).
    let input_descs = symbolic_node_inputs(chain, &topo);

    // Working state: (node_idx, output_idx) → Plane
    let mut state: HashMap<(u8, u8), Plane> = HashMap::new();

    // Seed from terminal planes.
    for (term, plane) in chain.terminals.iter().zip(ctx.terminal_planes.iter()) {
        state.insert((term.node_idx, term.output_idx), plane.clone());
    }

    // Traverse in reverse topological order.
    for &node_idx in topo.iter().rev() {
        if node_idx == 0 {
            // Source: its "input" is what we want to recover. The node
            // consuming Source's output has already run its inverse and
            // stored the recovered plane at (0, 0). Nothing to do.
            continue;
        }

        // Skip producer nodes — the consumer handles their inverse.
        if fused_producers_inv.contains(&(node_idx as u8)) {
            continue;
        }

        // Check whether this is a fusion consumer node in the inverse pass.
        if let Some(fused) = fused_consumers_inv.remove(&(node_idx as u8)) {
            // Collect the consumer's output planes from state — seeded as
            // terminal planes or populated by downstream inverse steps.
            let mut consumer_output_keys: Vec<(u8, u8)> = chain
                .edges
                .iter()
                .filter(|e| e.src_node == fused.consumer_node)
                .map(|e| (e.src_node, e.src_output_idx))
                .collect();
            for t in &chain.terminals {
                if t.node_idx == fused.consumer_node {
                    consumer_output_keys.push((t.node_idx, t.output_idx));
                }
            }
            consumer_output_keys.sort();
            consumer_output_keys.dedup();

            let mut consumer_outputs: Vec<Plane> = Vec::with_capacity(consumer_output_keys.len());
            for key in &consumer_output_keys {
                let plane = state.get(key).ok_or_else(|| {
                    PtwmCoreError::InvalidContainer(format!(
                        "inverse_chain: fused: missing consumer output plane ({}, {})",
                        key.0, key.1
                    ))
                })?;
                consumer_outputs.push(plane.clone());
            }

            // Run the fused inverse to recover the producer's INPUT plane.
            let reconstructed = (fused.run_inverse)(&consumer_outputs)?;

            // Store at the edge feeding the producer.
            let producer_input_edge = chain
                .edges
                .iter()
                .find(|e| e.dst_node == fused.producer_node)
                .ok_or_else(|| {
                    PtwmCoreError::InvalidContainer(format!(
                        "inverse_chain: fused: cannot find input edge for producer node {}",
                        fused.producer_node
                    ))
                })?;
            state.insert(
                (
                    producer_input_edge.src_node,
                    producer_input_edge.src_output_idx,
                ),
                reconstructed,
            );

            continue;
        }

        let node = &chain.nodes[node_idx];

        // Standard path: gather this node's output planes from state.
        // Outputs are indexed by their position in the chain's output slots.
        let mut output_keys: Vec<(u8, u8, u8)> = Vec::new(); // (node_idx, output_idx, slot_order)
        for edge in chain
            .edges
            .iter()
            .filter(|e| e.src_node as usize == node_idx)
        {
            output_keys.push((edge.src_node, edge.src_output_idx, edge.src_output_idx));
        }
        for term in chain
            .terminals
            .iter()
            .filter(|t| t.node_idx as usize == node_idx)
        {
            output_keys.push((term.node_idx, term.output_idx, term.output_idx));
        }
        output_keys.sort_by_key(|&(_, _, slot)| slot);

        let mut op_outputs: Vec<Plane> = Vec::with_capacity(output_keys.len());
        for &(n, out_idx, _) in &output_keys {
            let plane = state.get(&(n, out_idx)).ok_or_else(|| {
                PtwmCoreError::InvalidContainer(format!(
                    "inverse_chain: missing output plane (node={n}, output={out_idx})"
                ))
            })?;
            op_outputs.push(plane.clone());
        }

        // Cross-tensor delta ops need the dep plane appended.
        if is_delta_op(node.op) && !node.params.is_empty() {
            let dep_idx = delta_dep_idx(&node.params) as usize;
            let dep_plane = ctx.deps.get(dep_idx).ok_or_else(|| {
                PtwmCoreError::InvalidContainer(format!(
                    "inverse_chain: node[{node_idx}] dep_idx={dep_idx} out of range (deps.len={})",
                    ctx.deps.len()
                ))
            })?;
            op_outputs.push(dep_plane.clone());
        }

        let op = op_from_id(node.op, &node.params)?;
        let reconstructed_inputs =
            op.inverse_with_input_descs(&op_outputs, input_descs[node_idx].as_deref())?;

        // Store reconstructed inputs at the source (producer) slots.
        let mut input_edges: Vec<_> = chain
            .edges
            .iter()
            .filter(|e| e.dst_node as usize == node_idx)
            .collect();
        input_edges.sort_by_key(|e| e.dst_input_idx);

        for (inv_plane, edge) in reconstructed_inputs.into_iter().zip(input_edges.iter()) {
            state.insert((edge.src_node, edge.src_output_idx), inv_plane);
        }
    }

    // The reconstructed source plane is at (0, 0).
    state.remove(&(0u8, 0u8)).ok_or_else(|| {
        PtwmCoreError::InvalidContainer(
            "inverse_chain: source plane (0, 0) was not reconstructed".into(),
        )
    })
}

// ---------------------------------------------------------------------------
// Fusion picker
// ---------------------------------------------------------------------------

/// A fused execution pair: producer node merged with consumer node.
pub struct FusedExecution {
    pub producer_node: u8,
    pub consumer_node: u8,
    /// Run the fused forward transform on the producer's input, producing the
    /// consumer's outputs.
    pub run_forward: Box<dyn Fn(&Plane) -> Result<Vec<Plane>, PtwmCoreError>>,
    /// Run the fused inverse transform on the consumer's outputs, reproducing
    /// the producer's input (a single plane, which will be stored at the
    /// producer node's input edge).
    pub run_inverse: Box<dyn Fn(&[Plane]) -> Result<Plane, PtwmCoreError>>,
}

/// Check whether `node_idx` is the CONSUMER in a fusable pair. If so,
/// return a `FusedExecution` with `producer_node` set to the upstream
/// node.
///
/// Conditions for fusion:
/// 1. The connecting edge has no `role_override` and empty `vendor_bytes`.
/// 2. No other edge or terminal consumes the producer's output.
/// 3. The (producer.op, consumer.op) pair is in the fusion table.
pub fn try_fuse_at(chain: &Chain, node_idx: u8) -> Option<FusedExecution> {
    // Find the single edge whose dst_node == node_idx.
    let connecting_edges: Vec<_> = chain
        .edges
        .iter()
        .filter(|e| e.dst_node == node_idx)
        .collect();

    // Must have exactly one input edge (the producer's output).
    if connecting_edges.len() != 1 {
        return None;
    }
    let edge = connecting_edges[0];

    // No role_override or vendor_bytes allowed on the fusing edge.
    if edge.role_override.is_some() || !edge.vendor_bytes.is_empty() {
        return None;
    }

    let producer_node = edge.src_node;
    let consumer_node = node_idx;

    // Verify that the producer's output is consumed only by this one edge
    // (i.e., no other edge or terminal consumes (producer_node, src_output_idx)).
    let producer_output_key = (edge.src_node, edge.src_output_idx);
    let other_consumers: usize = chain
        .edges
        .iter()
        .filter(|e| (e.src_node, e.src_output_idx) == producer_output_key && e.dst_node != node_idx)
        .count()
        + chain
            .terminals
            .iter()
            .filter(|t| (t.node_idx, t.output_idx) == producer_output_key)
            .count();
    if other_consumers > 0 {
        return None;
    }

    let producer_op = chain.nodes[producer_node as usize].op;
    let consumer_op = chain.nodes[consumer_node as usize].op;
    let consumer_params = chain.nodes[consumer_node as usize].params.clone();

    match (producer_op, consumer_op) {
        // BitReorderIeee16 + ByteSplit{n=2}
        (OpId::BitReorderIeee16, OpId::ByteSplit) if consumer_params.first() == Some(&2) => {
            Some(FusedExecution {
                producer_node,
                consumer_node,
                run_forward: Box::new(FusedIeee16ByteSplit2::forward),
                run_inverse: Box::new(FusedIeee16ByteSplit2::inverse),
            })
        }

        // BitReorderIeee32 + ByteSplit{n=4}
        (OpId::BitReorderIeee32, OpId::ByteSplit) if consumer_params.first() == Some(&4) => {
            Some(FusedExecution {
                producer_node,
                consumer_node,
                run_forward: Box::new(FusedIeee32ByteSplit4::forward),
                run_inverse: Box::new(FusedIeee32ByteSplit4::inverse),
            })
        }

        // BitReorderFp8E4M3 + NibbleSplit
        (OpId::BitReorderFp8E4M3, OpId::NibbleSplit) => Some(FusedExecution {
            producer_node,
            consumer_node,
            run_forward: Box::new(FusedFp8E4M3NibbleSplit::forward),
            run_inverse: Box::new(FusedFp8E4M3NibbleSplit::inverse),
        }),

        // BitReorderFp8E5M2 + NibbleSplit
        (OpId::BitReorderFp8E5M2, OpId::NibbleSplit) => Some(FusedExecution {
            producer_node,
            consumer_node,
            run_forward: Box::new(FusedFp8E5M2NibbleSplit::forward),
            run_inverse: Box::new(FusedFp8E5M2NibbleSplit::inverse),
        }),

        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::graph::{Chain, ChainEdge, ChainNode, TerminalRef};
    use crate::transforms::op::OpId;
    use crate::types::descriptor::{ElementWidth, Layout, PlaneDescriptor};
    use crate::types::role::Role;

    // ── Shared helpers ──────────────────────────────────────────────────────

    fn raw_descriptor(len: u64) -> PlaneDescriptor {
        PlaneDescriptor {
            role: Role::Raw,
            element_width: ElementWidth::Byte,
            length_bytes: len,
            layout: Layout::Flat,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        }
    }

    fn source_params_for(n_elements: u32, dtype_code: u16) -> Vec<u8> {
        let mut p = vec![1u8]; // 1-dimensional
        p.extend_from_slice(&n_elements.to_le_bytes());
        p.extend_from_slice(&dtype_code.to_le_bytes());
        p
    }

    fn passthrough_terminal_chain() -> Chain {
        // Source(16 bytes, u8) → BytePassthrough → Terminal
        Chain {
            nodes: vec![
                ChainNode {
                    op: OpId::Source,
                    params: source_params_for(16, 0x0005), // int8 → 1 byte/elem
                },
                ChainNode {
                    op: OpId::BytePassthrough,
                    params: vec![],
                },
            ],
            edges: vec![ChainEdge {
                src_node: 0,
                src_output_idx: 0,
                dst_node: 1,
                dst_input_idx: 0,
                role_override: None,
                vendor_bytes: vec![],
            }],
            terminals: vec![TerminalRef {
                node_idx: 1,
                output_idx: 0,
                role: Role::Raw,
            }],
        }
    }

    fn byte_split_chain() -> Chain {
        // Source(8 bytes, u8) → ByteSplit{n=2} → 2× Terminal
        Chain {
            nodes: vec![
                ChainNode {
                    op: OpId::Source,
                    params: source_params_for(8, 0x0005),
                },
                ChainNode {
                    op: OpId::ByteSplit,
                    params: vec![2u8],
                },
            ],
            edges: vec![ChainEdge {
                src_node: 0,
                src_output_idx: 0,
                dst_node: 1,
                dst_input_idx: 0,
                role_override: None,
                vendor_bytes: vec![],
            }],
            terminals: vec![
                TerminalRef {
                    node_idx: 1,
                    output_idx: 0,
                    role: Role::Raw,
                },
                TerminalRef {
                    node_idx: 1,
                    output_idx: 1,
                    role: Role::Raw,
                },
            ],
        }
    }

    // ── Forward executor ──────────────────────────────────────────────────────────

    #[test]
    fn forward_simple_passthrough_chain() {
        let chain = passthrough_terminal_chain();
        let source_bytes: Vec<u8> = (0u8..16).collect();
        let ctx = ForwardContext {
            source_bytes: &source_bytes,
            source_descriptor: raw_descriptor(16),
            deps: &[],
        };
        let result = forward_chain(&chain, &ctx).unwrap();
        assert_eq!(result.terminal_planes.len(), 1);
        assert_eq!(result.terminal_planes[0].0.bytes.as_ref(), source_bytes);
    }

    #[test]
    fn forward_byte_split_chain() {
        let chain = byte_split_chain();
        let source_bytes: Vec<u8> = (0u8..8).collect();
        let ctx = ForwardContext {
            source_bytes: &source_bytes,
            source_descriptor: raw_descriptor(8),
            deps: &[],
        };
        let result = forward_chain(&chain, &ctx).unwrap();
        assert_eq!(result.terminal_planes.len(), 2);
        assert_eq!(result.terminal_planes[0].0.bytes.len(), 4);
        assert_eq!(result.terminal_planes[1].0.bytes.len(), 4);
        // Plane 0: bytes at indices 0, 2, 4, 6 = [0, 2, 4, 6]
        assert_eq!(result.terminal_planes[0].0.bytes.as_ref(), vec![0, 2, 4, 6]);
        // Plane 1: bytes at indices 1, 3, 5, 7 = [1, 3, 5, 7]
        assert_eq!(result.terminal_planes[1].0.bytes.as_ref(), vec![1, 3, 5, 7]);
    }

    #[test]
    fn forward_unknown_op_rejected() {
        // Build a chain with an invalid op id by exploiting OpId::from_u16 failure.
        // We can't directly inject an invalid OpId, but we can test via a chain
        // where the node's params are invalid (ByteSplit with n=3, which ByteSplit::new rejects).
        let chain = Chain {
            nodes: vec![
                ChainNode {
                    op: OpId::Source,
                    params: source_params_for(8, 0x0005),
                },
                ChainNode {
                    op: OpId::ByteSplit,
                    params: vec![3u8], // invalid n
                },
            ],
            edges: vec![ChainEdge {
                src_node: 0,
                src_output_idx: 0,
                dst_node: 1,
                dst_input_idx: 0,
                role_override: None,
                vendor_bytes: vec![],
            }],
            terminals: vec![TerminalRef {
                node_idx: 1,
                output_idx: 0,
                role: Role::Raw,
            }],
        };
        let ctx = ForwardContext {
            source_bytes: &[0u8; 8],
            source_descriptor: raw_descriptor(8),
            deps: &[],
        };
        assert!(forward_chain(&chain, &ctx).is_err());
    }

    // ── Inverse executor ──────────────────────────────────────────────────────────

    fn word2_descriptor(len: u64) -> PlaneDescriptor {
        PlaneDescriptor {
            role: Role::Raw,
            element_width: ElementWidth::Word2,
            length_bytes: len,
            layout: Layout::Flat,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        }
    }

    /// `Source(n bytes) → IndexBitwidthPack{bits} → Terminal`.
    fn index_pack_chain(n_indices: u32, bits: u8) -> Chain {
        Chain {
            nodes: vec![
                ChainNode {
                    op: OpId::Source,
                    params: source_params_for(n_indices, 0x0005), // int8 → 1 byte/elem
                },
                ChainNode {
                    op: OpId::IndexBitwidthPack,
                    params: vec![bits],
                },
            ],
            edges: vec![ChainEdge {
                src_node: 0,
                src_output_idx: 0,
                dst_node: 1,
                dst_input_idx: 0,
                role_override: None,
                vendor_bytes: vec![],
            }],
            terminals: vec![TerminalRef {
                node_idx: 1,
                output_idx: 0,
                role: Role::Index,
            }],
        }
    }

    #[test]
    fn forward_inverse_roundtrip_index_pack_exact_count() {
        // Counts 1..=40 cover every residue of `n * bits % 8`, including the
        // padded cases the inverse used to over-count.
        let mut padded = 0usize;
        for bits in 1u8..=8 {
            let modulus = 1u16 << bits;
            for n in 1u32..=40 {
                if !(n as u64 * bits as u64).is_multiple_of(8) {
                    padded += 1;
                }
                let chain = index_pack_chain(n, bits);
                let source_bytes: Vec<u8> = (0..n)
                    .map(|i| ((i as u16 * 7 + 3) % modulus) as u8)
                    .collect();
                let ctx = ForwardContext {
                    source_bytes: &source_bytes,
                    source_descriptor: raw_descriptor(n as u64),
                    deps: &[],
                };
                let fwd = forward_chain(&chain, &ctx).unwrap();
                let terminal_planes: Vec<Plane> =
                    fwd.terminal_planes.into_iter().map(|(p, _)| p).collect();
                assert_eq!(
                    terminal_planes[0].bytes.len() as u64,
                    (n as u64 * bits as u64).div_ceil(8),
                    "bits={bits} n={n}: packed length"
                );
                let inv_ctx = InverseContext {
                    terminal_planes: &terminal_planes,
                    deps: &[],
                };
                let recovered = inverse_chain(&chain, &inv_ctx).unwrap();
                assert_eq!(
                    recovered.bytes.as_ref(),
                    source_bytes.as_slice(),
                    "bits={bits} n={n}: round-trip mismatch"
                );
                assert_eq!(recovered.descriptor.length_bytes, n as u64);
            }
        }
        assert!(padded > 0);
    }

    #[test]
    fn inverse_index_pack_without_input_descs_uses_floor_count() {
        // A Word2 Source makes `IndexBitwidthPack::propagate_descriptors`
        // reject the node, so the symbolic walk gives it no input
        // descriptors. `forward` does not check the element width, so the
        // chain still encodes. Decoding must still succeed, with the
        // floor-based count: 6 bytes × 3 bits = 18 bits → 3 packed bytes
        // → floor(24 / 3) = 8 indices.
        let mut chain = index_pack_chain(3, 3);
        chain.nodes[0].params = source_params_for(3, 0x0002); // fp16 → 6 bytes
        let topo = topo_sort(&chain).unwrap();
        assert!(symbolic_node_inputs(&chain, &topo)[1].is_none());

        let source_bytes: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
        let ctx = ForwardContext {
            source_bytes: &source_bytes,
            source_descriptor: word2_descriptor(6),
            deps: &[],
        };
        let fwd = forward_chain(&chain, &ctx).unwrap();
        let terminal_planes: Vec<Plane> = fwd.terminal_planes.into_iter().map(|(p, _)| p).collect();
        assert_eq!(terminal_planes[0].bytes.len(), 3);
        let inv_ctx = InverseContext {
            terminal_planes: &terminal_planes,
            deps: &[],
        };
        let recovered = inverse_chain(&chain, &inv_ctx).unwrap();
        assert_eq!(recovered.bytes.len(), 8);
        assert_eq!(&recovered.bytes[..6], source_bytes.as_slice());
    }

    #[test]
    fn inverse_index_pack_tolerates_overstated_source() {
        // Legacy files may carry a Source that overstates the raw size by up
        // to 8×, and the non-strict decoder accepts them. With bits = 8 the
        // plain inverse recovers the real size, so the exact-count path must
        // not turn the mismatch into an error.
        let n = 5u32;
        let chain = index_pack_chain(8 * n, 8);
        let source_bytes: Vec<u8> = (0..n as u8).collect();
        let ctx = ForwardContext {
            source_bytes: &source_bytes,
            source_descriptor: raw_descriptor(n as u64),
            deps: &[],
        };
        let fwd = forward_chain(&chain, &ctx).unwrap();
        let terminal_planes: Vec<Plane> = fwd.terminal_planes.into_iter().map(|(p, _)| p).collect();
        let inv_ctx = InverseContext {
            terminal_planes: &terminal_planes,
            deps: &[],
        };
        let recovered = inverse_chain(&chain, &inv_ctx).unwrap();
        assert_eq!(recovered.bytes.as_ref(), source_bytes.as_slice());
    }

    #[test]
    fn forward_inverse_roundtrip_simple() {
        let chain = passthrough_terminal_chain();
        let source_bytes: Vec<u8> = (0u8..16).collect();
        let ctx = ForwardContext {
            source_bytes: &source_bytes,
            source_descriptor: raw_descriptor(16),
            deps: &[],
        };
        let fwd = forward_chain(&chain, &ctx).unwrap();
        let terminal_planes: Vec<Plane> = fwd.terminal_planes.into_iter().map(|(p, _)| p).collect();
        let inv_ctx = InverseContext {
            terminal_planes: &terminal_planes,
            deps: &[],
        };
        let reconstructed = inverse_chain(&chain, &inv_ctx).unwrap();
        assert_eq!(reconstructed.bytes.as_ref(), source_bytes);
    }

    #[test]
    fn forward_inverse_roundtrip_byte_split() {
        let chain = byte_split_chain();
        let source_bytes: Vec<u8> = (0u8..8).collect();
        let ctx = ForwardContext {
            source_bytes: &source_bytes,
            source_descriptor: raw_descriptor(8),
            deps: &[],
        };
        let fwd = forward_chain(&chain, &ctx).unwrap();
        let terminal_planes: Vec<Plane> = fwd.terminal_planes.into_iter().map(|(p, _)| p).collect();
        let inv_ctx = InverseContext {
            terminal_planes: &terminal_planes,
            deps: &[],
        };
        let reconstructed = inverse_chain(&chain, &inv_ctx).unwrap();
        assert_eq!(reconstructed.bytes.as_ref(), source_bytes);
    }

    #[test]
    fn forward_inverse_roundtrip_with_bit_reorder() {
        // Source(16 bytes, fp16) → BitReorderIeee16 → BytePassthrough → Terminal
        // (3-node chain to test multi-node roundtrip; BytePassthrough is
        // element-width agnostic unlike ByteSplit)
        let chain = Chain {
            nodes: vec![
                ChainNode {
                    op: OpId::Source,
                    params: source_params_for(8, 0x0002), // fp16, 8 elements × 2 bytes = 16 bytes
                },
                ChainNode {
                    op: OpId::BitReorderIeee16,
                    params: vec![],
                },
                ChainNode {
                    op: OpId::BytePassthrough,
                    params: vec![],
                },
            ],
            edges: vec![
                ChainEdge {
                    src_node: 0,
                    src_output_idx: 0,
                    dst_node: 1,
                    dst_input_idx: 0,
                    role_override: None,
                    vendor_bytes: vec![],
                },
                ChainEdge {
                    src_node: 1,
                    src_output_idx: 0,
                    dst_node: 2,
                    dst_input_idx: 0,
                    role_override: None,
                    vendor_bytes: vec![],
                },
            ],
            terminals: vec![TerminalRef {
                node_idx: 2,
                output_idx: 0,
                role: Role::Raw,
            }],
        };

        let source_bytes: Vec<u8> = (0u8..16).map(|i| i.wrapping_mul(37)).collect();
        let ctx = ForwardContext {
            source_bytes: &source_bytes,
            source_descriptor: word2_descriptor(16),
            deps: &[],
        };
        let fwd = forward_chain(&chain, &ctx).unwrap();
        let terminal_planes: Vec<Plane> = fwd.terminal_planes.into_iter().map(|(p, _)| p).collect();
        let inv_ctx = InverseContext {
            terminal_planes: &terminal_planes,
            deps: &[],
        };
        let reconstructed = inverse_chain(&chain, &inv_ctx).unwrap();
        assert_eq!(reconstructed.bytes.as_ref(), source_bytes);
    }

    #[test]
    fn forward_inverse_roundtrip_microscaling_repack() {
        // Source(BF16) → BlockMicroscalingRepack(32) → BitReorderIeee16 →
        // ByteSplit(2) → 2 element terminals + 1 E8M0 scale terminal.
        // Exercises a 2-output op whose output 0 feeds a
        // downstream sub-chain while output 1 is a terminal of its own.
        use crate::types::role::ScaleFormat;
        let n_elements = 70u32; // 3 blocks of 32 (last partial) → exercises framing
        let chain = Chain {
            nodes: vec![
                ChainNode {
                    op: OpId::Source,
                    params: source_params_for(n_elements, 0x000F), // BF16
                },
                ChainNode {
                    op: OpId::BlockMicroscalingRepack,
                    params: vec![32u8],
                },
                ChainNode {
                    op: OpId::BitReorderIeee16,
                    params: vec![],
                },
                ChainNode {
                    op: OpId::ByteSplit,
                    params: vec![2u8],
                },
            ],
            edges: vec![
                ChainEdge {
                    src_node: 0,
                    src_output_idx: 0,
                    dst_node: 1,
                    dst_input_idx: 0,
                    role_override: None,
                    vendor_bytes: vec![],
                },
                ChainEdge {
                    src_node: 1,
                    src_output_idx: 0, // element plane
                    dst_node: 2,
                    dst_input_idx: 0,
                    role_override: None,
                    vendor_bytes: vec![],
                },
                ChainEdge {
                    src_node: 2,
                    src_output_idx: 0,
                    dst_node: 3,
                    dst_input_idx: 0,
                    role_override: None,
                    vendor_bytes: vec![],
                },
            ],
            terminals: vec![
                TerminalRef {
                    node_idx: 3,
                    output_idx: 0,
                    role: Role::Raw,
                },
                TerminalRef {
                    node_idx: 3,
                    output_idx: 1,
                    role: Role::Raw,
                },
                TerminalRef {
                    node_idx: 1,
                    output_idx: 1, // E8M0 scale plane
                    role: Role::Scale {
                        format: ScaleFormat::E8M0,
                    },
                },
            ],
        };

        // Pseudo-BF16 payload: varied sign/exponent/mantissa patterns.
        let source_bytes: Vec<u8> = (0..n_elements * 2)
            .map(|i| (i.wrapping_mul(73).wrapping_add(11) & 0xFF) as u8)
            .collect();
        let ctx = ForwardContext {
            source_bytes: &source_bytes,
            source_descriptor: word2_descriptor(source_bytes.len() as u64),
            deps: &[],
        };
        let fwd = forward_chain(&chain, &ctx).unwrap();
        // Three terminals: two byte planes + one scale plane.
        assert_eq!(fwd.terminal_planes.len(), 3);
        let terminal_planes: Vec<Plane> = fwd.terminal_planes.into_iter().map(|(p, _)| p).collect();
        let inv_ctx = InverseContext {
            terminal_planes: &terminal_planes,
            deps: &[],
        };
        let reconstructed = inverse_chain(&chain, &inv_ctx).unwrap();
        assert_eq!(reconstructed.bytes.as_ref(), source_bytes);
    }

    // ── Fusion picker ─────────────────────────────────────────────────────────────

    fn ieee16_byte_split_chain() -> Chain {
        // Source → BitReorderIeee16 → ByteSplit{n=2} → 2× Terminal (no role_override)
        Chain {
            nodes: vec![
                ChainNode {
                    op: OpId::Source,
                    params: source_params_for(8, 0x0002),
                },
                ChainNode {
                    op: OpId::BitReorderIeee16,
                    params: vec![],
                },
                ChainNode {
                    op: OpId::ByteSplit,
                    params: vec![2u8],
                },
            ],
            edges: vec![
                ChainEdge {
                    src_node: 0,
                    src_output_idx: 0,
                    dst_node: 1,
                    dst_input_idx: 0,
                    role_override: None,
                    vendor_bytes: vec![],
                },
                ChainEdge {
                    src_node: 1,
                    src_output_idx: 0,
                    dst_node: 2,
                    dst_input_idx: 0,
                    role_override: None,
                    vendor_bytes: vec![],
                },
            ],
            terminals: vec![
                TerminalRef {
                    node_idx: 2,
                    output_idx: 0,
                    role: Role::Raw,
                },
                TerminalRef {
                    node_idx: 2,
                    output_idx: 1,
                    role: Role::Raw,
                },
            ],
        }
    }

    #[test]
    fn fusion_picker_recognises_ieee16_split_pair() {
        let chain = ieee16_byte_split_chain();
        // node 2 is ByteSplit, node 1 is BitReorderIeee16 → fused
        let fused = try_fuse_at(&chain, 2);
        assert!(fused.is_some(), "should recognise ieee16+byte_split pair");
        let f = fused.unwrap();
        assert_eq!(f.producer_node, 1);
        assert_eq!(f.consumer_node, 2);
    }

    #[test]
    fn fusion_picker_recognises_fp8e4m3_nibble_pair() {
        let chain = Chain {
            nodes: vec![
                ChainNode {
                    op: OpId::Source,
                    params: source_params_for(8, 0x0010), // fp8 e4m3, 1 byte/elem
                },
                ChainNode {
                    op: OpId::BitReorderFp8E4M3,
                    params: vec![],
                },
                ChainNode {
                    op: OpId::NibbleSplit,
                    params: vec![],
                },
            ],
            edges: vec![
                ChainEdge {
                    src_node: 0,
                    src_output_idx: 0,
                    dst_node: 1,
                    dst_input_idx: 0,
                    role_override: None,
                    vendor_bytes: vec![],
                },
                ChainEdge {
                    src_node: 1,
                    src_output_idx: 0,
                    dst_node: 2,
                    dst_input_idx: 0,
                    role_override: None,
                    vendor_bytes: vec![],
                },
            ],
            terminals: vec![
                TerminalRef {
                    node_idx: 2,
                    output_idx: 0,
                    role: Role::Raw,
                },
                TerminalRef {
                    node_idx: 2,
                    output_idx: 1,
                    role: Role::Raw,
                },
            ],
        };
        let fused = try_fuse_at(&chain, 2);
        assert!(
            fused.is_some(),
            "should recognise fp8e4m3+nibble_split pair"
        );
    }

    #[test]
    fn fusion_picker_skips_when_role_override_present() {
        let mut chain = ieee16_byte_split_chain();
        // Add a role_override to the connecting edge (node 1 → node 2).
        chain.edges[1].role_override = Some(Role::Raw);
        let fused = try_fuse_at(&chain, 2);
        assert!(
            fused.is_none(),
            "should skip fusion when role_override present"
        );
    }

    #[test]
    fn fusion_picker_skips_when_intermediate_consumed_twice() {
        // A terminal ALSO points to (node 1, output 0) — double consumption
        // of the producer's output. Fusion picker must reject this.
        let chain = Chain {
            nodes: vec![
                ChainNode {
                    op: OpId::Source,
                    params: source_params_for(8, 0x0002),
                },
                ChainNode {
                    op: OpId::BitReorderIeee16,
                    params: vec![],
                },
                ChainNode {
                    op: OpId::ByteSplit,
                    params: vec![2u8],
                },
            ],
            edges: vec![
                ChainEdge {
                    src_node: 0,
                    src_output_idx: 0,
                    dst_node: 1,
                    dst_input_idx: 0,
                    role_override: None,
                    vendor_bytes: vec![],
                },
                ChainEdge {
                    src_node: 1,
                    src_output_idx: 0,
                    dst_node: 2,
                    dst_input_idx: 0,
                    role_override: None,
                    vendor_bytes: vec![],
                },
            ],
            // Terminal also points to (node 1, output 0) — double consumption.
            terminals: vec![
                TerminalRef {
                    node_idx: 1,
                    output_idx: 0,
                    role: Role::Raw,
                },
                TerminalRef {
                    node_idx: 2,
                    output_idx: 0,
                    role: Role::Raw,
                },
                TerminalRef {
                    node_idx: 2,
                    output_idx: 1,
                    role: Role::Raw,
                },
            ],
        };
        let fused = try_fuse_at(&chain, 2);
        assert!(
            fused.is_none(),
            "should skip fusion when intermediate is consumed twice"
        );
    }

    #[test]
    fn forward_chain_uses_fused_path_when_matched() {
        use crate::transforms::fusion::FusedIeee16ByteSplit2;

        let chain = ieee16_byte_split_chain();
        let source_bytes: Vec<u8> = (0u8..16).map(|i| i.wrapping_mul(37)).collect();
        let ctx = ForwardContext {
            source_bytes: &source_bytes,
            source_descriptor: word2_descriptor(16),
            deps: &[],
        };
        let result = forward_chain(&chain, &ctx).unwrap();
        assert_eq!(result.terminal_planes.len(), 2);

        // Compare against the fused implementation directly.
        let input_plane = Plane {
            bytes: Arc::from(&source_bytes[..]),
            descriptor: word2_descriptor(16),
        };
        let fused_out = FusedIeee16ByteSplit2::forward(&input_plane).unwrap();
        assert_eq!(
            result.terminal_planes[0].0.bytes.as_ref(),
            fused_out[0].bytes.as_ref()
        );
        assert_eq!(
            result.terminal_planes[1].0.bytes.as_ref(),
            fused_out[1].bytes.as_ref()
        );
    }

    #[test]
    fn forward_inverse_roundtrip_fused_ieee16() {
        let chain = ieee16_byte_split_chain();
        let source_bytes: Vec<u8> = (0u8..16).map(|i| i.wrapping_mul(37)).collect();
        let ctx = ForwardContext {
            source_bytes: &source_bytes,
            source_descriptor: word2_descriptor(16),
            deps: &[],
        };
        let fwd = forward_chain(&chain, &ctx).unwrap();
        let terminal_planes: Vec<Plane> = fwd.terminal_planes.into_iter().map(|(p, _)| p).collect();
        let inv_ctx = InverseContext {
            terminal_planes: &terminal_planes,
            deps: &[],
        };
        let reconstructed = inverse_chain(&chain, &inv_ctx).unwrap();
        assert_eq!(reconstructed.bytes.as_ref(), source_bytes);
    }

    /// Defence-in-depth: a chain whose edges reference node indices outside
    /// `chain.nodes` must be rejected with [`PtwmCoreError::InvalidContainer`]
    /// rather than panicking via out-of-range indexing in `topo_sort`.
    /// This guards the v3 decode path against malformed wire-derived chains
    /// that bypass [`crate::chain::validate::validate_chain`].
    #[test]
    fn topo_sort_rejects_out_of_range_edge() {
        let mut chain = passthrough_terminal_chain();
        // Edge references a non-existent dst node (chain has 2 nodes: 0..=1).
        chain.edges.push(ChainEdge {
            src_node: 0,
            src_output_idx: 0,
            dst_node: 99,
            dst_input_idx: 0,
            role_override: None,
            vendor_bytes: vec![],
        });
        let placeholder = vec![0u8; 16];
        let ctx = ForwardContext {
            source_bytes: &placeholder,
            source_descriptor: raw_descriptor(16),
            deps: &[],
        };
        let err = match forward_chain(&chain, &ctx) {
            Ok(_) => panic!("forward_chain must reject out-of-range edges"),
            Err(e) => e,
        };
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("out-of-range") || msg.contains("out of range"),
            "expected out-of-range error from topo_sort, got: {msg}"
        );
    }

    /// Regression: a chain whose terminal declares `Role::Scale` must
    /// expose that role on the terminal plane's descriptor — not the
    /// `Role::Raw` propagated up from `Source`. Without this re-tag,
    /// `Order1ScaleAC.accepts()` rejects the descriptor and the codec
    /// never enters trial-encode on real-corpus FP8-E4M3 weight_scale
    /// tensors. See `compressor::trial_encode_terminal` and
    /// `dispatch::dispatch` for the consumer side of this invariant.
    #[test]
    fn forward_chain_applies_terminal_role_to_descriptor() {
        use crate::types::role::{Role, ScaleFormat};
        // Source(byte) → BytePassthrough → Terminal(Scale{E4M3})
        let chain = Chain {
            nodes: vec![
                ChainNode {
                    op: OpId::Source,
                    params: source_params_for(16, 0x0010), // FP8 E4M3
                },
                ChainNode {
                    op: OpId::BytePassthrough,
                    params: vec![],
                },
            ],
            edges: vec![ChainEdge {
                src_node: 0,
                src_output_idx: 0,
                dst_node: 1,
                dst_input_idx: 0,
                role_override: None,
                vendor_bytes: vec![],
            }],
            terminals: vec![TerminalRef {
                node_idx: 1,
                output_idx: 0,
                role: Role::Scale {
                    format: ScaleFormat::E4M3,
                },
            }],
        };
        let source_bytes: Vec<u8> = (0u8..16).collect();
        let ctx = ForwardContext {
            source_bytes: &source_bytes,
            source_descriptor: raw_descriptor(16),
            deps: &[],
        };
        let result = forward_chain(&chain, &ctx).unwrap();
        let (plane, role) = &result.terminal_planes[0];
        assert!(
            matches!(
                role,
                Role::Scale {
                    format: ScaleFormat::E4M3
                }
            ),
            "terminal role must round-trip"
        );
        assert!(
            matches!(
                plane.descriptor.role,
                Role::Scale {
                    format: ScaleFormat::E4M3
                }
            ),
            "plane descriptor role must be re-tagged from Raw to Scale, got {:?}",
            plane.descriptor.role,
        );
    }

    // ── Symbolic descriptor propagation (P5) ────────────────────────────────

    /// The earlier decoder path: run `forward_chain` over a zero buffer of
    /// the Source's declared size and read the terminal descriptors back.
    /// Kept here as the reference the symbolic walk must match.
    fn forward_descriptors(chain: &Chain) -> Result<Vec<PlaneDescriptor>, PtwmCoreError> {
        let source_descriptor = chain_source_descriptor(chain)?;
        let placeholder = vec![0u8; source_descriptor.length_bytes as usize];
        let ctx = ForwardContext {
            source_bytes: &placeholder,
            source_descriptor,
            deps: &[],
        };
        Ok(forward_chain(chain, &ctx)?
            .terminal_planes
            .into_iter()
            .map(|(plane, _)| plane.descriptor)
            .collect())
    }

    /// Compare the symbolic walk with the forward reference on one chain.
    /// Returns true when both ran and agreed; false when the forward pass
    /// rejects the chain (it checks some lengths `propagate_descriptors`
    /// does not, so the symbolic walk may accept those).
    fn assert_symbolic_matches_forward(label: &str, chain: &Chain) -> bool {
        let Ok(expected) = forward_descriptors(chain) else {
            return false;
        };
        let mut got = propagate_terminal_descriptors(chain)
            .unwrap_or_else(|e| panic!("{label}: forward succeeded but symbolic failed: {e}"));
        let mut expected = expected;
        // EntropyEstimate writes an estimate of the bytes it sees into
        // `vendor_bytes`. The forward reference computes it from placeholder
        // planes (zeros at the Source, arbitrary after ops such as
        // AlphaStableNormalize), the symbolic walk always reports zeros.
        // Neither is the encode-time value and no decoder reads it, so it
        // is left out of the comparison.
        if chain.nodes.iter().any(|n| n.op == OpId::EntropyEstimate) {
            for d in got.iter_mut().chain(expected.iter_mut()) {
                d.vendor_bytes.clear();
            }
        }
        assert_eq!(
            got, expected,
            "{label}: symbolic descriptors differ from forward"
        );
        true
    }

    #[test]
    fn symbolic_descriptors_match_forward_on_production_chains() {
        // Written by tests/ptwm/preprocessing/test_production_chain_fixture.py
        // from every Python chain builder and the explorer's first candidates.
        let data: &[u8] = include_bytes!("../../tests/fixtures/production_chains.bin");
        let mut pos = 0usize;
        let (mut total, mut valid, mut compared) = (0usize, 0usize, 0usize);
        let mut fused = 0usize;
        while pos < data.len() {
            let name_len = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
            pos += 2;
            let label = std::str::from_utf8(&data[pos..pos + name_len]).unwrap();
            pos += name_len;
            let blob_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            let blob = &data[pos..pos + blob_len];
            pos += blob_len;
            total += 1;
            let (chain, _) = crate::chain::wire::read_chain_blob(blob)
                .unwrap_or_else(|e| panic!("{label}: fixture blob does not parse: {e}"));
            // Some explorer chains do not fit every fixture shape (a 0-d
            // fp16 tensor cannot be split into 4 bytes); the decoder
            // rejects those through `validate_chain` before descriptors.
            if crate::chain::validate::validate_chain(&chain, 0).is_err() {
                assert!(label.starts_with("explore"), "{label}: invalid chain");
                continue;
            }
            valid += 1;
            if (1..chain.nodes.len()).any(|i| try_fuse_at(&chain, i as u8).is_some()) {
                fused += 1;
            }
            if assert_symbolic_matches_forward(label, &chain) {
                compared += 1;
            }
        }
        assert!(total > 500, "fixture holds only {total} chains");
        // Forward rejects a few builder/shape pairs (e.g. microscale repack
        // on lengths that are not whole blocks); nearly all must compare.
        assert!(
            compared * 10 >= valid * 9,
            "only {compared} of {valid} valid chains ran forward ({total} in fixture)"
        );
        assert!(fused > 50, "only {fused} chains exercise fusion");
    }

    /// Small deterministic PRNG (xorshift64*), so the test needs no crate.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
        fn pick<T: Clone>(&mut self, items: &[T]) -> T {
            items[self.below(items.len() as u64) as usize].clone()
        }
    }

    fn random_role(rng: &mut Rng) -> Role {
        use crate::types::role::{NibbleKind, ScaleFormat, ValueFormat};
        rng.pick(&[
            Role::Raw,
            Role::ExponentByte,
            Role::Index,
            Role::MantissaByte { index: 0, of: 2 },
            Role::Nibble {
                kind: NibbleKind::Exponent,
            },
            Role::Value {
                format: ValueFormat::IntN { bits: 8 },
            },
            Role::Scale {
                format: ScaleFormat::E8M0,
            },
            Role::GlobalScale {
                format: ScaleFormat::F32,
            },
        ])
    }

    /// A random op and its params, or a Concat of two open outputs.
    fn random_node(rng: &mut Rng) -> (OpId, Vec<u8>) {
        match rng.below(19) {
            0 => (OpId::BitReorderIeee16, vec![]),
            1 => (OpId::BitReorderIeee32, vec![]),
            2 => (OpId::BitReorderFp8E4M3, vec![]),
            3 => (OpId::BitReorderFp8E5M2, vec![]),
            4 => (OpId::ByteSplit, vec![rng.pick(&[2u8, 4, 8])]),
            5 => (OpId::NibbleSplit, vec![]),
            6 => (OpId::BytePassthrough, vec![]),
            7 => (OpId::MxFp4Deinterleave, vec![32]),
            8 => (OpId::BlockMicroscalingRepack, vec![32]),
            9 => (OpId::IntDelta, vec![]),
            10 => (OpId::PredictorXor, vec![]),
            11 => (OpId::IndexBitwidthPack, vec![1 + rng.below(8) as u8]),
            12 => (OpId::EntropyEstimate, vec![]),
            13 => (OpId::MantissaZeroStrip, vec![rng.below(5) as u8]),
            14 => (OpId::BurrowsWheeler, vec![]),
            15 => (OpId::MoveToFront, vec![]),
            16 => {
                let mut p = Vec::new();
                if rng.below(2) == 0 {
                    Layout::Flat.write(&mut p);
                } else {
                    Layout::Rows {
                        row_len: 1 + rng.below(16) as u32,
                    }
                    .write(&mut p);
                }
                (OpId::Reshape, p)
            }
            17 => (
                OpId::SphericalNormalize,
                vec![rng.below(2) as u8, 0, rng.below(3) as u8],
            ),
            _ => (OpId::AlphaStableNormalize, vec![0, rng.below(3) as u8]),
        }
    }

    /// Build a random well-formed chain: each step consumes one or two open
    /// outputs; whatever stays open becomes a terminal. Op choices whose
    /// descriptors do not propagate are dropped.
    fn random_chain(rng: &mut Rng) -> Chain {
        let dtype = rng.pick(&[
            0x0002u16, 0x000F, 0x0003, 0x0004, 0x0005, 0x0006, 0x0009, 0x0010, 0x0011, 0x001F,
        ]);
        let shape: Vec<u32> = (0..1 + rng.below(2))
            .map(|_| rng.below(48) as u32 + u32::from(rng.below(8) != 0))
            .collect();
        let mut src = vec![shape.len() as u8];
        for d in &shape {
            src.extend_from_slice(&d.to_le_bytes());
        }
        src.extend_from_slice(&dtype.to_le_bytes());
        let mut chain = Chain {
            nodes: vec![ChainNode {
                op: OpId::Source,
                params: src,
            }],
            edges: vec![],
            terminals: vec![],
        };
        let Ok(source) = chain_source_descriptor(&chain) else {
            return chain;
        };
        // Open outputs: (node, output, descriptor as seen by a consumer).
        let mut open: Vec<(u8, u8, PlaneDescriptor)> = vec![(0, 0, source)];
        for _ in 0..1 + rng.below(5) {
            if open.is_empty() {
                break;
            }
            let concat = open.len() >= 2 && rng.below(6) == 0;
            let n_in = if concat { 2 } else { 1 };
            let mut picked = Vec::new();
            for _ in 0..n_in {
                let i = rng.below(open.len() as u64) as usize;
                picked.push(open.swap_remove(i));
            }
            let overrides: Vec<Option<Role>> = picked
                .iter()
                .map(|_| (rng.below(4) == 0).then(|| random_role(rng)))
                .collect();
            let inputs: Vec<PlaneDescriptor> = picked
                .iter()
                .zip(&overrides)
                .map(|((_, _, d), o)| {
                    let mut d = d.clone();
                    if let Some(r) = o {
                        d.role = r.clone();
                    }
                    d
                })
                .collect();
            let (op, params) = if concat {
                let mut p = vec![2u8];
                for d in &inputs {
                    p.extend_from_slice(&(d.length_bytes as u32).to_le_bytes());
                }
                (OpId::Concat, p)
            } else {
                random_node(rng)
            };
            let produced = op_from_id(op, &params).and_then(|o| o.propagate_descriptors(&inputs));
            let Ok(produced) = produced else {
                open.extend(picked);
                continue;
            };
            let idx = chain.nodes.len() as u8;
            chain.nodes.push(ChainNode { op, params });
            for (slot, ((src_node, src_out, _), role_override)) in
                picked.into_iter().zip(overrides).enumerate()
            {
                chain.edges.push(ChainEdge {
                    src_node,
                    src_output_idx: src_out,
                    dst_node: idx,
                    dst_input_idx: slot as u8,
                    role_override,
                    vendor_bytes: vec![],
                });
            }
            for (o, d) in produced.into_iter().enumerate() {
                open.push((idx, o as u8, d));
            }
        }
        for (node_idx, output_idx, _) in open {
            chain.terminals.push(TerminalRef {
                node_idx,
                output_idx,
                role: random_role(rng),
            });
        }
        chain
    }

    #[test]
    fn symbolic_descriptors_match_forward_on_random_chains() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let (mut valid, mut compared, mut multi_node) = (0usize, 0usize, 0usize);
        for i in 0..3000 {
            let chain = random_chain(&mut rng);
            if crate::chain::validate::validate_chain(&chain, 0).is_err() {
                continue;
            }
            valid += 1;
            if assert_symbolic_matches_forward(&format!("random chain {i}: {chain:?}"), &chain) {
                compared += 1;
                if chain.nodes.len() > 2 {
                    multi_node += 1;
                }
            }
        }
        assert!(valid > 1000, "only {valid} random chains were valid");
        assert!(compared > 800, "only {compared} random chains ran forward");
        assert!(
            multi_node > 300,
            "only {multi_node} compared chains have 2+ ops"
        );
    }

    #[test]
    fn symbolic_applies_terminal_role() {
        use crate::types::role::ScaleFormat;
        let mut chain = passthrough_terminal_chain();
        chain.terminals[0].role = Role::Scale {
            format: ScaleFormat::E8M0,
        };
        let descs = propagate_terminal_descriptors(&chain).unwrap();
        assert_eq!(
            descs[0].role,
            Role::Scale {
                format: ScaleFormat::E8M0
            }
        );
        assert_eq!(descs, forward_descriptors(&chain).unwrap());
    }

    #[test]
    fn bounded_walk_rejects_huge_source_without_allocating() {
        // [u32::MAX, 64] fp64 declares about 2.2 TB. The bound from a 16-byte
        // tensor rejects it before any node runs; a forward pass would have
        // tried to allocate the whole buffer.
        let mut params = vec![2u8];
        params.extend_from_slice(&u32::MAX.to_le_bytes());
        params.extend_from_slice(&64u32.to_le_bytes());
        params.extend_from_slice(&0x0004u16.to_le_bytes());
        let mut chain = passthrough_terminal_chain();
        chain.nodes[0].params = params;
        let t0 = std::time::Instant::now();
        let err = propagate_terminal_descriptors_bounded(&chain, max_plane_len(16))
            .unwrap_err()
            .to_string();
        assert!(err.contains("Source declares"), "{err}");
        assert!(t0.elapsed() < std::time::Duration::from_secs(1));
    }

    #[test]
    fn bounded_walk_rejects_oversized_intermediate_plane() {
        // Source [256, 1] fp16 (512 bytes) → SphericalNormalize(mode 0,
        // fp16): the fp32 radius of 256 one-value rows is 1024 bytes, larger
        // than the source. A bound of 512 passes node 0 and stops at node 1.
        let mut params = vec![2u8];
        params.extend_from_slice(&256u32.to_le_bytes());
        params.extend_from_slice(&1u32.to_le_bytes());
        params.extend_from_slice(&0x0002u16.to_le_bytes());
        let mut chain = passthrough_terminal_chain();
        chain.nodes[0].params = params;
        chain.nodes[1] = ChainNode {
            op: OpId::SphericalNormalize,
            params: vec![0, 0, 2],
        };
        chain.terminals = (0..3)
            .map(|o| TerminalRef {
                node_idx: 1,
                output_idx: o,
                role: Role::Raw,
            })
            .collect();
        let descs = propagate_terminal_descriptors(&chain).unwrap();
        let largest = descs.iter().map(|d| d.length_bytes).max().unwrap();
        assert!(largest > 512, "planes {descs:?}");
        let err = propagate_terminal_descriptors_bounded(&chain, 512)
            .unwrap_err()
            .to_string();
        assert!(err.contains("node[1]"), "{err}");
        assert!(propagate_terminal_descriptors_bounded(&chain, largest).is_ok());
    }

    #[test]
    fn symbolic_walk_rejects_delta_ops() {
        let mut chain = passthrough_terminal_chain();
        chain.nodes[1] = ChainNode {
            op: OpId::XorDelta,
            params: vec![0],
        };
        let err = propagate_terminal_descriptors(&chain)
            .unwrap_err()
            .to_string();
        assert!(err.contains("delta"), "{err}");
    }

    #[test]
    fn symbolic_walk_reports_missing_planes_instead_of_panicking() {
        let mut chain = passthrough_terminal_chain();
        chain.edges[0].src_output_idx = 5;
        assert!(propagate_terminal_descriptors(&chain).is_err());
        let mut chain = passthrough_terminal_chain();
        chain.terminals[0].output_idx = 3;
        assert!(propagate_terminal_descriptors(&chain).is_err());
        let mut chain = passthrough_terminal_chain();
        chain.terminals[0].node_idx = 9;
        assert!(propagate_terminal_descriptors(&chain).is_err());
    }
}
