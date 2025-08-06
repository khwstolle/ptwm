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

/// Recover the [`PlaneDescriptor`] of every chain terminal in
/// `chain.terminals` order, without the original tensor bytes.
///
/// The reconstruction runs [`forward_chain`] over a **placeholder source
/// buffer** of zero bytes (length matching the `Source` op's declared
/// shape × dtype). Forward execution honours fusion, role overrides, and
/// every op's actual descriptor flow, so the descriptors returned here
/// match exactly what an end-to-end forward pass on the real tensor
/// would produce.
///
/// The decoder calls this to feed faithful `Plane` values into
/// [`inverse_chain`].
///
/// Cross-tensor delta ops are out of scope here: the placeholder forward
/// would need real dep planes. Chains containing `XorDelta` /
/// `FloatDelta` cannot reconstruct terminal descriptors via this helper
/// alone — the compressor driver owns that orchestration.
pub fn chain_terminal_descriptors(chain: &Chain) -> Result<Vec<PlaneDescriptor>, PtwmCoreError> {
    if chain.nodes.is_empty() {
        return Err(PtwmCoreError::InvalidContainer(
            "chain_terminal_descriptors: chain has no nodes".into(),
        ));
    }
    if chain.nodes[0].op != OpId::Source {
        return Err(PtwmCoreError::InvalidContainer(
            "chain_terminal_descriptors: node[0] must be Source".into(),
        ));
    }

    // Re-instantiate the Source op to call its `propagate_descriptors` on
    // no inputs, producing the canonical raw-tensor descriptor (with
    // element_width / layout / length_bytes derived from shape +
    // dtype_code).
    let source_op = op_from_id(OpId::Source, &chain.nodes[0].params)?;
    let source_descs = source_op.propagate_descriptors(&[])?;
    if source_descs.len() != 1 {
        return Err(PtwmCoreError::InvalidContainer(format!(
            "chain_terminal_descriptors: Source op produced {} descriptors (expected 1)",
            source_descs.len()
        )));
    }
    let source_descriptor = source_descs.into_iter().next().ok_or_else(|| {
        PtwmCoreError::InvalidContainer(
            "chain_terminal_descriptors: Source op produced no descriptors".into(),
        )
    })?;
    let placeholder_bytes = vec![0u8; source_descriptor.length_bytes as usize];

    let ctx = ForwardContext {
        source_bytes: &placeholder_bytes,
        source_descriptor,
        deps: &[],
    };
    let result = forward_chain(chain, &ctx)?;
    Ok(result
        .terminal_planes
        .into_iter()
        .map(|(plane, _role)| plane.descriptor)
        .collect())
}

// ---------------------------------------------------------------------------
// Inverse executor
// ---------------------------------------------------------------------------

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
        let reconstructed_inputs = op.inverse(&op_outputs)?;

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
}
