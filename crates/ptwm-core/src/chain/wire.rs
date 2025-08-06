//! Wire encoding / decoding for [`Chain`].
//!
//! Every node's op-id is stored as a **u16 index** into the
//! per-file Extension Table rather than the raw `OpId` discriminant. The
//! writer interns each op into an [`ExtensionTableBuilder`] and writes the
//! resulting index; the reader resolves each index back to an [`OpId`] via
//! the file's [`ExtensionTable`].
//!
//! # Blob format (Python-produced chain blobs)
//!
//! Chain blobs produced by the Python `Chain.to_bytes()` method use a
//! **self-contained** format:
//!
//! ```text
//! local_et_count:     u32
//! local_et_entries:   [ExtensionTableEntry; count]
//! chain_bytes:        (num_nodes, num_edges, num_terminals, lens, nodes, edges, terminals)
//! ```
//!
//! Use [`read_chain_blob`] to parse a self-contained blob (ET prefix + chain),
//! and [`write_chain_blob`] to produce one. These are the public APIs for
//! Python interop; the lower-level [`read_chain`] / [`write_chain`] functions
//! operate on already-parsed tables.

use crate::chain::graph::{Chain, ChainEdge, ChainNode, TerminalRef};
use crate::error::PtwmCoreError;
use crate::extension::dispatch::{BuiltinKind, dispatch_builtin};
use crate::extension::table::ExtensionTable;
use crate::extension::{CanonicalId, ExtensionTableBuilder, ExtensionTableEntry};
use crate::transforms::op::OpId;
use crate::types::role::Role;

/// Encode `chain` into `out`, interning each op's canonical id into
/// `builder` and writing the resulting `u16` table index in place of the
/// old `OpId` discriminant.
pub fn write_chain(
    chain: &Chain,
    builder: &mut ExtensionTableBuilder,
    out: &mut Vec<u8>,
) -> Result<(), PtwmCoreError> {
    // Validate counts fit in u8.
    if chain.nodes.len() > 255 {
        return Err(PtwmCoreError::InvalidContainer(format!(
            "Chain.write: nodes count {} exceeds u8 max (255)",
            chain.nodes.len()
        )));
    }
    if chain.edges.len() > 255 {
        return Err(PtwmCoreError::InvalidContainer(format!(
            "Chain.write: edges count {} exceeds u8 max (255)",
            chain.edges.len()
        )));
    }
    if chain.terminals.len() > 255 {
        return Err(PtwmCoreError::InvalidContainer(format!(
            "Chain.write: terminals count {} exceeds u8 max (255)",
            chain.terminals.len()
        )));
    }

    // Pre-encode each section into temporary buffers to measure their
    // lengths before writing the header.
    let mut nodes_buf: Vec<u8> = Vec::new();
    for node in &chain.nodes {
        if node.params.len() > 255 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "Chain.write: node params length {} exceeds u8 max (255)",
                node.params.len()
            )));
        }
        // Intern the op into the extension table and write the table index.
        let canonical_id = node.op.canonical_id();
        let entry = builtin_table_entry(canonical_id);
        let table_idx = builder.intern(entry);
        nodes_buf.extend_from_slice(&table_idx.to_le_bytes());
        nodes_buf.push(node.params.len() as u8);
        nodes_buf.extend_from_slice(&node.params);
    }

    let mut edges_buf: Vec<u8> = Vec::new();
    for edge in &chain.edges {
        if edge.vendor_bytes.len() > 65535 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "Chain.write: edge vendor_bytes length {} exceeds u16 max (65535)",
                edge.vendor_bytes.len()
            )));
        }
        edges_buf.push(edge.src_node);
        edges_buf.push(edge.src_output_idx);
        edges_buf.push(edge.dst_node);
        edges_buf.push(edge.dst_input_idx);
        if let Some(ref role) = edge.role_override {
            edges_buf.push(1u8);
            role.write(&mut edges_buf);
        } else {
            edges_buf.push(0u8);
        }
        edges_buf.extend_from_slice(&(edge.vendor_bytes.len() as u16).to_le_bytes());
        edges_buf.extend_from_slice(&edge.vendor_bytes);
    }

    let mut terminals_buf: Vec<u8> = Vec::new();
    for term in &chain.terminals {
        terminals_buf.push(term.node_idx);
        terminals_buf.push(term.output_idx);
        term.role.write(&mut terminals_buf);
    }

    // Validate section lengths fit in u16.
    if nodes_buf.len() > 65535 {
        return Err(PtwmCoreError::InvalidContainer(format!(
            "Chain.write: nodes_len {} exceeds u16 max",
            nodes_buf.len()
        )));
    }
    if edges_buf.len() > 65535 {
        return Err(PtwmCoreError::InvalidContainer(format!(
            "Chain.write: edges_len {} exceeds u16 max",
            edges_buf.len()
        )));
    }
    if terminals_buf.len() > 65535 {
        return Err(PtwmCoreError::InvalidContainer(format!(
            "Chain.write: terminals_len {} exceeds u16 max",
            terminals_buf.len()
        )));
    }

    // Write header counts.
    out.push(chain.nodes.len() as u8);
    out.push(chain.edges.len() as u8);
    out.push(chain.terminals.len() as u8);
    out.extend_from_slice(&(nodes_buf.len() as u16).to_le_bytes());
    out.extend_from_slice(&(edges_buf.len() as u16).to_le_bytes());
    out.extend_from_slice(&(terminals_buf.len() as u16).to_le_bytes());

    // Write sections.
    out.extend_from_slice(&nodes_buf);
    out.extend_from_slice(&edges_buf);
    out.extend_from_slice(&terminals_buf);

    Ok(())
}

/// Decode a `Chain` from `buf` using `table` to resolve each node's u16
/// table index back to an [`OpId`]. Returns `(chain, bytes_consumed)`.
pub fn read_chain(buf: &[u8], table: &ExtensionTable) -> Result<(Chain, usize), PtwmCoreError> {
    // Minimum header: 3 × u8 + 3 × u16 = 9 bytes.
    if buf.len() < 9 {
        return Err(PtwmCoreError::InvalidContainer(
            "Chain.read: header truncated (need ≥ 9 bytes)".into(),
        ));
    }
    let num_nodes = buf[0] as usize;
    let num_edges = buf[1] as usize;
    let num_terminals = buf[2] as usize;
    let nodes_len = u16::from_le_bytes([buf[3], buf[4]]) as usize;
    let edges_len = u16::from_le_bytes([buf[5], buf[6]]) as usize;
    let terminals_len = u16::from_le_bytes([buf[7], buf[8]]) as usize;

    let total = 9 + nodes_len + edges_len + terminals_len;
    if buf.len() < total {
        return Err(PtwmCoreError::InvalidContainer(format!(
            "Chain.read: buffer too short (need {total}, have {})",
            buf.len()
        )));
    }

    let nodes_start = 9;
    let edges_start = nodes_start + nodes_len;
    let terminals_start = edges_start + edges_len;

    let nodes_bytes = &buf[nodes_start..edges_start];
    let edges_bytes = &buf[edges_start..terminals_start];
    let terminals_bytes = &buf[terminals_start..terminals_start + terminals_len];

    // Parse nodes.
    let mut nodes = Vec::with_capacity(num_nodes);
    let mut pos = 0;
    for _ in 0..num_nodes {
        if pos + 3 > nodes_bytes.len() {
            return Err(PtwmCoreError::InvalidContainer(
                "Chain.read: node header truncated".into(),
            ));
        }
        let table_idx = u16::from_le_bytes([nodes_bytes[pos], nodes_bytes[pos + 1]]);
        let op = resolve_op_from_table(table_idx, table)?;
        let params_len = nodes_bytes[pos + 2] as usize;
        pos += 3;
        if pos + params_len > nodes_bytes.len() {
            return Err(PtwmCoreError::InvalidContainer(
                "Chain.read: node params truncated".into(),
            ));
        }
        let params = nodes_bytes[pos..pos + params_len].to_vec();
        pos += params_len;
        nodes.push(ChainNode { op, params });
    }
    if nodes.len() != num_nodes {
        return Err(PtwmCoreError::InvalidContainer(
            "Chain.read: node count mismatch".into(),
        ));
    }

    // Parse edges.
    let mut edges = Vec::with_capacity(num_edges);
    pos = 0;
    for _ in 0..num_edges {
        if pos + 5 > edges_bytes.len() {
            return Err(PtwmCoreError::InvalidContainer(
                "Chain.read: edge header truncated".into(),
            ));
        }
        let src_node = edges_bytes[pos];
        let src_output_idx = edges_bytes[pos + 1];
        let dst_node = edges_bytes[pos + 2];
        let dst_input_idx = edges_bytes[pos + 3];
        let role_present = edges_bytes[pos + 4];
        pos += 5;

        let role_override = match role_present {
            0 => None,
            1 => {
                let (role, n) = Role::read(&edges_bytes[pos..])?;
                pos += n;
                Some(role)
            }
            v => {
                return Err(PtwmCoreError::InvalidContainer(format!(
                    "Chain.read: edge role_present byte has unknown value {v} (expected 0 or 1)"
                )));
            }
        };

        if pos + 2 > edges_bytes.len() {
            return Err(PtwmCoreError::InvalidContainer(
                "Chain.read: edge vendor_bytes_len truncated".into(),
            ));
        }
        let vendor_len = u16::from_le_bytes([edges_bytes[pos], edges_bytes[pos + 1]]) as usize;
        pos += 2;

        if pos + vendor_len > edges_bytes.len() {
            return Err(PtwmCoreError::InvalidContainer(
                "Chain.read: edge vendor_bytes truncated".into(),
            ));
        }
        let vendor_bytes = edges_bytes[pos..pos + vendor_len].to_vec();
        pos += vendor_len;

        edges.push(ChainEdge {
            src_node,
            src_output_idx,
            dst_node,
            dst_input_idx,
            role_override,
            vendor_bytes,
        });
    }

    // Parse terminals.
    let mut terminals = Vec::with_capacity(num_terminals);
    pos = 0;
    for _ in 0..num_terminals {
        if pos + 2 > terminals_bytes.len() {
            return Err(PtwmCoreError::InvalidContainer(
                "Chain.read: terminal header truncated".into(),
            ));
        }
        let node_idx = terminals_bytes[pos];
        let output_idx = terminals_bytes[pos + 1];
        pos += 2;
        let (role, n) = Role::read(&terminals_bytes[pos..])?;
        pos += n;
        terminals.push(TerminalRef {
            node_idx,
            output_idx,
            role,
        });
    }

    Ok((
        Chain {
            nodes,
            edges,
            terminals,
        },
        total,
    ))
}

/// Resolve a u16 table index to an [`OpId`] using the file's Extension Table.
fn resolve_op_from_table(table_idx: u16, table: &ExtensionTable) -> Result<OpId, PtwmCoreError> {
    let entry = table.entries.get(table_idx as usize).ok_or_else(|| {
        PtwmCoreError::InvalidContainer(format!(
            "Chain.read: table index {table_idx} is out of range (table has {} entries)",
            table.entries.len()
        ))
    })?;
    match dispatch_builtin(&entry.canonical_id) {
        Some(BuiltinKind::Op(op)) => Ok(op),
        Some(BuiltinKind::Codec(_)) => Err(PtwmCoreError::InvalidContainer(format!(
            "Chain.read: table index {table_idx} resolves to a codec, not a transform op"
        ))),
        None => Err(PtwmCoreError::InvalidContainer(format!(
            "Chain.read: table index {table_idx} has unknown canonical id {:?}",
            entry.canonical_id
        ))),
    }
}

/// Look up the curated `ExtensionTableEntry` for an in-tree built-in by its
/// canonical id. Panics if the id is not found in the curated list: this
/// indicates an invariant violation (an `OpId` was added to
/// `transforms/op.rs` without a matching entry in `builtin_entries()`).
fn builtin_table_entry(id: CanonicalId) -> ExtensionTableEntry {
    crate::extension::builtins::builtin_entries()
        .into_iter()
        .find(|e| e.canonical_id == id)
        .expect("missing curated builtin entry for canonical id (invariant violated)")
}

/// Serialize `chain` into a **self-contained blob** suitable for use as a
/// Python chain blob or for any context where the caller does not have access
/// to a shared file-level Extension Table.
///
/// The blob format is:
/// ```text
/// local_et_count:    u32
/// local_et_entries:  [ExtensionTableEntry; count]
/// chain_bytes:       (num_nodes, …, nodes_section, edges_section, terminals_section)
/// ```
pub fn write_chain_blob(chain: &Chain) -> Result<Vec<u8>, PtwmCoreError> {
    let mut builder = ExtensionTableBuilder::new();
    let mut chain_bytes = Vec::new();
    write_chain(chain, &mut builder, &mut chain_bytes)?;
    let inline_table = ExtensionTable {
        entries: builder.finish(),
    };
    let et_bytes = inline_table
        .to_bytes()
        .map_err(|e| PtwmCoreError::InvalidContainer(format!("chain blob et: {e}")))?;
    let mut out = Vec::with_capacity(et_bytes.len() + chain_bytes.len());
    out.extend_from_slice(&et_bytes);
    out.extend_from_slice(&chain_bytes);
    Ok(out)
}

/// Parse a **self-contained chain blob** (ET prefix + chain bytes), as
/// produced by [`write_chain_blob`] or by the Python `Chain.to_bytes()` method.
///
/// Returns `(chain, bytes_consumed)`.
pub fn read_chain_blob(blob: &[u8]) -> Result<(Chain, usize), PtwmCoreError> {
    // Read the local extension table.
    let inline_table = ExtensionTable::from_bytes(blob)
        .map_err(|e| PtwmCoreError::InvalidContainer(format!("chain blob et: {e}")))?;
    // Measure how many bytes the table occupies by re-serializing it.
    let et_bytes = inline_table
        .to_bytes()
        .map_err(|e| PtwmCoreError::InvalidContainer(format!("chain blob et re-ser: {e}")))?;
    let et_consumed = et_bytes.len();
    if blob.len() < et_consumed {
        return Err(PtwmCoreError::InvalidContainer(
            "chain blob: buffer shorter than extension table".into(),
        ));
    }
    let (chain, chain_consumed) = read_chain(&blob[et_consumed..], &inline_table)?;
    Ok((chain, et_consumed + chain_consumed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extension::{ExtensionTable, ExtensionTableBuilder};
    use crate::types::role::{ScaleFormat, ValueFormat};

    fn make_source_params() -> Vec<u8> {
        // Source: dim_count=2, dims=[4,8], dtype=0x0002 (fp16)
        let mut p = vec![2u8]; // dim_count
        p.extend_from_slice(&4u32.to_le_bytes());
        p.extend_from_slice(&8u32.to_le_bytes());
        p.extend_from_slice(&0x0002u16.to_le_bytes());
        p
    }

    /// Write a chain to bytes and build an extension table from it.
    fn write_chain_to_buf(chain: &Chain) -> (Vec<u8>, ExtensionTable) {
        let mut builder = ExtensionTableBuilder::new();
        let mut buf = Vec::new();
        write_chain(chain, &mut builder, &mut buf).unwrap();
        let entries = builder.finish();
        let table = ExtensionTable { entries };
        (buf, table)
    }

    #[test]
    fn chain_full_roundtrip() {
        let chain = Chain {
            nodes: vec![
                ChainNode {
                    op: OpId::Source,
                    params: make_source_params(),
                },
                ChainNode {
                    op: OpId::BytePassthrough,
                    params: vec![],
                },
                ChainNode {
                    op: OpId::Terminal,
                    params: {
                        let mut p = vec![];
                        Role::Scale {
                            format: ScaleFormat::E4M3,
                        }
                        .write(&mut p);
                        p
                    },
                },
            ],
            edges: vec![
                ChainEdge {
                    src_node: 0,
                    src_output_idx: 0,
                    dst_node: 1,
                    dst_input_idx: 0,
                    role_override: Some(Role::Scale {
                        format: ScaleFormat::F32,
                    }),
                    vendor_bytes: vec![0xDE, 0xAD],
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
                role: Role::Value {
                    format: ValueFormat::Fp4E2m1,
                },
            }],
        };

        let (buf, table) = write_chain_to_buf(&chain);
        let (decoded, n) = read_chain(&buf, &table).unwrap();
        assert_eq!(n, buf.len());
        assert_eq!(decoded, chain);
    }

    #[test]
    fn chain_with_node_params() {
        // ByteSplit{n=4} carries n in params.
        let chain = Chain {
            nodes: vec![
                ChainNode {
                    op: OpId::Source,
                    params: make_source_params(),
                },
                ChainNode {
                    op: OpId::ByteSplit,
                    params: vec![4u8],
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
            terminals: vec![],
        };

        let (buf, table) = write_chain_to_buf(&chain);
        let (decoded, _n) = read_chain(&buf, &table).unwrap();
        assert_eq!(decoded.nodes[1].params, vec![4u8]);
    }

    #[test]
    fn chain_too_many_nodes_rejected() {
        // 256 nodes should error at encode time.
        let nodes: Vec<ChainNode> = (0..=255)
            .map(|_| ChainNode {
                op: OpId::BytePassthrough,
                params: vec![],
            })
            .collect();
        let chain = Chain {
            nodes,
            edges: vec![],
            terminals: vec![],
        };
        let mut builder = ExtensionTableBuilder::new();
        let mut buf = Vec::new();
        assert!(write_chain(&chain, &mut builder, &mut buf).is_err());
    }

    #[test]
    fn chain_rejects_unknown_role_present_byte() {
        let chain = Chain {
            nodes: vec![
                ChainNode {
                    op: OpId::Source,
                    params: make_source_params(),
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
            terminals: vec![],
        };
        let (mut buf, table) = write_chain_to_buf(&chain);

        // The edges section starts after the header (9 bytes) and the nodes
        // section. Locate the role_present byte: it's the 5th byte of the
        // first edge record (after src/src_out/dst/dst_in).
        let nodes_len = u16::from_le_bytes([buf[3], buf[4]]) as usize;
        let edges_start = 9 + nodes_len;
        let role_present_pos = edges_start + 4;
        assert_eq!(
            buf[role_present_pos], 0,
            "test setup: expected role_present=0"
        );
        buf[role_present_pos] = 2; // unknown value
        assert!(read_chain(&buf, &table).is_err());
    }

    #[test]
    fn chain_truncated_rejected() {
        let chain = Chain {
            nodes: vec![ChainNode {
                op: OpId::Source,
                params: make_source_params(),
            }],
            edges: vec![],
            terminals: vec![],
        };
        let (buf, table) = write_chain_to_buf(&chain);

        // Truncating any byte after the header should fail.
        for truncate_at in 1..buf.len() {
            let truncated = &buf[..truncate_at];
            assert!(
                read_chain(truncated, &table).is_err(),
                "expected error for truncation at byte {truncate_at}"
            );
        }
    }

    #[test]
    fn chain_rejects_table_index_out_of_range() {
        // Build a chain with Source, then mutate the table index to something
        // that does not exist in the extension table.
        let chain = Chain {
            nodes: vec![ChainNode {
                op: OpId::Source,
                params: make_source_params(),
            }],
            edges: vec![],
            terminals: vec![],
        };
        let (mut buf, _table) = write_chain_to_buf(&chain);
        // The first node's table index starts at byte 9 (after the 9-byte
        // chain header). Set it to 0xFF 0xFF — out of range for the 1-entry table.
        buf[9] = 0xFF;
        buf[10] = 0xFF;
        // Use the empty table to ensure the out-of-range check fires.
        let empty_table = ExtensionTable::empty();
        assert!(read_chain(&buf, &empty_table).is_err());
    }

    #[test]
    fn chain_with_two_distinct_ops_interns_two_entries() {
        // BitReorderIeee32 and ByteSplit are two distinct ops.
        let chain = Chain {
            nodes: vec![
                ChainNode {
                    op: OpId::BitReorderIeee32,
                    params: vec![],
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
            terminals: vec![],
        };
        let mut builder = ExtensionTableBuilder::new();
        let mut buf = Vec::new();
        write_chain(&chain, &mut builder, &mut buf).unwrap();
        let entries = builder.finish();
        assert_eq!(entries.len(), 2, "two distinct ops → two table entries");

        // The node bytes contain u16 indices 0 and 1, not OpId discriminants.
        let table = ExtensionTable { entries };
        let (decoded, _) = read_chain(&buf, &table).unwrap();
        assert_eq!(decoded, chain);
    }

    #[test]
    fn chain_with_same_op_twice_interns_once() {
        let chain = Chain {
            nodes: vec![
                ChainNode {
                    op: OpId::BytePassthrough,
                    params: vec![],
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
            terminals: vec![],
        };
        let mut builder = ExtensionTableBuilder::new();
        let mut buf = Vec::new();
        write_chain(&chain, &mut builder, &mut buf).unwrap();
        let entries = builder.finish();
        assert_eq!(entries.len(), 1, "same op twice → one table entry");

        let table = ExtensionTable { entries };
        let (decoded, _) = read_chain(&buf, &table).unwrap();
        assert_eq!(decoded, chain);
    }

    #[test]
    fn round_trip_chain_through_table_indices() {
        use crate::container::{ChainRegistry, ContainerReader, ContainerWriter};

        let chain = Chain {
            nodes: vec![
                ChainNode {
                    op: OpId::Source,
                    params: make_source_params(),
                },
                ChainNode {
                    op: OpId::BitReorderIeee32,
                    params: vec![],
                },
                ChainNode {
                    op: OpId::ByteSplit,
                    params: vec![4u8],
                },
                ChainNode {
                    op: OpId::Terminal,
                    params: {
                        let mut p = vec![];
                        Role::Scale {
                            format: ScaleFormat::E4M3,
                        }
                        .write(&mut p);
                        p
                    },
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
                ChainEdge {
                    src_node: 2,
                    src_output_idx: 0,
                    dst_node: 3,
                    dst_input_idx: 0,
                    role_override: None,
                    vendor_bytes: vec![],
                },
            ],
            terminals: vec![TerminalRef {
                node_idx: 3,
                output_idx: 0,
                role: Role::Scale {
                    format: ScaleFormat::E4M3,
                },
            }],
        };

        let registry = ChainRegistry {
            chains: vec![(0u16, chain.clone())],
        };

        // Write the container.
        let mut buf = std::io::Cursor::new(Vec::<u8>::new());
        let mut writer = ContainerWriter::new(&mut buf).unwrap();
        writer.set_chain_registry(registry).unwrap();
        writer.finalize().unwrap();

        // Read it back.
        let bytes = buf.into_inner();
        let reader = ContainerReader::open(&bytes).unwrap();
        let decoded = reader.chain_registry.lookup(0).expect("chain 0 not found");
        assert_eq!(decoded, &chain);
    }
}
