//! CW2, the compact chain wire.
//!
//! [`crate::chain::wire`] stores a chain with fixed-width fields and section
//! lengths, which costs 50-70 bytes for a typical production chain. CW2
//! stores the same [`Chain`] in about 15-20 bytes. The `\x89PTWS` compact
//! frame uses it, and the compact container will too. The legacy wire stays
//! unchanged for `\x89PTWM` containers and Python chain blobs.
//!
//! ```text
//!     n_nodes     varint  1..=255
//!     n_edges     varint  0..=255
//!     n_terms     varint  0..=255
//!     node × n_nodes:
//!         op_ref      varint  resolved by the caller (see below)
//!         param_len   varint  0..=255
//!         params      bytes
//!     [n_spine]   varint  0..=n_edges; present only when n_edges > 0
//!     edge × (n_edges − n_spine):
//!         src, src_out, dst, dst_in   varint each, 0..=255
//!         etag        u8      bit0 ROLE, bit1 VENDOR; other bits must be 0
//!         [role]      compact role (ROLE)
//!         [vendor]    varint len 1..=65535, then bytes (VENDOR)
//!     terminal × n_terms:
//!         ref         u8      node << 4 | out, for node <= 14 and out <= 15;
//!                             0xFF = escape, then node u8, out u8
//!         role        compact role
//! ```
//!
//! All varints are canonical unsigned LEB128 ([`crate::varint`]).
//!
//! **op_ref.** The caller maps ops to integers: the `\x89PTWS` frame uses
//! the `OpId` wire id, a container uses an Extension Table index.
//!
//! **Spine edges.** Most chains start with edges `(i, 0) → (i + 1, 0)` with
//! no role override and no vendor bytes. The first `n_spine` edges are of
//! that form for `i = 0..n_spine` and are not stored. The writer always
//! emits the longest such run, so the reader rejects an explicit first edge
//! that would extend it.
//!
//! **Bound Source.** A `Source` node with `param_len = 0` takes its params
//! from the tensor that the chain encodes: the caller passes
//! [`crate::transforms::source::bound_source_params`] of the tensor's shape
//! and dtype. The writer binds a Source exactly when its params equal those
//! bytes, and the reader rejects explicit params that equal them. Real
//! Source params are never empty (ndim and dtype take 3 bytes), so the
//! marker cannot collide with an explicit Source. Chains that differ only
//! in the tensor shape then encode to the same bytes.
//!
//! **Compact roles.** One byte for every role production chains use:
//!
//! | byte | role |
//! |---|---|
//! | `0x00..=0x04` | `Scale { format }` (`ScaleFormat` tag) |
//! | `0x08..=0x0C` | `GlobalScale { format }` |
//! | `0x10..=0x12` | `Nibble { kind }` (`NibbleKind` tag) |
//! | `0x18` | `Value { Fp4E2m1 }` |
//! | `0x19` | `ExponentByte` |
//! | `0x1A` | `Raw` |
//! | `0x1B` | `Index` |
//! | `0x1C` | `Residual { Xor }` |
//! | `0x40 + 8·(of − 1) + index` | `MantissaByte { index, of }`, `index < of <= 8` |
//! | `0x80 + 8·(of − 1) + index` | `IntegerByte { index, of }`, `index < of <= 8` |
//! | `0xC0 + bits − 1` | `Value { IntN { bits } }`, `1 <= bits <= 32` |
//! | `0xFF` | escape, followed by the [`Role::write`] bytes |
//!
//! Other bytes are rejected, and so is an escaped role that has a one-byte
//! form.

use crate::chain::graph::{Chain, ChainEdge, ChainNode, TerminalRef};
use crate::error::PtwmCoreError;
use crate::transforms::op::OpId;
use crate::types::role::{NibbleKind, ResidualFormat, Role, ScaleFormat, ValueFormat};
use crate::varint::{Cursor, write_varint};

/// Largest node, edge and terminal count (node indices are `u8`).
const MAX_COUNT: u64 = 255;
/// Largest params length per node (matches the legacy chain wire).
const MAX_PARAMS: u64 = 255;
/// Largest edge vendor-bytes length (matches the legacy chain wire).
const MAX_VENDOR: u64 = 65535;

const ETAG_ROLE: u8 = 0x01;
const ETAG_VENDOR: u8 = 0x02;

const TERM_ESCAPE: u8 = 0xFF;
const ROLE_ESCAPE: u8 = 0xFF;

/// `op_ref` resolver for the `\x89PTWS` frame: the `OpId` wire id.
pub fn op_wire_ref(op: OpId) -> Result<u64, PtwmCoreError> {
    Ok(u64::from(op.as_u16()))
}

/// Inverse of [`op_wire_ref`]; unknown ids are rejected.
pub fn op_from_wire_ref(op_ref: u64) -> Result<OpId, PtwmCoreError> {
    let wire = u16::try_from(op_ref).map_err(|_| {
        PtwmCoreError::InvalidContainer(format!("chain: op id {op_ref} exceeds u16"))
    })?;
    OpId::from_u16(wire)
}

/// Encode `chain` as CW2 into `out`.
///
/// `bound_source` is the tensor's [`bound_source_params`] (`None` when the
/// tensor cannot bind one); a Source whose params equal it is written as a
/// bound Source. `op_ref` maps each node's op to its stored integer.
///
/// [`bound_source_params`]: crate::transforms::source::bound_source_params
pub fn write_chain_compact(
    chain: &Chain,
    bound_source: Option<&[u8]>,
    mut op_ref: impl FnMut(OpId) -> Result<u64, PtwmCoreError>,
    out: &mut Vec<u8>,
) -> Result<(), PtwmCoreError> {
    let err = |msg: String| PtwmCoreError::InvalidContainer(format!("chain: {msg}"));
    for (what, n) in [
        ("nodes", chain.nodes.len()),
        ("edges", chain.edges.len()),
        ("terminals", chain.terminals.len()),
    ] {
        if n as u64 > MAX_COUNT {
            return Err(err(format!("{n} {what} (max {MAX_COUNT})")));
        }
    }
    if chain.nodes.is_empty() {
        return Err(err("no nodes".into()));
    }
    write_varint(out, chain.nodes.len() as u64);
    write_varint(out, chain.edges.len() as u64);
    write_varint(out, chain.terminals.len() as u64);

    for node in &chain.nodes {
        if node.params.len() as u64 > MAX_PARAMS {
            return Err(err(format!(
                "node params length {} (max {MAX_PARAMS})",
                node.params.len()
            )));
        }
        write_varint(out, op_ref(node.op)?);
        if node.op == OpId::Source {
            if node.params.is_empty() {
                // An empty Source would read back as a bound Source.
                return Err(err("Source node with empty params".into()));
            }
            if bound_source == Some(node.params.as_slice()) {
                write_varint(out, 0);
                continue;
            }
        }
        write_varint(out, node.params.len() as u64);
        out.extend_from_slice(&node.params);
    }

    if !chain.edges.is_empty() {
        let n_spine = spine_len(&chain.edges);
        write_varint(out, n_spine as u64);
        for edge in &chain.edges[n_spine..] {
            for v in [
                edge.src_node,
                edge.src_output_idx,
                edge.dst_node,
                edge.dst_input_idx,
            ] {
                write_varint(out, u64::from(v));
            }
            let mut etag = 0u8;
            if edge.role_override.is_some() {
                etag |= ETAG_ROLE;
            }
            if !edge.vendor_bytes.is_empty() {
                etag |= ETAG_VENDOR;
            }
            out.push(etag);
            if let Some(role) = &edge.role_override {
                write_role_compact(role, out)?;
            }
            if !edge.vendor_bytes.is_empty() {
                if edge.vendor_bytes.len() as u64 > MAX_VENDOR {
                    return Err(err(format!(
                        "edge vendor bytes length {} (max {MAX_VENDOR})",
                        edge.vendor_bytes.len()
                    )));
                }
                write_varint(out, edge.vendor_bytes.len() as u64);
                out.extend_from_slice(&edge.vendor_bytes);
            }
        }
    }

    for term in &chain.terminals {
        if term.node_idx <= 14 && term.output_idx <= 15 {
            out.push((term.node_idx << 4) | term.output_idx);
        } else {
            out.extend_from_slice(&[TERM_ESCAPE, term.node_idx, term.output_idx]);
        }
        write_role_compact(&term.role, out)?;
    }
    Ok(())
}

/// Decode one CW2 chain at the cursor. The cursor advances past the chain;
/// bytes after it are left for the caller.
///
/// `bound_source` supplies the params of a bound Source (see
/// [`write_chain_compact`]); a bound Source without it is an error. The
/// returned chain is not validated: callers run `validate_chain`.
pub fn read_chain_compact(
    r: &mut Cursor<'_>,
    bound_source: Option<&[u8]>,
    mut resolve: impl FnMut(u64) -> Result<OpId, PtwmCoreError>,
) -> Result<Chain, PtwmCoreError> {
    let n_nodes = r.varint_max(MAX_COUNT, "chain node count")? as usize;
    if n_nodes == 0 {
        return Err(r.err("chain has no nodes"));
    }
    let n_edges = r.varint_max(MAX_COUNT, "chain edge count")? as usize;
    let n_terms = r.varint_max(MAX_COUNT, "chain terminal count")? as usize;
    // Each node takes at least 2 bytes and each terminal 2, so a short
    // buffer fails here instead of after the allocations below.
    if r.remaining() < 2 * n_nodes + 2 * n_terms {
        return Err(r.err("chain truncated"));
    }

    let mut nodes = Vec::with_capacity(n_nodes);
    for _ in 0..n_nodes {
        let op_ref = r.varint()?;
        let op = resolve(op_ref)?;
        let param_len = r.varint_max(MAX_PARAMS, "chain node params length")? as usize;
        let params = if op == OpId::Source && param_len == 0 {
            bound_source
                .ok_or_else(|| {
                    r.err("chain has a bound Source but the tensor has no bindable shape and dtype")
                })?
                .to_vec()
        } else {
            let params = r.take(param_len)?;
            if op == OpId::Source && bound_source == Some(params) {
                return Err(r.err("chain Source params equal the bound Source (non-canonical)"));
            }
            params.to_vec()
        };
        nodes.push(ChainNode { op, params });
    }

    let mut edges = Vec::with_capacity(n_edges);
    if n_edges > 0 {
        let n_spine = r.varint_max(n_edges as u64, "chain spine edge count")? as usize;
        for i in 0..n_spine {
            edges.push(spine_edge(i as u8));
        }
        for i in n_spine..n_edges {
            let mut idx = [0u8; 4];
            for v in &mut idx {
                *v = r.varint_max(MAX_COUNT, "chain edge index")? as u8;
            }
            let etag = r.u8()?;
            if etag & !(ETAG_ROLE | ETAG_VENDOR) != 0 {
                return Err(r.err(format_args!("chain edge tag {etag:#04x} has unknown bits")));
            }
            let role_override = if etag & ETAG_ROLE != 0 {
                Some(read_role_compact(r)?)
            } else {
                None
            };
            let vendor_bytes = if etag & ETAG_VENDOR != 0 {
                let len = r.varint_max(MAX_VENDOR, "chain edge vendor length")?;
                if len == 0 {
                    return Err(r.err("chain edge VENDOR flag with no vendor bytes"));
                }
                r.take(len as usize)?.to_vec()
            } else {
                Vec::new()
            };
            let edge = ChainEdge {
                src_node: idx[0],
                src_output_idx: idx[1],
                dst_node: idx[2],
                dst_input_idx: idx[3],
                role_override,
                vendor_bytes,
            };
            if i == n_spine && edge == spine_edge(i as u8) {
                return Err(r.err("chain spine edge stored explicitly (non-canonical)"));
            }
            edges.push(edge);
        }
    }

    let mut terminals = Vec::with_capacity(n_terms);
    for _ in 0..n_terms {
        let b = r.u8()?;
        let (node_idx, output_idx) = if b == TERM_ESCAPE {
            let node = r.u8()?;
            let out = r.u8()?;
            if node <= 14 && out <= 15 {
                return Err(r.err("chain terminal ref escaped needlessly (non-canonical)"));
            }
            (node, out)
        } else if b >> 4 == 15 {
            return Err(r.err(format_args!("chain terminal ref byte {b:#04x} is reserved")));
        } else {
            (b >> 4, b & 0x0F)
        };
        let role = read_role_compact(r)?;
        terminals.push(TerminalRef {
            node_idx,
            output_idx,
            role,
        });
    }

    Ok(Chain {
        nodes,
        edges,
        terminals,
    })
}

/// Decode a CW2 chain that must fill `buf` exactly.
pub fn read_chain_compact_exact(
    buf: &[u8],
    bound_source: Option<&[u8]>,
    resolve: impl FnMut(u64) -> Result<OpId, PtwmCoreError>,
) -> Result<Chain, PtwmCoreError> {
    let mut r = Cursor::new(buf, "compact chain");
    let chain = read_chain_compact(&mut r, bound_source, resolve)?;
    r.finish("chain")?;
    Ok(chain)
}

fn spine_edge(i: u8) -> ChainEdge {
    ChainEdge {
        src_node: i,
        src_output_idx: 0,
        dst_node: i + 1,
        dst_input_idx: 0,
        role_override: None,
        vendor_bytes: Vec::new(),
    }
}

/// Length of the leading run of spine edges.
fn spine_len(edges: &[ChainEdge]) -> usize {
    edges
        .iter()
        .enumerate()
        .take_while(|&(i, e)| i < 255 && *e == spine_edge(i as u8))
        .count()
}

/// The one-byte form of `role`, if it has one.
fn role_code(role: &Role) -> Option<u8> {
    let small =
        |index: u8, of: u8| (index < of && (1..=8).contains(&of)).then(|| (of - 1) * 8 + index);
    match role {
        Role::Scale { format } => Some(format.as_u8()),
        Role::GlobalScale { format } => Some(0x08 + format.as_u8()),
        Role::Nibble { kind } => Some(0x10 + kind.as_u8()),
        Role::Value {
            format: ValueFormat::Fp4E2m1,
        } => Some(0x18),
        Role::ExponentByte => Some(0x19),
        Role::Raw => Some(0x1A),
        Role::Index => Some(0x1B),
        Role::Residual {
            format: ResidualFormat::Xor,
        } => Some(0x1C),
        Role::MantissaByte { index, of } => small(*index, *of).map(|c| 0x40 + c),
        Role::IntegerByte { index, of } => small(*index, *of).map(|c| 0x80 + c),
        Role::Value {
            format: ValueFormat::IntN { bits },
        } if (1..=32).contains(bits) => Some(0xC0 + bits - 1),
        _ => None,
    }
}

/// Append the compact encoding of `role`.
pub fn write_role_compact(role: &Role, out: &mut Vec<u8>) -> Result<(), PtwmCoreError> {
    if let Some(code) = role_code(role) {
        out.push(code);
        return Ok(());
    }
    if let Role::Vendor { bytes, .. } = role
        && bytes.len() > u16::MAX as usize
    {
        return Err(PtwmCoreError::InvalidContainer(format!(
            "chain: vendor role of {} bytes (max {})",
            bytes.len(),
            u16::MAX
        )));
    }
    out.push(ROLE_ESCAPE);
    role.write(out);
    Ok(())
}

/// Read one compact role at the cursor.
pub fn read_role_compact(r: &mut Cursor<'_>) -> Result<Role, PtwmCoreError> {
    let b = r.u8()?;
    let byte_role = |code: u8| ((code & 7), (code >> 3) + 1);
    let role = match b {
        0x00..=0x04 => Role::Scale {
            format: ScaleFormat::from_u8(b)?,
        },
        0x08..=0x0C => Role::GlobalScale {
            format: ScaleFormat::from_u8(b - 0x08)?,
        },
        0x10..=0x12 => Role::Nibble {
            kind: NibbleKind::from_u8(b - 0x10)?,
        },
        0x18 => Role::Value {
            format: ValueFormat::Fp4E2m1,
        },
        0x19 => Role::ExponentByte,
        0x1A => Role::Raw,
        0x1B => Role::Index,
        0x1C => Role::Residual {
            format: ResidualFormat::Xor,
        },
        0x40..=0x7F => {
            let (index, of) = byte_role(b - 0x40);
            if index >= of {
                return Err(r.err(format_args!("role byte {b:#04x} has index >= of")));
            }
            Role::MantissaByte { index, of }
        }
        0x80..=0xBF => {
            let (index, of) = byte_role(b - 0x80);
            if index >= of {
                return Err(r.err(format_args!("role byte {b:#04x} has index >= of")));
            }
            Role::IntegerByte { index, of }
        }
        0xC0..=0xDF => Role::Value {
            format: ValueFormat::IntN { bits: b - 0xC0 + 1 },
        },
        ROLE_ESCAPE => {
            let (role, n) = Role::read(r.rest())?;
            r.take(n)?;
            if role_code(&role).is_some() {
                return Err(r.err("role escaped needlessly (non-canonical)"));
            }
            role
        }
        _ => return Err(r.err(format_args!("unknown role byte {b:#04x}"))),
    };
    Ok(role)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transforms::source::bound_source_params;

    fn write(chain: &Chain, bound: Option<&[u8]>) -> Vec<u8> {
        let mut out = Vec::new();
        write_chain_compact(chain, bound, op_wire_ref, &mut out).unwrap();
        out
    }

    fn read(buf: &[u8], bound: Option<&[u8]>) -> Result<Chain, PtwmCoreError> {
        read_chain_compact_exact(buf, bound, op_from_wire_ref)
    }

    fn assert_round_trip(label: &str, chain: &Chain, bound: Option<&[u8]>) -> Vec<u8> {
        let bytes = write(chain, bound);
        let back = read(&bytes, bound).unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_eq!(&back, chain, "{label}");
        // Canonical: re-encoding gives the same bytes.
        assert_eq!(write(&back, bound), bytes, "{label}");
        bytes
    }

    fn node(op: OpId, params: &[u8]) -> ChainNode {
        ChainNode {
            op,
            params: params.to_vec(),
        }
    }

    fn edge(src: u8, out: u8, dst: u8, inp: u8) -> ChainEdge {
        ChainEdge {
            src_node: src,
            src_output_idx: out,
            dst_node: dst,
            dst_input_idx: inp,
            role_override: None,
            vendor_bytes: Vec::new(),
        }
    }

    fn term(node_idx: u8, output_idx: u8, role: Role) -> TerminalRef {
        TerminalRef {
            node_idx,
            output_idx,
            role,
        }
    }

    /// fp32 [64, 64]: Source → BitReorderIeee32 → ByteSplit(4), 4 terminals.
    fn fp32_split() -> (Chain, Vec<u8>) {
        let source = bound_source_params(&[64, 64], 1).unwrap();
        let chain = Chain {
            nodes: vec![
                node(OpId::Source, &source),
                node(OpId::BitReorderIeee32, &[]),
                node(OpId::ByteSplit, &[4]),
            ],
            edges: vec![edge(0, 0, 1, 0), edge(1, 0, 2, 0)],
            terminals: (0..4)
                .map(|i| term(2, i, Role::MantissaByte { index: i, of: 4 }))
                .collect(),
        };
        (chain, source)
    }

    #[test]
    fn production_chains_round_trip() {
        // Same fixture as the symbolic-descriptor test in `runtime.rs`.
        let data: &[u8] = include_bytes!("../../tests/fixtures/production_chains.bin");
        let mut pos = 0usize;
        let (mut total, mut bound_shorter) = (0usize, 0usize);
        let mut legacy_bytes = 0usize;
        let mut compact_bytes = 0usize;
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
            let (chain, _) = crate::chain::wire::read_chain_blob(blob).unwrap();
            let explicit = assert_round_trip(label, &chain, None);
            // Bind the chain's own Source params, as a writer does when
            // they match the tensor.
            let own = chain.nodes[0].params.clone();
            let bound = assert_round_trip(label, &chain, Some(&own));
            assert!(bound.len() < explicit.len(), "{label}");
            bound_shorter += 1;
            // A different binding leaves the Source explicit.
            assert_eq!(
                assert_round_trip(label, &chain, Some(&[0, 1, 0][..])),
                explicit
            );

            let mut b = crate::extension::ExtensionTableBuilder::new();
            let mut legacy = Vec::new();
            crate::chain::wire::write_chain(&chain, &mut b, &mut legacy).unwrap();
            legacy_bytes += legacy.len();
            compact_bytes += bound.len();
        }
        assert!(total > 500, "fixture holds only {total} chains");
        assert_eq!(bound_shorter, total);
        // CW2 with a bound Source is far smaller than the legacy wire.
        assert!(
            compact_bytes * 3 < legacy_bytes,
            "{compact_bytes} vs {legacy_bytes}"
        );
    }

    #[test]
    fn fp32_split_chain_is_small() {
        let (chain, source) = fp32_split();
        let bytes = assert_round_trip("fp32", &chain, Some(&source));
        // 3 counts, nodes 2 + 2 + 3, n_spine 1, 4 × (ref + role).
        assert_eq!(bytes.len(), 19, "{bytes:02x?}");
        assert_eq!(
            bytes,
            [
                3, 2, 4, // counts
                0x00, 0, // bound Source
                0x11, 0, // BitReorderIeee32
                0x20, 1, 4, // ByteSplit(4)
                2, // both edges are spine edges
                0x20, 0x58, 0x21, 0x59, 0x22, 0x5A, 0x23, 0x5B,
            ]
        );
    }

    #[test]
    fn edge_cases_round_trip() {
        let (base, source) = fp32_split();

        // Non-spine edges, role overrides and vendor bytes.
        let mut c = base.clone();
        c.edges[1].role_override = Some(Role::ExponentByte);
        c.edges.push(ChainEdge {
            vendor_bytes: vec![9; 300],
            ..edge(1, 1, 2, 1)
        });
        let bytes = assert_round_trip("non-spine", &c, Some(&source));
        assert_eq!(bytes[10], 1, "only the first edge is a spine edge");
        // An edge that matches the spine pattern after a break stays explicit.
        let mut c = base.clone();
        c.edges = vec![edge(0, 0, 1, 1), edge(1, 0, 2, 0)];
        assert_round_trip("late spine", &c, None);

        // Escaped terminal refs.
        let mut c = base.clone();
        c.terminals.push(term(15, 0, Role::Raw));
        c.terminals.push(term(1, 16, Role::Raw));
        c.terminals.push(term(255, 255, Role::Raw));
        assert_round_trip("terminal escape", &c, None);

        // Every one-byte role and escaped roles.
        let mut roles = vec![
            Role::ExponentByte,
            Role::Raw,
            Role::Index,
            Role::Residual {
                format: ResidualFormat::Xor,
            },
            Role::Residual {
                format: ResidualFormat::FloatDelta { dtype: 0x0203 },
            },
            Role::Value {
                format: ValueFormat::Fp4E2m1,
            },
            Role::Vendor {
                tag: 7,
                bytes: vec![1, 2, 3],
            },
            Role::MantissaByte { index: 9, of: 9 },
            Role::MantissaByte { index: 3, of: 2 },
            Role::IntegerByte { index: 0, of: 0 },
            Role::IntegerByte { index: 0, of: 12 },
        ];
        for f in [
            ScaleFormat::E8M0,
            ScaleFormat::E4M3,
            ScaleFormat::E5M2,
            ScaleFormat::F32,
            ScaleFormat::F16,
        ] {
            roles.push(Role::Scale { format: f });
            roles.push(Role::GlobalScale { format: f });
        }
        for kind in [
            NibbleKind::Exponent,
            NibbleKind::SignMantissa,
            NibbleKind::Value,
        ] {
            roles.push(Role::Nibble { kind });
        }
        for bits in [0u8, 1, 4, 8, 32, 33, 255] {
            roles.push(Role::Value {
                format: ValueFormat::IntN { bits },
            });
        }
        for of in 1..=8u8 {
            for index in 0..of {
                roles.push(Role::MantissaByte { index, of });
                roles.push(Role::IntegerByte { index, of });
            }
        }
        let mut c = base.clone();
        c.terminals = roles
            .iter()
            .enumerate()
            .map(|(i, role)| term(2, (i % 200) as u8, role.clone()))
            .collect();
        c.edges[0].role_override = Some(Role::Vendor {
            tag: 1,
            bytes: vec![],
        });
        assert_round_trip("roles", &c, None);
        for role in &roles {
            let mut out = Vec::new();
            write_role_compact(role, &mut out).unwrap();
            let escaped = out[0] == ROLE_ESCAPE;
            assert_eq!(escaped, role_code(role).is_none(), "{role:?}");
            if !escaped {
                assert_eq!(out.len(), 1, "{role:?}");
            }
        }

        // Large op refs (multi-byte varints) through a custom resolver.
        let mut out = Vec::new();
        write_chain_compact(
            &base,
            None,
            |op| Ok(u64::from(op.as_u16()) + 100_000),
            &mut out,
        )
        .unwrap();
        let back = read_chain_compact_exact(&out, None, |r| op_from_wire_ref(r - 100_000)).unwrap();
        assert_eq!(back, base);

        // 0-d tensor Source and a single-node chain.
        let scalar = bound_source_params(&[], 1).unwrap();
        let c = Chain {
            nodes: vec![node(OpId::Source, &scalar)],
            edges: vec![],
            terminals: vec![term(0, 0, Role::Raw)],
        };
        assert_eq!(
            assert_round_trip("scalar", &c, Some(&scalar)),
            [1, 0, 1, 0, 0, 0x00, 0x1A]
        );

        // 255 nodes and edges, all spine.
        let mut c = base.clone();
        c.nodes = std::iter::once(node(OpId::Source, &source))
            .chain((1..255).map(|_| node(OpId::BytePassthrough, &[])))
            .collect();
        c.edges = (0..254).map(|i| edge(i, 0, i + 1, 0)).collect();
        c.edges.push(edge(254, 1, 0, 0));
        assert_round_trip("max counts", &c, Some(&source));
    }

    #[test]
    fn unencodable_chains_are_rejected() {
        let (base, _) = fp32_split();
        let mut out = Vec::new();
        let mut c = base.clone();
        c.nodes[0].params.clear();
        let err = write_chain_compact(&c, None, op_wire_ref, &mut out).unwrap_err();
        assert!(err.to_string().contains("empty params"), "{err}");
        let mut c = base.clone();
        c.nodes[1].params = vec![0; 256];
        assert!(write_chain_compact(&c, None, op_wire_ref, &mut out).is_err());
        let mut c = base.clone();
        c.terminals[0].role = Role::Vendor {
            tag: 0,
            bytes: vec![0; 65536],
        };
        assert!(write_chain_compact(&c, None, op_wire_ref, &mut out).is_err());
        let mut c = base.clone();
        c.edges[1].vendor_bytes = vec![0; 65536];
        assert!(write_chain_compact(&c, None, op_wire_ref, &mut out).is_err());
        let c = Chain {
            nodes: vec![],
            edges: vec![],
            terminals: vec![],
        };
        assert!(write_chain_compact(&c, None, op_wire_ref, &mut out).is_err());
    }

    fn rejects(label: &str, buf: &[u8], bound: Option<&[u8]>, needle: &str) {
        let err = read(buf, bound)
            .err()
            .unwrap_or_else(|| panic!("{label}: accepted {buf:02x?}"))
            .to_string();
        assert!(err.contains(needle), "{label}: {err}");
    }

    #[test]
    fn non_canonical_and_corrupt_chains_are_rejected() {
        let (chain, source) = fp32_split();
        let good = write(&chain, Some(&source));
        let src = Some(source.as_slice());

        // Every truncation, and trailing bytes.
        for cut in 0..good.len() {
            assert!(read(&good[..cut], src).is_err(), "cut at {cut}");
        }
        let mut longer = good.clone();
        longer.push(0);
        rejects("trailing", &longer, src, "trailing");

        // Bound Source with no binding available.
        rejects("unbound", &good, None, "bound Source");
        // Explicit params equal to the binding.
        let explicit = write(&chain, None);
        rejects("explicit bound", &explicit, src, "non-canonical");

        // Non-minimal varint (n_nodes = 3 as 0x83 0x00).
        let mut b = vec![0x83, 0x00];
        b.extend_from_slice(&good[1..]);
        rejects("overlong", &b, src, "non-minimal");

        // Counts.
        rejects("no nodes", &[0, 0, 0], src, "no nodes");
        rejects("256 nodes", &[0x80, 0x02, 0, 0], src, "exceeds 255");

        // n_spine > n_edges.
        let mut b = good.clone();
        b[10] = 3;
        rejects("n_spine", &b, src, "spine edge count");

        // Unknown op.
        let mut b = good.clone();
        b[5] = 0x7F;
        assert!(read(&b, src).is_err());

        // Params longer than 255.
        let mut b = good[..6].to_vec();
        b.extend_from_slice(&[0x80, 0x02]);
        b.extend_from_slice(&[0; 300]);
        rejects("params len", &b, src, "params length");

        // Reserved terminal ref byte and needless escape.
        let mut b = good.clone();
        b[11] = 0xF0;
        rejects("ref 0xF0", &b, src, "reserved");
        let mut b = good[..11].to_vec();
        b.extend_from_slice(&[0xFF, 2, 0]);
        b.extend_from_slice(&good[12..]);
        rejects("ref escape", &b, src, "non-canonical");

        // Unknown and needlessly escaped roles.
        for code in [0x05u8, 0x0D, 0x13, 0x1D, 0x3F, 0xE0, 0xFE] {
            let mut b = good.clone();
            b[12] = code;
            rejects(&format!("role {code:#04x}"), &b, src, "role byte");
        }
        // MantissaByte index >= of: 0x40 + 8*0 + 1 = {index 1, of 1}.
        let mut b = good.clone();
        b[12] = 0x41;
        rejects("index >= of", &b, src, "index >= of");
        let mut b = good[..12].to_vec();
        b.extend_from_slice(&[0xFF, 0x08]); // Raw through the escape
        b.extend_from_slice(&good[13..]);
        rejects("role escape", &b, src, "non-canonical");

        // Explicit edges: unknown etag bits, VENDOR with no bytes, and an
        // explicit edge that extends the spine.
        let mut c = chain.clone();
        c.edges[0].dst_input_idx = 1;
        let ok = write(&c, Some(&source));
        // counts(3) nodes(7) n_spine(1) = 11; edge 0 indices at 11..15,
        // etag at 15.
        assert_eq!(ok[10], 0);
        assert_eq!(&ok[11..16], &[0, 0, 1, 1, 0]);
        for etag in [0x04u8, 0x80] {
            let mut b = ok.clone();
            b[15] = etag;
            rejects("etag", &b, src, "unknown bits");
        }
        let mut b = ok[..15].to_vec();
        b.extend_from_slice(&[ETAG_VENDOR, 0]);
        b.extend_from_slice(&ok[16..]);
        rejects("empty vendor", &b, src, "no vendor bytes");
        let mut b = ok.clone();
        b[14] = 0; // dst_in 1 → 0: now edge 0 is a spine edge
        rejects("explicit spine", &b, src, "spine edge stored explicitly");
    }

    #[test]
    fn mutated_chains_are_rejected_or_canonical() {
        // The invariant of the `chain_wire_compact` fuzz target, on random
        // mutations of a few chains: every accepted input re-encodes to
        // itself.
        let (base, source) = fp32_split();
        let mut with_extras = base.clone();
        with_extras.edges.push(ChainEdge {
            role_override: Some(Role::Vendor {
                tag: 3,
                bytes: vec![1, 2],
            }),
            vendor_bytes: vec![7],
            ..edge(1, 2, 2, 1)
        });
        with_extras
            .terminals
            .push(term(200, 3, Role::IntegerByte { index: 0, of: 9 }));
        let seeds = [
            write(&base, Some(&source)),
            write(&base, None),
            write(&with_extras, Some(&source)),
        ];
        let mut x: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut accepted = 0;
        for i in 0..60_000 {
            let mut b = seeds[i % seeds.len()].clone();
            for _ in 0..1 + next() % 3 {
                let at = (next() as usize) % b.len();
                match next() % 3 {
                    0 => b[at] = next() as u8,
                    1 => b[at] ^= 1 << (next() % 8),
                    _ => {
                        b.insert(at, next() as u8);
                    }
                }
            }
            let bound = (next() % 2 == 0).then_some(source.as_slice());
            if let Ok(chain) = read(&b, bound) {
                assert_eq!(write(&chain, bound), b, "{b:02x?}");
                accepted += 1;
            }
        }
        assert!(accepted > 1000, "only {accepted} mutations parsed");
    }
}
