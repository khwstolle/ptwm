//! PTWM container ⇄ canonical PTWM-JSON manifest transcoder.
//!
//! The binary `.ptwm` container is structurally *(a)* a registry of
//! codecs/ops/shared-state/chains, *(b)* per-tensor records, and *(c)* per-plane
//! compressed payloads. This module lifts that into a declarative manifest
//! (see [`manifest`]) plus loose binary members, and back again.
//!
//! * [`explode`] parses a container into `(registry_json, per-tensor manifests,
//!   loose members)`.
//! * [`implode`] reverses it, re-driving the deterministic container writer so
//!   that `implode(explode(blob)) == blob` byte-for-byte for every container the
//!   compression engine produces.
//!
//! The encoder may build a container transiently in memory; the stored artifact
//! (WebDataset shard / LMDB exploded value) never contains one.

mod base64;
mod jcs;
pub mod manifest;

use std::collections::HashMap;
use std::io::Cursor;

use crate::chain::graph::{Chain, ChainEdge, ChainNode, TerminalRef};
use crate::codec::{CodecId, StateSource};
use crate::container::{ChainRegistry, ContainerReader, ContainerWriter};
use crate::error::PtwmCoreError;
use crate::header::FLAG_PTWX;
use crate::layout::PlaneLayout;
use crate::metadata::{decode_shape_metadata, encode_shape_metadata};
use crate::plane_record::{ChunkEntry, ExternalStateRef, PlaneRecord};
use crate::prelude::PreludeEntry;
use crate::tensor_record::{
    CHAIN_REF_INLINE_BIT, Dependency, RefKind, TensorRecord, parse_tensor_record,
};
use crate::transforms::op::OpId;
use crate::types::PlaneRole;
use crate::types::role::Role;

use manifest::*;

/// Result of exploding a container.
pub struct Exploded {
    /// Canonical (JCS) archive-scoped registry manifest.
    pub registry_json: String,
    /// `(tensor_name, sample_key, canonical_tensor_manifest_json)` in file order.
    pub tensors: Vec<ExplodedTensor>,
    /// Loose binary members keyed by manifest member name (plane payloads,
    /// externalized state, shared state, embedded modules).
    pub members: Vec<(String, Vec<u8>)>,
}

pub struct ExplodedTensor {
    pub name: String,
    /// Zero-padded global tensor index used as the WebDataset `__key__`.
    pub key: String,
    pub manifest_json: String,
}

fn err(msg: impl Into<String>) -> PtwmCoreError {
    PtwmCoreError::InvalidContainer(msg.into())
}

// ── Naming helpers ────────────────────────────────────────────────────────────

fn codec_name(id: CodecId) -> String {
    crate::codec::REGISTRY
        .iter()
        .find(|(_, wire)| *wire == id.as_u16())
        .map(|(name, _)| (*name).to_string())
        .unwrap_or_else(|| format!("codec_{:#06x}", id.as_u16()))
}

fn op_name(op: OpId) -> String {
    format!("{op:?}")
}

fn plane_role_name(role: PlaneRole) -> String {
    format!("{role:?}")
}

/// Codecs whose inline state is a bounded-small table (always inlined as
/// base64 per Decision 2). Everything else with inline state is externalized
/// to a member.
fn codec_state_is_bounded(id: CodecId) -> bool {
    matches!(
        id,
        CodecId::Huffman | CodecId::HuffmanNibble | CodecId::Rans | CodecId::Tans
    )
}

fn state_source_label(source: StateSource, bounded: bool) -> &'static str {
    match source {
        StateSource::None => "none",
        StateSource::Inline => {
            if bounded {
                "inline"
            } else {
                "member"
            }
        }
        StateSource::Shared => "shared",
        StateSource::External => "external",
    }
}

// ── Chain ⇄ JSON ──────────────────────────────────────────────────────────────

fn node_to_json(node: &ChainNode) -> NodeJson {
    NodeJson {
        op: node.op.as_u16(),
        params_b64: if node.params.is_empty() {
            None
        } else {
            Some(base64::encode(&node.params))
        },
    }
}

fn edge_to_json(edge: &ChainEdge) -> EdgeJson {
    let role_override_b64 = edge.role_override.as_ref().map(|r| {
        let mut buf = Vec::new();
        r.write(&mut buf);
        base64::encode(&buf)
    });
    EdgeJson {
        src: edge.src_node,
        src_out: edge.src_output_idx,
        dst: edge.dst_node,
        dst_in: edge.dst_input_idx,
        role_override_b64,
        vendor_b64: if edge.vendor_bytes.is_empty() {
            None
        } else {
            Some(base64::encode(&edge.vendor_bytes))
        },
    }
}

fn terminal_to_json(term: &TerminalRef) -> TerminalJson {
    let mut buf = Vec::new();
    term.role.write(&mut buf);
    TerminalJson {
        node: term.node_idx,
        out: term.output_idx,
        role: format!("{:?}", term.role),
        role_b64: base64::encode(&buf),
    }
}

fn chain_to_json(chain: &Chain) -> (Vec<NodeJson>, Vec<EdgeJson>, Vec<TerminalJson>) {
    (
        chain.nodes.iter().map(node_to_json).collect(),
        chain.edges.iter().map(edge_to_json).collect(),
        chain.terminals.iter().map(terminal_to_json).collect(),
    )
}

fn node_from_json(node: &NodeJson) -> Result<ChainNode, PtwmCoreError> {
    let params = match &node.params_b64 {
        Some(s) => base64::decode(s).ok_or_else(|| err("node params_b64 not valid base64"))?,
        None => Vec::new(),
    };
    Ok(ChainNode {
        op: OpId::from_u16(node.op)?,
        params,
    })
}

fn role_from_b64(s: &str, what: &str) -> Result<Role, PtwmCoreError> {
    let bytes = base64::decode(s).ok_or_else(|| err(format!("{what}: not valid base64")))?;
    let (role, _) = Role::read(&bytes)?;
    Ok(role)
}

fn edge_from_json(edge: &EdgeJson) -> Result<ChainEdge, PtwmCoreError> {
    let role_override = match &edge.role_override_b64 {
        Some(s) => Some(role_from_b64(s, "edge role_override_b64")?),
        None => None,
    };
    let vendor_bytes = match &edge.vendor_b64 {
        Some(s) => base64::decode(s).ok_or_else(|| err("edge vendor_b64 not valid base64"))?,
        None => Vec::new(),
    };
    Ok(ChainEdge {
        src_node: edge.src,
        src_output_idx: edge.src_out,
        dst_node: edge.dst,
        dst_input_idx: edge.dst_in,
        role_override,
        vendor_bytes,
    })
}

fn terminal_from_json(term: &TerminalJson) -> Result<TerminalRef, PtwmCoreError> {
    Ok(TerminalRef {
        node_idx: term.node,
        output_idx: term.out,
        role: role_from_b64(&term.role_b64, "terminal role_b64")?,
    })
}

fn chain_from_json(
    nodes: &[NodeJson],
    edges: &[EdgeJson],
    terminals: &[TerminalJson],
) -> Result<Chain, PtwmCoreError> {
    Ok(Chain {
        nodes: nodes.iter().map(node_from_json).collect::<Result<_, _>>()?,
        edges: edges.iter().map(edge_from_json).collect::<Result<_, _>>()?,
        terminals: terminals
            .iter()
            .map(terminal_from_json)
            .collect::<Result<_, _>>()?,
    })
}

// ── Member keys ───────────────────────────────────────────────────────────────

fn plane_payload_key(sample_key: &str, plane_index: u32) -> String {
    format!("{sample_key}.p{plane_index}.bin")
}

fn plane_state_key(sample_key: &str, plane_index: u32) -> String {
    format!("{sample_key}.p{plane_index}.state")
}

fn shared_state_key(shared_state_id: u16) -> String {
    format!("shared/s{shared_state_id:05}.bin")
}

// ── Explode ───────────────────────────────────────────────────────────────────

/// Parse a `.ptwm` container into a registry manifest, per-tensor manifests,
/// and loose binary members.
pub fn explode(blob: &[u8]) -> Result<Exploded, PtwmCoreError> {
    let reader = ContainerReader::open_partial(blob)?;
    let embedded_wasm = reader.header.flags & FLAG_PTWX != 0;
    if embedded_wasm {
        return Err(err(
            "embedded-WASM (.ptwx) containers are not supported by the transcoder",
        ));
    }

    let mut members: Vec<(String, Vec<u8>)> = Vec::new();

    // Shared state → registry + members.
    let mut shared_state: Vec<SharedStateEntry> = Vec::with_capacity(reader.prelude.len());
    for entry in &reader.prelude {
        let payload = shared_state_key(entry.shared_state_id);
        members.push((payload.clone(), entry.state_bytes.clone()));
        shared_state.push(SharedStateEntry {
            id: entry.shared_state_id,
            codec: entry.codec_id.as_u16(),
            state_format_version: entry.state_format_version,
            applies_to_mask: entry.applies_to_mask,
            hash_xxh64: format!("{:016x}", entry.state_xxhash64),
            name: entry.name.clone(),
            payload,
        });
    }

    // Chains → registry + op projection.
    let mut chains: Vec<ChainEntry> = Vec::with_capacity(reader.chain_registry.chains.len());
    let mut op_ids: std::collections::BTreeSet<u16> = std::collections::BTreeSet::new();
    for (id, chain) in &reader.chain_registry.chains {
        let (nodes, edges, terminals) = chain_to_json(chain);
        for n in &chain.nodes {
            op_ids.insert(n.op.as_u16());
        }
        chains.push(ChainEntry {
            id: *id,
            nodes,
            edges,
            terminals,
        });
    }

    // File order = ascending tensor offset.
    let mut index: Vec<&crate::index::IndexEntry> = reader.index.iter().collect();
    index.sort_by_key(|e| e.tensor_offset);

    let mut codec_ids: std::collections::BTreeSet<u16> = std::collections::BTreeSet::new();
    let mut tensors: Vec<ExplodedTensor> = Vec::with_capacity(index.len());

    for (file_idx, entry) in index.iter().enumerate() {
        // `usize` is ≥ 32 bits on every supported platform; widening these
        // u64 index offsets matches the container's own decode path.
        let start = entry.tensor_offset as usize;
        let end = start
            .checked_add(entry.tensor_len as usize)
            .filter(|e| *e <= blob.len())
            .ok_or_else(|| err("tensor record slice out of bounds"))?;
        let (record, _) = parse_tensor_record(&blob[start..end])?;
        for plane in &record.terminals {
            codec_ids.insert(plane.codec_id.as_u16());
        }

        let sample_key = format!("{file_idx:06}");
        let tm = tensor_to_manifest(&record, &sample_key, &mut members)?;
        let value = serde_json::to_value(&tm)
            .map_err(|e| err(format!("serialize tensor manifest: {e}")))?;
        check_no_inline_chain_ops(&record, &mut op_ids);
        tensors.push(ExplodedTensor {
            name: record.name.clone(),
            key: sample_key,
            manifest_json: jcs::to_string(&value)?,
        });
    }

    let ops: Vec<OpEntry> = op_ids
        .into_iter()
        .map(|id| {
            Ok(OpEntry {
                id,
                op: op_name(OpId::from_u16(id)?),
            })
        })
        .collect::<Result<_, PtwmCoreError>>()?;
    let codecs: Vec<CodecEntry> = codec_ids
        .into_iter()
        .map(|id| CodecEntry {
            id,
            codec: CodecId::from_u16(id)
                .map(codec_name)
                .unwrap_or_else(|| format!("codec_{id:#06x}")),
            module: None,
        })
        .collect();

    let registry = RegistryManifest {
        ptwm_format: PTWM_MANIFEST_FORMAT.to_string(),
        manifest_version: MANIFEST_VERSION,
        container_flags: ContainerFlags { embedded_wasm },
        registry_hash_blake3: hex::encode(reader.header.extension_table_hash),
        registry: Registry {
            ops,
            codecs,
            shared_state,
            chains,
        },
    };
    let registry_value =
        serde_json::to_value(&registry).map_err(|e| err(format!("serialize registry: {e}")))?;

    Ok(Exploded {
        registry_json: jcs::to_string(&registry_value)?,
        tensors,
        members,
    })
}

/// Fold any inline-chain ops into the op projection so `registry.ops` is a
/// superset of every op referenced anywhere in the archive.
fn check_no_inline_chain_ops(record: &TensorRecord, op_ids: &mut std::collections::BTreeSet<u16>) {
    if let Some(chain) = &record.inline_chain {
        for n in &chain.nodes {
            op_ids.insert(n.op.as_u16());
        }
    }
}

fn tensor_to_manifest(
    record: &TensorRecord,
    sample_key: &str,
    members: &mut Vec<(String, Vec<u8>)>,
) -> Result<TensorManifest, PtwmCoreError> {
    // Metadata projection.
    let (dtype, shape, tensor_metadata_b64) = match &record.tensor_metadata {
        None => (None, None, None),
        Some(bytes) => match decode_shape_metadata(bytes) {
            Ok((shape, dtype)) => {
                // Confirm the standard encoding round-trips byte-exactly;
                // otherwise keep the verbatim bytes as a fallback.
                if encode_shape_metadata(&shape, &dtype) == *bytes {
                    (Some(dtype), Some(shape), None)
                } else {
                    (Some(dtype), Some(shape), Some(base64::encode(bytes)))
                }
            }
            Err(_) => (None, None, Some(base64::encode(bytes))),
        },
    };

    // Chain spec.
    let chain = if record.chain_ref & CHAIN_REF_INLINE_BIT != 0 {
        let inline = record
            .inline_chain
            .as_ref()
            .ok_or_else(|| err("inline chain bit set but inline_chain missing"))?;
        let (nodes, edges, terminals) = chain_to_json(inline);
        ChainSpec {
            chain_ref: None,
            nodes: Some(nodes),
            edges: Some(edges),
            terminals: Some(terminals),
        }
    } else {
        ChainSpec {
            chain_ref: Some(record.chain_ref),
            nodes: None,
            edges: None,
            terminals: None,
        }
    };

    let dependencies: Vec<DependencyJson> = record
        .dependencies
        .iter()
        .map(|dep| DependencyJson {
            ref_kind: match dep.ref_kind {
                RefKind::LocalTensor => "local_tensor",
                RefKind::ExternalPtwm => "external_ptwm",
                RefKind::ExternalSafetensors => "external_safetensors",
            }
            .to_string(),
            ref_b64: base64::encode(&dep.ref_bytes),
            expected_hash_blake3: dep.expected_hash.map(hex::encode),
        })
        .collect();

    let mut planes: Vec<PlaneJson> = Vec::with_capacity(record.terminals.len());
    for (plane_index, plane) in record.terminals.iter().enumerate() {
        let plane_index = plane_index as u32;
        let payload_key = plane_payload_key(sample_key, plane_index);
        members.push((payload_key.clone(), plane.payload_bytes.clone()));

        let bounded = codec_state_is_bounded(plane.codec_id);
        let (state_b64, state_payload) = if plane.inline_state_bytes.is_empty() {
            (None, None)
        } else if bounded {
            (Some(base64::encode(&plane.inline_state_bytes)), None)
        } else {
            let key = plane_state_key(sample_key, plane_index);
            members.push((key.clone(), plane.inline_state_bytes.clone()));
            (None, Some(key))
        };

        let external_state = plane.external_state.as_ref().map(|ext| ExternalStateJson {
            ref_kind: ext.ref_kind,
            ref_b64: base64::encode(&ext.ref_bytes),
            state_format_version: ext.state_format_version,
            expected_hash_blake3: ext.expected_hash.map(hex::encode),
        });

        let chunk_table = plane.chunk_table.as_ref().map(|chunks| {
            chunks
                .iter()
                .map(|c| ChunkJson {
                    offset_in_payload: c.offset_in_payload,
                    decoded_size: c.decoded_size,
                })
                .collect()
        });

        planes.push(PlaneJson {
            index: plane_index,
            role: plane_role_name(plane.role),
            role_code: plane.role as u8,
            codec: plane.codec_id.as_u16(),
            codec_table_idx: plane.codec_table_idx,
            state_source: state_source_label(plane.state_source, bounded).to_string(),
            state_source_code: plane.state_source as u8,
            state_version: plane.state_version,
            state_info: plane.state_info,
            state_b64,
            state_payload,
            external_state,
            crc32: plane.crc32,
            chunk_table,
            layout_kind: plane.layout.kind(),
            layout_row_len: plane.layout.row_len(),
            comp_len: plane.payload_len,
            payload: payload_key,
        });
    }

    Ok(TensorManifest {
        name: record.name.clone(),
        dtype,
        dtype_code: record.dtype_code,
        shape,
        tensor_metadata_b64,
        input_format: record.input_format,
        orig_size: record.orig_size,
        payload_hash_xxh64: record.payload_hash.map(|h| format!("{h:016x}")),
        chain,
        dependencies,
        planes,
    })
}

// ── Implode ───────────────────────────────────────────────────────────────────

/// Reconstruct a `.ptwm` container from a registry manifest, per-tensor
/// manifests (in file order), and loose binary members.
pub fn implode(
    registry_json: &str,
    tensor_jsons: &[String],
    mut members: HashMap<String, &[u8]>,
) -> Result<Vec<u8>, PtwmCoreError> {
    let registry: RegistryManifest = serde_json::from_str(registry_json)
        .map_err(|e| err(format!("parse registry manifest: {e}")))?;
    if registry.ptwm_format != PTWM_MANIFEST_FORMAT {
        return Err(err(format!(
            "unexpected ptwm_format {:?}",
            registry.ptwm_format
        )));
    }
    if registry.container_flags.embedded_wasm {
        return Err(err(
            "embedded-WASM (.ptwx) containers are not supported by the transcoder",
        ));
    }

    // Rebuild the shared-state prelude (order preserved).
    let mut prelude_entries: Vec<PreludeEntry> =
        Vec::with_capacity(registry.registry.shared_state.len());
    for s in &registry.registry.shared_state {
        // Materialize the (potentially large) state bytes here — this copy now
        // runs with the GIL released (the caller borrows the Python buffers).
        let state_bytes = members
            .remove(&s.payload)
            .ok_or_else(|| err(format!("missing shared-state member {:?}", s.payload)))?
            .to_vec();
        prelude_entries.push(PreludeEntry {
            shared_state_id: s.id,
            codec_id: CodecId::from_u16(s.codec)
                .ok_or_else(|| err(format!("unknown shared-state codec id {:#06x}", s.codec)))?,
            state_format_version: s.state_format_version,
            applies_to_mask: s.applies_to_mask,
            state_xxhash64: u64::from_str_radix(&s.hash_xxh64, 16)
                .map_err(|_| err("shared-state hash_xxh64 not valid hex"))?,
            name: s.name.clone(),
            state_bytes,
        });
    }

    // Rebuild the chain registry (order preserved).
    let mut chain_registry = ChainRegistry::new();
    for c in &registry.registry.chains {
        chain_registry
            .chains
            .push((c.id, chain_from_json(&c.nodes, &c.edges, &c.terminals)?));
    }

    // Rebuild each tensor record.
    let mut records: Vec<TensorRecord> = Vec::with_capacity(tensor_jsons.len());
    for json in tensor_jsons {
        let tm: TensorManifest =
            serde_json::from_str(json).map_err(|e| err(format!("parse tensor manifest: {e}")))?;
        records.push(record_from_manifest(&tm, &mut members)?);
    }

    // Re-drive the deterministic container writer.
    let mut cursor = Cursor::new(Vec::<u8>::new());
    {
        let mut writer = ContainerWriter::new(&mut cursor)?;
        writer.set_prelude(&prelude_entries)?;
        writer.set_chain_registry(chain_registry)?;
        for record in &records {
            writer.append_tensor(record)?;
        }
        writer.finalize()?;
    }
    Ok(cursor.into_inner())
}

fn record_from_manifest(
    tm: &TensorManifest,
    members: &mut HashMap<String, &[u8]>,
) -> Result<TensorRecord, PtwmCoreError> {
    // Chain ref / inline chain.
    let (chain_ref, inline_chain) = match tm.chain.chain_ref {
        Some(r) => {
            if r & CHAIN_REF_INLINE_BIT != 0 {
                return Err(err("chain ref must have the inline bit clear"));
            }
            (r, None)
        }
        None => {
            let nodes = tm.chain.nodes.as_deref().unwrap_or(&[]);
            let edges = tm.chain.edges.as_deref().unwrap_or(&[]);
            let terminals = tm.chain.terminals.as_deref().unwrap_or(&[]);
            (
                CHAIN_REF_INLINE_BIT,
                Some(chain_from_json(nodes, edges, terminals)?),
            )
        }
    };

    // Metadata.
    let tensor_metadata = match &tm.tensor_metadata_b64 {
        Some(s) => Some(base64::decode(s).ok_or_else(|| err("tensor_metadata_b64 not base64"))?),
        None => match (&tm.shape, &tm.dtype) {
            (Some(shape), Some(dtype)) => Some(encode_shape_metadata(shape, dtype)),
            _ => None,
        },
    };

    let payload_hash = match &tm.payload_hash_xxh64 {
        Some(s) => {
            Some(u64::from_str_radix(s, 16).map_err(|_| err("payload_hash_xxh64 not valid hex"))?)
        }
        None => None,
    };

    let dependencies: Vec<Dependency> = tm
        .dependencies
        .iter()
        .map(|d| {
            let ref_kind = match d.ref_kind.as_str() {
                "local_tensor" => RefKind::LocalTensor,
                "external_ptwm" => RefKind::ExternalPtwm,
                "external_safetensors" => RefKind::ExternalSafetensors,
                other => return Err(err(format!("unknown dependency ref_kind {other:?}"))),
            };
            let expected_hash = match &d.expected_hash_blake3 {
                Some(h) => Some(parse_hash32(h)?),
                None => None,
            };
            Ok(Dependency {
                ref_kind,
                expected_hash,
                ref_bytes: base64::decode(&d.ref_b64)
                    .ok_or_else(|| err("dependency ref_b64 not base64"))?,
            })
        })
        .collect::<Result<_, PtwmCoreError>>()?;

    let mut terminals: Vec<PlaneRecord> = Vec::with_capacity(tm.planes.len());
    for plane in &tm.planes {
        terminals.push(plane_from_manifest(plane, members)?);
    }

    Ok(TensorRecord {
        chain_ref,
        dtype_code: tm.dtype_code,
        input_format: tm.input_format,
        flags: 0, // recomputed by write_tensor_record
        orig_size: tm.orig_size,
        name: tm.name.clone(),
        payload_hash,
        inline_chain,
        dependencies,
        tensor_metadata,
        terminals,
    })
}

fn plane_from_manifest(
    plane: &PlaneJson,
    members: &mut HashMap<String, &[u8]>,
) -> Result<PlaneRecord, PtwmCoreError> {
    // Materialize the compressed payload (GIL-released copy, see `implode`).
    let payload_bytes = members
        .remove(&plane.payload)
        .ok_or_else(|| err(format!("missing plane payload member {:?}", plane.payload)))?
        .to_vec();
    if payload_bytes.len() as u64 != plane.comp_len {
        return Err(err(format!(
            "plane payload length {} does not match comp_len {}",
            payload_bytes.len(),
            plane.comp_len
        )));
    }

    let inline_state_bytes = match (&plane.state_b64, &plane.state_payload) {
        (Some(_), Some(_)) => {
            return Err(err("plane has both inline and externalized state"));
        }
        (Some(b64), None) => {
            base64::decode(b64).ok_or_else(|| err("plane state_b64 not base64"))?
        }
        (None, Some(key)) => members
            .remove(key)
            .ok_or_else(|| err(format!("missing plane state member {key:?}")))?
            .to_vec(),
        (None, None) => Vec::new(),
    };

    let external_state = match &plane.external_state {
        Some(ext) => Some(ExternalStateRef {
            ref_kind: ext.ref_kind,
            expected_hash: match &ext.expected_hash_blake3 {
                Some(h) => Some(parse_hash32(h)?),
                None => None,
            },
            ref_bytes: base64::decode(&ext.ref_b64)
                .ok_or_else(|| err("external_state ref_b64 not base64"))?,
            state_format_version: ext.state_format_version,
        }),
        None => None,
    };

    let chunk_table = plane.chunk_table.as_ref().map(|chunks| {
        chunks
            .iter()
            .map(|c| ChunkEntry {
                offset_in_payload: c.offset_in_payload,
                decoded_size: c.decoded_size,
            })
            .collect()
    });

    let layout = match plane.layout_kind {
        0 => PlaneLayout::Flat,
        1 => {
            let row_len = plane
                .layout_row_len
                .ok_or_else(|| err("rows layout missing layout_row_len"))?;
            PlaneLayout::rows(row_len).ok_or_else(|| err("rows layout row_len must be > 0"))?
        }
        other => return Err(err(format!("unknown layout kind {other}"))),
    };

    Ok(PlaneRecord {
        role: PlaneRole::from_u8(plane.role_code)
            .ok_or_else(|| err(format!("unknown plane role code {}", plane.role_code)))?,
        codec_id: CodecId::from_u16(plane.codec)
            .ok_or_else(|| err(format!("unknown codec id {:#06x}", plane.codec)))?,
        codec_table_idx: plane.codec_table_idx,
        state_source: StateSource::from_u8(plane.state_source_code)
            .ok_or_else(|| err(format!("unknown state_source {}", plane.state_source_code)))?,
        state_version: plane.state_version,
        state_info: plane.state_info,
        payload_len: payload_bytes.len() as u64,
        crc32: plane.crc32,
        chunk_table,
        external_state,
        inline_state_bytes,
        payload_bytes,
        layout,
    })
}

fn parse_hash32(h: &str) -> Result<[u8; 32], PtwmCoreError> {
    let bytes = hex::decode(h).map_err(|_| err("expected_hash not valid hex"))?;
    let arr: [u8; 32] = bytes
        .try_into()
        .map_err(|_| err("expected_hash must be 32 bytes"))?;
    Ok(arr)
}

#[cfg(test)]
mod tests;
