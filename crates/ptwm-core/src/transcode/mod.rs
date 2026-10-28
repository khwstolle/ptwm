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
//! Both container layouts are supported. A legacy (`\x89PTWM`) container gives
//! a [`RegistryManifest`] (`"ptwm-manifest"`, version 1); `implode` rebuilds
//! its Extension Table from the chains and codecs and checks the recorded
//! hash. A compact (`\x89PTWC`) container gives a [`CompactRegistryManifest`]
//! (`"ptwm-manifest-c"`, version 2) that lists the table entries, the
//! registry chains (a bound Source has `params_b64` `null`) and the tensors in
//! body order; `implode` writes them back as listed. Readers that predate the
//! compact manifest reject it on its format and version. Tensor manifests
//! have the same form for both layouts. `implode` with a subset of the tensor
//! manifests writes a container of those tensors with the archive-wide
//! tables, so each tensor decodes as in the full container.
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
use crate::container::{
    BuiltExtensionTable, ChainRegistry, CodecSet, ContainerLayout, ContainerReader,
    ContainerWriter, build_extension_table, encode_registry_for_table,
};
use crate::error::PtwmCoreError;
use crate::extension::compact_table::{
    CompactEntry, TAG_CODEC, TAG_FULL, TAG_NAMED, TAG_OP, classify, read_compact_table,
};
use crate::extension::{CURRENT_BUILTIN_VERSION, CanonicalId, ExtensionTable};
use crate::header::FLAG_PTWX;
use crate::layout::PlaneLayout;
use crate::metadata::{decode_shape_metadata, encode_shape_metadata};
use crate::plane_record::{ChunkEntry, ExternalStateRef, PlaneRecord};
use crate::prelude::{PreludeEntry, write_prelude};
use crate::tensor_record::{
    CHAIN_REF_INLINE_BIT, Dependency, RefKind, TensorRecord,
    parse_tensor_record_with_builtin_version, write_tensor_record_with_builtin_version,
};
use crate::transforms::op::OpId;
use crate::types::PlaneRole;
use crate::types::role::Role;
use crate::varint::write_varint;

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

// ── Extension Table reconstruction ────────────────────────────────────────────

/// Which table builder reproduced a container's Extension Table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TableLayout {
    /// Chain ops plus the codecs the planes use (current writers).
    Trimmed,
    /// Chain ops plus every built-in codec (v1.1 and earlier).
    AllBuiltins,
}

/// Current canonical ids of the manifest's codec list. The list is
/// archive-wide, so a one-tensor implode still rebuilds the archive's table.
fn used_codec_ids(codecs: &[CodecEntry]) -> Vec<CanonicalId> {
    codecs
        .iter()
        .filter_map(|c| CodecId::from_u16(c.id))
        .map(crate::compressor::codec_canonical_id)
        .collect()
}

/// Header-style hash of serialized table bytes.
fn table_hash(bytes: &[u8]) -> [u8; 32] {
    if bytes.is_empty() {
        [0u8; 32]
    } else {
        ExtensionTable::hash(bytes)
    }
}

/// Rebuild the Extension Table from the chain registry, trying the trimmed
/// layout and then the all-built-ins layout, both with built-in entries of
/// `version`. Returns the first layout whose table bytes hash to
/// `expected_hash`, or `None`.
fn rebuild_extension_table(
    registry: &ChainRegistry,
    used_codecs: &[CanonicalId],
    version: &str,
    expected_hash: &[u8; 32],
) -> Option<(TableLayout, BuiltExtensionTable)> {
    [
        (TableLayout::Trimmed, CodecSet::Only(used_codecs)),
        (TableLayout::AllBuiltins, CodecSet::AllBuiltins),
    ]
    .into_iter()
    .find_map(|(layout, codecs)| {
        let built = build_extension_table(registry, codecs, version).ok()?;
        let bytes = built.table.to_bytes().ok()?;
        (table_hash(&bytes) == *expected_hash).then_some((layout, built))
    })
}

/// The container's on-disk Extension Table bytes.
fn extension_table_bytes<'a>(
    reader: &ContainerReader<'_>,
    blob: &'a [u8],
) -> Result<&'a [u8], PtwmCoreError> {
    let start = usize::try_from(reader.header.extension_table_offset)
        .map_err(|_| err("extension_table_offset overflows usize"))?;
    let len = usize::try_from(reader.header.extension_table_length)
        .map_err(|_| err("extension_table_length overflows usize"))?;
    start
        .checked_add(len)
        .and_then(|end| blob.get(start..end))
        .ok_or_else(|| err("extension table region exceeds buffer"))
}

// ── Explode ───────────────────────────────────────────────────────────────────

/// Parse a `.ptwm` container (either layout) into a registry manifest,
/// per-tensor manifests in body order, and loose binary members.
pub fn explode(blob: &[u8]) -> Result<Exploded, PtwmCoreError> {
    let reader = ContainerReader::open_partial(blob)?;
    if reader.layout == ContainerLayout::Compact {
        return explode_compact(&reader);
    }
    let embedded_wasm = reader.header.flags & FLAG_PTWX != 0;
    if embedded_wasm {
        return Err(err(
            "embedded-WASM (.ptwx) containers are not supported by the transcoder",
        ));
    }

    let mut members: Vec<(String, Vec<u8>)> = Vec::new();

    // Shared state → registry + members.
    let shared_state = shared_state_to_manifest(&reader.prelude, &mut members);

    // Chains → registry + op projection.
    let chains = chains_to_manifest(&reader.chain_registry);
    let mut op_ids: std::collections::BTreeSet<u16> = std::collections::BTreeSet::new();
    for (_, chain) in &reader.chain_registry.chains {
        for n in &chain.nodes {
            op_ids.insert(n.op.as_u16());
        }
    }

    // File order = ascending tensor offset.
    let mut index: Vec<&crate::index::IndexEntry> = reader.index.iter().collect();
    index.sort_by_key(|e| e.tensor_offset);

    let mut codec_ids: std::collections::BTreeSet<u16> = std::collections::BTreeSet::new();
    let mut tensors: Vec<ExplodedTensor> = Vec::with_capacity(index.len());
    let mut inline_builtin_version: Option<String> = None;

    for (file_idx, entry) in index.iter().enumerate() {
        // `usize` is ≥ 32 bits on every supported platform; widening these
        // u64 index offsets matches the container's own decode path.
        let start = entry.tensor_offset as usize;
        let end = start
            .checked_add(entry.tensor_len as usize)
            .filter(|e| *e <= blob.len())
            .ok_or_else(|| err("tensor record slice out of bounds"))?;
        let (record, consumed, inline_version) =
            parse_tensor_record_with_builtin_version(&blob[start..end])?;
        // `implode` writes every inline-chain table with one built-in
        // version, so they must agree.
        if let Some(v) = inline_version {
            match &inline_builtin_version {
                None => inline_builtin_version = Some(v),
                Some(b) if *b != v => {
                    return Err(err(format!(
                        "tensor {:?}: inline chain built-in version {v} differs from \
                         {b} in an earlier inline chain; the container cannot be \
                         transcoded byte-exactly",
                        record.name
                    )));
                }
                Some(_) => {}
            }
        }
        // `implode` re-encodes inline-chain tables with a single built-in
        // version. A table that mixes versions (or is otherwise not what
        // that writer emits) would come back with different bytes, so
        // refuse it here instead of producing a silent mismatch.
        if record.inline_chain.is_some() {
            let version = inline_builtin_version
                .as_deref()
                .unwrap_or(CURRENT_BUILTIN_VERSION);
            let mut reencoded = Vec::new();
            write_tensor_record_with_builtin_version(&record, &mut reencoded, version)?;
            if blob.get(start..start + consumed) != Some(reencoded.as_slice()) {
                return Err(err(format!(
                    "tensor {:?}: inline chain record differs from its re-encoding \
                     (for example, its Extension Table mixes built-in versions); \
                     the container cannot be transcoded byte-exactly",
                    record.name
                )));
            }
        }
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

    // `implode` writes built-in entries with one version. Inline-chain
    // tables are always re-encoded, so their version wins; the file-level
    // table is rebuilt with it when that reproduces the bytes.
    let builtin_version = inline_builtin_version.or_else(|| reader.builtin_version.clone());
    let version = builtin_version
        .as_deref()
        .unwrap_or(CURRENT_BUILTIN_VERSION);

    // Check that implode can rebuild the Extension Table and the
    // shared-prelude section byte-exactly. When no table builder reproduces
    // the table, carry its bytes in the manifest.
    let used_codecs = used_codec_ids(&codecs);
    let et_bytes = extension_table_bytes(&reader, blob)?;
    let (built, extension_table_b64) = match rebuild_extension_table(
        &reader.chain_registry,
        &used_codecs,
        version,
        &reader.header.extension_table_hash,
    ) {
        Some((_, built)) => (built, None),
        None => (
            encode_registry_for_table(et_bytes, &reader.chain_registry)?,
            Some(base64::encode(et_bytes)),
        ),
    };
    let mut section = Vec::new();
    write_prelude(&reader.prelude, &mut section)?;
    section.extend_from_slice(&built.registry_bytes);
    let prelude_off = usize::try_from(reader.header.shared_prelude_offset)
        .map_err(|_| err("shared_prelude_offset overflows usize"))?;
    let prelude_end = usize::try_from(reader.tensors_offset)
        .map_err(|_| err("tensors_offset overflows usize"))?;
    if blob.get(prelude_off..prelude_end) != Some(section.as_slice()) {
        return Err(err(
            "the shared prelude or chain registry differs from its re-encoding; \
             the container cannot be transcoded byte-exactly",
        ));
    }

    let registry = RegistryManifest {
        ptwm_format: PTWM_MANIFEST_FORMAT.to_string(),
        manifest_version: MANIFEST_VERSION,
        container_flags: ContainerFlags {
            embedded_wasm,
            builtin_version: builtin_version.filter(|v| v != CURRENT_BUILTIN_VERSION),
        },
        registry_hash_blake3: hex::encode(reader.header.extension_table_hash),
        registry: Registry {
            ops,
            codecs,
            shared_state,
            chains,
            extension_table_b64,
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

/// Project the shared-state prelude to manifest entries, moving each state
/// into a member.
fn shared_state_to_manifest(
    prelude: &[PreludeEntry],
    members: &mut Vec<(String, Vec<u8>)>,
) -> Vec<SharedStateEntry> {
    prelude
        .iter()
        .map(|entry| {
            let payload = shared_state_key(entry.shared_state_id);
            members.push((payload.clone(), entry.state_bytes.clone()));
            SharedStateEntry {
                id: entry.shared_state_id,
                codec: entry.codec_id.as_u16(),
                state_format_version: entry.state_format_version,
                applies_to_mask: entry.applies_to_mask,
                hash_xxh64: format!("{:016x}", entry.state_xxhash64),
                name: entry.name.clone(),
                payload,
            }
        })
        .collect()
}

/// Project the chain registry to manifest entries, in registry order.
fn chains_to_manifest(registry: &ChainRegistry) -> Vec<ChainEntry> {
    registry
        .chains
        .iter()
        .map(|(id, chain)| {
            let (nodes, edges, terminals) = chain_to_json(chain);
            ChainEntry {
                id: *id,
                nodes,
                edges,
                terminals,
            }
        })
        .collect()
}

/// Project one Extension Table entry to its compact manifest form.
fn extension_entry_to_json(
    entry: &crate::extension::ExtensionTableEntry,
) -> Result<ExtensionEntryJson, PtwmCoreError> {
    Ok(match classify(entry)? {
        CompactEntry::Op(op) => ExtensionEntryJson::Op(op.as_u16()),
        CompactEntry::Codec(c) => ExtensionEntryJson::Codec(c.as_u16()),
        CompactEntry::Named(name) => ExtensionEntryJson::Named(name.to_owned()),
        CompactEntry::Full(e) => ExtensionEntryJson::FullB64(base64::encode(&e.to_bytes()?)),
    })
}

/// Rebuild the Extension Table of a compact manifest. The entries are
/// encoded as a compact table and parsed back, so the manifest gets every
/// check the compact reader applies (unknown ids and names, duplicates,
/// built-in ids in full entries, canonical entry bytes).
fn extension_table_from_json(
    entries: &[ExtensionEntryJson],
) -> Result<ExtensionTable, PtwmCoreError> {
    let mut bytes = Vec::new();
    write_varint(&mut bytes, entries.len() as u64);
    for entry in entries {
        match entry {
            ExtensionEntryJson::Op(id) => {
                bytes.push(TAG_OP);
                write_varint(&mut bytes, u64::from(*id));
            }
            ExtensionEntryJson::Codec(id) => {
                bytes.push(TAG_CODEC);
                write_varint(&mut bytes, u64::from(*id));
            }
            ExtensionEntryJson::Named(name) => {
                bytes.push(TAG_NAMED);
                write_varint(&mut bytes, name.len() as u64);
                bytes.extend_from_slice(name.as_bytes());
            }
            ExtensionEntryJson::FullB64(b64) => {
                let entry = base64::decode(b64)
                    .ok_or_else(|| err("extension_table full_b64 not valid base64"))?;
                bytes.push(TAG_FULL);
                write_varint(&mut bytes, entry.len() as u64);
                bytes.extend_from_slice(&entry);
            }
        }
    }
    read_compact_table(&bytes).map_err(|e| err(format!("registry.extension_table: {e}")))
}

/// [`explode`] for the compact layout (`\x89PTWC`).
///
/// The compact reader accepts only canonical encodings, so writing the
/// parsed tables, prelude, registry and records again (as [`implode`] does)
/// reproduces the file. Tensors come in body order.
fn explode_compact(reader: &ContainerReader<'_>) -> Result<Exploded, PtwmCoreError> {
    let mut members: Vec<(String, Vec<u8>)> = Vec::new();
    let shared_state = shared_state_to_manifest(&reader.prelude, &mut members);
    let chains = chains_to_manifest(&reader.chain_registry);
    let extension_table = reader
        .extension_table
        .entries
        .iter()
        .map(extension_entry_to_json)
        .collect::<Result<Vec<_>, _>>()?;

    let records = reader.tensor_records()?;
    let mut tensors: Vec<ExplodedTensor> = Vec::with_capacity(records.len());
    for (file_idx, record) in records.iter().enumerate() {
        let sample_key = format!("{file_idx:06}");
        let tm = tensor_to_manifest(record, &sample_key, &mut members)?;
        let value = serde_json::to_value(&tm)
            .map_err(|e| err(format!("serialize tensor manifest: {e}")))?;
        tensors.push(ExplodedTensor {
            name: record.name.clone(),
            key: sample_key,
            manifest_json: jcs::to_string(&value)?,
        });
    }

    let registry = CompactRegistryManifest {
        ptwm_format: PTWM_MANIFEST_FORMAT_COMPACT.to_string(),
        manifest_version: MANIFEST_VERSION_COMPACT,
        container_flags: CompactContainerFlags {
            embedded_wasm: false,
            layout: LAYOUT_COMPACT.to_string(),
        },
        registry: CompactRegistry {
            extension_table,
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
/// manifests (in file order), and loose binary members. The registry
/// manifest's `ptwm_format` and `manifest_version` select the layout; any
/// other pair is rejected.
pub fn implode(
    registry_json: &str,
    tensor_jsons: &[String],
    mut members: HashMap<String, &[u8]>,
) -> Result<Vec<u8>, PtwmCoreError> {
    // Check the format and version before the strict parse, so a manifest
    // of another format or version gets a clear error.
    let value: serde_json::Value = serde_json::from_str(registry_json)
        .map_err(|e| err(format!("parse registry manifest: {e}")))?;
    let format = value.get("ptwm_format").and_then(|v| v.as_str());
    let version = value.get("manifest_version").and_then(|v| v.as_u64());
    match (format, version) {
        (Some(PTWM_MANIFEST_FORMAT), Some(v)) if v == u64::from(MANIFEST_VERSION) => {}
        (Some(PTWM_MANIFEST_FORMAT_COMPACT), Some(v))
            if v == u64::from(MANIFEST_VERSION_COMPACT) =>
        {
            return implode_compact(value, tensor_jsons, members);
        }
        _ => {
            return Err(err(format!(
                "unsupported manifest: ptwm_format {format:?}, manifest_version {version:?} \
                 (this reader handles {PTWM_MANIFEST_FORMAT:?} version {MANIFEST_VERSION} \
                 and {PTWM_MANIFEST_FORMAT_COMPACT:?} version {MANIFEST_VERSION_COMPACT})"
            )));
        }
    }
    let registry: RegistryManifest =
        serde_json::from_value(value).map_err(|e| err(format!("parse registry manifest: {e}")))?;
    if registry.container_flags.embedded_wasm {
        return Err(err(
            "embedded-WASM (.ptwx) containers are not supported by the transcoder",
        ));
    }
    let expected_hash = parse_hash32(&registry.registry_hash_blake3)
        .map_err(|_| err("registry_hash_blake3 must be 32 bytes of hex"))?;
    let builtin_version = registry
        .container_flags
        .builtin_version
        .as_deref()
        .unwrap_or(CURRENT_BUILTIN_VERSION);

    let prelude_entries = prelude_from_manifest(&registry.registry.shared_state, &mut members)?;
    let chain_registry = chains_from_manifest(&registry.registry.chains)?;
    let records = records_from_manifests(tensor_jsons, &mut members)?;

    // Re-drive the deterministic container writer.
    let mut cursor = Cursor::new(Vec::<u8>::new());
    {
        let mut writer = ContainerWriter::new(&mut cursor)?;
        writer.set_prelude(&prelude_entries)?;
        // Built-in entries (file-level and inline-chain tables) carry the
        // ids and labels of the release that wrote the container.
        writer.set_builtin_version(builtin_version)?;
        // Containers written by v1.1 and earlier intern every built-in codec;
        // later writers intern only the codecs the planes reference. Rebuild
        // whichever layout reproduces the recorded table hash, so the planes'
        // `codec_table_idx` values resolve to the same entries. A table that
        // neither layout reproduces travels as `extension_table_b64`.
        //
        // Take the codec set from the archive-wide registry, not from
        // `records`: a caller may implode a subset of the tensors (the
        // exploded store layouts rebuild one tensor at a time).
        if let Some(b64) = &registry.registry.extension_table_b64 {
            let bytes =
                base64::decode(b64).ok_or_else(|| err("extension_table_b64 not valid base64"))?;
            if table_hash(&bytes) != expected_hash {
                return Err(err(
                    "extension_table_b64 does not match registry_hash_blake3",
                ));
            }
            writer.set_raw_extension_table(&bytes, chain_registry)?;
        } else {
            let used_codecs = used_codec_ids(&registry.registry.codecs);
            match rebuild_extension_table(
                &chain_registry,
                &used_codecs,
                builtin_version,
                &expected_hash,
            ) {
                Some((TableLayout::Trimmed, _)) => {
                    writer.set_chain_registry_with_codecs(chain_registry, &used_codecs)?;
                }
                Some((TableLayout::AllBuiltins, _)) => writer.set_chain_registry(chain_registry)?,
                None => {
                    return Err(err(format!(
                        "no Extension Table layout reproduces registry_hash_blake3 {} \
                         (built-in version {builtin_version})",
                        registry.registry_hash_blake3
                    )));
                }
            }
        }
        for record in &records {
            writer.append_tensor(record)?;
        }
        writer.finalize()?;
    }
    Ok(cursor.into_inner())
}

/// [`implode`] for a compact-layout manifest: writes the Extension Table,
/// registry and tensor order of the manifest verbatim in the compact layout.
fn implode_compact(
    value: serde_json::Value,
    tensor_jsons: &[String],
    mut members: HashMap<String, &[u8]>,
) -> Result<Vec<u8>, PtwmCoreError> {
    let registry: CompactRegistryManifest =
        serde_json::from_value(value).map_err(|e| err(format!("parse registry manifest: {e}")))?;
    if registry.container_flags.embedded_wasm {
        return Err(err("embedded WASM needs the legacy layout"));
    }
    if registry.container_flags.layout != LAYOUT_COMPACT {
        return Err(err(format!(
            "container_flags.layout {:?} in a {PTWM_MANIFEST_FORMAT_COMPACT:?} manifest \
             (expected {LAYOUT_COMPACT:?})",
            registry.container_flags.layout
        )));
    }
    let table = extension_table_from_json(&registry.registry.extension_table)?;
    let prelude_entries = prelude_from_manifest(&registry.registry.shared_state, &mut members)?;
    let chain_registry = chains_from_manifest(&registry.registry.chains)?;
    let records = records_from_manifests(tensor_jsons, &mut members)?;

    let mut cursor = Cursor::new(Vec::<u8>::new());
    {
        let mut writer = ContainerWriter::with_layout(&mut cursor, ContainerLayout::Compact)?;
        writer.set_prelude(&prelude_entries)?;
        // The table and registry are written as listed. The table is
        // archive-wide, so a one-tensor implode keeps every plane's
        // `codec_table_idx` valid.
        writer.set_compact_tables(table, chain_registry)?;
        for record in &records {
            writer.append_tensor(record)?;
        }
        writer.finalize()?;
    }
    Ok(cursor.into_inner())
}

/// Rebuild the shared-state prelude (order preserved), taking each state
/// out of `members`.
fn prelude_from_manifest(
    shared_state: &[SharedStateEntry],
    members: &mut HashMap<String, &[u8]>,
) -> Result<Vec<PreludeEntry>, PtwmCoreError> {
    let mut prelude_entries: Vec<PreludeEntry> = Vec::with_capacity(shared_state.len());
    for s in shared_state {
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
    Ok(prelude_entries)
}

/// Rebuild the chain registry (order preserved).
fn chains_from_manifest(chains: &[ChainEntry]) -> Result<ChainRegistry, PtwmCoreError> {
    let mut chain_registry = ChainRegistry::new();
    for c in chains {
        chain_registry
            .chains
            .push((c.id, chain_from_json(&c.nodes, &c.edges, &c.terminals)?));
    }
    Ok(chain_registry)
}

/// Rebuild each tensor record, taking its members out of `members`.
fn records_from_manifests(
    tensor_jsons: &[String],
    members: &mut HashMap<String, &[u8]>,
) -> Result<Vec<TensorRecord>, PtwmCoreError> {
    let mut records: Vec<TensorRecord> = Vec::with_capacity(tensor_jsons.len());
    for json in tensor_jsons {
        let tm: TensorManifest =
            serde_json::from_str(json).map_err(|e| err(format!("parse tensor manifest: {e}")))?;
        records.push(record_from_manifest(&tm, members)?);
    }
    Ok(records)
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
