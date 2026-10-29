//! Round-trip conformance tests for the container ⇄ manifest transcoder.

use std::collections::HashMap;
use std::io::Cursor;

use super::*;
use crate::chain::graph::{Chain, ChainEdge, ChainNode, TerminalRef};
use crate::codec::{CodecId, StateSource};
use crate::container::{ChainRegistry, ContainerWriter};
use crate::layout::PlaneLayout;
use crate::metadata::encode_shape_metadata;
use crate::plane_record::{ChunkEntry, PlaneRecord};
use crate::prelude::PreludeEntry;
use crate::tensor_record::{Dependency, RefKind, TensorRecord};
use crate::transforms::op::OpId;
use crate::types::PlaneRole;
use crate::types::role::{Role, ValueFormat};

fn passthrough_chain() -> Chain {
    Chain {
        nodes: vec![
            ChainNode {
                op: OpId::Source,
                params: {
                    // Source params: dim_count=1, dims=[8], dtype=0x0006 (bf16).
                    let mut p = vec![1u8];
                    p.extend_from_slice(&8u32.to_le_bytes());
                    p.extend_from_slice(&0x0006u16.to_le_bytes());
                    p
                },
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
            role: Role::Value {
                format: ValueFormat::Fp4E2m1,
            },
        }],
    }
}

fn members_map(exploded: &Exploded) -> HashMap<String, &[u8]> {
    exploded
        .members
        .iter()
        .map(|(k, v)| (k.clone(), v.as_slice()))
        .collect()
}

fn tensor_jsons(exploded: &Exploded) -> Vec<String> {
    exploded
        .tensors
        .iter()
        .map(|t| t.manifest_json.clone())
        .collect()
}

/// Build a container with the given prelude / chain registry / records.
fn build_container(
    prelude: &[PreludeEntry],
    registry: ChainRegistry,
    records: &[TensorRecord],
) -> Vec<u8> {
    let mut cursor = Cursor::new(Vec::<u8>::new());
    {
        let mut writer = ContainerWriter::new(&mut cursor).unwrap();
        writer.set_prelude(prelude).unwrap();
        writer.set_chain_registry(registry).unwrap();
        for rec in records {
            writer.append_tensor(rec).unwrap();
        }
        writer.finalize().unwrap();
    }
    cursor.into_inner()
}

fn assert_round_trip(blob: &[u8]) {
    let exploded = explode(blob).expect("explode");
    // Manifests must be valid canonical JSON (parseable).
    let _: serde_json::Value = serde_json::from_str(&exploded.registry_json).unwrap();
    for t in &exploded.tensors {
        let _: serde_json::Value = serde_json::from_str(&t.manifest_json).unwrap();
    }
    let members = members_map(&exploded);
    let rebuilt =
        implode(&exploded.registry_json, &tensor_jsons(&exploded), members).expect("implode");
    assert_eq!(blob, rebuilt.as_slice(), "implode(explode(blob)) != blob");
}

fn flat_plane(role: PlaneRole, codec: CodecId, payload: Vec<u8>) -> PlaneRecord {
    PlaneRecord {
        role,
        codec_id: codec,
        codec_table_idx: 0,
        state_source: StateSource::None,
        state_version: 0,
        state_info: 0,
        payload_len: payload.len() as u64,
        crc32: None,
        chunk_table: None,
        external_state: None,
        inline_state_bytes: Vec::new(),
        payload_bytes: payload,
        layout: PlaneLayout::Flat,
    }
}

#[test]
fn round_trip_minimal_identity() {
    let registry = ChainRegistry {
        chains: vec![(0u16, passthrough_chain())],
    };
    let rec = TensorRecord {
        chain_ref: 0,
        dtype_code: 6,
        input_format: 2,
        flags: 0,
        orig_size: 8,
        name: "model.weight".to_string(),
        payload_hash: None,
        inline_chain: None,
        dependencies: Vec::new(),
        tensor_metadata: None,
        terminals: vec![flat_plane(
            PlaneRole::Value,
            CodecId::Identity,
            vec![1, 2, 3, 4, 5, 6, 7, 8],
        )],
    };
    let blob = build_container(&[], registry, &[rec]);
    assert_round_trip(&blob);
}

#[test]
fn round_trip_full_features() {
    let registry = ChainRegistry {
        chains: vec![(0u16, passthrough_chain())],
    };

    // Plane with inline (bounded) Huffman state + crc + rows layout.
    let mut p0 = flat_plane(PlaneRole::Exponent, CodecId::Huffman, vec![9, 8, 7, 6]);
    p0.state_source = StateSource::Inline;
    p0.state_version = 1;
    p0.inline_state_bytes = vec![0xAA, 0xBB, 0xCC];
    p0.crc32 = Some(0xDEADBEEF);
    p0.layout = PlaneLayout::rows(2).unwrap();
    p0.codec_table_idx = 3;

    // Plane with externalized (unbounded) state + chunk table.
    let mut p1 = flat_plane(PlaneRole::Mantissa, CodecId::Order1ScaleAC, vec![1; 8]);
    p1.state_source = StateSource::Inline;
    p1.state_version = 2;
    p1.inline_state_bytes = vec![5; 64];
    p1.chunk_table = Some(vec![
        ChunkEntry {
            offset_in_payload: 0,
            decoded_size: 4,
        },
        ChunkEntry {
            offset_in_payload: 4,
            decoded_size: 4,
        },
    ]);

    let mut expected_hash = [0u8; 32];
    expected_hash[0] = 0xAB;
    expected_hash[31] = 0xCD;

    let rec = TensorRecord {
        chain_ref: 0,
        dtype_code: 6,
        input_format: 2,
        flags: 0,
        orig_size: 16,
        name: "model.layers.0.mlp.gate_proj.weight".to_string(),
        payload_hash: Some(0x0123_4567_89AB_CDEF),
        inline_chain: None,
        dependencies: vec![Dependency {
            ref_kind: RefKind::ExternalSafetensors,
            expected_hash: Some(expected_hash),
            ref_bytes: b"base/model.safetensors".to_vec(),
        }],
        tensor_metadata: Some(encode_shape_metadata(&[4, 4], "bfloat16")),
        terminals: vec![p0, p1],
    };
    let blob = build_container(&[], registry, &[rec]);
    assert_round_trip(&blob);
}

#[test]
fn round_trip_shared_state_and_multi_tensor() {
    let prelude = vec![PreludeEntry {
        shared_state_id: 0,
        codec_id: CodecId::Order1ScaleAC,
        state_format_version: 1,
        applies_to_mask: 0,
        state_xxhash64: 0xFEED_FACE_DEAD_BEEF,
        name: "codebook0".to_string(),
        state_bytes: vec![7; 100],
    }];
    let registry = ChainRegistry {
        chains: vec![(0u16, passthrough_chain())],
    };

    let mut p_shared = flat_plane(PlaneRole::Scale, CodecId::Order1ScaleAC, vec![3; 12]);
    p_shared.state_source = StateSource::Shared;
    p_shared.state_info = 0; // shared_state_id

    let rec_a = TensorRecord {
        chain_ref: 0,
        dtype_code: 6,
        input_format: 2,
        flags: 0,
        orig_size: 8,
        name: "a.weight".to_string(),
        payload_hash: Some(1),
        inline_chain: None,
        dependencies: Vec::new(),
        tensor_metadata: Some(encode_shape_metadata(&[2, 2], "bf16")),
        terminals: vec![p_shared],
    };
    let rec_b = TensorRecord {
        chain_ref: 0,
        dtype_code: 6,
        input_format: 2,
        flags: 0,
        orig_size: 8,
        name: "b.weight".to_string(),
        payload_hash: Some(2),
        inline_chain: None,
        dependencies: Vec::new(),
        tensor_metadata: None,
        terminals: vec![flat_plane(PlaneRole::Value, CodecId::Rans, vec![4; 6])],
    };
    let blob = build_container(&prelude, registry, &[rec_a, rec_b]);
    assert_round_trip(&blob);

    // The registry manifest must reference the externalized shared state.
    let exploded = explode(&blob).unwrap();
    assert!(exploded.registry_json.contains("shared/s00000.bin"));
    assert!(members_map(&exploded).contains_key("shared/s00000.bin"));
}

#[test]
fn explode_rejects_garbage() {
    assert!(explode(b"not a container").is_err());
}

#[test]
fn implode_rejects_missing_member() {
    let registry = ChainRegistry {
        chains: vec![(0u16, passthrough_chain())],
    };
    let rec = TensorRecord {
        chain_ref: 0,
        dtype_code: 6,
        input_format: 2,
        flags: 0,
        orig_size: 8,
        name: "w".to_string(),
        payload_hash: None,
        inline_chain: None,
        dependencies: Vec::new(),
        tensor_metadata: None,
        terminals: vec![flat_plane(
            PlaneRole::Value,
            CodecId::Identity,
            vec![1, 2, 3, 4],
        )],
    };
    let blob = build_container(&[], registry, &[rec]);
    let exploded = explode(&blob).unwrap();
    let empty: HashMap<String, &[u8]> = HashMap::new();
    assert!(implode(&exploded.registry_json, &tensor_jsons(&exploded), empty).is_err());
}

/// Containers whose Extension Table holds only the referenced codecs must
/// round-trip byte-exactly, and imploding one tensor must keep its codec
/// index pointing at the same codec.
#[test]
fn round_trip_trimmed_codec_table() {
    use crate::compressor::codec_canonical_id;

    let registry = ChainRegistry {
        chains: vec![(0u16, passthrough_chain())],
    };
    let rec = |name: &str, codec: CodecId| TensorRecord {
        chain_ref: 0,
        dtype_code: 6,
        input_format: 2,
        flags: 0,
        orig_size: 8,
        name: name.to_string(),
        payload_hash: None,
        inline_chain: None,
        dependencies: Vec::new(),
        tensor_metadata: None,
        terminals: vec![flat_plane(PlaneRole::Value, codec, vec![5; 8])],
    };
    let codecs = [
        codec_canonical_id(CodecId::Zstd),
        codec_canonical_id(CodecId::Rans),
    ];

    let mut cursor = Cursor::new(Vec::<u8>::new());
    {
        let mut writer = ContainerWriter::new(&mut cursor).unwrap();
        writer
            .set_chain_registry_with_codecs(registry, &codecs)
            .unwrap();
        let mut a = rec("a.weight", CodecId::Zstd);
        a.terminals[0].codec_table_idx = writer.codec_table_idx_for(&codecs[0]).unwrap();
        let mut b = rec("b.weight", CodecId::Rans);
        b.terminals[0].codec_table_idx = writer.codec_table_idx_for(&codecs[1]).unwrap();
        writer.append_tensor(&a).unwrap();
        writer.append_tensor(&b).unwrap();
        writer.finalize().unwrap();
    }
    let blob = cursor.into_inner();

    let reader = ContainerReader::open(&blob).unwrap();
    // Two chain ops plus two codecs; no unreferenced built-in codecs.
    assert_eq!(reader.extension_table.entries.len(), 4);
    assert_round_trip(&blob);

    // Implode only "b.weight" with the archive-wide registry.
    let exploded = explode(&blob).unwrap();
    let b = exploded
        .tensors
        .iter()
        .find(|t| t.name == "b.weight")
        .unwrap();
    let rebuilt = implode(
        &exploded.registry_json,
        std::slice::from_ref(&b.manifest_json),
        members_map(&exploded),
    )
    .unwrap();
    let reader = ContainerReader::open(&rebuilt).unwrap();
    let rec_bytes = reader.get_tensor_record_bytes("b.weight").unwrap();
    let (parsed, _) = crate::tensor_record::parse_tensor_record(rec_bytes).unwrap();
    let idx = parsed.terminals[0].codec_table_idx as usize;
    assert_eq!(
        reader.extension_table.entries[idx].canonical_id, codecs[1],
        "imploded subset must resolve b.weight's plane to rANS"
    );
}

fn minimal_container() -> Vec<u8> {
    let registry = ChainRegistry {
        chains: vec![(0u16, passthrough_chain())],
    };
    let rec = TensorRecord {
        chain_ref: 0,
        dtype_code: 6,
        input_format: 2,
        flags: 0,
        orig_size: 8,
        name: "w".to_string(),
        payload_hash: None,
        inline_chain: None,
        dependencies: Vec::new(),
        tensor_metadata: None,
        terminals: vec![flat_plane(
            PlaneRole::Value,
            CodecId::Identity,
            vec![1, 2, 3, 4, 5, 6, 7, 8],
        )],
    };
    build_container(&[], registry, &[rec])
}

/// Implode `registry` (a JSON value) with the tensors and members of
/// `exploded`.
fn implode_value(
    exploded: &Exploded,
    registry: &serde_json::Value,
) -> Result<Vec<u8>, PtwmCoreError> {
    implode(
        &registry.to_string(),
        &tensor_jsons(exploded),
        members_map(exploded),
    )
}

#[test]
fn implode_rejects_other_manifest_format_or_version() {
    let blob = minimal_container();
    let exploded = explode(&blob).unwrap();
    let reg: serde_json::Value = serde_json::from_str(&exploded.registry_json).unwrap();
    assert_eq!(implode_value(&exploded, &reg).unwrap(), blob);

    for (field, value) in [
        ("ptwm_format", serde_json::json!("ptwm-manifest-c")),
        ("manifest_version", serde_json::json!(2)),
        ("manifest_version", serde_json::json!(0)),
    ] {
        let mut bad = reg.clone();
        bad[field] = value.clone();
        let e = implode_value(&exploded, &bad).unwrap_err();
        assert!(
            e.to_string().contains("unsupported manifest"),
            "{field}={value}: {e}"
        );
    }
    let mut missing = reg.clone();
    missing.as_object_mut().unwrap().remove("manifest_version");
    assert!(implode_value(&exploded, &missing).is_err());
}

#[test]
fn implode_rejects_unknown_fields() {
    let blob = minimal_container();
    let exploded = explode(&blob).unwrap();
    let reg: serde_json::Value = serde_json::from_str(&exploded.registry_json).unwrap();

    let mut cases = Vec::new();
    let mut top = reg.clone();
    top["layout"] = serde_json::json!("compact");
    cases.push(top);
    let mut flags = reg.clone();
    flags["container_flags"]["layout"] = serde_json::json!("compact");
    cases.push(flags);
    let mut registry = reg.clone();
    registry["registry"]["extension_table"] = serde_json::json!([]);
    cases.push(registry);
    let mut chain = reg.clone();
    chain["registry"]["chains"][0]["extra"] = serde_json::json!(1);
    cases.push(chain);
    for bad in cases {
        let e = implode_value(&exploded, &bad).unwrap_err();
        assert!(e.to_string().contains("unknown field"), "{e}");
    }

    // Unknown fields in a tensor manifest are rejected too.
    let mut tm: serde_json::Value =
        serde_json::from_str(&exploded.tensors[0].manifest_json).unwrap();
    tm["planes"][0]["extra"] = serde_json::json!(true);
    let e = implode(
        &exploded.registry_json,
        &[tm.to_string()],
        members_map(&exploded),
    )
    .unwrap_err();
    assert!(e.to_string().contains("unknown field"), "{e}");
}

/// When no table layout reproduces the recorded hash, implode errors
/// instead of writing a container with another table.
#[test]
fn implode_rejects_unreproducible_table_hash() {
    let blob = minimal_container();
    let exploded = explode(&blob).unwrap();
    let mut reg: serde_json::Value = serde_json::from_str(&exploded.registry_json).unwrap();
    reg["registry_hash_blake3"] = serde_json::json!("00".repeat(32));
    let e = implode_value(&exploded, &reg).unwrap_err();
    assert!(e.to_string().contains("no Extension Table layout"), "{e}");

    // The wrong built-in version does not reproduce the hash either.
    let mut reg: serde_json::Value = serde_json::from_str(&exploded.registry_json).unwrap();
    reg["container_flags"]["builtin_version"] = serde_json::json!("0.9.0");
    let e = implode_value(&exploded, &reg).unwrap_err();
    assert!(e.to_string().contains("no Extension Table layout"), "{e}");
}

#[test]
fn implode_writes_raw_extension_table() {
    let blob = minimal_container();
    let exploded = explode(&blob).unwrap();
    let reader = ContainerReader::open(&blob).unwrap();
    let et = extension_table_bytes(&reader, &blob).unwrap();

    // A raw table equal to the on-disk one reproduces the container.
    let mut reg: serde_json::Value = serde_json::from_str(&exploded.registry_json).unwrap();
    reg["registry"]["extension_table_b64"] = serde_json::json!(base64::encode(et));
    assert_eq!(implode_value(&exploded, &reg).unwrap(), blob);

    // Raw bytes that do not match the recorded hash are rejected.
    let mut other = et.to_vec();
    other.push(0);
    reg["registry"]["extension_table_b64"] = serde_json::json!(base64::encode(&other));
    let e = implode_value(&exploded, &reg).unwrap_err();
    assert!(e.to_string().contains("does not match"), "{e}");

    // So are bytes that match a forged hash but do not parse cleanly.
    reg["registry_hash_blake3"] = serde_json::json!(hex::encode(ExtensionTable::hash(&other)));
    let e = implode_value(&exploded, &reg).unwrap_err();
    assert!(e.to_string().contains("trailing bytes"), "{e}");
}

/// A container with built-ins of `version`: tensor "a" uses the chain
/// registry, tensor "b" an inline chain.
fn container_at_builtin_version(version: &str) -> Vec<u8> {
    let registry = ChainRegistry {
        chains: vec![(0u16, passthrough_chain())],
    };
    let rec = |name: &str, inline: bool| TensorRecord {
        chain_ref: if inline { CHAIN_REF_INLINE_BIT } else { 0 },
        dtype_code: 6,
        input_format: 2,
        flags: 0,
        orig_size: 8,
        name: name.to_string(),
        payload_hash: None,
        inline_chain: inline.then(passthrough_chain),
        dependencies: Vec::new(),
        tensor_metadata: None,
        terminals: vec![flat_plane(PlaneRole::Value, CodecId::Identity, vec![9; 8])],
    };
    let mut cursor = Cursor::new(Vec::<u8>::new());
    {
        let mut writer = ContainerWriter::new(&mut cursor).unwrap();
        writer.set_builtin_version(version).unwrap();
        writer.set_chain_registry(registry).unwrap();
        let idx = writer
            .codec_table_idx_for(&crate::compressor::codec_canonical_id(CodecId::Identity))
            .unwrap();
        for (name, inline) in [("a", false), ("b", true)] {
            let mut r = rec(name, inline);
            r.terminals[0].codec_table_idx = idx;
            writer.append_tensor(&r).unwrap();
        }
        writer.finalize().unwrap();
    }
    cursor.into_inner()
}

/// A container written with the built-ins of another release explodes with
/// that version and implodes to the same bytes.
#[test]
fn round_trip_other_builtin_version() {
    let blob = container_at_builtin_version("0.9.0");
    assert!(
        blob.windows(b"@0.9.0".len()).any(|w| w == b"@0.9.0"),
        "labels carry the requested version"
    );
    let reader = ContainerReader::open(&blob).unwrap();
    assert_eq!(reader.builtin_version.as_deref(), Some("0.9.0"));
    assert_eq!(reader.decode_tensor("b").unwrap(), vec![9; 8]);

    let exploded = explode(&blob).unwrap();
    let reg: serde_json::Value = serde_json::from_str(&exploded.registry_json).unwrap();
    assert_eq!(reg["container_flags"]["builtin_version"], "0.9.0");
    assert!(reg["registry"].get("extension_table_b64").is_none());
    assert_round_trip(&blob);
}

fn find_last(haystack: &[u8], needle: &[u8]) -> usize {
    haystack
        .windows(needle.len())
        .rposition(|w| w == needle)
        .expect("needle present")
}

/// Rewrite the built-in entry labelled `old_label` (its last occurrence) to
/// `<name>@<new_version>` with the id derived for that version. Both
/// versions must have the same length. An entry's id directly precedes its
/// u16 label length.
fn relabel_builtin(blob: &mut [u8], old_label: &str, new_version: &str) {
    let (name, old_version) = old_label
        .strip_prefix(crate::extension::BUILTIN_LABEL_PREFIX)
        .unwrap()
        .split_once('@')
        .unwrap();
    assert_eq!(old_version.len(), new_version.len());
    let pos = find_last(blob, old_label.as_bytes());
    let id = CanonicalId::derive(&crate::extension::BUILTIN_PUBKEY, name, new_version);
    blob[pos - 34..pos - 2].copy_from_slice(id.as_bytes());
    let ver_pos = pos + old_label.len() - old_version.len();
    blob[ver_pos..pos + old_label.len()].copy_from_slice(new_version.as_bytes());
}

/// One inline-chain table that mixes built-in versions cannot be rebuilt by
/// implode, which writes each inline table with a single version. Explode
/// must refuse it rather than produce a manifest that implodes to other
/// bytes.
#[test]
fn explode_rejects_inline_table_with_mixed_versions() {
    let mut blob = container_at_builtin_version("0.9.0");
    // The last `source@0.9.0` label is the one in the inline table of "b".
    relabel_builtin(&mut blob, "io.ptwm.builtin.source@0.9.0", "0.8.0");
    let reader = ContainerReader::open(&blob).unwrap();
    assert_eq!(reader.decode_tensor("b").unwrap(), vec![9; 8]);
    let Err(e) = explode(&blob) else {
        panic!("explode accepted an inline table with mixed versions");
    };
    let e = e.to_string();
    assert!(e.contains("differs from its re-encoding"), "{e}");

    // The control: relabelling every entry of that table to one version
    // round-trips.
    let mut blob = container_at_builtin_version("0.9.0");
    relabel_builtin(&mut blob, "io.ptwm.builtin.source@0.9.0", "0.8.0");
    relabel_builtin(&mut blob, "io.ptwm.builtin.byte_passthrough@0.9.0", "0.8.0");
    let exploded = explode(&blob).unwrap();
    let reg: serde_json::Value = serde_json::from_str(&exploded.registry_json).unwrap();
    assert_eq!(reg["container_flags"]["builtin_version"], "0.8.0");
    assert_round_trip(&blob);
}

/// Built-in labels with a version that no writer accepts are not
/// normalised: the writer refuses such a table, and explode never records
/// the version in a manifest that implode would then reject.
#[test]
fn invalid_builtin_versions_are_not_recorded() {
    // Writer side: a raw table whose built-ins carry an invalid version
    // does not cover the chain ops, so it is refused.
    let long = "7".repeat(70);
    for bad in ["9.9@x", long.as_str(), ""] {
        let registry = ChainRegistry {
            chains: vec![(0u16, passthrough_chain())],
        };
        let mut built =
            crate::container::build_extension_table(&registry, CodecSet::AllBuiltins, "1.0.0")
                .unwrap();
        for e in &mut built.table.entries {
            let name = e
                .human_label
                .strip_prefix(crate::extension::BUILTIN_LABEL_PREFIX)
                .unwrap()
                .split_once('@')
                .unwrap()
                .0
                .to_owned();
            e.canonical_id = CanonicalId::derive(&crate::extension::BUILTIN_PUBKEY, &name, bad);
            e.human_label = format!("{}{name}@{bad}", crate::extension::BUILTIN_LABEL_PREFIX);
        }
        let et = built.table.to_bytes().unwrap();
        let mut writer = ContainerWriter::new(Cursor::new(Vec::<u8>::new())).unwrap();
        let e = writer.set_raw_extension_table(&et, registry).unwrap_err();
        assert!(
            e.to_string().contains("missing from the table"),
            "{bad:?}: {e}"
        );
    }

    // Reader side: patch every built-in entry of a 1.0.0 file to the
    // same-length invalid version "9.9@x" and fix the table hash.
    let mut blob = container_at_builtin_version("1.0.0");
    let names: Vec<String> = ContainerReader::open(&blob)
        .unwrap()
        .extension_table
        .entries
        .iter()
        .map(|e| {
            e.human_label
                .strip_prefix(crate::extension::BUILTIN_LABEL_PREFIX)
                .unwrap()
                .split_once('@')
                .unwrap()
                .0
                .to_owned()
        })
        .collect();
    for name in &names {
        let label = format!("{}{name}@1.0.0", crate::extension::BUILTIN_LABEL_PREFIX);
        while blob.windows(label.len()).any(|w| w == label.as_bytes()) {
            relabel_builtin(&mut blob, &label, "9.9@x");
        }
    }
    let mut header = crate::header::Header::from_bytes(&blob).unwrap();
    let start = header.extension_table_offset as usize;
    let end = start + header.extension_table_length as usize;
    header.extension_table_hash = ExtensionTable::hash(&blob[start..end]);
    let header_bytes = header.to_bytes().unwrap();
    blob[..header_bytes.len()].copy_from_slice(&header_bytes);

    // The entries are not built-ins, so the chain ops do not resolve and
    // the file is rejected on read instead of exploding to a manifest with
    // builtin_version "9.9@x".
    assert!(ContainerReader::open(&blob).is_err());
    let Err(e) = explode(&blob) else {
        panic!("explode accepted built-ins with an invalid version");
    };
    let msg = e.to_string();
    assert!(msg.contains("unknown canonical id"), "{msg}");
}
