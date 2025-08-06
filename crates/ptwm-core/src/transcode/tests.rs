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
