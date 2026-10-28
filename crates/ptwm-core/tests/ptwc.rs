//! Compact container (`\x89PTWC`) round trips, size bounds and old-reader
//! rejection.
//!
//! The legacy fixtures in `tests/fixtures/` hold every record feature the
//! writers produce (registry and inline chains, a shared prelude state,
//! chunked planes, a dependency, the Rows layout, bf16/fp16/fp32/fp8/int8).
//! Rewriting them in the compact layout and decoding both files covers the
//! compact record, plane, registry and Extension Table codecs on real data.
//! `compress_model` tests cover the compact writer path, bound Sources and
//! the overhead bounds of synthesis section 9 item 14.

use std::collections::HashMap;
use std::io::Cursor;
use std::path::PathBuf;

use ptwm_core::chain::{Chain, ChainEdge, ChainNode, TerminalRef};
use ptwm_core::codec::CodecId;
use ptwm_core::compressor::{CompressorOptions, InputTensor, compress_model};
use ptwm_core::container::{ContainerLayout, ContainerReader, ContainerWriter};
use ptwm_core::extension::compact_table::{CompactEntry, classify};
use ptwm_core::extension::{
    Attestation, CanonicalId, CapabilityMap, ExtensionTableEntry, Kind, Lifecycle, builtin_entries,
    builtin_registry,
};
use ptwm_core::header::{HEADER_LEN, Header, MAGIC_PTWC};
use ptwm_core::tensor_record::TensorRecord;
use ptwm_core::tensor_record_compact::TF_RAW_META;
use ptwm_core::transcode::{Exploded, explode, implode};
use ptwm_core::transforms::op::OpId;
use ptwm_core::types::{Role, ScaleFormat, ValueFormat};

const FIXTURES: &[&str] = &[
    "legacy_v1_1_0.ptwm",
    "legacy_v1_0_0.ptwm",
    "legacy_v1_0_0_native.ptwm",
    "legacy_trimmed.ptwm",
];

fn load(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Rewrite the container `blob` (either layout) in the compact layout:
/// same Extension Table, registry, prelude and records, in body order.
fn to_compact(blob: &[u8]) -> Vec<u8> {
    let reader = ContainerReader::open(blob).expect("open");
    let mut out = Vec::new();
    let mut w =
        ContainerWriter::with_layout(Cursor::new(&mut out), ContainerLayout::Compact).unwrap();
    w.set_prelude(&reader.prelude).unwrap();
    w.set_compact_tables(
        reader.extension_table.clone(),
        reader.chain_registry.clone(),
    )
    .unwrap();
    for rec in reader.tensor_records().unwrap() {
        w.append_tensor(&rec).unwrap();
    }
    w.finalize().unwrap();
    out
}

fn decode_all(blob: &[u8]) -> HashMap<String, Vec<u8>> {
    ContainerReader::open(blob)
        .unwrap()
        .decode_model()
        .unwrap()
        .into_iter()
        .collect()
}

#[test]
fn legacy_fixtures_round_trip_through_the_compact_layout() {
    for fixture in FIXTURES {
        let legacy = load(fixture);
        let compact = to_compact(&legacy);
        assert_eq!(compact[..9], MAGIC_PTWC, "{fixture}");
        assert!(
            compact.len() < legacy.len(),
            "{fixture}: compact {} B, legacy {} B",
            compact.len(),
            legacy.len()
        );

        let lr = ContainerReader::open(&legacy).unwrap();
        let cr = ContainerReader::open(&compact).unwrap();
        assert_eq!(cr.layout, ContainerLayout::Compact);
        assert_eq!(lr.layout, ContainerLayout::Legacy);
        assert_eq!(cr.prelude, lr.prelude, "{fixture}: prelude");
        assert_eq!(cr.chain_registry, lr.chain_registry, "{fixture}: registry");
        assert_eq!(
            cr.extension_table, lr.extension_table,
            "{fixture}: extension table"
        );

        // Every record parses back to the legacy record.
        let lrecs = lr.tensor_records().unwrap();
        let crecs = cr.tensor_records().unwrap();
        assert_eq!(crecs, lrecs, "{fixture}: records");
        // Names come from the index in body order.
        let names: Vec<String> = lrecs.iter().map(|r| r.name.clone()).collect();
        assert_eq!(cr.tensor_names().unwrap(), names, "{fixture}: names");

        // Per-tensor and whole-model decode agree with the legacy file.
        let want = decode_all(&legacy);
        assert_eq!(decode_all(&compact), want, "{fixture}: decode_model");
        for name in &names {
            assert_eq!(&cr.decode_tensor(name).unwrap(), &want[name], "{fixture}");
            assert_eq!(
                cr.tensor_shape(name).unwrap(),
                lr.tensor_shape(name).unwrap(),
                "{fixture}: {name}"
            );
        }
        assert!(cr.get_tensor_record_bytes("no.such.tensor").is_none());

        // One encoding per content: rewriting the compact file reproduces it.
        assert!(
            to_compact(&compact) == compact,
            "{fixture}: rewrite differs"
        );
        assert_transcodes(&compact, fixture);
    }
}

#[test]
fn every_byte_of_a_compact_fixture_is_checked() {
    // Flipping any header, table, prelude, registry or index byte must fail
    // the open (meta_xxh64 or a structural check), never decode quietly.
    let compact = to_compact(&load("legacy_v1_1_0.ptwm"));
    let reader = ContainerReader::open(&compact).unwrap();
    let body_start = reader.tensors_offset as usize;
    let index_start = reader.header.tensor_index_offset as usize;
    let checked = (0..body_start).chain(index_start..compact.len());
    for pos in checked {
        let mut bad = compact.clone();
        bad[pos] ^= 0x01;
        let opened = ContainerReader::open(&bad);
        if let Ok(r) = opened {
            // An index name byte can turn into another valid name; the file
            // is then a different, still consistent container.
            assert!(
                pos >= index_start,
                "byte {pos} of the metadata sections is not checked"
            );
            assert_ne!(
                r.tensor_names().unwrap(),
                reader.tensor_names().unwrap(),
                "byte {pos}"
            );
        }
    }
}

#[test]
fn old_reader_rejects_compact_files_on_the_magic() {
    let compact = to_compact(&load("legacy_trimmed.ptwm"));
    assert!(compact.len() >= HEADER_LEN);
    let err = Header::from_bytes(&compact[..HEADER_LEN])
        .unwrap_err()
        .to_string();
    assert!(err.contains("bad magic"), "{err}");
}

#[test]
fn p4_source_slack_is_rejected_in_the_compact_layout() {
    // Legacy files from v1.1 BYTE input declare a Source 4x the real size;
    // the legacy reader tolerates it, the compact reader requires exact
    // sizes.
    let legacy = load("legacy_p4_byte_fp32.ptwm");
    let compact = to_compact(&legacy);
    let reader = ContainerReader::open(&compact).unwrap();
    let err = reader.decode_model().unwrap_err().to_string();
    assert!(err.contains("Source declares"), "{err}");
    // The transcoder carries the file unchanged; per-tensor implodes fail to
    // decode the same way.
    assert_transcodes(&compact, "legacy_p4_byte_fp32.ptwm");
}

// ── compress_model in the compact layout ───────────────────────────────────

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn weight(&mut self) -> f32 {
        let mut s = 0.0f32;
        for _ in 0..4 {
            s += (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        }
        (s - 2.0) * 0.05
    }
}

fn source(shape: &[u64], internal_dtype: u16) -> ChainNode {
    let mut params = vec![shape.len() as u8];
    for d in shape {
        params.extend_from_slice(&(*d as u32).to_le_bytes());
    }
    params.extend_from_slice(&internal_dtype.to_le_bytes());
    ChainNode {
        op: OpId::Source,
        params,
    }
}

fn edge(src: u8, dst: u8, role_override: Option<Role>) -> ChainEdge {
    ChainEdge {
        src_node: src,
        src_output_idx: 0,
        dst_node: dst,
        dst_input_idx: 0,
        role_override,
        vendor_bytes: vec![],
    }
}

fn term(node_idx: u8, output_idx: u8, role: Role) -> TerminalRef {
    TerminalRef {
        node_idx,
        output_idx,
        role,
    }
}

fn fp32_split_chain(shape: &[u64]) -> Chain {
    Chain {
        nodes: vec![
            source(shape, 0x0003),
            ChainNode {
                op: OpId::BitReorderIeee32,
                params: vec![],
            },
            ChainNode {
                op: OpId::ByteSplit,
                params: vec![4],
            },
        ],
        edges: vec![edge(0, 1, None), edge(1, 2, None)],
        terminals: vec![
            term(2, 0, Role::MantissaByte { index: 0, of: 4 }),
            term(2, 1, Role::MantissaByte { index: 1, of: 4 }),
            term(2, 2, Role::MantissaByte { index: 2, of: 4 }),
            term(2, 3, Role::ExponentByte),
        ],
    }
}

fn bf16_split_chain(shape: &[u64]) -> Chain {
    Chain {
        nodes: vec![
            source(shape, 0x000F),
            ChainNode {
                op: OpId::BitReorderIeee16,
                params: vec![],
            },
            ChainNode {
                op: OpId::ByteSplit,
                params: vec![2],
            },
        ],
        edges: vec![edge(0, 1, None), edge(1, 2, None)],
        terminals: vec![
            term(2, 0, Role::MantissaByte { index: 0, of: 2 }),
            term(2, 1, Role::ExponentByte),
        ],
    }
}

fn byte_chain(shape: &[u64], scale: bool) -> Chain {
    let role = if scale {
        Role::Scale {
            format: ScaleFormat::E8M0,
        }
    } else {
        Role::Value {
            format: ValueFormat::IntN { bits: 8 },
        }
    };
    Chain {
        nodes: vec![
            source(shape, 0x0006),
            ChainNode {
                op: OpId::BytePassthrough,
                params: vec![],
            },
        ],
        edges: vec![edge(0, 1, scale.then(|| role.clone()))],
        terminals: vec![term(1, 0, role)],
    }
}

struct Tensor {
    name: String,
    dtype_code: u16,
    dtype_name: String,
    shape: Vec<u64>,
    bytes: Vec<u8>,
    chain: Chain,
    delta: bool,
}

fn fp32(name: &str, shape: Vec<u64>, rng: &mut Rng) -> Tensor {
    let n: u64 = shape.iter().product();
    Tensor {
        name: name.into(),
        dtype_code: 1,
        dtype_name: "float32".into(),
        chain: fp32_split_chain(&shape),
        bytes: (0..n).flat_map(|_| rng.weight().to_le_bytes()).collect(),
        shape,
        delta: false,
    }
}

/// One tensor per feature the compressor emits: dtypes, 0-d, a
/// non-canonical dtype name, a dependency, Rows scale planes that share a
/// prelude state.
fn feature_tensors() -> Vec<Tensor> {
    let mut rng = Rng(0x5054_5743_5445_5354);
    let mut out = vec![
        fp32("model.norm.weight", vec![4, 32], &mut rng),
        fp32("model.scalar", vec![], &mut rng),
    ];
    let mut custom = fp32("model.custom.weight", vec![16], &mut rng);
    custom.dtype_name = "my_float".into();
    out.push(custom);
    let mut delta = fp32("model.delta.weight", vec![8, 8], &mut rng);
    delta.delta = true;
    out.push(delta);
    let shape = vec![8u64, 32];
    out.push(Tensor {
        name: "model.layers.0.q.weight".into(),
        dtype_code: 6,
        dtype_name: "bfloat16".into(),
        chain: bf16_split_chain(&shape),
        bytes: (0..256)
            .flat_map(|_| ((rng.weight().to_bits() >> 16) as u16).to_le_bytes())
            .collect(),
        shape,
        delta: false,
    });
    for name in ["model.layers.0.q.scale", "model.layers.0.k.scale"] {
        let mut bytes = Vec::with_capacity(2048);
        for _ in 0..64 {
            let mut prev: u8 = 0x80;
            for _ in 0..32 {
                prev = prev
                    .wrapping_add((rng.next_u64() % 3) as u8)
                    .wrapping_sub(1);
                bytes.push(prev);
            }
        }
        out.push(Tensor {
            name: name.into(),
            dtype_code: 17,
            dtype_name: "int8".into(),
            chain: byte_chain(&[2048], true),
            bytes,
            shape: vec![64, 32],
            delta: false,
        });
    }
    out.push(Tensor {
        name: "model.embed.int8".into(),
        dtype_code: 17,
        dtype_name: "int8".into(),
        chain: byte_chain(&[256], false),
        bytes: (0..256)
            .map(|_| ((rng.next_u64() % 9) as i8 - 4) as u8)
            .collect(),
        shape: vec![256],
        delta: false,
    });
    out
}

fn inputs(data: &[Tensor]) -> Vec<InputTensor<'_>> {
    data.iter()
        .map(|t| InputTensor {
            name: t.name.clone(),
            candidate_chains: vec![t.chain.clone()],
            dtype_code: t.dtype_code,
            input_format: 2,
            raw_bytes: &t.bytes,
            shape: Some(t.shape.clone()),
            dtype_name: Some(t.dtype_name.clone()),
            delta_reference_blake3: t.delta.then_some([7u8; 32]),
        })
        .collect()
}

fn options(
    hash: bool,
    crc: bool,
    chunk_size: Option<u32>,
    layout: ContainerLayout,
) -> CompressorOptions {
    CompressorOptions {
        method_hint: 1,
        emit_payload_hash: hash,
        emit_plane_crc: crc,
        chunk_size,
        container_layout: layout,
        ..Default::default()
    }
}

fn compress(data: &[Tensor], opts: CompressorOptions) -> Vec<u8> {
    let mut out = Cursor::new(Vec::new());
    compress_model(&mut out, &inputs(data), opts).expect("compress_model");
    out.into_inner()
}

#[test]
fn compress_model_matrix_round_trips_in_both_layouts() {
    let data = feature_tensors();
    for (hash, crc) in [(true, true), (false, false), (true, false), (false, true)] {
        for chunk in [None, Some(256)] {
            let mut decoded = Vec::new();
            for layout in [ContainerLayout::Legacy, ContainerLayout::Compact] {
                let blob = compress(&data, options(hash, crc, chunk, layout));
                let reader = ContainerReader::open(&blob).unwrap();
                assert_eq!(reader.layout, layout);
                for t in &data {
                    assert_eq!(
                        reader.decode_tensor(&t.name).unwrap(),
                        t.bytes,
                        "{layout:?} hash={hash} crc={crc} chunk={chunk:?}: {}",
                        t.name
                    );
                    let (shape, name) = reader.tensor_shape(&t.name).unwrap().unwrap();
                    assert_eq!((shape, name), (t.shape.clone(), t.dtype_name.clone()));
                    let rec = reader.tensor_record(&t.name).unwrap();
                    assert_eq!(rec.name, t.name);
                    assert_eq!(rec.payload_hash.is_some(), hash);
                    assert_eq!(rec.dependencies.len(), usize::from(t.delta));
                    for p in &rec.terminals {
                        assert_eq!(p.crc32.is_some(), crc);
                        assert_eq!(p.chunk_table.is_some(), chunk.is_some());
                    }
                }
                let mut all = reader.decode_model().unwrap();
                all.sort();
                decoded.push(all);
                assert_transcodes(
                    &blob,
                    &format!("{layout:?} hash={hash} crc={crc} chunk={chunk:?}"),
                );
                if layout == ContainerLayout::Compact {
                    // parse then write is the identity.
                    assert!(to_compact(&blob) == blob, "rewrite differs");
                    if chunk.is_none() {
                        assert!(!reader.prelude.is_empty(), "no shared state");
                    }
                }
            }
            assert_eq!(decoded[0], decoded[1]);
        }
    }
}

/// New multi-tensor writes default to the compact layout; the legacy
/// layout is still available on request.
#[test]
fn compact_layout_is_the_default() {
    let data = feature_tensors();
    let default = compress(&data, options(true, true, None, ContainerLayout::default()));
    let compact = compress(&data, options(true, true, None, ContainerLayout::Compact));
    assert!(default == compact);
    assert_eq!(default[..5], *b"\x89PTWC");
    let legacy = compress(&data, options(true, true, None, ContainerLayout::Legacy));
    assert_eq!(legacy[..5], *b"\x89PTWM");
    let from_default = compress(&data, CompressorOptions::default());
    assert_eq!(from_default[..5], *b"\x89PTWC");
}

#[test]
fn bound_source_shares_one_chain_across_shapes() {
    let mut rng = Rng(1);
    let data: Vec<Tensor> = (1..=64)
        .map(|i| fp32(&format!("t{i}"), vec![i, 2], &mut rng))
        .collect();
    let legacy = compress(&data, options(true, true, None, ContainerLayout::Legacy));
    let compact = compress(&data, options(true, true, None, ContainerLayout::Compact));
    let lr = ContainerReader::open(&legacy).unwrap();
    let cr = ContainerReader::open(&compact).unwrap();
    assert_eq!(lr.chain_registry.chains.len(), 64);
    assert_eq!(cr.chain_registry.chains.len(), 1);
    for t in &data {
        assert_eq!(cr.decode_tensor(&t.name).unwrap(), t.bytes);
    }
    assert_transcodes(&compact, "bound Source");

    // A record without shape metadata cannot use the bound chain: the writer
    // refuses it.
    let mut rec = cr.tensor_record("t1").unwrap();
    rec.tensor_metadata = None;
    rec.flags &= !ptwm_core::tensor_record::TENSOR_FLAG_METADATA;
    rec.name = "t_no_shape".into();
    let mut out = Vec::new();
    let mut w =
        ContainerWriter::with_layout(Cursor::new(&mut out), ContainerLayout::Compact).unwrap();
    w.set_compact_tables(cr.extension_table.clone(), cr.chain_registry.clone())
        .unwrap();
    let err = w.append_tensor(&rec).unwrap_err().to_string();
    assert!(err.contains("bound Source"), "{err}");
}

#[test]
fn duplicate_names_are_rejected_and_lookup_is_exact() {
    let compact = to_compact(&load("legacy_trimmed.ptwm"));
    let reader = ContainerReader::open(&compact).unwrap();
    let rec = reader.tensor_records().unwrap().remove(0);
    let mut out = Vec::new();
    let mut w =
        ContainerWriter::with_layout(Cursor::new(&mut out), ContainerLayout::Compact).unwrap();
    w.set_prelude(&reader.prelude).unwrap();
    w.set_compact_tables(
        reader.extension_table.clone(),
        reader.chain_registry.clone(),
    )
    .unwrap();
    w.append_tensor(&rec).unwrap();
    let err = w.append_tensor(&rec).unwrap_err().to_string();
    assert!(err.contains("duplicate tensor name"), "{err}");

    // Lookup compares names, so a name that shares a prefix or differs in
    // one byte is not found.
    for name in reader.tensor_names().unwrap() {
        assert!(reader.get_tensor_record_bytes(&name).is_some());
        let mut other = name.clone();
        other.push('x');
        assert!(reader.get_tensor_record_bytes(&other).is_none());
        assert!(
            reader
                .get_tensor_record_bytes(&name[..name.len() - 1])
                .is_none()
        );
    }
}

/// Rewrite the compact or legacy container `blob` in the compact layout,
/// applying `f` to each record first.
fn rewrite_compact(blob: &[u8], mut f: impl FnMut(&mut TensorRecord)) -> Vec<u8> {
    let reader = ContainerReader::open(blob).expect("open");
    let mut out = Vec::new();
    let mut w =
        ContainerWriter::with_layout(Cursor::new(&mut out), ContainerLayout::Compact).unwrap();
    w.set_prelude(&reader.prelude).unwrap();
    w.set_compact_tables(
        reader.extension_table.clone(),
        reader.chain_registry.clone(),
    )
    .unwrap();
    for mut rec in reader.tensor_records().unwrap() {
        f(&mut rec);
        w.append_tensor(&rec).unwrap();
    }
    w.finalize().unwrap();
    out
}

/// A BYTE-input fp32 tensor: no shape metadata, an exact Source over the
/// raw byte count (the P4-fixed Python byte path).
fn byte_input_blob(hash: bool, crc: bool) -> (Vec<u8>, Vec<u8>) {
    let mut rng = Rng(0xB7E);
    let t = fp32("model.raw", vec![32], &mut rng);
    let input = InputTensor {
        name: t.name.clone(),
        candidate_chains: vec![t.chain.clone()],
        dtype_code: t.dtype_code,
        input_format: 1,
        raw_bytes: &t.bytes,
        shape: None,
        dtype_name: None,
        delta_reference_blake3: None,
    };
    let mut out = Cursor::new(Vec::new());
    compress_model(
        &mut out,
        &[input],
        options(hash, crc, None, ContainerLayout::Compact),
    )
    .expect("compress_model");
    (out.into_inner(), t.bytes)
}

#[test]
fn byte_input_tensor_round_trips_in_the_compact_layout() {
    for (hash, crc) in [(true, true), (false, false)] {
        let (blob, raw) = byte_input_blob(hash, crc);
        let reader = ContainerReader::open(&blob).unwrap();
        assert_eq!(reader.layout, ContainerLayout::Compact);
        let rec = reader.tensor_record("model.raw").unwrap();
        assert_eq!(rec.input_format, 1);
        assert!(rec.tensor_metadata.is_none());
        assert_eq!(reader.tensor_shape("model.raw").unwrap(), None);
        assert_eq!(reader.decode_tensor("model.raw").unwrap(), raw);
        assert!(to_compact(&blob) == blob, "rewrite differs");
        assert_transcodes(&blob, "BYTE input");
    }
}

#[test]
fn raw_cbor_metadata_round_trips_in_a_compact_container() {
    // Metadata that is not the canonical shape map is stored as TF_RAW_META
    // and must come back byte for byte. The tensor uses an explicit
    // registry chain, so its Source does not depend on the metadata.
    let (blob, raw) = byte_input_blob(true, true);
    let metas: [&[u8]; 3] = [
        // {"a": 1}
        &[0xA1, 0x61, b'a', 0x01],
        // Shape map with a non-canonical (two-byte) length for the shape.
        &[
            0xA2, 0x65, b's', b'h', b'a', b'p', b'e', 0x98, 0x01, 0x18, 0x20, 0x65, b'd', b't',
            b'y', b'p', b'e', 0x67, b'f', b'l', b'o', b'a', b't', b'3', b'2',
        ],
        // Arbitrary bytes, not CBOR at all.
        &[0xFF, 0x00, 0x13],
    ];
    for meta in metas {
        let compact = rewrite_compact(&blob, |rec| {
            rec.tensor_metadata = Some(meta.to_vec());
            rec.flags |= ptwm_core::tensor_record::TENSOR_FLAG_METADATA;
        });
        let reader = ContainerReader::open(&compact).unwrap();
        let rec = reader.tensor_record("model.raw").unwrap();
        assert_eq!(rec.tensor_metadata.as_deref(), Some(meta));
        let bytes = reader.get_tensor_record_bytes("model.raw").unwrap();
        assert_ne!(bytes[0] & TF_RAW_META, 0, "stored as TF_RAW_META");
        assert_eq!(reader.decode_tensor("model.raw").unwrap(), raw);
        assert!(to_compact(&compact) == compact, "rewrite differs");
        assert_transcodes(&compact, "raw metadata");
    }
}

#[test]
fn external_state_plane_round_trips_in_a_compact_container() {
    // The reader does not resolve external state at open time; the record
    // and the file must still round-trip exactly.
    use ptwm_core::codec::StateSource;
    use ptwm_core::plane_record::ExternalStateRef;
    let data = feature_tensors();
    let blob = compress(&data, options(true, true, None, ContainerLayout::Compact));
    let target = "model.norm.weight";
    let ext = ExternalStateRef {
        ref_kind: 1,
        expected_hash: Some([0x5A; 32]),
        ref_bytes: b"dict/zstd-v1.bin".to_vec(),
        state_format_version: 3,
    };
    let compact = rewrite_compact(&blob, |rec| {
        if rec.name == target {
            let p = &mut rec.terminals[0];
            p.state_source = StateSource::External;
            p.state_version = 0;
            p.state_info = 0;
            p.inline_state_bytes.clear();
            p.external_state = Some(ext.clone());
        }
    });
    let reader = ContainerReader::open(&compact).unwrap();
    let rec = reader.tensor_record(target).unwrap();
    assert_eq!(rec.terminals[0].state_source, StateSource::External);
    assert_eq!(rec.terminals[0].external_state.as_ref(), Some(&ext));
    assert!(to_compact(&compact) == compact, "rewrite differs");
    assert_transcodes(&compact, "external state");
    // The other tensors are unaffected.
    for t in data.iter().filter(|t| t.name != target) {
        assert_eq!(
            reader.decode_tensor(&t.name).unwrap(),
            t.bytes,
            "{}",
            t.name
        );
    }
}

#[test]
fn compact_source_of_4_gib_fails_without_allocating() {
    // Shape metadata that declares [1 << 30] fp32 (4 GiB) binds the
    // registry chain's Source to 4 GiB for a 256-byte tensor. Decoding must
    // fail on the size check, not try to size buffers from the chain.
    use ptwm_core::metadata::encode_shape_metadata;
    let mut rng = Rng(0xB0B);
    let one = vec![fp32("t", vec![64], &mut rng)];
    let blob = compress(&one, options(true, true, None, ContainerLayout::Compact));
    assert_eq!(
        ContainerReader::open(&blob)
            .unwrap()
            .chain_registry
            .chains
            .len(),
        1
    );
    let bomb = rewrite_compact(&blob, |rec| {
        rec.tensor_metadata = Some(encode_shape_metadata(&[1 << 30], "float32"));
        rec.orig_size = 4 << 30;
    });
    let reader = ContainerReader::open(&bomb).unwrap();
    assert_eq!(reader.tensor_shape("t").unwrap().unwrap().0, vec![1 << 30]);
    let check = |err: String, t0: std::time::Instant| {
        eprintln!("4 GiB Source: {err}");
        assert!(t0.elapsed() < std::time::Duration::from_secs(1), "{err}");
        assert!(
            err.contains("1073741824") || err.contains("Source declares"),
            "{err}"
        );
    };
    let t0 = std::time::Instant::now();
    check(reader.decode_tensor("t").unwrap_err().to_string(), t0);
    let t0 = std::time::Instant::now();
    check(reader.decode_model().unwrap_err().to_string(), t0);
}

// ── Size bounds (synthesis section 9 item 14) ─────────────────────────────

/// Bytes of a compact file that are not plane payload, inline state or
/// shared prelude state, per tensor.
fn overhead_per_tensor(blob: &[u8]) -> f64 {
    let reader = ContainerReader::open(blob).unwrap();
    let mut data = 0usize;
    for rec in reader.tensor_records().unwrap() {
        for p in &rec.terminals {
            data += p.payload_bytes.len() + p.inline_state_bytes.len();
        }
    }
    data += reader
        .prelude
        .iter()
        .map(|e| e.state_bytes.len())
        .sum::<usize>();
    (blob.len() - data) as f64 / reader.index.len() as f64
}

fn case_a(distinct_shapes: bool) -> Vec<Tensor> {
    let mut rng = Rng(0xA);
    (0..64)
        .map(|i| {
            let shape = if distinct_shapes {
                vec![64 + i as u64]
            } else {
                vec![64]
            };
            fp32(&format!("model.layers.{i:02}.attn.q.bias"), shape, &mut rng)
        })
        .collect()
}

#[test]
fn compact_overhead_bounds() {
    let a = compress(
        &case_a(false),
        options(true, true, None, ContainerLayout::Compact),
    );
    let b = compress(
        &case_a(true),
        options(true, true, None, ContainerLayout::Compact),
    );
    let bare = compress(
        &case_a(false),
        options(false, false, None, ContainerLayout::Compact),
    );
    let legacy = compress(
        &case_a(false),
        options(true, true, None, ContainerLayout::Legacy),
    );
    let (oa, ob, obare, olegacy) = (
        overhead_per_tensor(&a),
        overhead_per_tensor(&b),
        overhead_per_tensor(&bare),
        overhead_per_tensor(&legacy),
    );
    eprintln!("overhead B/tensor: A {oa:.1}, B {ob:.1}, bare {obare:.1}, legacy {olegacy:.1}");
    assert!(oa <= 95.0, "case A: {oa:.1} B/tensor");
    assert!(ob <= 100.0, "case B: {ob:.1} B/tensor");
    assert!(obare <= 70.0, "no hash/CRC: {obare:.1} B/tensor");
    assert!(oa < olegacy / 2.0, "compact {oa:.1} vs legacy {olegacy:.1}");

    // A single tensor pays the whole fixed overhead (header, tables,
    // registry, index, footer) alone.
    let mut rng = Rng(0x51);
    let one = vec![fp32("model.bias", vec![64], &mut rng)];
    let single = compress(&one, options(true, true, None, ContainerLayout::Compact));
    let osingle = overhead_per_tensor(&single);
    eprintln!("single-tensor overhead: {osingle:.0} B");
    assert!(osingle <= 160.0, "single tensor: {osingle:.0} B");

    // A built-in-only Extension Table costs at most 4 B per entry.
    let reader = ContainerReader::open(&a).unwrap();
    let et_len = reader.header.extension_table_length as usize;
    let n = reader.extension_table.entries.len();
    assert!(et_len <= 4 * n, "{et_len} B for {n} entries");
}

// ── Mutations and fuzz seeds ───────────────────────────────────────────────

#[test]
fn mutated_records_are_rejected_or_re_encode_identically() {
    let compact = to_compact(&load("legacy_v1_1_0.ptwm"));
    let reader = ContainerReader::open(&compact).unwrap();
    let mut w =
        ContainerWriter::with_layout(Cursor::new(Vec::new()), ContainerLayout::Compact).unwrap();
    w.set_compact_tables(
        reader.extension_table.clone(),
        reader.chain_registry.clone(),
    )
    .unwrap();
    let encoder = w.record_encoder();
    let records: Vec<&[u8]> = reader
        .tensor_names()
        .unwrap()
        .iter()
        .map(|n| reader.get_tensor_record_bytes(n).unwrap())
        .collect();
    let mut rng = Rng(0x4D55_5441_5445);
    let mut accepted = 0;
    for i in 0..30_000 {
        let base = records[i % records.len()];
        let mut m = base.to_vec();
        // Mutate the record head, where the structure is; payload bytes are
        // opaque to the parser.
        let head = m.len().min(64);
        match rng.next_u64() % 4 {
            0 => {
                let p = (rng.next_u64() as usize) % head;
                m[p] ^= 1 << (rng.next_u64() % 8);
            }
            1 => {
                let p = (rng.next_u64() as usize) % head;
                m[p] = rng.next_u64() as u8;
            }
            2 => {
                let p = (rng.next_u64() as usize) % head;
                m.insert(p, rng.next_u64() as u8);
            }
            _ => {
                let p = (rng.next_u64() as usize) % head;
                m.remove(p);
            }
        }
        if let Ok(rec) = reader.parse_record(&m) {
            accepted += 1;
            assert_eq!(encoder.encode(&rec).unwrap(), m, "mutation {i}");
        }
    }
    eprintln!("{accepted} of 30000 mutated records parsed");
}

#[test]
fn mutated_files_never_panic() {
    let compact = to_compact(&load("legacy_trimmed.ptwm"));
    let mut rng = Rng(0x4649_4C45);
    for _ in 0..3_000 {
        let mut m = compact.clone();
        for _ in 0..1 + rng.next_u64() % 3 {
            let p = (rng.next_u64() as usize) % m.len();
            m[p] = rng.next_u64() as u8;
        }
        if let Ok(r) = ContainerReader::open(&m) {
            let _ = r.decode_model();
        }
    }
}

/// Write PTWC seeds for the fuzz corpora:
///
/// ```text
/// PTWM_WRITE_FUZZ_SEEDS=1 cargo test -p ptwm-core --test ptwc \
///     -- --ignored write_fuzz_seeds
/// ```
#[test]
#[ignore]
fn write_fuzz_seeds() {
    if std::env::var_os("PTWM_WRITE_FUZZ_SEEDS").is_none() {
        return;
    }
    let corpus = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fuzz/corpus");
    let write = |dir: &str, name: &str, bytes: &[u8]| {
        let d = corpus.join(dir);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join(name), bytes).unwrap();
    };
    let trimmed = to_compact(&load("legacy_trimmed.ptwm"));
    write("container_open", "ptwc_trimmed", &trimmed);
    let mut rng = Rng(3);
    let small: Vec<Tensor> = (1..=4)
        .map(|i| fp32(&format!("t{i}"), vec![i, 2], &mut rng))
        .collect();
    let small = compress(&small, options(true, true, None, ContainerLayout::Compact));
    write("container_open", "ptwc_bound_source", &small);

    // Compact Extension Tables of both files.
    for (name, blob) in [("trimmed", &trimmed), ("bound_source", &small)] {
        let r = ContainerReader::open(blob).unwrap();
        let start = r.header.extension_table_offset as usize;
        let len = r.header.extension_table_length as usize;
        write("extension_table_compact", name, &blob[start..start + len]);
    }

    // Records parsed against the fuzz target's fixed context: Extension
    // Table entry 0 a transform, 1 Zstd, 2 Identity; registry chain 0
    // explicit, 1 bound.
    use ptwm_core::codec::CodecId;
    use ptwm_core::tensor_record_compact::{RecordContext, write_compact_record};
    let ctx = RecordContext {
        codec_ids: &[None, Some(CodecId::Zstd), Some(CodecId::Identity)],
        registry_bound: &[false, true],
    };
    let reader = ContainerReader::open(&small).unwrap();
    for (i, mut rec) in reader.tensor_records().unwrap().into_iter().enumerate() {
        rec.chain_ref = 1;
        for p in &mut rec.terminals {
            p.codec_id = CodecId::Zstd;
            p.codec_table_idx = 1;
        }
        let mut out = Vec::new();
        write_compact_record(&rec, &ctx, &mut out).unwrap();
        write("parse_wire_compact", &format!("record_{i}"), &out);
    }
}

// ── Transcode (synthesis section 6) ─────────────────────────────────────────

fn members_of(exploded: &Exploded) -> HashMap<String, &[u8]> {
    exploded
        .members
        .iter()
        .map(|(k, v)| (k.clone(), v.as_slice()))
        .collect()
}

fn manifests_of(exploded: &Exploded) -> Vec<String> {
    exploded
        .tensors
        .iter()
        .map(|t| t.manifest_json.clone())
        .collect()
}

/// The members one tensor needs: its own (`<key>.`) and the archive-wide
/// shared state, as the exploded stores keep them.
fn tensor_members<'a>(exploded: &'a Exploded, key: &str) -> HashMap<String, &'a [u8]> {
    let prefix = format!("{key}.");
    exploded
        .members
        .iter()
        .filter(|(k, _)| k.starts_with(&prefix) || k.starts_with("shared/"))
        .map(|(k, v)| (k.clone(), v.as_slice()))
        .collect()
}

/// `implode(explode(blob)) == blob`, the manifest has the layout's form,
/// tensors come in body order, and each tensor imploded on its own (as the
/// exploded stores do) decodes like the full file.
fn assert_transcodes(blob: &[u8], what: &str) {
    let reader = ContainerReader::open_partial(blob).unwrap();
    let exploded = explode(blob).unwrap_or_else(|e| panic!("{what}: explode: {e}"));
    let rebuilt = implode(
        &exploded.registry_json,
        &manifests_of(&exploded),
        members_of(&exploded),
    )
    .unwrap_or_else(|e| panic!("{what}: implode: {e}"));
    assert!(rebuilt == blob, "{what}: implode(explode(x)) != x");

    let reg: serde_json::Value = serde_json::from_str(&exploded.registry_json).unwrap();
    match reader.layout {
        ContainerLayout::Compact => {
            assert_eq!(reg["ptwm_format"], "ptwm-manifest-c", "{what}");
            assert_eq!(reg["manifest_version"], 2, "{what}");
            assert_eq!(reg["container_flags"]["layout"], "compact", "{what}");
            assert!(reg.get("registry_hash_blake3").is_none(), "{what}");
            let table = reg["registry"]["extension_table"].as_array().unwrap();
            assert_eq!(table.len(), reader.extension_table.entries.len(), "{what}");
            let chains = reg["registry"]["chains"].as_array().unwrap();
            assert_eq!(chains.len(), reader.chain_registry.chains.len(), "{what}");
            for (pos, c) in chains.iter().enumerate() {
                assert_eq!(c["id"], pos, "{what}: chains in registry order");
            }
        }
        ContainerLayout::Legacy => {
            assert_eq!(reg["ptwm_format"], "ptwm-manifest", "{what}");
            assert_eq!(reg["manifest_version"], 1, "{what}");
        }
    }

    // Tensors in body order.
    let names: Vec<String> = reader
        .tensor_records()
        .unwrap()
        .into_iter()
        .map(|r| r.name)
        .collect();
    let exploded_names: Vec<&str> = exploded.tensors.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(exploded_names, names, "{what}: tensor order");

    // Per-tensor implode with the archive-wide registry.
    for t in &exploded.tensors {
        let single = implode(
            &exploded.registry_json,
            std::slice::from_ref(&t.manifest_json),
            tensor_members(&exploded, &t.key),
        )
        .unwrap_or_else(|e| panic!("{what}: implode {}: {e}", t.name));
        let one = ContainerReader::open_partial(&single).unwrap();
        assert_eq!(one.layout, reader.layout, "{what}");
        assert_eq!(one.tensor_names().unwrap(), vec![t.name.clone()], "{what}");
        match reader.decode_tensor(&t.name) {
            Ok(want) => assert_eq!(
                one.decode_tensor(&t.name).unwrap(),
                want,
                "{what}: subset implode of {}",
                t.name
            ),
            Err(_) => assert!(one.decode_tensor(&t.name).is_err(), "{what}: {}", t.name),
        }
    }
}

fn compact_feature_blob() -> Vec<u8> {
    compress(
        &feature_tensors(),
        options(true, true, None, ContainerLayout::Compact),
    )
}

#[test]
fn compact_manifest_lists_the_table_and_binds_sources() {
    let blob = compact_feature_blob();
    let reader = ContainerReader::open(&blob).unwrap();
    let exploded = explode(&blob).unwrap();
    let reg: serde_json::Value = serde_json::from_str(&exploded.registry_json).unwrap();
    assert_eq!(
        reg["container_flags"],
        serde_json::json!({"embedded_wasm": false, "layout": "compact"})
    );
    let keys: Vec<&String> = reg["registry"].as_object().unwrap().keys().collect();
    assert_eq!(keys, ["chains", "extension_table", "shared_state"]);

    // Every entry is a one-key object naming a built-in by wire id.
    let table = reg["registry"]["extension_table"].as_array().unwrap();
    for (i, entry) in table.iter().enumerate() {
        let obj = entry.as_object().unwrap();
        assert_eq!(obj.len(), 1, "{entry}");
        let (kind, value) = obj.iter().next().unwrap();
        let e = &reader.extension_table.entries[i];
        match kind.as_str() {
            "op" => {
                let op = OpId::from_u16(value.as_u64().unwrap() as u16).unwrap();
                assert_eq!(
                    e.human_label,
                    format!(
                        "io.ptwm.builtin.{}@{}",
                        op.builtin_name(),
                        env!("CARGO_PKG_VERSION")
                    )
                );
            }
            "codec" => assert!(value.is_u64()),
            other => panic!("unexpected entry kind {other} in {entry}"),
        }
    }

    // Registry chains with a bound Source carry params_b64 null.
    let chains = reg["registry"]["chains"].as_array().unwrap();
    let bound = chains
        .iter()
        .filter(|c| {
            c["nodes"]
                .as_array()
                .unwrap()
                .iter()
                .any(|n| n["op"] == OpId::Source.as_u16() && n["params_b64"].is_null())
        })
        .count();
    assert!(bound > 0, "no bound Source in {chains:?}");

    // Plane codec_table_idx names a codec entry of the listed table.
    for t in &exploded.tensors {
        let tm: serde_json::Value = serde_json::from_str(&t.manifest_json).unwrap();
        for p in tm["planes"].as_array().unwrap() {
            let idx = p["codec_table_idx"].as_u64().unwrap() as usize;
            assert_eq!(
                table[idx],
                serde_json::json!({"codec": p["codec"]}),
                "{}",
                t.name
            );
        }
    }
}

/// Rewrite `blob` in the compact layout with `extra` appended to its
/// Extension Table. Existing table indices stay valid.
fn with_extra_entries(blob: &[u8], extra: Vec<ExtensionTableEntry>) -> Vec<u8> {
    let reader = ContainerReader::open(blob).unwrap();
    let mut table = reader.extension_table.clone();
    table.entries.extend(extra);
    let mut out = Vec::new();
    let mut w =
        ContainerWriter::with_layout(Cursor::new(&mut out), ContainerLayout::Compact).unwrap();
    w.set_prelude(&reader.prelude).unwrap();
    w.set_compact_tables(table, reader.chain_registry.clone())
        .unwrap();
    for rec in reader.tensor_records().unwrap() {
        w.append_tensor(&rec).unwrap();
    }
    w.finalize().unwrap();
    out
}

fn third_party_entry() -> ExtensionTableEntry {
    ExtensionTableEntry {
        canonical_id: CanonicalId::derive(&[7; 32], "acme_codec", "0.1.0"),
        human_label: "com.acme.codec@0.1.0".into(),
        kind: Kind::PlaneCodec,
        abi_version: 1,
        lifecycle: Lifecycle::Thread,
        flavor_hints: ptwm_core::extension::table::FLAVOR_NATIVE,
        capabilities: CapabilityMap::new(),
        attestation: Attestation::PgpSignature(vec![1, 2, 3]),
        install_hint: Some("pip install acme".into()),
        embedded_wasm_offset: None,
        embedded_wasm_length: None,
    }
}

#[test]
fn compact_manifest_carries_named_and_third_party_entries() {
    let blob = compact_feature_blob();
    let reader = ContainerReader::open(&blob).unwrap();
    // A built-in without a wire id (stored by name).
    let named: Vec<ExtensionTableEntry> = builtin_entries()
        .into_iter()
        .filter(|e| {
            matches!(classify(e), Ok(CompactEntry::Named(_)))
                && !reader.extension_table.entries.contains(e)
        })
        .take(1)
        .collect();
    assert_eq!(named.len(), 1, "no built-in without a wire id");
    let mut extra = named.clone();
    extra.push(third_party_entry());
    let file = with_extra_entries(&blob, extra);

    let exploded = explode(&file).unwrap();
    let reg: serde_json::Value = serde_json::from_str(&exploded.registry_json).unwrap();
    let table = reg["registry"]["extension_table"].as_array().unwrap();
    let last = table.last().unwrap();
    assert!(last["full_b64"].is_string(), "{last}");
    assert!(table[table.len() - 2]["named"].is_string(), "{table:?}");
    let rebuilt = implode(
        &exploded.registry_json,
        &manifests_of(&exploded),
        members_of(&exploded),
    )
    .unwrap();
    assert!(rebuilt == file, "implode(explode(x)) != x");
}

/// Implode `registry` (a JSON value) with the tensors of `exploded`.
fn implode_with(
    exploded: &Exploded,
    registry: &serde_json::Value,
) -> Result<Vec<u8>, ptwm_core::error::PtwmCoreError> {
    implode(
        &registry.to_string(),
        &manifests_of(exploded),
        members_of(exploded),
    )
}

#[test]
fn implode_rejects_malformed_compact_manifests() {
    let blob = compact_feature_blob();
    let exploded = explode(&blob).unwrap();
    let reg: serde_json::Value = serde_json::from_str(&exploded.registry_json).unwrap();
    assert!(implode_with(&exploded, &reg).unwrap() == blob);

    let cases: Vec<(&str, Box<dyn Fn(&mut serde_json::Value)>, &str)> = vec![
        (
            "compact format, legacy version",
            Box::new(|r| r["manifest_version"] = 1.into()),
            "unsupported manifest",
        ),
        (
            "compact format, version 3",
            Box::new(|r| r["manifest_version"] = 3.into()),
            "unsupported manifest",
        ),
        (
            "legacy format, compact version",
            Box::new(|r| r["ptwm_format"] = "ptwm-manifest".into()),
            "unsupported manifest",
        ),
        (
            "legacy layout name",
            Box::new(|r| r["container_flags"]["layout"] = "legacy".into()),
            "container_flags.layout",
        ),
        (
            "missing layout",
            Box::new(|r| {
                r["container_flags"]
                    .as_object_mut()
                    .unwrap()
                    .remove("layout");
            }),
            "missing field",
        ),
        (
            "embedded WASM",
            Box::new(|r| r["container_flags"]["embedded_wasm"] = true.into()),
            "embedded WASM",
        ),
        (
            "legacy registry hash",
            Box::new(|r| r["registry_hash_blake3"] = "00".repeat(32).into()),
            "unknown field",
        ),
        (
            "legacy op list",
            Box::new(|r| r["registry"]["ops"] = serde_json::json!([])),
            "unknown field",
        ),
        (
            "unknown entry kind",
            Box::new(|r| r["registry"]["extension_table"][0] = serde_json::json!({"wasm": 1})),
            "unknown variant",
        ),
        (
            "two keys in an entry",
            Box::new(|r| {
                r["registry"]["extension_table"][0] = serde_json::json!({"op": 1, "codec": 1})
            }),
            "parse registry manifest",
        ),
        (
            "unknown op id",
            Box::new(|r| r["registry"]["extension_table"][0] = serde_json::json!({"op": 0x7FFF})),
            "unknown op id",
        ),
        (
            "unknown built-in name",
            Box::new(|r| {
                r["registry"]["extension_table"][0] = serde_json::json!({"named": "no_such"})
            }),
            "unknown built-in name",
        ),
        (
            "duplicate entry",
            Box::new(|r| {
                let first = r["registry"]["extension_table"][0].clone();
                r["registry"]["extension_table"]
                    .as_array_mut()
                    .unwrap()
                    .push(first);
            }),
            "duplicate entry",
        ),
        (
            "chain id not its position",
            Box::new(|r| r["registry"]["chains"][0]["id"] = 7.into()),
            "compact chain ids are positions",
        ),
    ];
    for (what, mutate, needle) in cases {
        let mut bad = reg.clone();
        mutate(&mut bad);
        let e = implode_with(&exploded, &bad).unwrap_err().to_string();
        assert!(e.contains(needle), "{what}: {e}");
    }

    // A built-in stored as a full entry is not canonical.
    let mut bad = reg.clone();
    let zstd = builtin_registry()
        .entry_by_name("zstd")
        .unwrap()
        .to_bytes()
        .unwrap();
    bad["registry"]["extension_table"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::json!({ "full_b64": b64(&zstd) }));
    let e = implode_with(&exploded, &bad).unwrap_err().to_string();
    assert!(e.contains("built-in id"), "{e}");

    // codec is informational, but must agree with the table entry that
    // codec_table_idx names.
    let mut tm: serde_json::Value =
        serde_json::from_str(&exploded.tensors[0].manifest_json).unwrap();
    tm["planes"][0]["codec"] = 0x7FFE.into();
    let e = implode(
        &exploded.registry_json,
        &[tm.to_string()],
        tensor_members(&exploded, &exploded.tensors[0].key),
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("codec"), "{e}");
    let mut tm: serde_json::Value =
        serde_json::from_str(&exploded.tensors[0].manifest_json).unwrap();
    let other = if tm["planes"][0]["codec"] == CodecId::Zstd.as_u16() {
        CodecId::Identity
    } else {
        CodecId::Zstd
    };
    tm["planes"][0]["codec"] = other.as_u16().into();
    let e = implode(
        &exploded.registry_json,
        &[tm.to_string()],
        tensor_members(&exploded, &exploded.tensors[0].key),
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("codec"), "{e}");
}

/// Standard base64 with padding, for building manifests by hand.
fn b64(bytes: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(A[(n >> (18 - 6 * i)) as usize & 63] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// The registry-manifest gate of the v1.1 `implode`, frozen: the v1.1
/// manifest struct (no `deny_unknown_fields`) and its `ptwm_format` check.
fn v1_1_implode_gate(registry_json: &str) -> Result<(), String> {
    #[derive(serde::Deserialize)]
    #[allow(dead_code)]
    struct Flags {
        embedded_wasm: bool,
    }
    #[derive(serde::Deserialize)]
    #[allow(dead_code)]
    struct Registry {
        ops: Vec<serde_json::Value>,
        codecs: Vec<serde_json::Value>,
        shared_state: Vec<serde_json::Value>,
        chains: Vec<serde_json::Value>,
    }
    #[derive(serde::Deserialize)]
    #[allow(dead_code)]
    struct Manifest {
        ptwm_format: String,
        manifest_version: u32,
        container_flags: Flags,
        registry_hash_blake3: String,
        registry: Registry,
    }
    let m: Manifest = serde_json::from_str(registry_json).map_err(|e| e.to_string())?;
    if m.ptwm_format != "ptwm-manifest" {
        return Err(format!("unexpected ptwm_format {:?}", m.ptwm_format));
    }
    Ok(())
}

/// The gate of an implode that accepts only `("ptwm-manifest", 1)`.
fn legacy_only_implode_gate(registry_json: &str) -> Result<(), String> {
    let v: serde_json::Value = serde_json::from_str(registry_json).map_err(|e| e.to_string())?;
    if v["ptwm_format"] != "ptwm-manifest" || v["manifest_version"] != 1 {
        return Err(format!(
            "unsupported manifest {} version {}",
            v["ptwm_format"], v["manifest_version"]
        ));
    }
    Ok(())
}

#[test]
fn old_implode_rejects_compact_manifests() {
    let data = feature_tensors();
    let legacy = compress(&data, options(true, true, None, ContainerLayout::Legacy));
    let legacy_reg = explode(&legacy).unwrap().registry_json;
    let compact_reg = explode(&compact_feature_blob()).unwrap().registry_json;

    for gate in [v1_1_implode_gate, legacy_only_implode_gate] {
        assert_eq!(gate(&legacy_reg), Ok(()));
        let e = gate(&compact_reg).unwrap_err();
        assert!(!e.is_empty());
    }
    // The v1.1 gate fails on the manifest structure before the format check.
    let e = v1_1_implode_gate(&compact_reg).unwrap_err();
    assert!(e.contains("missing field"), "{e}");
}

// ── read_extension_table and tensor_names (synthesis section 3, E3 / E14) ──

#[test]
fn read_extension_table_dispatches_on_the_magic() {
    use ptwm_core::container::read_extension_table;

    for name in FIXTURES {
        let legacy = load(name);
        let compact = to_compact(&legacy);
        for blob in [&legacy, &compact] {
            let reader = ContainerReader::open(blob).unwrap();
            let table = read_extension_table(blob).unwrap();
            assert_eq!(table.entries, reader.extension_table.entries, "{name}");
        }
    }

    let data = feature_tensors();
    let compact = compress(&data, options(true, true, None, ContainerLayout::Compact));
    let table = read_extension_table(&compact).unwrap();
    assert!(!table.entries.is_empty());
    assert!(
        table
            .entries
            .iter()
            .all(|e| ptwm_core::extension::is_builtin(&e.canonical_id))
    );

    assert!(read_extension_table(b"not a container").is_err());
    assert!(read_extension_table(b"").is_err());
}

#[test]
fn read_extension_table_checks_the_legacy_region_and_hash() {
    use ptwm_core::container::read_extension_table;

    let legacy = load("legacy_v1_1_0.ptwm");
    let header = Header::from_bytes(&legacy[..HEADER_LEN]).unwrap();
    assert!(header.extension_table_length > 0);

    let patched = |h: &Header| {
        let mut out = legacy.clone();
        out[..HEADER_LEN].copy_from_slice(&h.to_bytes().unwrap());
        out
    };

    // offset + length overflows u64 / usize.
    let mut h = header;
    h.extension_table_offset = u64::MAX - 4;
    let err = read_extension_table(&patched(&h)).unwrap_err().to_string();
    assert!(err.contains("exceeds the buffer"), "{err}");

    let mut h = header;
    h.extension_table_length = u64::MAX;
    let err = read_extension_table(&patched(&h)).unwrap_err().to_string();
    assert!(err.contains("exceeds the buffer"), "{err}");

    // A region that runs past the end of the file.
    let mut h = header;
    h.extension_table_length = legacy.len() as u64;
    assert!(read_extension_table(&patched(&h)).is_err());

    // A table byte that no longer matches the header hash.
    let mut corrupt = legacy.clone();
    corrupt[header.extension_table_offset as usize + 2] ^= 0x01;
    let err = read_extension_table(&corrupt).unwrap_err();
    assert!(
        matches!(err, ptwm_core::PtwmCoreError::ExtensionTableHashMismatch),
        "{err}"
    );
}

#[test]
fn read_extension_table_checks_the_compact_meta_hash() {
    use ptwm_core::container::read_extension_table;

    let compact = to_compact(&load("legacy_v1_1_0.ptwm"));
    // meta_xxh64 covers the Extension Table: flip its first byte.
    let sec = ptwm_core::ptwc::locate_sections(&compact).unwrap();
    let mut corrupt = compact.clone();
    corrupt[sec.extension_table.start] ^= 0x01;
    assert!(read_extension_table(&corrupt).is_err());
}

#[test]
fn tensor_names_and_decode_model_come_in_write_order_in_both_layouts() {
    let mut rng = Rng(7);
    let names = ["zeta", "alpha", "mid", "beta", "omega", "a.b.c"];
    let data: Vec<Tensor> = names
        .iter()
        .map(|n| fp32(n, vec![8, 4], &mut rng))
        .collect();
    for layout in [ContainerLayout::Legacy, ContainerLayout::Compact] {
        let blob = compress(&data, options(true, true, None, layout));
        let reader = ContainerReader::open(&blob).unwrap();
        assert_eq!(reader.tensor_names().unwrap(), names, "{layout:?}");
        let decoded: Vec<String> = reader
            .decode_model()
            .unwrap()
            .into_iter()
            .map(|(name, _)| name)
            .collect();
        assert_eq!(decoded, names, "{layout:?}");
    }
}
