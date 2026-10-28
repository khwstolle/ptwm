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
use ptwm_core::compressor::{CompressorOptions, InputTensor, compress_model};
use ptwm_core::container::{ContainerLayout, ContainerReader, ContainerWriter};
use ptwm_core::header::{HEADER_LEN, Header, MAGIC_PTWC};
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

#[test]
fn legacy_layout_stays_the_default() {
    let data = feature_tensors();
    let default = compress(&data, options(true, true, None, ContainerLayout::default()));
    let legacy = compress(&data, options(true, true, None, ContainerLayout::Legacy));
    assert!(default == legacy);
    assert_eq!(default[..5], *b"\x89PTWM");
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
