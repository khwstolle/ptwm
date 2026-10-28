//! Golden legacy `.ptwm` fixtures.
//!
//! The files in `tests/fixtures/` pin the legacy (`\x89PTWM`) wire format as
//! written by released and current writers:
//!
//! | file | content |
//! |---|---|
//! | `legacy_v1_1_0.ptwm` | Multi-tensor container with the v1.1.0 Extension Table layout (every built-in codec interned, `ContainerWriter::set_chain_registry`). Holds bf16, fp16, fp32, fp8 and int8 tensors, two scale tensors that share an `Order1ScaleAC` prelude state, a chunked bf16 tensor, an fp8 tensor with an inline chain, and a tensor with a dependency. Built by [`rewrite_as_v1_1`] from current `compress_model` output; built-in ids and labels are derived for version `1.1.0`. Byte-identical to what this generator writes when built against the v1.1.0 release, whose `compress_model` emits the same layout. |
//! | `legacy_v1_0_0.ptwm` | The v1.1.0 fixture with every built-in id and label re-derived for version `1.0.0` and the Extension Table hash fixed up. It is not v1.0.0 writer output: for these inputs the v1.0.0 `compress_model` writes no shared `Order1ScaleAC` prelude state. It covers v1.0.0 ids on a file that has the v1.1.0 features. |
//! | `legacy_v1_0_0_native.ptwm` | Written once by the v1.0.0 release (commit `a17d403`): this generator, built against that tree, run as `rewrite_as_v1_1(&[main, chunked], INLINE_TENSOR)`. Real v1.0.0 writer output, with v1.0.0 built-in ids. `regenerate_fixtures` does not rewrite it. |
//! | `legacy_trimmed.ptwm` | `compress_model` output of the current writer, whose Extension Table holds only the codecs the planes use. Same tensors as the v1.1.0 file minus the two scale tensors and the chunked tensor. |
//! | `legacy_p4_byte_fp32_v1_1_0.ptwm` | An fp32 tensor from `Format.BYTE` input, with the chain built the way the v1.1 Python path builds it: the Source node declares `[n_bytes]` fp32 elements, four times the real size. v1.1.0 layout (full Extension Table), built-in ids for `1.1.0`. This is the layout of P4 files written by v1.1.x; it is byte-identical to the v1.1.0 `compress_model` output for the same input. |
//! | `legacy_p4_byte_fp32.ptwm` | The same tensor written by the current writer (trimmed Extension Table). |
//!
//! The tests check that every fixture decodes to the expected bytes and that
//! `implode(explode(x)) == x`. The expected bytes are stored as
//! `(length, xxh64)` pairs in [`EXPECTED`]. The reader maps built-in ids
//! of other releases to the crate version
//! (`ptwm_core::extension::normalize_builtin_ids`), so fixtures with
//! v1.0.0 ids decode and transcode as they are.
//!
//! # Regenerating
//!
//! ```text
//! PTWM_REGEN_FIXTURES=1 cargo test -p ptwm-core --test legacy_fixtures \
//!     -- --ignored regenerate_fixtures --nocapture
//! ```
//!
//! This rewrites the four files and prints the `EXPECTED` table and the
//! golden hash for [`LEGACY_WRITER_GOLDEN_BLAKE3`]. Regenerate only when a
//! format change is intended: the point of the fixtures is that they do not
//! change. A crate version bump changes the built-in ids that the current
//! writer emits, so it changes `legacy_trimmed.ptwm`,
//! `legacy_p4_byte_fp32.ptwm` and the golden hash; the fixtures pinned to
//! v1.1.0 and v1.0.0 do not change.
//!
//! [`rederive_builtins`] rewrites the version inside the built-in labels in
//! place, so it only maps between versions whose strings have the same
//! length (`1.1.0` and `1.0.0`, but not `1.10.0`). A bump to a version of
//! another length must extend it first.

use std::collections::HashMap;
use std::io::Cursor;
use std::path::PathBuf;

use ptwm_core::chain::{Chain, ChainEdge, ChainNode, TerminalRef};
use ptwm_core::compressor::{CompressorOptions, InputTensor, compress_model};
use ptwm_core::container::{ChainRegistry, ContainerReader, ContainerWriter};
use ptwm_core::extension::{BUILTIN_PUBKEY, CanonicalId};
use ptwm_core::tensor_record::{CHAIN_REF_INLINE_BIT, TensorRecord, parse_tensor_record};
use ptwm_core::transcode::{explode, implode};
use ptwm_core::transforms::op::OpId;
use ptwm_core::types::{NibbleKind, Role, ScaleFormat, ValueFormat};
use xxhash_rust::xxh64::xxh64;

const V1_1_0: &str = "legacy_v1_1_0.ptwm";
const V1_0_0: &str = "legacy_v1_0_0.ptwm";
const TRIMMED: &str = "legacy_trimmed.ptwm";
const V1_0_0_NATIVE: &str = "legacy_v1_0_0_native.ptwm";
const P4: &str = "legacy_p4_byte_fp32.ptwm";
const P4_V1_1_0: &str = "legacy_p4_byte_fp32_v1_1_0.ptwm";

/// Expected decoded bytes per fixture: `(tensor name, length, xxh64 seed 0)`.
/// Printed by `regenerate_fixtures`.
const EXPECTED: &[(&str, &[(&str, usize, u64)])] = &[
    // EXPECTED-BEGIN
    (
        V1_1_0,
        &[
            ("model.layers.0.q.weight", 512, 0xc82d5962a46d19ff),
            ("model.layers.0.k.weight", 512, 0x73173bba2f17ef23),
            ("model.norm.weight", 512, 0x946ca638752e3bfc),
            ("model.layers.0.mlp.fp8.weight", 256, 0x84e10734e3fb3518),
            ("model.embed.int8", 256, 0xf6193be8f075d386),
            ("model.layers.0.q.scale", 2048, 0x95a622e512233480),
            ("model.layers.0.k.scale", 2048, 0xda44a3268d62821d),
            ("model.layers.1.q.weight", 256, 0x0d7f3dd0787bd569),
            ("model.layers.2.chunked.weight", 1024, 0xf9e05338b3a99b14),
        ],
    ),
    (
        V1_0_0,
        &[
            ("model.layers.0.q.weight", 512, 0xc82d5962a46d19ff),
            ("model.layers.0.k.weight", 512, 0x73173bba2f17ef23),
            ("model.norm.weight", 512, 0x946ca638752e3bfc),
            ("model.layers.0.mlp.fp8.weight", 256, 0x84e10734e3fb3518),
            ("model.embed.int8", 256, 0xf6193be8f075d386),
            ("model.layers.0.q.scale", 2048, 0x95a622e512233480),
            ("model.layers.0.k.scale", 2048, 0xda44a3268d62821d),
            ("model.layers.1.q.weight", 256, 0x0d7f3dd0787bd569),
            ("model.layers.2.chunked.weight", 1024, 0xf9e05338b3a99b14),
        ],
    ),
    (
        V1_0_0_NATIVE,
        &[
            ("model.layers.0.q.weight", 512, 0xc82d5962a46d19ff),
            ("model.layers.0.k.weight", 512, 0x73173bba2f17ef23),
            ("model.norm.weight", 512, 0x946ca638752e3bfc),
            ("model.layers.0.mlp.fp8.weight", 256, 0x84e10734e3fb3518),
            ("model.embed.int8", 256, 0xf6193be8f075d386),
            ("model.layers.0.q.scale", 2048, 0x95a622e512233480),
            ("model.layers.0.k.scale", 2048, 0xda44a3268d62821d),
            ("model.layers.1.q.weight", 256, 0x0d7f3dd0787bd569),
            ("model.layers.2.chunked.weight", 1024, 0xf9e05338b3a99b14),
        ],
    ),
    (
        TRIMMED,
        &[
            ("model.layers.0.q.weight", 512, 0xc82d5962a46d19ff),
            ("model.layers.0.k.weight", 512, 0x73173bba2f17ef23),
            ("model.norm.weight", 512, 0x946ca638752e3bfc),
            ("model.layers.0.mlp.fp8.weight", 256, 0x84e10734e3fb3518),
            ("model.embed.int8", 256, 0xf6193be8f075d386),
            ("model.layers.1.q.weight", 256, 0x0d7f3dd0787bd569),
        ],
    ),
    (P4, &[("p4.byte.fp32", 512, 0x57e3af5eb7d32ec1)]),
    (P4_V1_1_0, &[("p4.byte.fp32", 512, 0x57e3af5eb7d32ec1)]),
    // EXPECTED-END
];

/// blake3 of `compress_model` output for [`tensors`] with the options of
/// [`main_options`] (see [`main_container`]). Printed by `regenerate_fixtures`.
const LEGACY_WRITER_GOLDEN_BLAKE3: &str =
    "b74c3fd1d81d5bbf4703fb55fb6427781d9aa6d2a92a44127cf4dc91ee4b9791";

// ── Inputs ───────────────────────────────────────────────────────────────────

/// Deterministic SplitMix64 byte stream.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Roughly normal value in [-1, 1] (sum of uniforms).
    fn weight(&mut self) -> f32 {
        let mut s = 0.0f32;
        for _ in 0..4 {
            s += (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        }
        (s - 2.0) * 0.05
    }
}

fn f32_to_bf16_bits(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

fn f32_to_f16_bits(v: f32) -> u16 {
    // Values are small and normal or zero; truncate the mantissa.
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xFF) as i32 - 127 + 15;
    if exp <= 0 {
        return sign;
    }
    sign | ((exp as u16) << 10) | ((b >> 13) & 0x3FF) as u16
}

struct TensorData {
    name: &'static str,
    dtype_code: u16,
    dtype_name: &'static str,
    shape: Vec<u64>,
    bytes: Vec<u8>,
}

fn bf16_tensor(name: &'static str, rng: &mut Rng, shape: Vec<u64>) -> TensorData {
    let n: u64 = shape.iter().product();
    let bytes = (0..n)
        .flat_map(|_| f32_to_bf16_bits(rng.weight()).to_le_bytes())
        .collect();
    TensorData {
        name,
        dtype_code: 6,
        dtype_name: "bfloat16",
        shape,
        bytes,
    }
}

fn tensors() -> Vec<TensorData> {
    let mut rng = Rng(0x5054_574D_4649_5854);
    let mut out = vec![bf16_tensor(
        "model.layers.0.q.weight",
        &mut rng,
        vec![8, 32],
    )];

    let n = 8 * 32;
    out.push(TensorData {
        name: "model.layers.0.k.weight",
        dtype_code: 4,
        dtype_name: "float16",
        shape: vec![8, 32],
        bytes: (0..n)
            .flat_map(|_| f32_to_f16_bits(rng.weight()).to_le_bytes())
            .collect(),
    });

    let n = 4 * 32;
    out.push(TensorData {
        name: "model.norm.weight",
        dtype_code: 1,
        dtype_name: "float32",
        shape: vec![4, 32],
        bytes: (0..n)
            .flat_map(|_| (1.0 + rng.weight()).to_le_bytes())
            .collect(),
    });

    // fp8 e4m3: small exponents, random sign and mantissa.
    out.push(TensorData {
        name: "model.layers.0.mlp.fp8.weight",
        dtype_code: 29,
        dtype_name: "float8_e4m3fn",
        shape: vec![8, 32],
        bytes: (0..256)
            .map(|_| {
                let r = rng.next_u64();
                let exp = 3 + (r % 5) as u8;
                ((r >> 8) as u8 & 0x80) | (exp << 3) | ((r >> 16) as u8 & 0x07)
            })
            .collect(),
    });

    out.push(TensorData {
        name: "model.embed.int8",
        dtype_code: 17,
        dtype_name: "int8",
        shape: vec![256],
        bytes: (0..256)
            .map(|_| ((rng.next_u64() % 9) as i8 - 4) as u8)
            .collect(),
    });

    // Two auto-correlated scale planes so the shared Order1ScaleAC state wins.
    for name in ["model.layers.0.q.scale", "model.layers.0.k.scale"] {
        let mut bytes = Vec::with_capacity(2048);
        for _ in 0..64 {
            let mut prev: u8 = 0x80;
            for _ in 0..32 {
                let next = prev
                    .wrapping_add((rng.next_u64() % 3) as u8)
                    .wrapping_sub(1);
                bytes.push(next);
                prev = next;
            }
        }
        out.push(TensorData {
            name,
            dtype_code: 17,
            dtype_name: "int8",
            shape: vec![64, 32],
            bytes,
        });
    }

    // Delta residual against an external safetensors reference.
    let mut residual = bf16_tensor("model.layers.1.q.weight", &mut rng, vec![4, 32]);
    residual.bytes.iter_mut().step_by(3).for_each(|b| *b = 0);
    out.push(residual);
    out
}

/// Tensor name that carries an `ExternalSafetensors` dependency.
const DELTA_TENSOR: &str = "model.layers.1.q.weight";
/// Tensor whose chain the v1.1.0 fixture stores inline.
const INLINE_TENSOR: &str = "model.layers.0.mlp.fp8.weight";

fn chunked_tensors() -> Vec<TensorData> {
    let mut rng = Rng(0x4348_554E_4B45_4421);
    vec![bf16_tensor(
        "model.layers.2.chunked.weight",
        &mut rng,
        vec![16, 32],
    )]
}

// ── Chains (mirrors of python/ptwm/preprocessing/_chains.py) ────────────────

/// Source params: ndim u8, dims u32 LE, chain-internal dtype u16 LE.
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

fn node(op: OpId, params: Vec<u8>) -> ChainNode {
    ChainNode { op, params }
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

/// `_bf16_split_chain`: Source → BitReorderIeee16 → ByteSplit(2). The Python
/// builder uses the bf16 Source code for fp16 too.
fn ieee16_split_chain(shape: &[u64]) -> Chain {
    Chain {
        nodes: vec![
            source(shape, 0x000F),
            node(OpId::BitReorderIeee16, vec![]),
            node(OpId::ByteSplit, vec![2]),
        ],
        edges: vec![edge(0, 1, None), edge(1, 2, None)],
        terminals: vec![
            term(2, 0, Role::MantissaByte { index: 0, of: 2 }),
            term(2, 1, Role::ExponentByte),
        ],
    }
}

/// `_fp32_split_chain`: Source → BitReorderIeee32 → ByteSplit(4).
fn fp32_split_chain(shape: &[u64]) -> Chain {
    Chain {
        nodes: vec![
            source(shape, 0x0003),
            node(OpId::BitReorderIeee32, vec![]),
            node(OpId::ByteSplit, vec![4]),
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

/// `_fp8e4m3_nibble_chain`: Source → BitReorderFp8E4M3 → NibbleSplit.
fn fp8_nibble_chain(shape: &[u64]) -> Chain {
    Chain {
        nodes: vec![
            source(shape, 0x0010),
            node(OpId::BitReorderFp8E4M3, vec![]),
            node(OpId::NibbleSplit, vec![]),
        ],
        edges: vec![edge(0, 1, None), edge(1, 2, None)],
        terminals: vec![
            term(
                2,
                0,
                Role::Nibble {
                    kind: NibbleKind::Exponent,
                },
            ),
            term(
                2,
                1,
                Role::Nibble {
                    kind: NibbleKind::SignMantissa,
                },
            ),
        ],
    }
}

/// `_u8_byte_passthrough_value_chain`: Source(u8) → BytePassthrough.
fn byte_passthrough_chain(n_bytes: u64) -> Chain {
    Chain {
        nodes: vec![
            source(&[n_bytes], 0x0006),
            node(OpId::BytePassthrough, vec![]),
        ],
        edges: vec![edge(0, 1, None)],
        terminals: vec![term(
            1,
            0,
            Role::Value {
                format: ValueFormat::IntN { bits: 8 },
            },
        )],
    }
}

/// Byte passthrough re-tagged as an E8M0 scale plane (routes to
/// `Order1ScaleAC` together with the Rows layout from the tensor shape).
fn scale_passthrough_chain(n_bytes: u64) -> Chain {
    let scale = Role::Scale {
        format: ScaleFormat::E8M0,
    };
    Chain {
        nodes: vec![
            source(&[n_bytes], 0x0006),
            node(OpId::BytePassthrough, vec![]),
        ],
        edges: vec![edge(0, 1, Some(scale.clone()))],
        terminals: vec![term(1, 0, scale)],
    }
}

fn chain_for(t: &TensorData) -> Chain {
    match t.dtype_code {
        4 | 6 => ieee16_split_chain(&t.shape),
        1 => fp32_split_chain(&t.shape),
        29 => fp8_nibble_chain(&t.shape),
        17 if t.name.ends_with(".scale") => scale_passthrough_chain(t.bytes.len() as u64),
        17 => byte_passthrough_chain(t.bytes.len() as u64),
        other => panic!("no chain for dtype {other}"),
    }
}

fn inputs(data: &[TensorData]) -> Vec<InputTensor<'_>> {
    data.iter()
        .map(|t| InputTensor {
            name: t.name.to_string(),
            candidate_chains: vec![chain_for(t)],
            dtype_code: t.dtype_code,
            // Format.TORCH
            input_format: 2,
            raw_bytes: &t.bytes,
            shape: Some(t.shape.clone()),
            dtype_name: Some(t.dtype_name.to_string()),
            delta_reference_blake3: (t.name == DELTA_TENSOR)
                .then(|| *blake3::hash(b"reference.safetensors").as_bytes()),
        })
        .collect()
}

/// Options of the Python default (`emit_payload_hash`, `emit_plane_crc`).
fn main_options(chunk_size: Option<u32>) -> CompressorOptions {
    CompressorOptions {
        method_hint: 1,
        emit_payload_hash: true,
        emit_plane_crc: true,
        forced_codec: None,
        allow_codec_ids: None,
        chunk_size,
        ..Default::default()
    }
}

fn compress(inputs: &[InputTensor<'_>], opts: CompressorOptions) -> Vec<u8> {
    let mut cursor = Cursor::new(Vec::new());
    compress_model(&mut cursor, inputs, opts).expect("compress_model");
    cursor.into_inner()
}

/// The current legacy writer's output for the main inputs.
fn main_container() -> Vec<u8> {
    let data = tensors();
    compress(&inputs(&data), main_options(None))
}

/// Main inputs without the scale tensors. Their shared `Order1ScaleAC`
/// state is about 15 KB; leaving it out keeps `legacy_trimmed.ptwm` small.
/// The v1.1.0 fixture and [`main_container`] cover the shared state.
fn trimmed_tensors() -> Vec<TensorData> {
    tensors()
        .into_iter()
        .filter(|t| !t.name.ends_with(".scale"))
        .collect()
}

/// The current legacy writer's output for [`trimmed_tensors`].
fn trimmed_container() -> Vec<u8> {
    let data = trimmed_tensors();
    compress(&inputs(&data), main_options(None))
}

// ── v1.1.0 layout rewrite ────────────────────────────────────────────────────

/// Records of `blob` in file (offset) order.
fn records_in_file_order(reader: &ContainerReader<'_>) -> Vec<TensorRecord> {
    let mut index: Vec<_> = reader.index.iter().collect();
    index.sort_by_key(|e| e.tensor_offset);
    index
        .iter()
        .map(|e| {
            let start = e.tensor_offset as usize;
            let end = start + e.tensor_len as usize;
            parse_tensor_record(&reader.buf[start..end]).unwrap().0
        })
        .collect()
}

/// Rewrite trimmed-table containers into one container with the v1.1.0
/// table layout (`set_chain_registry`, every built-in codec interned).
///
/// Records keep their order: first all records of `parts[0]`, then of
/// `parts[1]`, and so on. Chains are merged and deduplicated; prelude
/// entries are taken from the parts unchanged (ids must not clash). The
/// record of `inline_tensor` gets its chain stored inline.
fn rewrite_as_v1_1(parts: &[&[u8]], inline_tensor: &str) -> Vec<u8> {
    let readers: Vec<ContainerReader<'_>> = parts
        .iter()
        .map(|b| ContainerReader::open(b).expect("open part"))
        .collect();

    let mut prelude = Vec::new();
    let mut chains: Vec<Chain> = Vec::new();
    let mut records: Vec<(TensorRecord, usize)> = Vec::new();
    for (part, reader) in readers.iter().enumerate() {
        for entry in &reader.prelude {
            assert!(
                prelude
                    .iter()
                    .all(|p: &ptwm_core::prelude::PreludeEntry| p.shared_state_id
                        != entry.shared_state_id),
                "shared state id clash"
            );
            prelude.push(entry.clone());
        }
        for mut rec in records_in_file_order(reader) {
            assert_eq!(rec.chain_ref & CHAIN_REF_INLINE_BIT, 0);
            let chain = reader.chain_registry.lookup(rec.chain_ref).unwrap().clone();
            if rec.name == inline_tensor {
                rec.chain_ref = CHAIN_REF_INLINE_BIT;
                rec.inline_chain = Some(chain);
            } else {
                let idx = match chains.iter().position(|c| *c == chain) {
                    Some(i) => i,
                    None => {
                        chains.push(chain);
                        chains.len() - 1
                    }
                };
                rec.chain_ref = idx as u16;
            }
            records.push((rec, part));
        }
    }

    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer = ContainerWriter::new(&mut cursor).unwrap();
        writer.set_prelude(&prelude).unwrap();
        writer
            .set_chain_registry(ChainRegistry {
                chains: chains
                    .into_iter()
                    .enumerate()
                    .map(|(i, c)| (i as u16, c))
                    .collect(),
            })
            .unwrap();
        for (mut rec, part) in records {
            let table = &readers[part].extension_table;
            for plane in &mut rec.terminals {
                let id = table.entries[plane.codec_table_idx as usize].canonical_id;
                plane.codec_table_idx = writer.codec_table_idx_for(&id).unwrap();
            }
            writer.append_tensor(&rec).unwrap();
        }
        writer.finalize().unwrap();
    }
    cursor.into_inner()
}

/// Rewrite every built-in Extension Table entry (file-level and inline-chain
/// tables) from version `from` to version `to`: the canonical id is
/// re-derived and the `io.ptwm.builtin.<name>@<ver>` label is changed. The
/// file-level table hash in the header is recomputed. Returns the number of
/// entries rewritten.
///
/// The header is parsed by hand, not with `ContainerReader`, because the
/// reader rejects built-in ids of other versions. The version is replaced
/// in place, so `from` and `to` must have the same length.
fn rederive_builtins(blob: &mut [u8], from: &str, to: &str) -> usize {
    assert_eq!(
        from.len(),
        to.len(),
        "rederive_builtins only maps between versions of equal length"
    );
    if from == to {
        return 0;
    }
    // Header: magic[9] flags u16 | et_offset u64 | et_length u64 | et_hash [32].
    let u64_at = |off: usize| u64::from_le_bytes(blob[off..off + 8].try_into().unwrap());
    let et_off = u64_at(9 + 2) as usize;
    let et_len = u64_at(9 + 2 + 8) as usize;
    let hash_off = 9 + 2 + 8 + 8;
    assert_eq!(
        &blob[hash_off..hash_off + 32],
        blake3::hash(&blob[et_off..et_off + et_len]).as_bytes(),
        "Extension Table hash"
    );

    let prefix = b"io.ptwm.builtin.";
    let mut count = 0;
    let mut p = 0;
    while let Some(pos) = find(&blob[p..], prefix) {
        let label_start = p + pos;
        // Entry layout: canonical_id [32] | label_len u16 | label.
        let label_len = u16::from_le_bytes([blob[label_start - 2], blob[label_start - 1]]) as usize;
        let label = std::str::from_utf8(&blob[label_start..label_start + label_len])
            .unwrap()
            .to_string();
        let (name, ver) = label[prefix.len()..].split_once('@').unwrap();
        assert_eq!(ver, from, "unexpected builtin version in {label}");
        let id_start = label_start - 34;
        let old = CanonicalId::derive(&BUILTIN_PUBKEY, name, from);
        assert_eq!(&blob[id_start..id_start + 32], old.as_bytes(), "{label}");
        let new = CanonicalId::derive(&BUILTIN_PUBKEY, name, to);
        blob[id_start..id_start + 32].copy_from_slice(new.as_bytes());
        let ver_start = label_start + label_len - from.len();
        blob[ver_start..label_start + label_len].copy_from_slice(to.as_bytes());
        count += 1;
        p = label_start + label_len;
    }

    let hash = blake3::hash(&blob[et_off..et_off + et_len]);
    blob[hash_off..hash_off + 32].copy_from_slice(hash.as_bytes());
    count
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// The v1.1.0 fixture: main and chunked containers merged and rewritten with
/// the v1.1.0 table layout, built-in ids pinned to version 1.1.0.
fn v1_1_0_container() -> Vec<u8> {
    let main = main_container();
    let chunked_data = chunked_tensors();
    let chunked = compress(&inputs(&chunked_data), main_options(Some(256)));
    let mut blob = rewrite_as_v1_1(&[&main, &chunked], INLINE_TENSOR);
    rederive_builtins(&mut blob, env!("CARGO_PKG_VERSION"), "1.1.0");
    blob
}

fn p4_bytes() -> Vec<u8> {
    let mut rng = Rng(0x0050_3442_5954_4521);
    (0..128).flat_map(|_| rng.weight().to_le_bytes()).collect()
}

/// The P4 fixture: fp32 from `Format.BYTE` input. The v1.1 Python path
/// builds the chain with `chain_shape = [len(bytes)]`, so the fp32 Source
/// declares four times the real byte count.
///
/// The writer now skips chains whose Source does not declare the raw byte
/// count, so this compresses with the exact chain (`[len / 4]`) and then
/// rewrites the stored Source dimension to `len(bytes)`. The encoder never
/// read the Source size (it sizes the source plane from the raw bytes), and
/// the dimension is a fixed-width u32, so the result is byte-identical to
/// what v1.1 `compress_model` wrote for the overstated chain.
fn p4_container() -> Vec<u8> {
    let bytes = p4_bytes();
    let n = bytes.len() as u64;
    let input = InputTensor {
        name: "p4.byte.fp32".to_string(),
        candidate_chains: vec![fp32_split_chain(&[n / 4])],
        dtype_code: 1,
        // Format.BYTE
        input_format: 1,
        raw_bytes: &bytes,
        shape: None,
        dtype_name: None,
        delta_reference_blake3: None,
    };
    let mut blob = compress(&[input], main_options(None));
    let exact = fp32_split_chain(&[n / 4]).nodes[0].params.clone();
    let overstated = fp32_split_chain(&[n]).nodes[0].params.clone();
    assert_eq!(exact.len(), overstated.len());
    let at: Vec<usize> = blob
        .windows(exact.len())
        .enumerate()
        .filter(|(_, w)| *w == exact.as_slice())
        .map(|(i, _)| i)
        .collect();
    assert_eq!(at.len(), 1, "Source params must occur once in the P4 blob");
    blob[at[0]..at[0] + exact.len()].copy_from_slice(&overstated);
    blob
}

/// The P4 tensor in the v1.1.0 table layout, built-in ids pinned to 1.1.0.
fn p4_v1_1_0_container() -> Vec<u8> {
    let mut blob = rewrite_as_v1_1(&[&p4_container()], "");
    rederive_builtins(&mut blob, env!("CARGO_PKG_VERSION"), "1.1.0");
    blob
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn load(name: &str) -> Vec<u8> {
    let path = fixture_dir().join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn expected(fixture: &str) -> &'static [(&'static str, usize, u64)] {
    EXPECTED
        .iter()
        .find(|(f, _)| *f == fixture)
        .unwrap_or_else(|| panic!("no EXPECTED entry for {fixture}"))
        .1
}

/// `blob` with built-in ids of version `from` re-derived for the crate
/// version, so the current reader accepts it.
fn as_current_version(mut blob: Vec<u8>, from: &str) -> Vec<u8> {
    rederive_builtins(&mut blob, from, env!("CARGO_PKG_VERSION"));
    blob
}

fn assert_decodes(fixture: &str) {
    assert_blob_decodes(fixture, &load(fixture), expected(fixture));
}

fn assert_blob_decodes(fixture: &str, blob: &[u8], want: &[(&str, usize, u64)]) {
    let reader = ContainerReader::open(blob).unwrap_or_else(|e| panic!("{fixture}: open: {e}"));
    assert_eq!(reader.index.len(), want.len(), "{fixture}: tensor count");
    for (name, len, hash) in want {
        let got = reader
            .decode_tensor(name)
            .unwrap_or_else(|e| panic!("{fixture}: decode {name}: {e}"));
        assert_eq!(got.len(), *len, "{fixture}: {name} length");
        assert_eq!(xxh64(&got, 0), *hash, "{fixture}: {name} xxh64");
    }
    // decode_model must agree with per-tensor decode.
    let all: HashMap<String, Vec<u8>> = reader.decode_model().unwrap().into_iter().collect();
    for (name, len, hash) in want {
        let got = &all[*name];
        assert_eq!(
            (got.len(), xxh64(got, 0)),
            (*len, *hash),
            "{fixture}: {name}"
        );
    }
}

fn plane_codec_count(reader: &ContainerReader<'_>) -> usize {
    reader
        .extension_table
        .entries
        .iter()
        .filter(|e| e.kind == ptwm_core::extension::Kind::PlaneCodec)
        .count()
}

fn assert_transcode_round_trip(fixture: &str) {
    assert_blob_transcode_round_trip(fixture, &load(fixture));
}

fn assert_blob_transcode_round_trip(fixture: &str, blob: &[u8]) {
    let exploded = explode(blob).unwrap_or_else(|e| panic!("{fixture}: explode: {e}"));
    let members: HashMap<String, &[u8]> = exploded
        .members
        .iter()
        .map(|(k, v)| (k.clone(), v.as_slice()))
        .collect();
    let tensor_jsons: Vec<String> = exploded
        .tensors
        .iter()
        .map(|t| t.manifest_json.clone())
        .collect();
    let rebuilt = implode(&exploded.registry_json, &tensor_jsons, members)
        .unwrap_or_else(|e| panic!("{fixture}: implode: {e}"));
    assert!(
        rebuilt == blob,
        "{fixture}: implode(explode(x)) differs from x ({} vs {} bytes)",
        rebuilt.len(),
        blob.len()
    );
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[test]
fn fixtures_stay_small() {
    let total: u64 = [V1_1_0, V1_0_0, V1_0_0_NATIVE, TRIMMED, P4, P4_V1_1_0]
        .iter()
        .map(|f| load(f).len() as u64)
        .sum();
    assert!(total < 96 * 1024, "fixtures total {total} B");
}

#[test]
fn v1_1_0_fixture_decodes() {
    assert_decodes(V1_1_0);
}

#[test]
fn v1_1_0_fixture_transcode_round_trip() {
    assert_transcode_round_trip(V1_1_0);
}

#[test]
fn v1_1_0_fixture_has_expected_features() {
    let blob = load(V1_1_0);
    let reader = ContainerReader::open(&blob).unwrap();
    assert!(!reader.prelude.is_empty(), "shared prelude state");
    let records = records_in_file_order(&reader);
    assert!(
        records.iter().any(|r| r.inline_chain.is_some()),
        "inline chain"
    );
    assert!(
        records
            .iter()
            .flat_map(|r| &r.terminals)
            .any(|p| p.chunk_table.is_some()),
        "chunked plane"
    );
    assert!(
        records
            .iter()
            .flat_map(|r| &r.terminals)
            .any(|p| p.state_source == ptwm_core::codec::StateSource::Shared),
        "plane with shared state"
    );
    assert!(
        records.iter().any(|r| !r.dependencies.is_empty()),
        "dependency"
    );
    // The v1.1.0 table interns all 16 built-in codecs.
    assert_eq!(plane_codec_count(&reader), 16);
}

#[test]
fn v1_0_0_fixture_decodes() {
    assert_decodes(V1_0_0);
}

#[test]
fn v1_0_0_fixture_transcode_round_trip() {
    assert_transcode_round_trip(V1_0_0);
}

#[test]
fn v1_0_0_fixture_differs_from_v1_1_0_only_in_builtin_ids() {
    let old = load(V1_0_0);
    let mut cur = load(V1_1_0);
    assert_eq!(old.len(), cur.len());
    assert!(rederive_builtins(&mut cur, "1.1.0", "1.0.0") > 0);
    assert!(cur == old);
}

#[test]
fn trimmed_fixture_decodes() {
    assert_decodes(TRIMMED);
}

#[test]
fn trimmed_fixture_transcode_round_trip() {
    assert_transcode_round_trip(TRIMMED);
}

#[test]
fn p4_fixture_decodes() {
    assert_decodes(P4);
}

#[test]
fn p4_fixture_transcode_round_trip() {
    assert_transcode_round_trip(P4);
}

#[test]
fn p4_fixture_source_overstates_size() {
    for fixture in [P4, P4_V1_1_0] {
        let blob = load(fixture);
        let reader = ContainerReader::open(&blob).unwrap();
        let rec = &records_in_file_order(&reader)[0];
        let chain = reader.chain_registry.lookup(rec.chain_ref).unwrap();
        let p = &chain.nodes[0].params;
        assert_eq!(p[0], 1, "{fixture}: 1-D Source");
        let dim = u32::from_le_bytes(p[1..5].try_into().unwrap()) as u64;
        assert_eq!(u16::from_le_bytes([p[5], p[6]]), 0x0003, "{fixture}: fp32");
        // dim fp32 elements = 4 * dim bytes, against orig_size bytes.
        assert_eq!(dim, rec.orig_size, "{fixture}: Source declares 4x size");
    }
}

#[test]
fn p4_v1_1_0_fixture_decodes() {
    assert_decodes(P4_V1_1_0);
}

#[test]
fn p4_v1_1_0_fixture_transcode_round_trip() {
    assert_transcode_round_trip(P4_V1_1_0);
}

#[test]
fn p4_v1_1_0_fixture_has_full_table() {
    let blob = load(P4_V1_1_0);
    let reader = ContainerReader::open(&blob).unwrap();
    assert_eq!(plane_codec_count(&reader), 16);
    let trimmed = load(P4);
    let reader = ContainerReader::open(&trimmed).unwrap();
    assert!(plane_codec_count(&reader) < 16);
}

/// The v1.1.0 table layout must keep producing the same bytes for the P4
/// input.
#[test]
fn p4_v1_1_0_layout_writer_output_is_golden() {
    assert!(p4_v1_1_0_container() == load(P4_V1_1_0));
}

#[test]
fn v1_0_0_native_fixture_decodes() {
    assert_decodes(V1_0_0_NATIVE);
}

#[test]
fn v1_0_0_native_fixture_transcode_round_trip() {
    assert_transcode_round_trip(V1_0_0_NATIVE);
}

/// The v1.0.0 writer output decodes and transcodes byte-exactly once its
/// built-in ids are mapped to the crate version.
#[test]
fn v1_0_0_native_fixture_decodes_with_current_ids() {
    let blob = load(V1_0_0_NATIVE);
    let mut cur = blob.clone();
    assert!(rederive_builtins(&mut cur, "1.0.0", env!("CARGO_PKG_VERSION")) > 0);
    let reader = ContainerReader::open(&cur).unwrap();
    assert!(
        reader.prelude.is_empty(),
        "v1.0.0 writes no shared state here"
    );
    assert_eq!(plane_codec_count(&reader), 16);
    assert_blob_decodes(V1_0_0_NATIVE, &cur, expected(V1_0_0_NATIVE));
    assert_blob_transcode_round_trip(V1_0_0_NATIVE, &cur);
    assert!(as_current_version(blob, "1.0.0") == cur);
}

/// `rederive_builtins` must work on blobs whose built-in ids are not those
/// of the crate version, in both directions.
#[test]
fn rederive_builtins_handles_foreign_versions() {
    let orig = load(V1_0_0_NATIVE);
    let mut blob = orig.clone();
    let n = rederive_builtins(&mut blob, "1.0.0", "1.2.0");
    assert!(n > 0);
    assert_eq!(rederive_builtins(&mut blob, "1.2.0", "1.0.0"), n);
    assert!(blob == orig);
}

#[test]
#[should_panic(expected = "versions of equal length")]
fn rederive_builtins_rejects_length_change() {
    let mut blob = load(V1_1_0);
    rederive_builtins(&mut blob, "1.1.0", "1.10.0");
}

/// Current writer output with a trimmed table and shared `Order1ScaleAC`
/// prelude state (the default for models with scale planes) decodes to the
/// inputs and transcodes byte-exactly. No fixture file: the bytes come from
/// [`main_container`], whose hash is pinned by
/// [`legacy_writer_output_is_golden`].
#[test]
fn current_writer_shared_state_transcode_round_trip() {
    let blob = main_container();
    let reader = ContainerReader::open(&blob).unwrap();
    assert!(!reader.prelude.is_empty(), "shared prelude state");
    assert!(plane_codec_count(&reader) < 16, "trimmed table");
    assert!(
        records_in_file_order(&reader)
            .iter()
            .flat_map(|r| &r.terminals)
            .any(|p| p.state_source == ptwm_core::codec::StateSource::Shared),
        "plane with shared state"
    );
    let data = tensors();
    let want: Vec<(&str, usize, u64)> = data
        .iter()
        .map(|t| (t.name, t.bytes.len(), xxh64(&t.bytes, 0)))
        .collect();
    assert_blob_decodes("main_container", &blob, &want);
    assert_blob_transcode_round_trip("main_container", &blob);
}

/// The legacy writer must keep producing the same bytes. A failure here
/// means a change altered `\x89PTWM` output; if the change is intended
/// (for example a crate version bump, which changes built-in ids), follow
/// the regeneration steps in the module docs.
#[test]
fn legacy_writer_output_is_golden() {
    let blob = main_container();
    assert_eq!(
        blake3::hash(&blob).to_hex().as_str(),
        LEGACY_WRITER_GOLDEN_BLAKE3,
        "compress_model legacy output changed"
    );
    assert!(
        trimmed_container() == load(TRIMMED),
        "compress_model output != {TRIMMED}"
    );
}

/// The v1.1.0 table layout (`set_chain_registry`) must keep producing the
/// same bytes.
#[test]
fn v1_1_0_layout_writer_output_is_golden() {
    assert!(v1_1_0_container() == load(V1_1_0));
}

/// Writes the fixtures (all but `legacy_v1_0_0_native.ptwm`, which only the
/// v1.0.0 release can write) and prints the `EXPECTED` table and the golden hash.
#[test]
#[ignore = "regenerates tests/fixtures; run with PTWM_REGEN_FIXTURES=1"]
fn regenerate_fixtures() {
    assert!(
        std::env::var_os("PTWM_REGEN_FIXTURES").is_some(),
        "set PTWM_REGEN_FIXTURES=1 to overwrite tests/fixtures"
    );
    let v1_1_0 = v1_1_0_container();
    let mut v1_0_0 = v1_1_0.clone();
    assert!(rederive_builtins(&mut v1_0_0, "1.1.0", "1.0.0") > 0);
    let trimmed = trimmed_container();
    let golden = main_container();
    let p4 = p4_container();
    let p4_v1_1_0 = p4_v1_1_0_container();

    let dir = fixture_dir();
    std::fs::create_dir_all(&dir).unwrap();
    let files = [
        (V1_1_0, &v1_1_0),
        (V1_0_0, &v1_0_0),
        (TRIMMED, &trimmed),
        (P4, &p4),
        (P4_V1_1_0, &p4_v1_1_0),
    ];
    for (name, bytes) in files {
        std::fs::write(dir.join(name), bytes).unwrap();
    }

    // Expected decoded bytes come from the inputs, not from a decode.
    let mut main: Vec<(String, Vec<u8>)> = tensors()
        .into_iter()
        .map(|t| (t.name.to_string(), t.bytes))
        .collect();
    let trimmed_expected: Vec<(String, Vec<u8>)> = trimmed_tensors()
        .into_iter()
        .map(|t| (t.name.to_string(), t.bytes))
        .collect();
    main.extend(
        chunked_tensors()
            .into_iter()
            .map(|t| (t.name.to_string(), t.bytes)),
    );
    let p4_expected = vec![("p4.byte.fp32".to_string(), p4_bytes())];

    println!("    // EXPECTED-BEGIN");
    for (fixture, list) in [
        ("V1_1_0", &main),
        ("V1_0_0", &main),
        ("V1_0_0_NATIVE", &main),
        ("TRIMMED", &trimmed_expected),
        ("P4", &p4_expected),
        ("P4_V1_1_0", &p4_expected),
    ] {
        println!("    (\n        {fixture},\n        &[");
        for (name, bytes) in list.iter() {
            println!(
                "            ({name:?}, {}, 0x{:016x}),",
                bytes.len(),
                xxh64(bytes, 0)
            );
        }
        println!("        ],\n    ),");
    }
    println!("    // EXPECTED-END");
    println!(
        "const LEGACY_WRITER_GOLDEN_BLAKE3: &str = \"{}\";",
        blake3::hash(&golden).to_hex()
    );
    for (name, bytes) in files {
        println!("{name}: {} B", bytes.len());
    }
}

// ── Cross-version built-ins ──────────────────────────────────────────────────

fn explode_implode(blob: &[u8]) -> (String, Vec<u8>) {
    let exploded = explode(blob).expect("explode");
    let members: HashMap<String, &[u8]> = exploded
        .members
        .iter()
        .map(|(k, v)| (k.clone(), v.as_slice()))
        .collect();
    let tensor_jsons: Vec<String> = exploded
        .tensors
        .iter()
        .map(|t| t.manifest_json.clone())
        .collect();
    let rebuilt = implode(&exploded.registry_json, &tensor_jsons, members).expect("implode");
    (exploded.registry_json, rebuilt)
}

/// Offset of the Extension Table entry whose label is `label`, as
/// `(id_start, label_start)`.
fn find_label(blob: &[u8], label: &str) -> (usize, usize) {
    let label_start = find(blob, label.as_bytes()).unwrap_or_else(|| panic!("no {label}"));
    (label_start - 34, label_start)
}

/// Recompute the header's Extension Table hash after an in-place edit.
fn fix_table_hash(blob: &mut [u8]) {
    let u64_at = |b: &[u8], off: usize| u64::from_le_bytes(b[off..off + 8].try_into().unwrap());
    let et_off = u64_at(blob, 11) as usize;
    let et_len = u64_at(blob, 19) as usize;
    let hash = blake3::hash(&blob[et_off..et_off + et_len]);
    blob[27..59].copy_from_slice(hash.as_bytes());
}

#[test]
fn v1_0_0_fixture_reader_reports_builtin_version() {
    for fixture in [V1_0_0, V1_0_0_NATIVE] {
        let blob = load(fixture);
        let reader = ContainerReader::open(&blob).unwrap();
        assert_eq!(
            reader.builtin_version.as_deref(),
            Some("1.0.0"),
            "{fixture}"
        );
        // The in-memory table holds current built-in ids.
        assert!(
            reader
                .extension_table
                .entries
                .iter()
                .all(|e| ptwm_core::extension::is_builtin(&e.canonical_id)),
            "{fixture}"
        );
    }
    let blob = load(V1_1_0);
    let reader = ContainerReader::open(&blob).unwrap();
    assert_eq!(reader.builtin_version.as_deref(), Some("1.1.0"));
}

#[test]
fn v1_0_0_fixture_manifest_records_builtin_version() {
    let blob = load(V1_0_0);
    let (registry_json, rebuilt) = explode_implode(&blob);
    assert!(rebuilt == blob);
    let reg: serde_json::Value = serde_json::from_str(&registry_json).unwrap();
    let want = if env!("CARGO_PKG_VERSION") == "1.0.0" {
        serde_json::Value::Null
    } else {
        serde_json::Value::String("1.0.0".into())
    };
    assert_eq!(reg["container_flags"]["builtin_version"], want);
    // The table is rebuilt from the chains, so its bytes are not carried.
    assert!(reg["registry"].get("extension_table_b64").is_none());

    // A crate-version file carries neither field.
    let (registry_json, _) = explode_implode(&main_container());
    let reg: serde_json::Value = serde_json::from_str(&registry_json).unwrap();
    assert!(reg["container_flags"].get("builtin_version").is_none());
    assert!(reg["registry"].get("extension_table_b64").is_none());
}

/// The exploded stores implode one tensor at a time with the archive-wide
/// registry; that must work for files of other releases too.
#[test]
fn v1_0_0_fixture_single_tensor_implode_decodes() {
    let blob = load(V1_0_0);
    let exploded = explode(&blob).unwrap();
    let members: HashMap<String, &[u8]> = exploded
        .members
        .iter()
        .map(|(k, v)| (k.clone(), v.as_slice()))
        .collect();
    let want: HashMap<&str, (usize, u64)> = expected(V1_0_0)
        .iter()
        .map(|(n, l, h)| (*n, (*l, *h)))
        .collect();
    for t in &exploded.tensors {
        let one = implode(
            &exploded.registry_json,
            std::slice::from_ref(&t.manifest_json),
            members.clone(),
        )
        .unwrap_or_else(|e| panic!("implode {}: {e}", t.name));
        let reader = ContainerReader::open(&one).unwrap();
        assert_eq!(reader.builtin_version.as_deref(), Some("1.0.0"));
        let got = reader.decode_tensor(&t.name).unwrap();
        assert_eq!(
            (got.len(), xxh64(&got, 0)),
            want[t.name.as_str()],
            "{}",
            t.name
        );
    }
}

/// A table that mixes built-in versions is not rebuilt by any table
/// builder; its bytes travel in the manifest and implode stays byte-exact.
#[test]
fn mixed_version_table_round_trips_through_raw_bytes() {
    let mut blob = load(V1_1_0);
    // Move the file-level `identity` codec entry (and only it) to 1.0.0.
    let (id_start, label_start) = find_label(&blob, "io.ptwm.builtin.identity@1.1.0");
    let et_end = {
        let u64_at = |off: usize| u64::from_le_bytes(blob[off..off + 8].try_into().unwrap());
        (u64_at(11) + u64_at(19)) as usize
    };
    assert!(label_start < et_end, "entry is in the file-level table");
    let new = CanonicalId::derive(&BUILTIN_PUBKEY, "identity", "1.0.0");
    blob[id_start..id_start + 32].copy_from_slice(new.as_bytes());
    let ver = label_start + "io.ptwm.builtin.identity@".len();
    blob[ver..ver + 5].copy_from_slice(b"1.0.0");
    fix_table_hash(&mut blob);

    assert_blob_decodes("mixed", &blob, expected(V1_1_0));
    let (registry_json, rebuilt) = explode_implode(&blob);
    let reg: serde_json::Value = serde_json::from_str(&registry_json).unwrap();
    assert!(reg["registry"]["extension_table_b64"].is_string());
    assert!(rebuilt == blob, "implode(explode(x)) != x");
}

/// An entry with a built-in label whose id does not re-derive from that
/// label is not a built-in: the reader treats it as a missing third-party
/// contribution.
#[test]
fn forged_builtin_label_stays_non_builtin() {
    let mut blob = load(V1_0_0);
    let (id_start, _) = find_label(&blob, "io.ptwm.builtin.huffman@1.0.0");
    blob[id_start..id_start + 32].copy_from_slice(&[0x5A; 32]);
    fix_table_hash(&mut blob);

    match ContainerReader::open(&blob) {
        Err(ptwm_core::error::PtwmCoreError::MissingFlavor { needs }) => {
            assert_eq!(needs.len(), 1);
            assert_eq!(needs[0].id, CanonicalId::from_bytes([0x5A; 32]));
        }
        other => panic!("expected MissingFlavor, got {other:?}"),
    }
    let reader = ContainerReader::open_partial(&blob).unwrap();
    assert_eq!(reader.missing_extensions.len(), 1);
    assert!(
        !reader.extension_table.entries.iter().any(|e| e.canonical_id
            == CanonicalId::derive(&BUILTIN_PUBKEY, "huffman", env!("CARGO_PKG_VERSION"))),
        "the forged entry is not mapped to the current huffman id"
    );
}

/// Rewrite every built-in label (and id) at or after `from` to `to`.
fn rederive_labels_after(blob: &mut [u8], from: usize, to: &str) -> usize {
    let prefix = b"io.ptwm.builtin.";
    let mut p = from;
    let mut n = 0;
    while let Some(pos) = find(&blob[p..], prefix) {
        let label_start = p + pos;
        let label_len = u16::from_le_bytes([blob[label_start - 2], blob[label_start - 1]]) as usize;
        let label = std::str::from_utf8(&blob[label_start..label_start + label_len])
            .unwrap()
            .to_string();
        let (name, ver) = label[prefix.len()..].split_once('@').unwrap();
        assert_eq!(ver.len(), to.len());
        let id = CanonicalId::derive(&BUILTIN_PUBKEY, name, to);
        blob[label_start - 34..label_start - 2].copy_from_slice(id.as_bytes());
        blob[label_start + label_len - to.len()..label_start + label_len]
            .copy_from_slice(to.as_bytes());
        p = label_start + label_len;
        n += 1;
    }
    n
}

fn table_end(blob: &[u8]) -> usize {
    let u64_at = |off: usize| u64::from_le_bytes(blob[off..off + 8].try_into().unwrap());
    (u64_at(11) + u64_at(19)) as usize
}

/// An inline chain whose table version differs from the file-level table:
/// the inline version is used for implode and the file-level table travels
/// as raw bytes.
#[test]
fn inline_chain_of_another_version_round_trips() {
    let mut blob = load(V1_1_0);
    let end = table_end(&blob);
    assert!(
        rederive_labels_after(&mut blob, end, "1.0.0") > 0,
        "fixture has an inline chain"
    );
    assert_blob_decodes("inline-1.0.0", &blob, expected(V1_1_0));
    let (registry_json, rebuilt) = explode_implode(&blob);
    let reg: serde_json::Value = serde_json::from_str(&registry_json).unwrap();
    assert_eq!(reg["container_flags"]["builtin_version"], "1.0.0");
    assert!(reg["registry"]["extension_table_b64"].is_string());
    assert!(rebuilt == blob, "implode(explode(x)) != x");
}

/// Inline chains of two different built-in versions cannot be rebuilt with
/// one version, so explode refuses the container instead of producing
/// different bytes.
#[test]
fn explode_rejects_inline_chains_of_two_versions() {
    let blob = load(V1_1_0);
    let reader = ContainerReader::open(&blob).unwrap();
    let mut records = records_in_file_order(&reader);
    // Store a second registry chain inline, on the last record.
    let last = records.last_mut().unwrap();
    assert!(last.inline_chain.is_none());
    last.inline_chain = Some(
        reader
            .chain_registry
            .lookup(last.chain_ref)
            .unwrap()
            .clone(),
    );
    last.chain_ref = CHAIN_REF_INLINE_BIT;
    let mut cursor = Cursor::new(Vec::new());
    {
        let mut w = ContainerWriter::new(&mut cursor).unwrap();
        w.set_prelude(&reader.prelude).unwrap();
        w.set_chain_registry(reader.chain_registry.clone()).unwrap();
        for r in &records {
            w.append_tensor(r).unwrap();
        }
        w.finalize().unwrap();
    }
    let mut blob = cursor.into_inner();
    // The last record's inline table is the last run of built-in labels.
    let last_off = ContainerReader::open(&blob)
        .unwrap()
        .index
        .iter()
        .map(|e| e.tensor_offset)
        .max()
        .unwrap() as usize;
    let from = env!("CARGO_PKG_VERSION");
    let to = if from == "1.0.0" { "1.0.1" } else { "1.0.0" };
    if to.len() != from.len() {
        return; // `rederive_labels_after` needs equal-length versions.
    }
    assert!(rederive_labels_after(&mut blob, last_off, to) > 0);
    let e = explode(&blob).err().expect("explode must fail");
    assert!(
        e.to_string().contains("inline chain built-in version"),
        "{e}"
    );
}
