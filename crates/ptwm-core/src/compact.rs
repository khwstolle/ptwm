//! Compact single-tensor frame (magic `\x89PTWS`).
//!
//! A full `.ptwm` container carries a fixed header, an Extension Table, a
//! prelude, a chain registry, a tensor index and an EOF sentinel. For one
//! small tensor those sections cost more than the payload. The compact frame
//! holds one tensor with only the fields its decoder needs:
//!
//! ```text
//!     magic:          [u8; 5]   \x89PTWS
//!     flags:          u8        see FLAG_*; unknown bits are rejected,
//!                               and FLAG_PLANE_CRC with FLAG_PAYLOAD_HASH
//!                               is rejected
//!     dtype_code:     varint    at most u16::MAX
//!     input_format:   u8
//!     [orig_size]     varint    raw byte count                  (FLAG_SIZE)
//!     [shape]         varint ndim (<= 64), then varint per dimension
//!                                                              (FLAG_SHAPE)
//!     [dtype_name]    varint len, then UTF-8 bytes          (FLAG_DTYPE_NAME)
//!     [delta_ref]     [u8; 32]  BLAKE3 of the delta reference (FLAG_DELTA_REF)
//!     [payload_hash]  u64       xxh64 of the raw bytes     (FLAG_PAYLOAD_HASH)
//!     body
//! ```
//!
//! `orig_size` is stored only when the shape does not give it: without
//! `FLAG_SHAPE`, for a dtype code without a known element size, or when
//! `product(shape) × element_size` differs from the raw byte count (for
//! example a byte buffer with a shape in other units). A reader rejects
//! `FLAG_SIZE` when the shape gives the same value. `FLAG_DTYPE_NAME`
//! requires `FLAG_SHAPE` and a name that differs from the canonical name of
//! `dtype_code`; `FLAG_SHAPE` without it requires a known dtype code.
//!
//! When `FLAG_STORED` is set, the body is the `orig_size` raw bytes. Else the
//! body is a coded tensor:
//!
//! ```text
//!     chain:          CW2 chain (chain::wire_compact); op_ref is the OpId
//!                     wire id, and a bound Source takes the frame's shape
//!                     and dtype_code
//!     per terminal plane, in chain terminal order:
//!         head:           varint      CodecId wire id << 2 | state
//!                                     state 0 = none, 1 = inline,
//!                                     2 = inline, version 0, empty;
//!                                     3 is rejected
//!         [state_version] u8          (state 1)
//!         [state_len]     varint      (state 1)
//!         [state]         bytes       (state 1)
//!         payload_len:    varint
//!         [crc32]         u32         CRC32 of the raw plane (FLAG_PLANE_CRC)
//!         payload:        bytes
//! ```
//!
//! State 1 with version 0 and an empty state is rejected: state 2 is its
//! only encoding. States 0 and 2 decode alike; the frame keeps the
//! distinction the codec reported, and the reader accepts either. All
//! integers are little-endian; varints are canonical unsigned LEB128
//! ([`crate::varint`]). A stored body must be exactly `orig_size` bytes, and
//! a coded body must end exactly with the last plane's payload, so the reader
//! rejects bytes appended to either.
//!
//! The payload hash covers the decoded bytes of the one tensor, and so every
//! plane, so the writer stores plane CRCs only when it does not store the
//! hash. Both checks verify decoded bytes; they do not cover the coded
//! payload.
//!
//! Each header, chain and plane field has one encoding for the value it
//! holds. The frame as a whole is not unique for a tensor: the writer
//! chooses between a stored and a coded body, between plane state codes 0
//! and 2, and between codecs and chains, and the reader accepts any valid
//! choice.
//!
//! The plane layout, the decoded plane length and the plane count all come
//! from the chain, so the frame does not store them. Planes that need shared
//! state (a prelude) or chunking cannot use this frame; the caller falls back
//! to the full container for those.
//!
//! The magic differs from `\x89PTWM`, so readers that predate this frame
//! reject it as "not a .ptwm container" instead of misparsing it. Like the
//! container, the frame has no numeric version field: a reader rejects
//! unknown flag bits, and an incompatible layout gets a new magic.

use crate::chain::Chain;
use crate::chain::runtime::{
    InverseContext, chain_source_descriptor, inverse_chain, max_plane_len,
    propagate_terminal_descriptors_bounded,
};
use crate::chain::validate::validate_chain;
use crate::chain::wire_compact::{
    op_from_wire_ref, op_wire_ref, read_chain_compact, write_chain_compact,
};
use crate::codec::{CodecId, StateSource, codec_for};
use crate::compressor::{CompressorOptions, InputTensor, legacy_plane_layout, plan_model};
use crate::dtype::Dtype;
use crate::error::PtwmCoreError;
use crate::transforms::op::Plane;
use crate::transforms::source::bound_source_params;
use crate::varint::{Cursor, write_varint};

use std::sync::Arc;
use xxhash_rust::xxh64::xxh64;

/// Compact-frame magic. Shares the `\x89PTW` prefix with the full container.
pub const MAGIC_COMPACT: [u8; 5] = [0x89, b'P', b'T', b'W', b'S'];

/// The frame carries an xxh64 of the raw bytes.
pub const FLAG_PAYLOAD_HASH: u8 = 0x01;
/// Every coded plane carries a CRC32 of its raw bytes. Not combined with
/// [`FLAG_PAYLOAD_HASH`], which already covers every plane.
pub const FLAG_PLANE_CRC: u8 = 0x02;
/// The frame carries the tensor shape.
pub const FLAG_SHAPE: u8 = 0x04;
/// The frame carries a dtype name that differs from the canonical name of
/// `dtype_code`.
pub const FLAG_DTYPE_NAME: u8 = 0x08;
/// The frame carries the BLAKE3 hash of a delta reference.
pub const FLAG_DELTA_REF: u8 = 0x10;
/// The body is the raw bytes, not a coded tensor.
pub const FLAG_STORED: u8 = 0x20;
/// The frame stores `orig_size` because the shape does not give it.
pub const FLAG_SIZE: u8 = 0x40;
const KNOWN_FLAGS: u8 = FLAG_PAYLOAD_HASH
    | FLAG_PLANE_CRC
    | FLAG_SHAPE
    | FLAG_DTYPE_NAME
    | FLAG_DELTA_REF
    | FLAG_STORED
    | FLAG_SIZE;

/// Upper bound on the dimension count, so a corrupt `ndim` cannot drive a
/// large allocation.
const MAX_NDIM: u64 = 64;

/// Plane state codes in the plane head.
const STATE_NONE: u64 = 0;
const STATE_INLINE: u64 = 1;
const STATE_INLINE_EMPTY: u64 = 2;

/// A tensor decoded from a compact frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactTensor {
    pub dtype_code: u16,
    pub input_format: u8,
    /// Shape, when the writer supplied one.
    pub shape: Option<Vec<u64>>,
    /// Dtype name paired with `shape`. When the frame does not store a name,
    /// this is the canonical name of `dtype_code`.
    pub dtype_name: Option<String>,
    /// BLAKE3 hash of the delta reference, when the tensor is delta-coded.
    pub delta_reference_blake3: Option<[u8; 32]>,
    /// Decoded raw bytes.
    pub raw: Vec<u8>,
    /// Bytes of codec payload and inline codec state in the frame (the raw
    /// byte count for a stored frame). The frame length minus this value is
    /// the frame's fixed overhead.
    pub payload_len: u64,
}

/// Return `true` when `buf` starts with the compact-frame magic.
pub fn is_compact(buf: &[u8]) -> bool {
    buf.len() >= MAGIC_COMPACT.len() && buf[..MAGIC_COMPACT.len()] == MAGIC_COMPACT
}

fn canonical_dtype_name(dtype_code: u16) -> Option<&'static str> {
    u8::try_from(dtype_code)
        .ok()
        .and_then(|c| Dtype::from_code(c).ok())
        .map(Dtype::canonical_name)
}

/// Raw byte count implied by a shape and a public dtype code:
/// `product(shape) × element_size`. `None` for an unknown dtype code or on
/// overflow.
pub fn size_from_shape(shape: &[u64], dtype_code: u16) -> Option<u64> {
    let dtype = Dtype::from_code(u8::try_from(dtype_code).ok()?).ok()?;
    shape
        .iter()
        .try_fold(u64::from(dtype.element_size()), |acc, &d| {
            acc.checked_mul(d)
        })
}

/// Encode one tensor as a compact frame.
///
/// Runs the same chain and codec search as
/// [`crate::compressor::compress_model`], then writes the smaller of the
/// coded body and the raw bytes. Returns `Ok(None)` when the tensor cannot use
/// this frame: chunked encoding (`opts.chunk_size`) or a plane that selected
/// shared state. The caller then writes a full container.
pub fn compress_compact(
    input: &InputTensor<'_>,
    opts: &CompressorOptions,
) -> Result<Option<Vec<u8>>, PtwmCoreError> {
    if opts.chunk_size.is_some() {
        return Ok(None);
    }
    let raw = input.raw_bytes;
    let orig_size = raw.len() as u64;

    let mut out = Vec::with_capacity(64);
    out.extend_from_slice(&MAGIC_COMPACT);
    let flags_pos = out.len();
    out.push(0);
    let mut flags = 0u8;
    write_varint(&mut out, u64::from(input.dtype_code));
    out.push(input.input_format);
    // Shape metadata is recorded only when both shape and dtype name are
    // known, matching the full container's tensor metadata.
    let shape = match (&input.shape, &input.dtype_name) {
        (Some(shape), Some(_)) => Some(shape.as_slice()),
        _ => None,
    };
    if shape.and_then(|s| size_from_shape(s, input.dtype_code)) != Some(orig_size) {
        flags |= FLAG_SIZE;
        write_varint(&mut out, orig_size);
    }
    let mut bound_source = None;
    if let (Some(shape), Some(name)) = (shape, &input.dtype_name) {
        if shape.len() as u64 > MAX_NDIM {
            // The full container has no such limit.
            return Ok(None);
        }
        flags |= FLAG_SHAPE;
        write_varint(&mut out, shape.len() as u64);
        for &d in shape {
            write_varint(&mut out, d);
        }
        if canonical_dtype_name(input.dtype_code) != Some(name.as_str()) {
            flags |= FLAG_DTYPE_NAME;
            write_varint(&mut out, name.len() as u64);
            out.extend_from_slice(name.as_bytes());
        }
        bound_source = bound_source_params(shape, input.dtype_code);
    }
    if let Some(h) = &input.delta_reference_blake3 {
        flags |= FLAG_DELTA_REF;
        out.extend_from_slice(h);
    }
    if opts.emit_payload_hash {
        flags |= FLAG_PAYLOAD_HASH;
        out.extend_from_slice(&xxh64(raw, 0).to_le_bytes());
    }

    // The payload hash covers the decoded bytes of every plane, so plane
    // CRCs would only repeat it.
    let plane_crc = opts.emit_plane_crc && !opts.emit_payload_hash;
    let coded = encode_body(input, opts, bound_source.as_deref(), plane_crc)?;
    match coded {
        None => return Ok(None),
        Some(body) if body.len() < raw.len() => {
            if plane_crc {
                flags |= FLAG_PLANE_CRC;
            }
            out.extend_from_slice(&body);
        }
        Some(_) => {
            flags |= FLAG_STORED;
            out.extend_from_slice(raw);
        }
    }
    out[flags_pos] = flags;
    Ok(Some(out))
}

/// Encode the coded body (chain, planes). `Ok(None)` when a plane needs
/// shared or external state.
fn encode_body(
    input: &InputTensor<'_>,
    opts: &CompressorOptions,
    bound_source: Option<&[u8]>,
    plane_crc: bool,
) -> Result<Option<Vec<u8>>, PtwmCoreError> {
    let plan = plan_model(std::slice::from_ref(input), opts)?;
    if !plan.prelude_entries.is_empty() {
        return Ok(None);
    }
    let chain: &Chain = &input.candidate_chains[plan.pass1[0].chain_idx];
    let planes = &plan.chosen_per_tensor[0];

    let mut body = Vec::with_capacity(
        64 + planes
            .iter()
            .map(|p| p.payload_bytes.len() + p.encoded_state_bytes.len() + 16)
            .sum::<usize>(),
    );
    write_chain_compact(chain, bound_source, op_wire_ref, &mut body)?;

    for cp in planes {
        let codec = u64::from(cp.codec_id.as_u16()) << 2;
        match cp.state_source {
            StateSource::None => write_varint(&mut body, codec | STATE_NONE),
            StateSource::Inline
                if cp.encoded_state_version == 0 && cp.encoded_state_bytes.is_empty() =>
            {
                write_varint(&mut body, codec | STATE_INLINE_EMPTY)
            }
            StateSource::Inline => {
                write_varint(&mut body, codec | STATE_INLINE);
                body.push(cp.encoded_state_version);
                write_varint(&mut body, cp.encoded_state_bytes.len() as u64);
                body.extend_from_slice(&cp.encoded_state_bytes);
            }
            StateSource::Shared | StateSource::External => return Ok(None),
        }
        write_varint(&mut body, cp.payload_bytes.len() as u64);
        if plane_crc {
            body.extend_from_slice(&crc32fast::hash(&cp.raw_plane_for_crc).to_le_bytes());
        }
        body.extend_from_slice(&cp.payload_bytes);
    }
    Ok(Some(body))
}

/// The frame fields before the body.
struct FrameHeader {
    flags: u8,
    dtype_code: u16,
    input_format: u8,
    orig_size: u64,
    shape: Option<Vec<u64>>,
    dtype_name: Option<String>,
    delta_reference_blake3: Option<[u8; 32]>,
    payload_hash: Option<u64>,
}

fn read_header(r: &mut Cursor<'_>) -> Result<FrameHeader, PtwmCoreError> {
    if !is_compact(r.rest()) {
        return Err(r.err("bad magic"));
    }
    r.take(MAGIC_COMPACT.len())?;
    let flags = r.u8()?;
    if flags & !KNOWN_FLAGS != 0 {
        return Err(r.err(format_args!(
            "unknown flag bits {:#04x}",
            flags & !KNOWN_FLAGS
        )));
    }
    if flags & FLAG_PLANE_CRC != 0 && flags & FLAG_PAYLOAD_HASH != 0 {
        return Err(r.err("plane CRC flag with payload hash"));
    }
    let dtype_code = r.varint_max(u64::from(u16::MAX), "dtype_code")? as u16;
    let input_format = r.u8()?;
    let stored_size = if flags & FLAG_SIZE != 0 {
        Some(r.varint()?)
    } else {
        None
    };

    let mut shape = None;
    let mut dtype_name = None;
    if flags & FLAG_SHAPE != 0 {
        let ndim = r.varint_max(MAX_NDIM, "ndim")?;
        let mut dims = Vec::with_capacity(ndim as usize);
        for _ in 0..ndim {
            dims.push(r.varint()?);
        }
        shape = Some(dims);
        dtype_name = canonical_dtype_name(dtype_code).map(str::to_owned);
    }
    if flags & FLAG_DTYPE_NAME != 0 {
        if flags & FLAG_SHAPE == 0 {
            return Err(r.err("dtype name without shape"));
        }
        let bytes = r.len_bytes()?;
        let name = std::str::from_utf8(bytes).map_err(|_| r.err("dtype name is not UTF-8"))?;
        if dtype_name.as_deref() == Some(name) {
            return Err(r.err("stored dtype name equals the canonical name"));
        }
        dtype_name = Some(name.to_owned());
    } else if shape.is_some() && dtype_name.is_none() {
        return Err(r.err(format_args!(
            "shape without a dtype name for unknown dtype_code {dtype_code}"
        )));
    }

    let derived_size = shape
        .as_deref()
        .and_then(|s| size_from_shape(s, dtype_code));
    let orig_size = match (stored_size, derived_size) {
        (Some(s), Some(d)) if s == d => {
            return Err(r.err("stored orig_size equals the size the shape gives"));
        }
        (Some(s), _) => s,
        (None, Some(d)) => d,
        (None, None) => return Err(r.err("no orig_size and no shape that gives it")),
    };

    let delta_reference_blake3 = if flags & FLAG_DELTA_REF != 0 {
        Some(r.array::<32>()?)
    } else {
        None
    };
    let payload_hash = if flags & FLAG_PAYLOAD_HASH != 0 {
        Some(r.u64_le()?)
    } else {
        None
    };
    Ok(FrameHeader {
        flags,
        dtype_code,
        input_format,
        orig_size,
        shape,
        dtype_name,
        delta_reference_blake3,
        payload_hash,
    })
}

/// Decode a compact frame. Verifies the payload hash and plane CRCs when the
/// frame carries them.
pub fn decode_compact(buf: &[u8]) -> Result<CompactTensor, PtwmCoreError> {
    let mut r = Cursor::new(buf, "compact frame");
    let h = read_header(&mut r)?;
    let orig_size = h.orig_size;

    let mut payload_len = 0u64;
    let raw = if h.flags & FLAG_STORED != 0 {
        if h.flags & FLAG_PLANE_CRC != 0 {
            return Err(r.err("stored frame with plane CRC flag"));
        }
        let rest = r.take(r.remaining())?;
        if rest.len() as u64 != orig_size {
            return Err(r.err(format_args!(
                "stored body is {} bytes, header says {orig_size}",
                rest.len()
            )));
        }
        payload_len = orig_size;
        rest.to_vec()
    } else {
        let bound = h
            .shape
            .as_deref()
            .and_then(|s| bound_source_params(s, h.dtype_code));
        decode_body(
            &mut r,
            h.flags & FLAG_PLANE_CRC != 0,
            orig_size,
            bound.as_deref(),
            &mut payload_len,
        )?
    };
    r.finish("body")?;

    if let Some(expected) = h.payload_hash
        && xxh64(&raw, 0) != expected
    {
        return Err(r.err("payload hash mismatch"));
    }

    Ok(CompactTensor {
        dtype_code: h.dtype_code,
        input_format: h.input_format,
        shape: h.shape,
        dtype_name: h.dtype_name,
        delta_reference_blake3: h.delta_reference_blake3,
        raw,
        payload_len,
    })
}

fn decode_body(
    r: &mut Cursor<'_>,
    has_crc: bool,
    orig_size: u64,
    bound_source: Option<&[u8]>,
    payload_len_out: &mut u64,
) -> Result<Vec<u8>, PtwmCoreError> {
    let chain = read_chain_compact(r, bound_source, op_from_wire_ref)?;
    // Writers check that the chain's Source declares exactly the raw byte
    // count, so a frame whose Source disagrees with `orig_size` is corrupt.
    // Checking before `validate_chain` and the descriptor walk (both call
    // every op's `propagate_descriptors`) keeps a corrupt chain from sizing
    // planes (and codec allocations) beyond the tensor.
    let source = chain_source_descriptor(&chain)?;
    if source.length_bytes != orig_size {
        return Err(r.err(format_args!(
            "chain Source declares {} bytes, header says {orig_size}",
            source.length_bytes
        )));
    }
    validate_chain(&chain, 0)?;
    let descriptors = propagate_terminal_descriptors_bounded(&chain, max_plane_len(orig_size))?;

    let n_planes = descriptors.len();
    let mut terminal_planes: Vec<Plane> = Vec::with_capacity(n_planes);
    for descriptor in descriptors {
        let head = r.varint()?;
        let wire = u16::try_from(head >> 2).map_err(|_| r.err("codec id exceeds u16"))?;
        let codec_id = CodecId::from_u16(wire)
            .ok_or_else(|| r.err(format_args!("unknown codec {wire:#06x}")))?;
        let (state_version, state) = match head & 3 {
            STATE_NONE | STATE_INLINE_EMPTY => (0u8, &[][..]),
            STATE_INLINE => {
                let v = r.u8()?;
                let state = r.len_bytes()?;
                if v == 0 && state.is_empty() {
                    return Err(r.err("empty version-0 inline state must use state code 2"));
                }
                (v, state)
            }
            s => return Err(r.err(format_args!("unknown state code {s}"))),
        };
        let payload_len = r.len_varint()?;
        let crc = if has_crc { Some(r.u32_le()?) } else { None };
        let payload = r.take(payload_len)?;
        *payload_len_out += (state.len() + payload.len()) as u64;

        let codec = codec_for(codec_id)
            .ok_or_else(|| r.err(format_args!("no codec implementation for {codec_id:?}")))?;
        let layout = legacy_plane_layout(descriptor.layout);
        let decoded_len = usize::try_from(descriptor.length_bytes)
            .map_err(|_| r.err("plane length overflows usize"))?;
        // The Source is exact, so every plane decodes to its descriptor's
        // length. The codec stops at that length, so a corrupt payload cannot
        // allocate past it.
        let bytes = codec.decode_bounded(
            state_version,
            state,
            payload,
            &layout,
            decoded_len,
            decoded_len,
        )?;
        if bytes.len() != decoded_len {
            return Err(r.err(format_args!(
                "plane decoded to {} bytes, chain says {decoded_len}",
                bytes.len()
            )));
        }
        if let Some(expected) = crc
            && crc32fast::hash(&bytes) != expected
        {
            return Err(r.err("plane CRC mismatch"));
        }
        terminal_planes.push(Plane {
            bytes: Arc::from(bytes.into_boxed_slice()),
            descriptor,
        });
    }

    let ctx = InverseContext {
        terminal_planes: &terminal_planes,
        deps: &[],
    };
    let source = inverse_chain(&chain, &ctx)?;
    if source.bytes.len() as u64 != orig_size {
        return Err(r.err(format_args!(
            "decoded {} bytes, header says {orig_size}",
            source.bytes.len()
        )));
    }
    Ok(source.bytes.as_ref().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::graph::{ChainEdge, ChainNode, TerminalRef};
    use crate::transforms::op::OpId;
    use crate::types::role::Role;

    const FLAGS_AT: usize = MAGIC_COMPACT.len();

    fn source_params(n_elements: u32, dtype_code: u16) -> Vec<u8> {
        // 1-D shape encoding (matches `Source::write_params`).
        let mut p = vec![1u8];
        p.extend_from_slice(&n_elements.to_le_bytes());
        p.extend_from_slice(&dtype_code.to_le_bytes());
        p
    }

    /// Source(int8) → BytePassthrough → one Value terminal.
    fn passthrough_chain(n_bytes: u32) -> Chain {
        let role = Role::Value {
            format: crate::types::role::ValueFormat::IntN { bits: 8 },
        };
        Chain {
            nodes: vec![
                ChainNode {
                    op: OpId::Source,
                    params: source_params(n_bytes, 17),
                },
                ChainNode {
                    op: OpId::BytePassthrough,
                    params: Vec::new(),
                },
            ],
            edges: vec![ChainEdge {
                src_node: 0,
                src_output_idx: 0,
                dst_node: 1,
                dst_input_idx: 0,
                role_override: Some(role.clone()),
                vendor_bytes: Vec::new(),
            }],
            terminals: vec![TerminalRef {
                node_idx: 1,
                output_idx: 0,
                role,
            }],
        }
    }

    fn opts(hash: bool, crc: bool) -> CompressorOptions {
        CompressorOptions {
            method_hint: 1,
            emit_payload_hash: hash,
            emit_plane_crc: crc,
            forced_codec: None,
            allow_codec_ids: None,
            chunk_size: None,
        }
    }

    fn input<'a>(raw: &'a [u8], shape: Option<Vec<u64>>, name: Option<&str>) -> InputTensor<'a> {
        InputTensor {
            name: "_".into(),
            candidate_chains: vec![passthrough_chain(raw.len() as u32)],
            dtype_code: 0,
            input_format: 1,
            raw_bytes: raw,
            shape,
            dtype_name: name.map(str::to_owned),
            delta_reference_blake3: None,
        }
    }

    #[test]
    fn constant_tensor_round_trips_coded() {
        let raw = vec![7u8; 4096];
        for (h, c) in [(true, true), (false, false), (true, false), (false, true)] {
            let blob = compress_compact(&input(&raw, None, None), &opts(h, c))
                .unwrap()
                .unwrap();
            assert!(blob.len() < 128, "constant frame is {} bytes", blob.len());
            assert_eq!(
                blob[FLAGS_AT] & FLAG_STORED,
                0,
                "constant data must be coded"
            );
            let t = decode_compact(&blob).unwrap();
            assert_eq!(t.raw, raw);
            assert_eq!(t.shape, None);
        }
    }

    #[test]
    fn incompressible_tensor_is_stored() {
        // xorshift bytes: no codec beats the raw size.
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        let raw: Vec<u8> = (0..300)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x as u8
            })
            .collect();
        let blob = compress_compact(
            &input(&raw, Some(vec![300]), Some("uint8")),
            &opts(true, true),
        )
        .unwrap()
        .unwrap();
        assert_ne!(blob[FLAGS_AT] & FLAG_STORED, 0);
        assert!(blob.len() <= raw.len() + 32);
        let t = decode_compact(&blob).unwrap();
        assert_eq!(t.raw, raw);
        assert_eq!(t.shape, Some(vec![300]));
        assert_eq!(t.dtype_name.as_deref(), Some("uint8"));
    }

    #[test]
    fn noncanonical_dtype_name_is_kept() {
        let raw = vec![1u8; 64];
        let blob = compress_compact(
            &input(&raw, Some(vec![8, 8]), Some("custom")),
            &opts(true, true),
        )
        .unwrap()
        .unwrap();
        assert_ne!(blob[FLAGS_AT] & FLAG_DTYPE_NAME, 0);
        let t = decode_compact(&blob).unwrap();
        assert_eq!(t.shape, Some(vec![8, 8]));
        assert_eq!(t.dtype_name.as_deref(), Some("custom"));
    }

    #[test]
    fn empty_tensor_round_trips() {
        let raw: Vec<u8> = Vec::new();
        let blob = compress_compact(
            &input(&raw, Some(vec![0]), Some("uint8")),
            &opts(true, true),
        )
        .unwrap()
        .unwrap();
        let t = decode_compact(&blob).unwrap();
        assert!(t.raw.is_empty());
    }

    #[test]
    fn chunked_request_is_not_eligible() {
        let raw = vec![0u8; 64];
        let mut o = opts(true, true);
        o.chunk_size = Some(16);
        assert!(
            compress_compact(&input(&raw, None, None), &o)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn corruption_is_detected() {
        let raw: Vec<u8> = (0..2048u32).map(|i| (i % 7) as u8).collect();
        let blob = compress_compact(&input(&raw, None, None), &opts(true, true))
            .unwrap()
            .unwrap();
        // Every truncation fails cleanly.
        for cut in 0..blob.len() {
            assert!(
                decode_compact(&blob[..cut]).is_err(),
                "cut at {cut} decoded"
            );
        }
        // Trailing bytes are rejected.
        let mut longer = blob.clone();
        longer.push(0);
        assert!(decode_compact(&longer).is_err());
        // Unknown flag bits are rejected.
        let mut flagged = blob.clone();
        flagged[FLAGS_AT] |= 0x80;
        assert!(decode_compact(&flagged).is_err());
        // A flipped payload byte fails a check or decodes to different bytes
        // that the payload hash rejects; it must never return wrong data.
        for i in FLAGS_AT + 1..blob.len() {
            let mut b = blob.clone();
            b[i] ^= 0x55;
            if let Ok(t) = decode_compact(&b) {
                assert_eq!(t.raw, raw, "byte {i} flip returned wrong data");
            }
        }
    }

    #[test]
    fn oversized_chain_source_is_rejected() {
        let raw: Vec<u8> = (0..3000u32).map(|i| (i % 5) as u8).collect();
        let blob = compress_compact(&input(&raw, None, None), &opts(false, false))
            .unwrap()
            .unwrap();
        assert_eq!(blob[FLAGS_AT] & FLAG_STORED, 0);
        // Rewrite the Source dimension (1-D, 3000) to about 4 G elements.
        let needle: Vec<u8> = [&[1u8][..], &3000u32.to_le_bytes()].concat();
        let at = blob
            .windows(needle.len())
            .position(|w| w == needle.as_slice())
            .unwrap();
        let mut b = blob.clone();
        b[at + 1..at + 5].copy_from_slice(&0xFFFF_FFF0u32.to_le_bytes());
        let t0 = std::time::Instant::now();
        let err = decode_compact(&b).unwrap_err().to_string();
        assert!(err.contains("Source declares"), "{err}");
        assert!(t0.elapsed() < std::time::Duration::from_secs(1));
    }

    #[test]
    fn huge_source_is_rejected_before_op_descriptor_arithmetic() {
        // Source [2, 1500] int8 → BurrowsWheeler, then the dims are patched
        // in place to [u32::MAX, u32::MAX] (about 2^64 bytes). The Source
        // check must run before `validate_chain`, which calls
        // `BurrowsWheeler::propagate_descriptors`; before the fix, a build
        // with overflow checks panicked there.
        let raw: Vec<u8> = (0..3000u32).map(|i| (i % 5) as u8).collect();
        let mut chain = passthrough_chain(raw.len() as u32);
        chain.nodes[0].params = [&[2u8][..], &2u32.to_le_bytes(), &1500u32.to_le_bytes()]
            .concat()
            .into_iter()
            .chain(17u16.to_le_bytes())
            .collect();
        chain.nodes[1].op = OpId::BurrowsWheeler;
        let mut inp = input(&raw, None, None);
        inp.candidate_chains = vec![chain];
        let blob = compress_compact(&inp, &opts(false, false))
            .unwrap()
            .unwrap();
        assert_eq!(blob[FLAGS_AT] & FLAG_STORED, 0);
        assert_eq!(decode_compact(&blob).unwrap().raw, raw);
        let needle: Vec<u8> = [&[2u8][..], &2u32.to_le_bytes(), &1500u32.to_le_bytes()].concat();
        let at = blob
            .windows(needle.len())
            .position(|w| w == needle.as_slice())
            .unwrap();
        let mut b = blob.clone();
        b[at + 1..at + 9].copy_from_slice(&[0xFF; 8]);
        let err = decode_compact(&b).unwrap_err().to_string();
        assert!(err.contains("Source declares"), "{err}");
    }

    #[test]
    fn source_must_declare_orig_size_exactly() {
        // Unlike legacy containers, compact frames carry no files written
        // before the writer check, so any Source size other than
        // `orig_size` is corrupt, overstated or understated.
        let raw: Vec<u8> = (0..3000u32).map(|i| (i % 5) as u8).collect();
        let blob = compress_compact(&input(&raw, None, None), &opts(false, false))
            .unwrap()
            .unwrap();
        let needle: Vec<u8> = [&[1u8][..], &3000u32.to_le_bytes()].concat();
        let at = blob
            .windows(needle.len())
            .position(|w| w == needle.as_slice())
            .unwrap();
        for dim in [3001u32, 1500, 6000] {
            let mut b = blob.clone();
            b[at + 1..at + 5].copy_from_slice(&dim.to_le_bytes());
            let err = decode_compact(&b).unwrap_err().to_string();
            assert!(err.contains("Source declares"), "{dim}: {err}");
        }
    }

    /// Source(fp32, bound) → BitReorderIeee32 → ByteSplit(4), the
    /// production fp32 split chain.
    fn fp32_chain(shape: &[u64]) -> Chain {
        let mut terminals: Vec<TerminalRef> = (0..3)
            .map(|i| TerminalRef {
                node_idx: 2,
                output_idx: i,
                role: Role::MantissaByte { index: i, of: 4 },
            })
            .collect();
        terminals.push(TerminalRef {
            node_idx: 2,
            output_idx: 3,
            role: Role::ExponentByte,
        });
        let spine = |i: u8| ChainEdge {
            src_node: i,
            src_output_idx: 0,
            dst_node: i + 1,
            dst_input_idx: 0,
            role_override: None,
            vendor_bytes: Vec::new(),
        };
        Chain {
            nodes: vec![
                ChainNode {
                    op: OpId::Source,
                    params: bound_source_params(shape, 1).unwrap(),
                },
                ChainNode {
                    op: OpId::BitReorderIeee32,
                    params: Vec::new(),
                },
                ChainNode {
                    op: OpId::ByteSplit,
                    params: vec![4],
                },
            ],
            edges: vec![spine(0), spine(1)],
            terminals,
        }
    }

    fn fp32_raw(n: usize) -> Vec<u8> {
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        (0..n)
            .flat_map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                // Normal-ish magnitudes: a narrow exponent range.
                let v = ((x >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 4.0;
                v.to_le_bytes()
            })
            .collect()
    }

    fn fp32_input<'a>(raw: &'a [u8], shape: &[u64]) -> InputTensor<'a> {
        InputTensor {
            name: "_".into(),
            candidate_chains: vec![fp32_chain(shape)],
            dtype_code: 1,
            input_format: 1,
            raw_bytes: raw,
            shape: Some(shape.to_vec()),
            dtype_name: Some("float32".into()),
            delta_reference_blake3: None,
        }
    }

    /// Byte offset where the frame body starts.
    fn body_start(blob: &[u8]) -> usize {
        let mut r = Cursor::new(blob, "test");
        read_header(&mut r).unwrap();
        r.pos()
    }

    /// Byte offset of the first plane head (after the chain).
    fn planes_start(blob: &[u8]) -> usize {
        let mut r = Cursor::new(blob, "test");
        let h = read_header(&mut r).unwrap();
        let bound = bound_source_params(h.shape.as_deref().unwrap_or(&[]), h.dtype_code);
        let bound = h.shape.as_ref().and(bound);
        read_chain_compact(&mut r, bound.as_deref(), op_from_wire_ref).unwrap();
        r.pos()
    }

    #[test]
    fn fp32_frame_binds_source_and_stays_small() {
        let raw = fp32_raw(4096);
        for (h, c) in [(true, true), (false, false), (false, true)] {
            let blob = compress_compact(&fp32_input(&raw, &[64, 64]), &opts(h, c))
                .unwrap()
                .unwrap();
            let flags = blob[FLAGS_AT];
            assert_eq!(flags & (FLAG_STORED | FLAG_SIZE), 0, "{flags:#04x}");
            let t = decode_compact(&blob).unwrap();
            assert_eq!(t.raw, raw);
            assert_eq!(t.shape, Some(vec![64, 64]));
            assert_eq!(t.dtype_name.as_deref(), Some("float32"));
            let overhead = blob.len() as u64 - t.payload_len;
            // magic 5, flags 1, dtype 1, format 1, shape 3, chain 19,
            // 4 plane heads, 4 payload lengths (up to 2 bytes each); plus
            // the hash (8) or 4 CRCs (16), never both.
            let integrity = match (h, c) {
                (true, _) => 8,
                (false, true) => 16,
                (false, false) => 0,
            };
            let limit = 31 + 4 + 8 + integrity;
            assert!(
                overhead <= limit,
                "overhead {overhead} > {limit} (hash {h}, crc {c})"
            );
            assert_eq!(
                flags & (FLAG_PAYLOAD_HASH | FLAG_PLANE_CRC),
                if h {
                    FLAG_PAYLOAD_HASH
                } else if c {
                    FLAG_PLANE_CRC
                } else {
                    0
                }
            );
            // The Source dims are not stored.
            let dims: Vec<u8> = [&64u32.to_le_bytes()[..], &64u32.to_le_bytes()].concat();
            assert!(!blob.windows(8).any(|w| w == dims.as_slice()));
        }
        // Chains for other shapes encode to the same bytes.
        let raw2 = fp32_raw(1000);
        let a = compress_compact(&fp32_input(&raw, &[64, 64]), &opts(false, false))
            .unwrap()
            .unwrap();
        let b = compress_compact(&fp32_input(&raw2, &[10, 100]), &opts(false, false))
            .unwrap()
            .unwrap();
        let chain = |blob: &[u8]| blob[body_start(blob)..planes_start(blob)].to_vec();
        assert_eq!(chain(&a), chain(&b));
    }

    #[test]
    fn bound_source_needs_the_frame_shape() {
        let raw = fp32_raw(256);
        let blob = compress_compact(&fp32_input(&raw, &[256]), &opts(false, false))
            .unwrap()
            .unwrap();
        // magic, flags, dtype 1, format 1, shape (1, 256 = 0x80 0x02), body.
        assert_eq!(
            &blob[FLAGS_AT..FLAGS_AT + 7],
            &[FLAG_SHAPE, 1, 1, 1, 0x80, 0x02, 3]
        );
        // The same body in a frame without a shape: the size is explicit and
        // the bound Source has nothing to bind to.
        let mut b = blob[..FLAGS_AT].to_vec();
        b.extend_from_slice(&[FLAG_SIZE, 1, 1]);
        write_varint(&mut b, raw.len() as u64);
        b.extend_from_slice(&blob[FLAGS_AT + 6..]);
        let err = decode_compact(&b).unwrap_err().to_string();
        assert!(err.contains("bound Source"), "{err}");
    }

    #[test]
    fn orig_size_is_stored_only_when_the_shape_does_not_give_it() {
        // A byte count the shape does not give: stored.
        let raw = vec![3u8; 64];
        let mut inp = input(&raw, Some(vec![10]), Some("float32"));
        inp.dtype_code = 1;
        let blob = compress_compact(&inp, &opts(true, true)).unwrap().unwrap();
        assert_ne!(blob[FLAGS_AT] & FLAG_SIZE, 0);
        let t = decode_compact(&blob).unwrap();
        assert_eq!((t.raw, t.shape), (raw.clone(), Some(vec![10])));

        // Packed fp4 and 0-d tensors derive it.
        for (shape, code, name, len) in [
            (vec![16u64], 31u16, "float4_e2m1fn_x2", 16usize),
            (vec![], 1, "float32", 4),
            (vec![0, 7], 6, "bfloat16", 0),
        ] {
            let raw = vec![0x11u8; len];
            let mut inp = input(&raw, Some(shape.clone()), Some(name));
            inp.dtype_code = code;
            let blob = compress_compact(&inp, &opts(true, false)).unwrap().unwrap();
            assert_eq!(blob[FLAGS_AT] & FLAG_SIZE, 0, "{shape:?}");
            let t = decode_compact(&blob).unwrap();
            assert_eq!((t.raw, t.shape), (raw, Some(shape)));
        }
    }

    #[test]
    fn non_canonical_headers_are_rejected() {
        let raw = fp32_raw(256);
        let blob = compress_compact(&fp32_input(&raw, &[256]), &opts(false, false))
            .unwrap()
            .unwrap();
        let rest = &blob[FLAGS_AT + 6..];
        let frame = |head: &[u8]| [&MAGIC_COMPACT[..], head, rest].concat();
        let check = |head: &[u8], needle: &str| {
            let err = decode_compact(&frame(head)).unwrap_err().to_string();
            assert!(err.contains(needle), "{head:02x?}: {err}");
        };
        // The original decodes.
        decode_compact(&frame(&[FLAG_SHAPE, 1, 1, 1, 0x80, 0x02])).unwrap();
        // orig_size stored although the shape gives it.
        check(
            &[FLAG_SHAPE | FLAG_SIZE, 1, 1, 0x80, 0x08, 1, 0x80, 0x02],
            "equals the size",
        );
        // Neither orig_size nor shape.
        check(&[0, 1, 1], "no orig_size");
        // Dtype name equal to the canonical name.
        let mut h = vec![FLAG_SHAPE | FLAG_DTYPE_NAME, 1, 1, 1, 0x80, 0x02, 7];
        h.extend_from_slice(b"float32");
        check(&h, "canonical name");
        // Shape without a name for a dtype code with no canonical name.
        check(
            &[
                FLAG_SHAPE | FLAG_SIZE,
                0xC8,
                0x01,
                1,
                0x80,
                0x08,
                1,
                0x80,
                0x02,
            ],
            "dtype name",
        );
        // Non-minimal varints.
        check(&[FLAG_SHAPE, 0x81, 0x00, 1, 1, 0x80, 0x02], "non-minimal");
        check(&[FLAG_SHAPE, 1, 1, 0x81, 0x00, 0x80, 0x02], "non-minimal");
        // dtype_code above u16, unknown flag bit.
        check(&[FLAG_SIZE, 0x80, 0x80, 0x04, 1, 0x80, 0x08], "dtype_code");
        check(&[FLAG_SHAPE | 0x80, 1, 1, 1, 0x80, 0x02], "unknown flag");
        // ndim above 64.
        check(&[FLAG_SHAPE, 1, 1, 65], "ndim");
    }

    #[test]
    fn plane_state_codes_are_checked() {
        let raw: Vec<u8> = (0..4096u32).map(|i| (i % 7) as u8).collect();
        let mut o = opts(false, false);
        o.forced_codec = Some(CodecId::Zstd);
        let blob = compress_compact(&input(&raw, None, None), &o)
            .unwrap()
            .unwrap();
        assert_eq!(decode_compact(&blob).unwrap().raw, raw);
        let at = planes_start(&blob);
        // Zstd carries an empty version-0 inline state: state code 2.
        assert_eq!(blob[at], (4 << 2) | 2);
        let with_head = |head: &[u8]| [&blob[..at], head, &blob[at + 1..]].concat();
        // The same plane with state code 0 decodes: decode treats both
        // alike, and the frame keeps the state kind the codec reported, so
        // these two frames are both valid for one tensor.
        assert_eq!(decode_compact(&with_head(&[4 << 2])).unwrap().raw, raw);
        // State code 1 with an empty version-0 state is non-canonical.
        let err = decode_compact(&with_head(&[(4 << 2) | 1, 0, 0]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("state code 2"), "{err}");
        let err = decode_compact(&with_head(&[(4 << 2) | 3]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("unknown state code"), "{err}");
    }

    #[test]
    fn appended_bytes_are_rejected() {
        let fp32 = fp32_raw(1024);
        let bytes: Vec<u8> = (0..4096u32).map(|i| (i % 7) as u8).collect();
        let mut blobs = Vec::new();
        for (h, c) in [(true, true), (false, true), (false, false)] {
            blobs.push(
                compress_compact(&fp32_input(&fp32, &[1024]), &opts(h, c))
                    .unwrap()
                    .unwrap(),
            );
            blobs.push(
                compress_compact(&input(&bytes, None, None), &opts(h, c))
                    .unwrap()
                    .unwrap(),
            );
        }
        let mut zstd = opts(true, false);
        zstd.forced_codec = Some(CodecId::Zstd);
        blobs.push(
            compress_compact(&input(&bytes, None, None), &zstd)
                .unwrap()
                .unwrap(),
        );
        for blob in &blobs {
            assert_eq!(blob[FLAGS_AT] & FLAG_STORED, 0);
            let t = decode_compact(blob).unwrap();
            // Every payload byte is accounted for: the coded body ends with
            // the last payload.
            assert!(t.payload_len > 0);
            for cut in 0..blob.len() {
                assert!(decode_compact(&blob[..cut]).is_err(), "cut at {cut}");
            }
            for tail in [&[0u8][..], &[0u8; 7], &[0xFFu8; 100]] {
                let longer = [blob.as_slice(), tail].concat();
                let err = decode_compact(&longer).unwrap_err().to_string();
                assert!(err.contains("trailing"), "{} appended: {err}", tail.len());
            }
        }
        // A stored body must be exactly orig_size bytes.
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        let noise: Vec<u8> = (0..300)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x as u8
            })
            .collect();
        let stored = compress_compact(&input(&noise, None, None), &opts(true, true))
            .unwrap()
            .unwrap();
        assert_ne!(stored[FLAGS_AT] & FLAG_STORED, 0);
        assert!(decode_compact(&[stored.as_slice(), &[0]].concat()).is_err());
    }

    #[test]
    fn plane_crc_is_not_combined_with_the_payload_hash() {
        let raw = fp32_raw(256);
        let with_crc = compress_compact(&fp32_input(&raw, &[256]), &opts(false, true))
            .unwrap()
            .unwrap();
        assert_eq!(with_crc[FLAGS_AT], FLAG_SHAPE | FLAG_PLANE_CRC);
        decode_compact(&with_crc).unwrap();
        // Add the payload hash to the CRC frame: the writer never does.
        let at = body_start(&with_crc);
        let mut b = [
            &with_crc[..at],
            &xxh64(&raw, 0).to_le_bytes()[..],
            &with_crc[at..],
        ]
        .concat();
        b[FLAGS_AT] |= FLAG_PAYLOAD_HASH;
        let err = decode_compact(&b).unwrap_err().to_string();
        assert!(err.contains("plane CRC flag with payload hash"), "{err}");
    }
}
