//! Compact single-tensor frame (magic `\x89PTWS`).
//!
//! A full `.ptwm` container carries a fixed header, an Extension Table, a
//! prelude, a chain registry, a tensor index and an EOF sentinel. For one
//! small tensor those sections cost more than the payload. The compact frame
//! holds one tensor with only the fields its decoder needs:
//!
//! ```text
//!     magic:          [u8; 5]   \x89PTWS
//!     flags:          u8        see FLAG_*; unknown bits are rejected
//!     dtype_code:     varint
//!     input_format:   u8
//!     orig_size:      varint    raw byte count
//!     [shape]         varint ndim, then varint per dimension   (FLAG_SHAPE)
//!     [dtype_name]    varint len, then UTF-8 bytes           (FLAG_DTYPE_NAME)
//!     [delta_ref]     [u8; 32]  BLAKE3 of the delta reference (FLAG_DELTA_REF)
//!     [payload_hash]  u64       xxh64 of the raw bytes      (FLAG_PAYLOAD_HASH)
//!     body
//! ```
//!
//! When `FLAG_STORED` is set, the body is the `orig_size` raw bytes. Else the
//! body is a coded tensor:
//!
//! ```text
//!     n_ops:          u8
//!     op_ids:         [u16; n_ops]    OpId wire ids; the chain's u16 op
//!                                     indices refer to this list
//!     chain:          chain::wire::write_chain bytes
//!     per terminal plane, in chain terminal order:
//!         codec_id:       varint      CodecId wire id
//!         state_source:   u8          0 = none, 1 = inline
//!         [state_version] u8          (inline only)
//!         [state_len]     varint      (inline only)
//!         [state]         bytes       (inline only)
//!         payload_len:    varint
//!         [crc32]         u32         CRC32 of the raw plane (FLAG_PLANE_CRC)
//!         payload:        bytes
//! ```
//!
//! All integers are little-endian; varints are unsigned LEB128. The frame
//! ends exactly after the body; trailing bytes are rejected.
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
use crate::chain::wire::{read_chain, write_chain};
use crate::codec::{CodecId, StateSource, codec_for};
use crate::compressor::{CompressorOptions, InputTensor, legacy_plane_layout, plan_model};
use crate::dtype::Dtype;
use crate::error::PtwmCoreError;
use crate::extension::dispatch::{BuiltinKind, dispatch_builtin};
use crate::extension::{ExtensionTable, ExtensionTableBuilder, builtin_entries};
use crate::transforms::op::{OpId, Plane};

use std::sync::Arc;
use xxhash_rust::xxh64::xxh64;

/// Compact-frame magic. Shares the `\x89PTW` prefix with the full container.
pub const MAGIC_COMPACT: [u8; 5] = [0x89, b'P', b'T', b'W', b'S'];

/// The frame carries an xxh64 of the raw bytes.
pub const FLAG_PAYLOAD_HASH: u8 = 0x01;
/// Every coded plane carries a CRC32 of its raw bytes.
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
const KNOWN_FLAGS: u8 = FLAG_PAYLOAD_HASH
    | FLAG_PLANE_CRC
    | FLAG_SHAPE
    | FLAG_DTYPE_NAME
    | FLAG_DELTA_REF
    | FLAG_STORED;

/// Upper bound on the dimension count, so a corrupt `ndim` cannot drive a
/// large allocation.
const MAX_NDIM: u64 = 64;

/// A tensor decoded from a compact frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactTensor {
    pub dtype_code: u16,
    pub input_format: u8,
    /// Shape, when the writer supplied one.
    pub shape: Option<Vec<u64>>,
    /// Dtype name paired with `shape`. When the frame does not store a name,
    /// this is the canonical name of `dtype_code` (or `None` if the code is
    /// unknown).
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

    let mut out = Vec::with_capacity(64);
    out.extend_from_slice(&MAGIC_COMPACT);
    let flags_pos = out.len();
    out.push(0);
    let mut flags = 0u8;
    write_varint(&mut out, u64::from(input.dtype_code));
    out.push(input.input_format);
    write_varint(&mut out, raw.len() as u64);
    // Shape metadata is recorded only when both shape and dtype name are
    // known, matching the full container's tensor metadata.
    if let (Some(shape), Some(name)) = (&input.shape, &input.dtype_name) {
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
    }
    if let Some(h) = &input.delta_reference_blake3 {
        flags |= FLAG_DELTA_REF;
        out.extend_from_slice(h);
    }
    if opts.emit_payload_hash {
        flags |= FLAG_PAYLOAD_HASH;
        out.extend_from_slice(&xxh64(raw, 0).to_le_bytes());
    }

    let coded = encode_body(input, opts)?;
    match coded {
        None => return Ok(None),
        Some(body) if body.len() < raw.len() => {
            if opts.emit_plane_crc {
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

/// Encode the coded body (ops, chain, planes). `Ok(None)` when a plane needs
/// shared state.
fn encode_body(
    input: &InputTensor<'_>,
    opts: &CompressorOptions,
) -> Result<Option<Vec<u8>>, PtwmCoreError> {
    let plan = plan_model(std::slice::from_ref(input), opts)?;
    if !plan.prelude_entries.is_empty() {
        return Ok(None);
    }
    let chain: &Chain = &input.candidate_chains[plan.pass1[0].chain_idx];
    let planes = &plan.chosen_per_tensor[0];

    let mut builder = ExtensionTableBuilder::new();
    let mut chain_bytes = Vec::new();
    write_chain(chain, &mut builder, &mut chain_bytes)?;
    let entries = builder.finish();
    let n_ops = u8::try_from(entries.len()).map_err(|_| {
        PtwmCoreError::InvalidContainer(format!(
            "compact frame: chain uses {} ops (max 255)",
            entries.len()
        ))
    })?;

    let mut body = Vec::with_capacity(
        16 + chain_bytes.len()
            + planes
                .iter()
                .map(|p| p.payload_bytes.len() + 16)
                .sum::<usize>(),
    );
    body.push(n_ops);
    for entry in &entries {
        let op = match dispatch_builtin(&entry.canonical_id) {
            Some(BuiltinKind::Op(op)) => op,
            _ => {
                return Err(PtwmCoreError::InvalidContainer(
                    "compact frame: chain op is not a built-in transform".into(),
                ));
            }
        };
        body.extend_from_slice(&op.as_u16().to_le_bytes());
    }
    body.extend_from_slice(&chain_bytes);

    for cp in planes {
        write_varint(&mut body, u64::from(cp.codec_id.as_u16()));
        match cp.state_source {
            StateSource::None => body.push(0),
            StateSource::Inline => {
                body.push(1);
                body.push(cp.encoded_state_version);
                write_varint(&mut body, cp.encoded_state_bytes.len() as u64);
                body.extend_from_slice(&cp.encoded_state_bytes);
            }
            StateSource::Shared | StateSource::External => return Ok(None),
        }
        write_varint(&mut body, cp.payload_bytes.len() as u64);
        if opts.emit_plane_crc {
            body.extend_from_slice(&crc32fast::hash(&cp.raw_plane_for_crc).to_le_bytes());
        }
        body.extend_from_slice(&cp.payload_bytes);
    }
    Ok(Some(body))
}

/// Decode a compact frame. Verifies the payload hash and plane CRCs when the
/// frame carries them.
pub fn decode_compact(buf: &[u8]) -> Result<CompactTensor, PtwmCoreError> {
    let mut r = Reader { buf, pos: 0 };
    if !is_compact(buf) {
        return Err(bad("bad magic"));
    }
    r.pos = MAGIC_COMPACT.len();
    let flags = r.u8()?;
    if flags & !KNOWN_FLAGS != 0 {
        return Err(bad(&format!(
            "unknown flag bits {:#04x}",
            flags & !KNOWN_FLAGS
        )));
    }
    let dtype_code = u16::try_from(r.varint()?).map_err(|_| bad("dtype_code exceeds u16"))?;
    let input_format = r.u8()?;
    let orig_size = r.varint()?;

    let mut shape = None;
    let mut dtype_name = None;
    if flags & FLAG_SHAPE != 0 {
        let ndim = r.varint()?;
        if ndim > MAX_NDIM {
            return Err(bad(&format!("ndim {ndim} exceeds {MAX_NDIM}")));
        }
        let mut dims = Vec::with_capacity(ndim as usize);
        for _ in 0..ndim {
            dims.push(r.varint()?);
        }
        shape = Some(dims);
        dtype_name = canonical_dtype_name(dtype_code).map(str::to_owned);
    }
    if flags & FLAG_DTYPE_NAME != 0 {
        if flags & FLAG_SHAPE == 0 {
            return Err(bad("dtype name without shape"));
        }
        let len = r.len_varint()?;
        let bytes = r.take(len)?;
        let name = std::str::from_utf8(bytes).map_err(|_| bad("dtype name is not UTF-8"))?;
        dtype_name = Some(name.to_owned());
    }
    let delta_reference_blake3 = if flags & FLAG_DELTA_REF != 0 {
        let mut h = [0u8; 32];
        h.copy_from_slice(r.take(32)?);
        Some(h)
    } else {
        None
    };
    let payload_hash = if flags & FLAG_PAYLOAD_HASH != 0 {
        Some(u64::from_le_bytes(r.take(8)?.try_into().unwrap()))
    } else {
        None
    };

    let mut payload_len = 0u64;
    let raw = if flags & FLAG_STORED != 0 {
        if flags & FLAG_PLANE_CRC != 0 {
            return Err(bad("stored frame with plane CRC flag"));
        }
        let rest = r.rest();
        r.pos = buf.len();
        if rest.len() as u64 != orig_size {
            return Err(bad(&format!(
                "stored body is {} bytes, header says {orig_size}",
                rest.len()
            )));
        }
        payload_len = orig_size;
        rest.to_vec()
    } else {
        decode_body(
            &mut r,
            flags & FLAG_PLANE_CRC != 0,
            orig_size,
            &mut payload_len,
        )?
    };
    if r.pos != buf.len() {
        return Err(bad("trailing bytes after body"));
    }

    if let Some(expected) = payload_hash
        && xxh64(&raw, 0) != expected
    {
        return Err(bad("payload hash mismatch"));
    }

    Ok(CompactTensor {
        dtype_code,
        input_format,
        shape,
        dtype_name,
        delta_reference_blake3,
        raw,
        payload_len,
    })
}

fn decode_body(
    r: &mut Reader<'_>,
    has_crc: bool,
    orig_size: u64,
    payload_len_out: &mut u64,
) -> Result<Vec<u8>, PtwmCoreError> {
    let n_ops = r.u8()? as usize;
    let all = builtin_entries();
    let mut table = ExtensionTable::empty();
    for _ in 0..n_ops {
        let wire = u16::from_le_bytes(r.take(2)?.try_into().unwrap());
        let op = OpId::from_u16(wire)?;
        let id = op.canonical_id();
        let entry = all
            .iter()
            .find(|e| e.canonical_id == id)
            .ok_or_else(|| bad(&format!("op {wire:#06x} has no built-in entry")))?;
        table.entries.push(entry.clone());
    }
    let (chain, consumed) = read_chain(r.rest(), &table)?;
    r.pos += consumed;
    validate_chain(&chain, 0)?;
    // Writers check that the chain's Source declares exactly the raw byte
    // count, so a frame whose Source disagrees with `orig_size` is corrupt.
    // Checking before the descriptor walk keeps a corrupt chain from sizing
    // planes (and codec allocations) beyond the tensor.
    let source = chain_source_descriptor(&chain)?;
    if source.length_bytes != orig_size {
        return Err(bad(&format!(
            "chain Source declares {} bytes, header says {orig_size}",
            source.length_bytes
        )));
    }
    let descriptors = propagate_terminal_descriptors_bounded(&chain, max_plane_len(orig_size))?;

    let mut terminal_planes: Vec<Plane> = Vec::with_capacity(descriptors.len());
    for descriptor in descriptors {
        let wire = u16::try_from(r.varint()?).map_err(|_| bad("codec id exceeds u16"))?;
        let codec_id =
            CodecId::from_u16(wire).ok_or_else(|| bad(&format!("unknown codec {wire:#06x}")))?;
        let (state_version, state) = match r.u8()? {
            0 => (0u8, &[][..]),
            1 => {
                let v = r.u8()?;
                let len = r.len_varint()?;
                (v, r.take(len)?)
            }
            s => return Err(bad(&format!("unknown state source {s}"))),
        };
        let payload_len = r.len_varint()?;
        let crc = if has_crc {
            Some(u32::from_le_bytes(r.take(4)?.try_into().unwrap()))
        } else {
            None
        };
        let payload = r.take(payload_len)?;
        *payload_len_out += (state.len() + payload.len()) as u64;

        let codec = codec_for(codec_id)
            .ok_or_else(|| bad(&format!("no codec implementation for {codec_id:?}")))?;
        let layout = legacy_plane_layout(descriptor.layout);
        let decoded_len = usize::try_from(descriptor.length_bytes)
            .map_err(|_| bad("plane length overflows usize"))?;
        let bytes = codec.decode(state_version, state, payload, &layout, decoded_len)?;
        // The Source is exact, so every plane decodes to its descriptor's
        // length.
        if bytes.len() != decoded_len {
            return Err(bad(&format!(
                "plane decoded to {} bytes, chain says {decoded_len}",
                bytes.len()
            )));
        }
        if let Some(expected) = crc
            && crc32fast::hash(&bytes) != expected
        {
            return Err(bad("plane CRC mismatch"));
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
        return Err(bad(&format!(
            "decoded {} bytes, header says {orig_size}",
            source.bytes.len()
        )));
    }
    Ok(source.bytes.as_ref().to_vec())
}

fn bad(msg: &str) -> PtwmCoreError {
    PtwmCoreError::InvalidContainer(format!("compact frame: {msg}"))
}

fn write_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], PtwmCoreError> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.buf.len())
            .ok_or_else(|| bad("truncated"))?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8, PtwmCoreError> {
        Ok(self.take(1)?[0])
    }

    fn rest(&self) -> &'a [u8] {
        &self.buf[self.pos..]
    }

    fn varint(&mut self) -> Result<u64, PtwmCoreError> {
        let mut v: u64 = 0;
        for i in 0..10 {
            let b = self.u8()?;
            let bits = u64::from(b & 0x7F);
            if i == 9 && b > 1 {
                return Err(bad("varint overflows u64"));
            }
            v |= bits << (7 * i);
            if b & 0x80 == 0 {
                return Ok(v);
            }
        }
        Err(bad("varint overflows u64"))
    }

    /// A varint used as a byte length; must fit in the remaining buffer.
    fn len_varint(&mut self) -> Result<usize, PtwmCoreError> {
        let v = self.varint()?;
        let remaining = (self.buf.len() - self.pos) as u64;
        if v > remaining {
            return Err(bad("length exceeds remaining bytes"));
        }
        Ok(v as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::graph::{ChainEdge, ChainNode, TerminalRef};
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

    #[test]
    fn varint_round_trips() {
        for v in [0u64, 1, 127, 128, 300, u32::MAX as u64, u64::MAX] {
            let mut out = Vec::new();
            write_varint(&mut out, v);
            let mut r = Reader { buf: &out, pos: 0 };
            assert_eq!(r.varint().unwrap(), v);
            assert_eq!(r.pos, out.len());
        }
    }
}
