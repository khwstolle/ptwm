//! Tensor and plane records of the compact container (`\x89PTWC`).
//!
//! A compact record holds the same [`TensorRecord`] as a legacy record, with
//! varints instead of fixed-width fields, without the name (the container
//! index stores it), and without fields the reader can derive. All varints
//! are canonical unsigned LEB128 ([`crate::varint`]); integers are
//! little-endian.
//!
//! ```text
//! TENSOR RECORD
//!     tflags          u8      TF_* bits; bits 6-7 must be 0
//!     [chain_id]      varint  registry index, < registry count (no TF_INLINE_CHAIN)
//!     dtype_code      varint  at most u16::MAX
//!     input_format    u8
//!     orig_size       varint
//!     [shape]         varint ndim (<= 64), then varint per dimension (TF_SHAPE)
//!     [dtype_name]    varint len, then UTF-8 bytes          (TF_DTYPE_NAME)
//!     [raw_meta]      varint len, then bytes                (TF_RAW_META)
//!     [payload_hash]  u64                                   (TF_PAYLOAD_HASH)
//!     [inline chain]  varint et_len, compact Extension Table of the chain's
//!                     ops in first-use node order, varint chain_len, CW2
//!                     chain                               (TF_INLINE_CHAIN)
//!     [deps]          varint count >= 1, then per dependency: ref_kind u8,
//!                     ref_flags u8 (bit 0: hash), varint ref_len, bytes,
//!                     [hash 32 B]                               (TF_DEPS)
//!     n_planes        varint  at most 255
//!     plane × n_planes
//!
//! PLANE
//!     pflags          u8      PF_* bits, state source in bits 2-3; bits 6-7
//!                             must be 0
//!     role            u8      PlaneRole
//!     codec_ref       varint  Extension Table index of a plane codec
//!     [state_version] u8                                  (PF_STATE_EXTRA)
//!     [state_info]    varint  at most u16::MAX            (PF_STATE_EXTRA)
//!     [inline_len]    varint  inline state length         (state source Inline)
//!     payload_len     varint
//!     [row_len]       varint  1..=u32::MAX                (PF_LAYOUT_ROWS)
//!     [crc32]         u32                                 (PF_CRC32)
//!     [chunks]        varint n (<= payload_len), then n × (varint offset,
//!                     varint decoded_size), each at most u32::MAX (PF_CHUNKS)
//!     [external]      ref_kind u8, ref_flags u8 (bit 0: hash), varint
//!                     ref_len, bytes, state_format_version u8, [hash 32 B]
//!                                                         (state source External)
//!     [inline state]  bytes
//!     payload         bytes
//! ```
//!
//! **Metadata.** `TF_SHAPE` stands for the CBOR shape map of
//! [`encode_shape_metadata`] with the canonical dtype name of `dtype_code`,
//! or with the stored name under `TF_DTYPE_NAME` (which must then differ
//! from the canonical name). Other metadata is stored as `TF_RAW_META`. The
//! writer uses `TF_SHAPE` exactly when it reproduces the record's metadata
//! bytes, so the reader rejects `TF_RAW_META` bytes that `TF_SHAPE` could
//! store.
//!
//! **Chains.** A registry chain may have a bound Source (see
//! [`crate::chain::wire_compact::write_chain_compact_deferred`]); the record
//! then needs `TF_SHAPE` and a shape and dtype that bind
//! ([`bound_source_params`]). An inline chain binds its Source to the
//! record the same way as the `\x89PTWS` frame does.
//!
//! **Derived fields.** The in-memory `flags`, `chain_ref` (`0x8000` for an
//! inline chain), `payload_len`, and each plane's `codec_id` (from the
//! Extension Table entry its `codec_ref` names) are not stored. The writer
//! rejects a record whose stored values disagree with them, so each record
//! has one encoding and parsing then writing gives the same bytes.

use crate::chain::Chain;
use crate::chain::wire_compact::{read_chain_compact_exact, write_chain_compact};
use crate::codec::{CodecId, StateSource};
use crate::dtype::canonical_dtype_name;
use crate::error::PtwmCoreError;
use crate::extension::compact_table::{read_compact_table, write_compact_table};
use crate::extension::{
    BuiltinKind, ExtensionTable, Kind, builtin_registry, dispatch_builtin,
};
use crate::layout::PlaneLayout;
use crate::metadata::{decode_shape_metadata, encode_shape_metadata};
use crate::plane_record::{ChunkEntry, ExternalStateRef, PlaneRecord};
use crate::tensor_record::{
    CHAIN_REF_INLINE_BIT, Dependency, RefKind, TENSOR_FLAG_DEPENDENCIES, TENSOR_FLAG_METADATA,
    TENSOR_FLAG_PAYLOAD_HASH, TensorRecord,
};
use crate::transforms::op::OpId;
use crate::transforms::source::bound_source_params;
use crate::types::PlaneRole;
use crate::varint::{Cursor, write_varint};

pub const TF_PAYLOAD_HASH: u8 = 0x01;
pub const TF_INLINE_CHAIN: u8 = 0x02;
pub const TF_DEPS: u8 = 0x04;
pub const TF_SHAPE: u8 = 0x08;
pub const TF_DTYPE_NAME: u8 = 0x10;
pub const TF_RAW_META: u8 = 0x20;
const TF_KNOWN: u8 =
    TF_PAYLOAD_HASH | TF_INLINE_CHAIN | TF_DEPS | TF_SHAPE | TF_DTYPE_NAME | TF_RAW_META;

pub const PF_CRC32: u8 = 0x01;
pub const PF_CHUNKS: u8 = 0x02;
/// Plane state source (`StateSource` value) in bits 2-3.
pub const PF_STATE_SHIFT: u8 = 2;
pub const PF_STATE_MASK: u8 = 0x0C;
pub const PF_LAYOUT_ROWS: u8 = 0x10;
pub const PF_STATE_EXTRA: u8 = 0x20;
const PF_KNOWN: u8 = PF_CRC32 | PF_CHUNKS | PF_STATE_MASK | PF_LAYOUT_ROWS | PF_STATE_EXTRA;

/// Largest dimension count a compact record stores. Shape metadata with
/// more dimensions is stored as `TF_RAW_META`.
pub const MAX_NDIM: usize = 64;
/// Largest plane count (the legacy record stores it as `u8`).
const MAX_PLANES: u64 = 255;
/// Smallest compact record: tflags, dtype_code, input_format, orig_size,
/// n_planes would be 5; the index requires at least 3.
pub const MIN_RECORD_LEN: u64 = 3;

const DEP_FLAG_HASH: u8 = 0x01;

/// What a compact record refers to outside itself.
#[derive(Debug, Clone, Copy)]
pub struct RecordContext<'a> {
    /// Per Extension Table index: the codec id of a plane that references
    /// the entry, or `None` when the entry is not a plane codec. See
    /// [`plane_codec_ids`].
    pub codec_ids: &'a [Option<CodecId>],
    /// Per registry chain: whether it has a bound Source that the record
    /// must supply.
    pub registry_bound: &'a [bool],
}

/// The codec id each Extension Table entry gives a plane that references
/// it: the built-in codec for a built-in codec entry, `Identity` for a
/// third-party plane codec (which has no wire id), and `None` for entries
/// that are not plane codecs.
pub fn plane_codec_ids(table: &ExtensionTable) -> Vec<Option<CodecId>> {
    table
        .entries
        .iter()
        .map(|e| match dispatch_builtin(&e.canonical_id) {
            Some(BuiltinKind::Codec(c)) => Some(c),
            Some(BuiltinKind::Op(_)) => None,
            None if e.kind == Kind::PlaneCodec => Some(CodecId::Identity),
            None => None,
        })
        .collect()
}

fn err(msg: impl std::fmt::Display) -> PtwmCoreError {
    PtwmCoreError::InvalidContainer(format!("compact tensor record: {msg}"))
}

/// How a record's metadata is stored.
enum Meta<'a> {
    None,
    Shape {
        shape: Vec<u64>,
        /// Stored name; `None` when it is the canonical name.
        stored_name: Option<String>,
    },
    Raw(&'a [u8]),
}

fn classify_meta(meta: Option<&[u8]>, dtype_code: u16) -> Meta<'_> {
    let Some(m) = meta else {
        return Meta::None;
    };
    match decode_shape_metadata(m) {
        Ok((shape, name))
            if shape.len() <= MAX_NDIM && encode_shape_metadata(&shape, &name) == m =>
        {
            let stored_name = (canonical_dtype_name(dtype_code) != Some(name.as_str())).then_some(name);
            Meta::Shape { shape, stored_name }
        }
        _ => Meta::Raw(m),
    }
}

/// Bound Source params for a record's shape metadata as the compact
/// container stores it: `Some` only when the metadata is stored as
/// `TF_SHAPE` and the shape and dtype bind.
pub fn record_binding(meta: Option<&[u8]>, dtype_code: u16) -> Option<Vec<u8>> {
    match classify_meta(meta, dtype_code) {
        Meta::Shape { shape, .. } => bound_source_params(&shape, dtype_code),
        _ => None,
    }
}

/// Ops of `chain` in first-use node order, without repeats.
fn ops_in_first_use_order(chain: &Chain) -> Vec<OpId> {
    let mut ops: Vec<OpId> = Vec::new();
    for n in &chain.nodes {
        if !ops.contains(&n.op) {
            ops.push(n.op);
        }
    }
    ops
}

/// Append the compact encoding of `rec` to `out`. `rec.name` is not
/// written. On error `out` is left unchanged.
pub fn write_compact_record(
    rec: &TensorRecord,
    ctx: &RecordContext<'_>,
    out: &mut Vec<u8>,
) -> Result<(), PtwmCoreError> {
    let start = out.len();
    let result = write_unchecked(rec, ctx, out);
    if result.is_err() {
        out.truncate(start);
    }
    result
}

fn write_unchecked(
    rec: &TensorRecord,
    ctx: &RecordContext<'_>,
    out: &mut Vec<u8>,
) -> Result<(), PtwmCoreError> {
    let inline = rec.chain_ref & CHAIN_REF_INLINE_BIT != 0;
    if inline != rec.inline_chain.is_some() {
        return Err(err("chain_ref inline bit and inline_chain disagree"));
    }
    if inline && rec.chain_ref != CHAIN_REF_INLINE_BIT {
        return Err(err(format_args!(
            "inline chain_ref {:#06x} has low bits set; the compact record stores none",
            rec.chain_ref
        )));
    }
    if rec.terminals.len() as u64 > MAX_PLANES {
        return Err(err(format_args!("{} planes (max {MAX_PLANES})", rec.terminals.len())));
    }
    if rec.dependencies.len() > u16::MAX as usize {
        return Err(err(format_args!(
            "{} dependencies (max {})",
            rec.dependencies.len(),
            u16::MAX
        )));
    }
    if let Some(m) = &rec.tensor_metadata
        && m.len() as u64 > u64::from(u32::MAX)
    {
        return Err(err("tensor metadata longer than u32::MAX bytes"));
    }
    let meta = classify_meta(rec.tensor_metadata.as_deref(), rec.dtype_code);
    let binding = match &meta {
        Meta::Shape { shape, .. } => bound_source_params(shape, rec.dtype_code),
        _ => None,
    };

    let mut tflags = 0u8;
    if rec.payload_hash.is_some() {
        tflags |= TF_PAYLOAD_HASH;
    }
    if inline {
        tflags |= TF_INLINE_CHAIN;
    }
    if !rec.dependencies.is_empty() {
        tflags |= TF_DEPS;
    }
    match &meta {
        Meta::None => {}
        Meta::Shape { stored_name, .. } => {
            tflags |= TF_SHAPE;
            if stored_name.is_some() {
                tflags |= TF_DTYPE_NAME;
            }
        }
        Meta::Raw(_) => tflags |= TF_RAW_META,
    }
    out.push(tflags);

    if !inline {
        let id = usize::from(rec.chain_ref);
        let bound = ctx.registry_bound.get(id).ok_or_else(|| {
            err(format_args!(
                "chain_ref {id} is not in the registry ({} chains)",
                ctx.registry_bound.len()
            ))
        })?;
        if *bound && binding.is_none() {
            return Err(err(format_args!(
                "registry chain {id} has a bound Source, but the record has no shape \
                 metadata that binds it"
            )));
        }
        write_varint(out, id as u64);
    }
    write_varint(out, u64::from(rec.dtype_code));
    out.push(rec.input_format);
    write_varint(out, rec.orig_size);
    match &meta {
        Meta::None => {}
        Meta::Shape { shape, stored_name } => {
            write_varint(out, shape.len() as u64);
            for &d in shape {
                write_varint(out, d);
            }
            if let Some(name) = stored_name {
                write_varint(out, name.len() as u64);
                out.extend_from_slice(name.as_bytes());
            }
        }
        Meta::Raw(m) => {
            write_varint(out, m.len() as u64);
            out.extend_from_slice(m);
        }
    }
    if let Some(h) = rec.payload_hash {
        out.extend_from_slice(&h.to_le_bytes());
    }
    if let Some(chain) = &rec.inline_chain {
        let ops = ops_in_first_use_order(chain);
        let reg = builtin_registry();
        let table = ExtensionTable {
            entries: ops
                .iter()
                .map(|op| {
                    reg.entry_by_name(op.builtin_name())
                        .cloned()
                        .ok_or_else(|| err(format_args!("op {op:?} has no built-in entry")))
                })
                .collect::<Result<_, _>>()?,
        };
        let mut et = Vec::new();
        write_compact_table(&table, &mut et)?;
        let mut chain_bytes = Vec::new();
        write_chain_compact(
            chain,
            binding.as_deref(),
            |op| {
                Ok(ops
                    .iter()
                    .position(|&o| o == op)
                    .expect("every op of the chain is listed") as u64)
            },
            &mut chain_bytes,
        )?;
        write_varint(out, et.len() as u64);
        out.extend_from_slice(&et);
        write_varint(out, chain_bytes.len() as u64);
        out.extend_from_slice(&chain_bytes);
    }
    if !rec.dependencies.is_empty() {
        write_varint(out, rec.dependencies.len() as u64);
        for dep in &rec.dependencies {
            if dep.ref_bytes.len() > u16::MAX as usize {
                return Err(err(format_args!(
                    "dependency ref of {} bytes (max {})",
                    dep.ref_bytes.len(),
                    u16::MAX
                )));
            }
            out.push(dep.ref_kind as u8);
            out.push(if dep.expected_hash.is_some() {
                DEP_FLAG_HASH
            } else {
                0
            });
            write_varint(out, dep.ref_bytes.len() as u64);
            out.extend_from_slice(&dep.ref_bytes);
            if let Some(h) = &dep.expected_hash {
                out.extend_from_slice(h);
            }
        }
    }
    write_varint(out, rec.terminals.len() as u64);
    for (i, plane) in rec.terminals.iter().enumerate() {
        write_plane(plane, ctx, out).map_err(|e| err(format_args!("plane {i}: {e}")))?;
    }
    Ok(())
}

fn write_plane(
    p: &PlaneRecord,
    ctx: &RecordContext<'_>,
    out: &mut Vec<u8>,
) -> Result<(), PtwmCoreError> {
    let idx = usize::from(p.codec_table_idx);
    match ctx.codec_ids.get(idx) {
        Some(Some(c)) if *c == p.codec_id => {}
        Some(Some(c)) => {
            return Err(err(format_args!(
                "codec_id {:?} differs from {c:?}, the codec of Extension Table entry {idx}",
                p.codec_id
            )));
        }
        Some(None) => {
            return Err(err(format_args!(
                "Extension Table entry {idx} is not a plane codec"
            )));
        }
        None => {
            return Err(err(format_args!(
                "codec_table_idx {idx} is outside the Extension Table ({} entries)",
                ctx.codec_ids.len()
            )));
        }
    }
    if p.payload_len != p.payload_bytes.len() as u64 {
        return Err(err(format_args!(
            "payload_len {} differs from the {} payload bytes",
            p.payload_len,
            p.payload_bytes.len()
        )));
    }
    let is_inline = p.state_source == StateSource::Inline;
    if !is_inline && !p.inline_state_bytes.is_empty() {
        return Err(err("inline state bytes without an Inline state source"));
    }
    let is_external = p.state_source == StateSource::External;
    if is_external != p.external_state.is_some() {
        return Err(err("external state ref and External state source disagree"));
    }
    let mut pflags = (p.state_source as u8) << PF_STATE_SHIFT;
    if p.crc32.is_some() {
        pflags |= PF_CRC32;
    }
    if p.chunk_table.is_some() {
        pflags |= PF_CHUNKS;
    }
    if p.layout.row_len().is_some() {
        pflags |= PF_LAYOUT_ROWS;
    }
    let extra = p.state_version != 0 || p.state_info != 0;
    if extra {
        pflags |= PF_STATE_EXTRA;
    }
    out.push(pflags);
    out.push(p.role as u8);
    write_varint(out, idx as u64);
    if extra {
        out.push(p.state_version);
        write_varint(out, u64::from(p.state_info));
    }
    if is_inline {
        write_varint(out, p.inline_state_bytes.len() as u64);
    }
    write_varint(out, p.payload_bytes.len() as u64);
    if let Some(row_len) = p.layout.row_len() {
        write_varint(out, u64::from(row_len));
    }
    if let Some(crc) = p.crc32 {
        out.extend_from_slice(&crc.to_le_bytes());
    }
    if let Some(chunks) = &p.chunk_table {
        if chunks.len() > p.payload_bytes.len() {
            return Err(err(format_args!(
                "{} chunks for a {}-byte payload",
                chunks.len(),
                p.payload_bytes.len()
            )));
        }
        write_varint(out, chunks.len() as u64);
        for c in chunks {
            write_varint(out, u64::from(c.offset_in_payload));
            write_varint(out, u64::from(c.decoded_size));
        }
    }
    if let Some(ext) = &p.external_state {
        if ext.ref_bytes.len() > u16::MAX as usize {
            return Err(err(format_args!(
                "external state ref of {} bytes (max {})",
                ext.ref_bytes.len(),
                u16::MAX
            )));
        }
        out.push(ext.ref_kind);
        out.push(if ext.expected_hash.is_some() {
            DEP_FLAG_HASH
        } else {
            0
        });
        write_varint(out, ext.ref_bytes.len() as u64);
        out.extend_from_slice(&ext.ref_bytes);
        out.push(ext.state_format_version);
        if let Some(h) = &ext.expected_hash {
            out.extend_from_slice(h);
        }
    }
    out.extend_from_slice(&p.inline_state_bytes);
    out.extend_from_slice(&p.payload_bytes);
    Ok(())
}

/// Parse a compact record that fills `buf` exactly. The returned record has
/// an empty `name`; the container index supplies it.
pub fn parse_compact_record(
    buf: &[u8],
    ctx: &RecordContext<'_>,
) -> Result<TensorRecord, PtwmCoreError> {
    let mut r = Cursor::new(buf, "compact tensor record");
    let tflags = r.u8()?;
    if tflags & !TF_KNOWN != 0 {
        return Err(r.err(format_args!(
            "unknown tflags bits {:#04x}",
            tflags & !TF_KNOWN
        )));
    }
    let inline = tflags & TF_INLINE_CHAIN != 0;
    if tflags & TF_DTYPE_NAME != 0 && tflags & TF_SHAPE == 0 {
        return Err(r.err("TF_DTYPE_NAME without TF_SHAPE"));
    }
    if tflags & TF_RAW_META != 0 && tflags & TF_SHAPE != 0 {
        return Err(r.err("TF_RAW_META with TF_SHAPE"));
    }

    let chain_id = if inline {
        None
    } else {
        let n = ctx.registry_bound.len() as u64;
        let id = r.varint()?;
        if id >= n {
            return Err(r.err(format_args!("chain id {id} is not in the registry ({n} chains)")));
        }
        Some(id as usize)
    };
    let dtype_code = r.varint_max(u64::from(u16::MAX), "dtype_code")? as u16;
    let input_format = r.u8()?;
    let orig_size = r.varint()?;

    let mut tensor_metadata = None;
    let mut binding = None;
    if tflags & TF_SHAPE != 0 {
        let ndim = r.varint_max(MAX_NDIM as u64, "ndim")? as usize;
        if ndim > r.remaining() {
            return Err(r.err("truncated shape"));
        }
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            shape.push(r.varint()?);
        }
        let canonical = canonical_dtype_name(dtype_code);
        let name: String = if tflags & TF_DTYPE_NAME != 0 {
            let bytes = r.len_bytes()?;
            let name =
                std::str::from_utf8(bytes).map_err(|_| r.err("dtype name is not UTF-8"))?;
            if canonical == Some(name) {
                return Err(r.err("stored dtype name equals the canonical name (non-canonical)"));
            }
            name.to_owned()
        } else {
            canonical
                .ok_or_else(|| {
                    r.err(format_args!(
                        "shape without a dtype name for dtype_code {dtype_code}, which has no \
                         canonical name"
                    ))
                })?
                .to_owned()
        };
        binding = bound_source_params(&shape, dtype_code);
        tensor_metadata = Some(encode_shape_metadata(&shape, &name));
    } else if tflags & TF_RAW_META != 0 {
        let bytes = r.len_bytes()?;
        if bytes.len() as u64 > u64::from(u32::MAX) {
            return Err(r.err("raw metadata longer than u32::MAX bytes"));
        }
        if !matches!(classify_meta(Some(bytes), dtype_code), Meta::Raw(_)) {
            return Err(r.err("raw metadata that TF_SHAPE stores (non-canonical)"));
        }
        tensor_metadata = Some(bytes.to_vec());
    }

    if let Some(id) = chain_id
        && ctx.registry_bound[id]
        && binding.is_none()
    {
        return Err(r.err(format_args!(
            "registry chain {id} has a bound Source, but the record has no shape that binds it"
        )));
    }

    let payload_hash = if tflags & TF_PAYLOAD_HASH != 0 {
        Some(r.u64_le()?)
    } else {
        None
    };

    let inline_chain = if inline {
        let et_bytes = r.len_bytes()?;
        let table = read_compact_table(et_bytes)
            .map_err(|e| r.err(format_args!("inline chain: {e}")))?;
        let ops: Vec<OpId> = table
            .entries
            .iter()
            .map(|e| match dispatch_builtin(&e.canonical_id) {
                Some(BuiltinKind::Op(op)) => Ok(op),
                _ => Err(r.err(format_args!(
                    "inline chain table entry {:?} is not a transform",
                    e.human_label
                ))),
            })
            .collect::<Result<_, _>>()?;
        let chain_bytes = r.len_bytes()?;
        let chain = read_chain_compact_exact(chain_bytes, binding.as_deref(), |i| {
            usize::try_from(i)
                .ok()
                .and_then(|i| ops.get(i).copied())
                .ok_or_else(|| err(format_args!("inline chain op_ref {i} outside its table")))
        })?;
        if ops_in_first_use_order(&chain) != ops {
            return Err(r.err(
                "inline chain table does not list the ops in first-use order (non-canonical)",
            ));
        }
        Some(chain)
    } else {
        None
    };

    let mut dependencies = Vec::new();
    if tflags & TF_DEPS != 0 {
        let n = r.varint_max(u64::from(u16::MAX), "dependency count")?;
        if n == 0 {
            return Err(r.err("TF_DEPS with no dependencies"));
        }
        // Each dependency takes at least 3 bytes.
        if n > (r.remaining() / 3) as u64 {
            return Err(r.err("dependency count exceeds the record"));
        }
        dependencies.reserve_exact(n as usize);
        for _ in 0..n {
            let kind = r.u8()?;
            let ref_kind = RefKind::from_u8(kind)
                .ok_or_else(|| r.err(format_args!("unknown ref_kind {kind:#04x}")))?;
            let flags = r.u8()?;
            if flags & !DEP_FLAG_HASH != 0 {
                return Err(r.err(format_args!("unknown dependency flag bits {flags:#04x}")));
            }
            let len = r.varint_max(u64::from(u16::MAX), "dependency ref length")? as usize;
            let ref_bytes = r.take(len)?.to_vec();
            let expected_hash = if flags & DEP_FLAG_HASH != 0 {
                Some(r.array::<32>()?)
            } else {
                None
            };
            dependencies.push(Dependency {
                ref_kind,
                expected_hash,
                ref_bytes,
            });
        }
    }

    let n_planes = r.varint_max(MAX_PLANES, "plane count")?;
    // Each plane takes at least 4 bytes.
    if n_planes > (r.remaining() / 4) as u64 {
        return Err(r.err("plane count exceeds the record"));
    }
    let mut terminals = Vec::with_capacity(n_planes as usize);
    for _ in 0..n_planes {
        terminals.push(read_plane(&mut r, ctx)?);
    }
    r.finish("the last plane")?;

    let mut flags = 0u8;
    if !dependencies.is_empty() {
        flags |= TENSOR_FLAG_DEPENDENCIES;
    }
    if tensor_metadata.is_some() {
        flags |= TENSOR_FLAG_METADATA;
    }
    if payload_hash.is_some() {
        flags |= TENSOR_FLAG_PAYLOAD_HASH;
    }
    Ok(TensorRecord {
        chain_ref: match chain_id {
            Some(id) => id as u16,
            None => CHAIN_REF_INLINE_BIT,
        },
        dtype_code,
        input_format,
        flags,
        orig_size,
        name: String::new(),
        payload_hash,
        inline_chain,
        dependencies,
        tensor_metadata,
        terminals,
    })
}

fn read_plane(r: &mut Cursor<'_>, ctx: &RecordContext<'_>) -> Result<PlaneRecord, PtwmCoreError> {
    let pflags = r.u8()?;
    if pflags & !PF_KNOWN != 0 {
        return Err(r.err(format_args!(
            "unknown pflags bits {:#04x}",
            pflags & !PF_KNOWN
        )));
    }
    let state_source = StateSource::from_u8((pflags & PF_STATE_MASK) >> PF_STATE_SHIFT)
        .expect("two bits always name a state source");
    let role_byte = r.u8()?;
    let role = PlaneRole::from_u8(role_byte)
        .ok_or_else(|| r.err(format_args!("unknown plane role {role_byte:#04x}")))?;
    let codec_ref = r.varint_max(u64::from(u16::MAX), "codec_ref")? as u16;
    let codec_id = match ctx.codec_ids.get(usize::from(codec_ref)) {
        Some(Some(c)) => *c,
        Some(None) => {
            return Err(r.err(format_args!(
                "codec_ref {codec_ref} names an entry that is not a plane codec"
            )));
        }
        None => {
            return Err(r.err(format_args!(
                "codec_ref {codec_ref} is outside the Extension Table ({} entries)",
                ctx.codec_ids.len()
            )));
        }
    };
    let (state_version, state_info) = if pflags & PF_STATE_EXTRA != 0 {
        let v = r.u8()?;
        let info = r.varint_max(u64::from(u16::MAX), "state_info")? as u16;
        if v == 0 && info == 0 {
            return Err(r.err("PF_STATE_EXTRA with zero state_version and state_info"));
        }
        (v, info)
    } else {
        (0, 0)
    };
    let inline_len = if state_source == StateSource::Inline {
        r.len_varint()?
    } else {
        0
    };
    let payload_len = r.len_varint()?;
    let layout = if pflags & PF_LAYOUT_ROWS != 0 {
        let row_len = r.varint_max(u64::from(u32::MAX), "row_len")? as u32;
        PlaneLayout::rows(row_len).ok_or_else(|| r.err("row_len 0"))?
    } else {
        PlaneLayout::Flat
    };
    let crc32 = if pflags & PF_CRC32 != 0 {
        Some(r.u32_le()?)
    } else {
        None
    };
    let chunk_table = if pflags & PF_CHUNKS != 0 {
        let n = r.varint_max(payload_len as u64, "chunk count")?;
        // Each chunk entry takes at least 2 bytes.
        if n > (r.remaining() / 2) as u64 {
            return Err(r.err("chunk count exceeds the record"));
        }
        let mut chunks = Vec::with_capacity(n as usize);
        for _ in 0..n {
            let offset = r.varint_max(u64::from(u32::MAX), "chunk offset")? as u32;
            let size = r.varint_max(u64::from(u32::MAX), "chunk decoded size")? as u32;
            chunks.push(ChunkEntry {
                offset_in_payload: offset,
                decoded_size: size,
            });
        }
        Some(chunks)
    } else {
        None
    };
    let external_state = if state_source == StateSource::External {
        let ref_kind = r.u8()?;
        let flags = r.u8()?;
        if flags & !DEP_FLAG_HASH != 0 {
            return Err(r.err(format_args!("unknown external state flag bits {flags:#04x}")));
        }
        let len = r.varint_max(u64::from(u16::MAX), "external state ref length")? as usize;
        let ref_bytes = r.take(len)?.to_vec();
        let state_format_version = r.u8()?;
        let expected_hash = if flags & DEP_FLAG_HASH != 0 {
            Some(r.array::<32>()?)
        } else {
            None
        };
        Some(ExternalStateRef {
            ref_kind,
            expected_hash,
            ref_bytes,
            state_format_version,
        })
    } else {
        None
    };
    let inline_state_bytes = r.take(inline_len)?.to_vec();
    let payload_bytes = r.take(payload_len)?.to_vec();
    Ok(PlaneRecord {
        role,
        codec_id,
        codec_table_idx: codec_ref,
        state_source,
        state_version,
        state_info,
        payload_len: payload_len as u64,
        crc32,
        chunk_table,
        external_state,
        inline_state_bytes,
        payload_bytes,
        layout,
    })
}
