//! Sections of the compact container (`\x89PTWC`): header, chain registry,
//! index and footer. [`crate::container`] writes and reads whole files;
//! [`crate::extension::compact_table`] and [`crate::tensor_record_compact`]
//! hold the Extension Table and record codecs.
//!
//! ```text
//! FILE = HEADER | EXTENSION TABLE | PRELUDE | REGISTRY | BODY | INDEX | FOOTER
//!
//! HEADER
//!     magic           [u8; 9]   \x89PTWC\r\n\x1a\n
//!     hflags          u16       must be 0 (every bit is reserved)
//!     et_len          varint    compact Extension Table bytes
//!     prelude_len     varint    shared-state prelude bytes (legacy prelude layout)
//!     registry_len    varint    chain registry bytes
//!     meta_xxh64      u64       xxh64 (seed 0) of the header bytes before this
//!                               field, then the Extension Table, prelude and
//!                               registry bytes
//! REGISTRY
//!     count           varint    at most 32768
//!     chain × count   varint chain_len, then a CW2 chain (chain::wire_compact)
//!                     whose op_ref is an Extension Table index of a
//!                     built-in transform; chain id = position
//! BODY                tensor records (tensor_record_compact), back to back
//! INDEX
//!     count           varint    at most (index bytes) / 3
//!     entry × count   varint name_len (1..=65535), UTF-8 name, varint rec_len
//!                     (>= 3), in body order
//! FOOTER
//!     index_len       u64
//!     sentinel        [u8; 8]   \x89TWMEOF\x1a (the legacy EOF sentinel)
//! ```
//!
//! The body starts right after the registry and the index right after the
//! body: the record lengths must add up to exactly the bytes between them,
//! and names must be unique. Each section parser gets exactly its slice and
//! must consume all of it. Integers are little-endian and varints are
//! canonical unsigned LEB128 ([`crate::varint`]), so each file has one
//! encoding for what it holds.
//!
//! There is no version field. A future layout change sets an `hflags` bit,
//! which current readers reject.

use std::collections::HashSet;
use std::ops::Range;

use xxhash_rust::xxh64::Xxh64;

use crate::chain::wire_compact::{read_chain_compact_deferred_exact, write_chain_compact_deferred};
use crate::container::{ChainRegistry, MAX_REGISTRY_CHAINS};
use crate::error::PtwmCoreError;
use crate::extension::{BuiltinKind, ExtensionTable, dispatch_builtin};
use crate::header::MAGIC_PTWC;
use crate::index::{EOF_SENTINEL, IndexEntry, name_hash};
use crate::tensor_record_compact::MIN_RECORD_LEN;
use crate::transforms::op::OpId;
use crate::varint::{Cursor, write_varint};

/// Footer size: `index_len` and the EOF sentinel.
pub const FOOTER_LEN: usize = 16;
/// Longest tensor name.
pub const MAX_NAME_LEN: usize = u16::MAX as usize;

fn err(msg: impl std::fmt::Display) -> PtwmCoreError {
    PtwmCoreError::InvalidContainer(format!("compact container: {msg}"))
}

/// Serialize the header for sections of the given bytes.
pub fn header_bytes(et: &[u8], prelude: &[u8], registry: &[u8]) -> Vec<u8> {
    let mut h = Vec::with_capacity(9 + 2 + 3 * 10 + 8);
    h.extend_from_slice(&MAGIC_PTWC);
    h.extend_from_slice(&0u16.to_le_bytes());
    write_varint(&mut h, et.len() as u64);
    write_varint(&mut h, prelude.len() as u64);
    write_varint(&mut h, registry.len() as u64);
    let mut hasher = Xxh64::new(0);
    hasher.update(&h);
    hasher.update(et);
    hasher.update(prelude);
    hasher.update(registry);
    h.extend_from_slice(&hasher.digest().to_le_bytes());
    h
}

/// Byte ranges of each section of a compact container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sections {
    pub header: Range<usize>,
    pub extension_table: Range<usize>,
    pub prelude: Range<usize>,
    pub registry: Range<usize>,
    pub body: Range<usize>,
    pub index: Range<usize>,
}

/// Locate and check the sections of a compact container: magic, `hflags`,
/// section lengths, footer, and `meta_xxh64`. The caller checks the EOF
/// sentinel first.
pub fn locate_sections(buf: &[u8]) -> Result<Sections, PtwmCoreError> {
    if buf.len() < MAGIC_PTWC.len() + 2 + FOOTER_LEN || buf[..MAGIC_PTWC.len()] != MAGIC_PTWC {
        return Err(err("bad magic or file too short"));
    }
    let footer_start = buf.len() - FOOTER_LEN;
    let mut r = Cursor::new(&buf[..footer_start], "compact container header");
    r.take(MAGIC_PTWC.len())?;
    let hflags = u16::from_le_bytes(r.array()?);
    if hflags != 0 {
        return Err(r.err(format_args!("unknown hflags bits {hflags:#06x}")));
    }
    let et_len = r.varint()?;
    let prelude_len = r.varint()?;
    let registry_len = r.varint()?;
    let hash_pos = r.pos();
    let stored_hash = r.u64_le()?;
    let header_end = r.pos();

    let section = |start: usize, len: u64, what: &str| -> Result<Range<usize>, PtwmCoreError> {
        usize::try_from(len)
            .ok()
            .and_then(|len| start.checked_add(len))
            .filter(|&end| end <= footer_start)
            .map(|end| start..end)
            .ok_or_else(|| err(format_args!("{what} section exceeds the file")))
    };
    let extension_table = section(header_end, et_len, "extension table")?;
    let prelude = section(extension_table.end, prelude_len, "prelude")?;
    let registry = section(prelude.end, registry_len, "registry")?;

    let index_len = u64::from_le_bytes(buf[footer_start..footer_start + 8].try_into().unwrap());
    let index_start = usize::try_from(index_len)
        .ok()
        .and_then(|len| footer_start.checked_sub(len))
        .ok_or_else(|| err("index_len exceeds the file"))?;
    if registry.end > index_start {
        return Err(err("index overlaps the registry"));
    }

    let mut hasher = Xxh64::new(0);
    hasher.update(&buf[..hash_pos]);
    hasher.update(&buf[header_end..registry.end]);
    if hasher.digest() != stored_hash {
        return Err(err("meta_xxh64 mismatch"));
    }
    Ok(Sections {
        header: 0..header_end,
        extension_table,
        prelude,
        body: registry.end..index_start,
        registry,
        index: index_start..footer_start,
    })
}

/// Encode the chain registry. Chain ids must equal their positions; a
/// Source node with empty params is written as a bound Source. `op_index`
/// maps an op to its Extension Table index.
pub fn write_registry(
    registry: &ChainRegistry,
    mut op_index: impl FnMut(OpId) -> Result<u64, PtwmCoreError>,
) -> Result<Vec<u8>, PtwmCoreError> {
    if registry.chains.len() > MAX_REGISTRY_CHAINS {
        return Err(err(format_args!(
            "{} chains exceed the limit of {MAX_REGISTRY_CHAINS}",
            registry.chains.len()
        )));
    }
    let mut out = Vec::new();
    write_varint(&mut out, registry.chains.len() as u64);
    let mut chain_bytes = Vec::new();
    for (pos, (id, chain)) in registry.chains.iter().enumerate() {
        if usize::from(*id) != pos {
            return Err(err(format_args!(
                "registry chain at position {pos} has id {id}; compact chain ids are positions"
            )));
        }
        chain_bytes.clear();
        write_chain_compact_deferred(chain, &mut op_index, &mut chain_bytes)
            .map_err(|e| err(format_args!("registry chain {id}: {e}")))?;
        write_varint(&mut out, chain_bytes.len() as u64);
        out.extend_from_slice(&chain_bytes);
    }
    Ok(out)
}

/// Parse a chain registry that fills `buf` exactly. Op references resolve
/// through `table`, whose entry must be a built-in transform.
pub fn read_registry(buf: &[u8], table: &ExtensionTable) -> Result<ChainRegistry, PtwmCoreError> {
    let mut r = Cursor::new(buf, "compact chain registry");
    let count = r.varint_max(MAX_REGISTRY_CHAINS as u64, "chain count")? as usize;
    // Each chain takes at least its length byte and 5 chain bytes.
    if count > buf.len() / 6 {
        return Err(r.err("chain count exceeds the registry bytes"));
    }
    let resolve = |op_ref: u64| -> Result<OpId, PtwmCoreError> {
        let entry = usize::try_from(op_ref)
            .ok()
            .and_then(|i| table.entries.get(i))
            .ok_or_else(|| err(format_args!("op_ref {op_ref} is outside the Extension Table")))?;
        match dispatch_builtin(&entry.canonical_id) {
            Some(BuiltinKind::Op(op)) => Ok(op),
            _ => Err(err(format_args!(
                "op_ref {op_ref} names {:?}, which is not a built-in transform",
                entry.human_label
            ))),
        }
    };
    let mut chains = Vec::with_capacity(count);
    for id in 0..count {
        let bytes = r.len_bytes()?;
        let chain = read_chain_compact_deferred_exact(bytes, resolve)
            .map_err(|e| r.err(format_args!("chain {id}: {e}")))?;
        chains.push((id as u16, chain));
    }
    r.finish("the last chain")?;
    Ok(ChainRegistry { chains })
}

/// Encode the index for `(name, record length)` pairs in body order.
pub fn write_index(entries: &[(String, u64)]) -> Result<Vec<u8>, PtwmCoreError> {
    let mut out = Vec::new();
    write_varint(&mut out, entries.len() as u64);
    for (name, len) in entries {
        if name.is_empty() || name.len() > MAX_NAME_LEN {
            return Err(err(format_args!(
                "tensor name of {} bytes (must be 1..={MAX_NAME_LEN})",
                name.len()
            )));
        }
        if *len < MIN_RECORD_LEN {
            return Err(err(format_args!("record of {len} bytes for {name:?}")));
        }
        write_varint(&mut out, name.len() as u64);
        out.extend_from_slice(name.as_bytes());
        write_varint(&mut out, *len);
    }
    Ok(out)
}

/// Parse the index of a body at `body` (absolute range). Returns entries
/// sorted by `(name_hash, name)` with absolute offsets; their names are set.
pub fn read_index(buf: &[u8], body: Range<usize>) -> Result<Vec<IndexEntry>, PtwmCoreError> {
    let mut r = Cursor::new(buf, "compact container index");
    let count = r.varint()?;
    // Each entry takes at least 3 bytes.
    if count > (buf.len() / 3) as u64 {
        return Err(r.err("entry count exceeds the index bytes"));
    }
    let mut entries = Vec::with_capacity(count as usize);
    let mut names = HashSet::with_capacity(count as usize);
    let mut offset = body.start as u64;
    for _ in 0..count {
        let name_len = r.varint_max(MAX_NAME_LEN as u64, "name length")? as usize;
        if name_len == 0 {
            return Err(r.err("empty tensor name"));
        }
        let name = std::str::from_utf8(r.take(name_len)?)
            .map_err(|_| r.err("tensor name is not UTF-8"))?;
        if !names.insert(name) {
            return Err(r.err(format_args!("duplicate tensor name {name:?}")));
        }
        let len = r.varint()?;
        if len < MIN_RECORD_LEN {
            return Err(r.err(format_args!("record of {len} bytes for {name:?}")));
        }
        let end = offset
            .checked_add(len)
            .filter(|&end| end <= body.end as u64)
            .ok_or_else(|| r.err(format_args!("record of {name:?} runs past the body")))?;
        entries.push(IndexEntry {
            name_hash: name_hash(name),
            tensor_offset: offset,
            tensor_len: len,
            name: Some(name.into()),
        });
        offset = end;
    }
    r.finish("the last index entry")?;
    if offset != body.end as u64 {
        return Err(err(format_args!(
            "records cover {} of {} body bytes",
            offset - body.start as u64,
            body.len()
        )));
    }
    entries.sort_by(|a, b| (a.name_hash, &a.name).cmp(&(b.name_hash, &b.name)));
    Ok(entries)
}

/// Footer bytes for an index of `index_len` bytes.
pub fn footer_bytes(index_len: u64) -> [u8; FOOTER_LEN] {
    let mut f = [0u8; FOOTER_LEN];
    f[..8].copy_from_slice(&index_len.to_le_bytes());
    f[8..].copy_from_slice(&EOF_SENTINEL);
    f
}
