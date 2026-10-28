//! Compact Extension Table, used by the compact container (`\x89PTWC`).
//!
//! The legacy table stores every entry in full: id, label, kind, ABI,
//! capabilities and attestation, about 125 bytes per built-in. A built-in
//! entry is fully determined by its name and the crate version, so the
//! compact table stores a built-in as a tag and a wire id or name, and
//! stores third-party entries in full:
//!
//! ```text
//!     count       varint  at most 65536 and at most (table bytes) / 2
//!     entry × count:
//!         tag     u8
//!         0x01    OpId wire id (varint)     built-in transform
//!         0x02    CodecId wire id (varint)  built-in plane codec
//!         0x03    name_len varint (1..=255), then the name
//!                                           built-in without a wire id
//!         0x7F    len varint, then one legacy table entry of exactly len
//!                 bytes                     third-party entry
//! ```
//!
//! The reader expands tags 0x01-0x03 to the built-in entry of the reader's
//! own version ([`builtin_registry`]), so dispatch, trust and the
//! missing-flavor check see the same entries as for a legacy file, and a
//! compact file does not depend on the version that wrote it.
//!
//! Each entry has one encoding:
//! - a built-in with an op or codec wire id must use tag 0x01 or 0x02; tag
//!   0x03 is only for built-ins without one;
//! - a codec id without a built-in entry (`huffman_nibble`) is rejected;
//! - a 0x7F entry must not carry a built-in id, of this version or (by its
//!   label) of another one, must re-serialize to the same bytes, and must
//!   not reference embedded WASM;
//! - unknown tags, ids and names are rejected, and so are duplicate ids.
//!
//! The writer refuses entries it cannot store this way: a built-in id whose
//! other fields differ from the built-in entry, or a built-in of another
//! release.

use std::collections::HashSet;

use super::{
    BuiltinKind, ExtensionTable, ExtensionTableEntry, builtin_registry, normalize_builtin_ids,
};
use crate::codec::CodecId;
use crate::error::PtwmCoreError;
use crate::transforms::op::OpId;
use crate::varint::{Cursor, write_varint};

/// Built-in transform, stored by `OpId` wire id.
pub const TAG_OP: u8 = 0x01;
/// Built-in plane codec, stored by `CodecId` wire id.
pub const TAG_CODEC: u8 = 0x02;
/// Built-in without a wire id, stored by name.
pub const TAG_NAMED: u8 = 0x03;
/// Third-party entry, stored in the legacy entry layout.
pub const TAG_FULL: u8 = 0x7F;

/// Largest entry count: table indices are `u16`.
pub const MAX_COMPACT_ENTRIES: u64 = 1 << 16;
/// Longest built-in name stored under [`TAG_NAMED`].
const MAX_NAME_LEN: u64 = 255;

/// How [`write_compact_table`] stores one entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactEntry<'a> {
    Op(OpId),
    Codec(CodecId),
    Named(&'static str),
    Full(&'a ExtensionTableEntry),
}

fn err(msg: impl std::fmt::Display) -> PtwmCoreError {
    PtwmCoreError::InvalidContainer(format!("compact extension table: {msg}"))
}

/// Whether `entry` is a built-in of another release: its label names a
/// built-in and its id re-derives from that label (see
/// [`normalize_builtin_ids`]).
fn is_other_version_builtin(entry: &ExtensionTableEntry) -> bool {
    let mut one = ExtensionTable {
        entries: vec![entry.clone()],
    };
    normalize_builtin_ids(&mut one);
    one.entries[0].canonical_id != entry.canonical_id
}

/// Decide how `entry` is stored, or fail when the compact table cannot
/// store it.
pub fn classify(entry: &ExtensionTableEntry) -> Result<CompactEntry<'_>, PtwmCoreError> {
    let reg = builtin_registry();
    if let Some(r) = reg.get(&entry.canonical_id) {
        let canonical = r.entry_index.and_then(|i| reg.entry_at(i));
        if canonical != Some(entry) {
            return Err(err(format_args!(
                "entry with the id of built-in {:?} differs from that built-in's entry",
                r.name
            )));
        }
        return Ok(match r.dispatch {
            Some(BuiltinKind::Op(op)) => CompactEntry::Op(op),
            Some(BuiltinKind::Codec(c)) => CompactEntry::Codec(c),
            None => CompactEntry::Named(r.name),
        });
    }
    if is_other_version_builtin(entry) {
        return Err(err(format_args!(
            "{:?} is a built-in of another release; the compact table stores \
             built-ins of the reader's version only",
            entry.human_label
        )));
    }
    if entry.embedded_wasm_offset.is_some() || entry.embedded_wasm_length.is_some() {
        return Err(err(format_args!(
            "{:?} references embedded WASM, which needs the legacy layout",
            entry.human_label
        )));
    }
    Ok(CompactEntry::Full(entry))
}

/// Append the compact encoding of `table` to `out`. On error `out` is left
/// unchanged.
pub fn write_compact_table(
    table: &ExtensionTable,
    out: &mut Vec<u8>,
) -> Result<(), PtwmCoreError> {
    let n = table.entries.len() as u64;
    if n > MAX_COMPACT_ENTRIES {
        return Err(err(format_args!(
            "{n} entries (max {MAX_COMPACT_ENTRIES})"
        )));
    }
    let mut buf = Vec::with_capacity(1 + 3 * table.entries.len());
    write_varint(&mut buf, n);
    let mut seen = HashSet::with_capacity(table.entries.len());
    for entry in &table.entries {
        if !seen.insert(entry.canonical_id) {
            return Err(err(format_args!(
                "duplicate entry {:?}",
                entry.human_label
            )));
        }
        match classify(entry)? {
            CompactEntry::Op(op) => {
                buf.push(TAG_OP);
                write_varint(&mut buf, u64::from(op.as_u16()));
            }
            CompactEntry::Codec(c) => {
                buf.push(TAG_CODEC);
                write_varint(&mut buf, u64::from(c.as_u16()));
            }
            CompactEntry::Named(name) => {
                buf.push(TAG_NAMED);
                write_varint(&mut buf, name.len() as u64);
                buf.extend_from_slice(name.as_bytes());
            }
            CompactEntry::Full(e) => {
                let bytes = e.to_bytes().map_err(err)?;
                buf.push(TAG_FULL);
                write_varint(&mut buf, bytes.len() as u64);
                buf.extend_from_slice(&bytes);
            }
        }
    }
    out.extend_from_slice(&buf);
    Ok(())
}

/// Serialize `table` in the compact layout.
pub fn compact_table_bytes(table: &ExtensionTable) -> Result<Vec<u8>, PtwmCoreError> {
    let mut out = Vec::new();
    write_compact_table(table, &mut out)?;
    Ok(out)
}

/// The built-in entry for a wire id or name, or an error naming `what`.
fn builtin_by_name(name: &str, what: impl std::fmt::Display) -> Result<ExtensionTableEntry, PtwmCoreError> {
    builtin_registry()
        .entry_by_name(name)
        .cloned()
        .ok_or_else(|| err(format_args!("{what} has no built-in entry")))
}

/// Parse one entry at the cursor.
fn read_entry(r: &mut Cursor<'_>) -> Result<ExtensionTableEntry, PtwmCoreError> {
    let tag = r.u8()?;
    match tag {
        TAG_OP => {
            let id = r.varint_max(u64::from(u16::MAX), "op id")? as u16;
            let op = OpId::from_u16(id).map_err(|_| r.err(format_args!("unknown op id {id:#06x}")))?;
            builtin_by_name(op.builtin_name(), format_args!("op {op:?}"))
        }
        TAG_CODEC => {
            let id = r.varint_max(u64::from(u16::MAX), "codec id")? as u16;
            let name = crate::codec::REGISTRY
                .iter()
                .find(|&&(_, raw)| raw == id)
                .map(|&(name, _)| name)
                .ok_or_else(|| r.err(format_args!("unknown codec id {id:#06x}")))?;
            builtin_by_name(name, format_args!("codec {name:?}"))
        }
        TAG_NAMED => {
            let len = r.varint_max(MAX_NAME_LEN, "built-in name length")?;
            if len == 0 {
                return Err(r.err("empty built-in name"));
            }
            let bytes = r.take(len as usize)?;
            let name = std::str::from_utf8(bytes).map_err(|_| r.err("built-in name is not UTF-8"))?;
            let reg = builtin_registry();
            let id = reg
                .id_for_entry_name(name)
                .ok_or_else(|| r.err(format_args!("unknown built-in name {name:?}")))?;
            if reg.get(&id).is_some_and(|b| b.dispatch.is_some()) {
                return Err(r.err(format_args!(
                    "built-in {name:?} has a wire id and must use its tag (non-canonical)"
                )));
            }
            builtin_by_name(name, format_args!("{name:?}"))
        }
        TAG_FULL => {
            let bytes = r.len_bytes()?;
            let entry = ExtensionTableEntry::from_bytes_exact(bytes)
                .map_err(|e| r.err(format_args!("third-party entry: {e}")))?;
            if entry.to_bytes().map_err(err)? != bytes {
                return Err(r.err("third-party entry is not in canonical form"));
            }
            if builtin_registry().get(&entry.canonical_id).is_some()
                || is_other_version_builtin(&entry)
            {
                return Err(r.err(format_args!(
                    "full entry {:?} carries a built-in id (non-canonical)",
                    entry.human_label
                )));
            }
            if entry.embedded_wasm_offset.is_some() || entry.embedded_wasm_length.is_some() {
                return Err(r.err("embedded WASM entries are not allowed in a compact table"));
            }
            Ok(entry)
        }
        other => Err(r.err(format_args!("unknown entry tag {other:#04x}"))),
    }
}

/// Parse a compact table that fills `buf` exactly.
pub fn read_compact_table(buf: &[u8]) -> Result<ExtensionTable, PtwmCoreError> {
    let mut r = Cursor::new(buf, "compact extension table");
    let count = r.varint_max(MAX_COMPACT_ENTRIES, "entry count")?;
    // Every entry takes at least a tag and one more byte.
    if count > (buf.len() / 2) as u64 {
        return Err(r.err(format_args!("entry count {count} exceeds the table bytes")));
    }
    let count = count as usize;
    let mut entries = Vec::with_capacity(count);
    let mut seen = HashSet::with_capacity(count);
    for _ in 0..count {
        let entry = read_entry(&mut r)?;
        if !seen.insert(entry.canonical_id) {
            return Err(r.err(format_args!("duplicate entry {:?}", entry.human_label)));
        }
        entries.push(entry);
    }
    r.finish("the last entry")?;
    Ok(ExtensionTable { entries })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extension::{
        Attestation, CanonicalId, CapabilityMap, Kind, Lifecycle, builtin_entries,
        builtin_entry_at_version,
    };

    fn third_party() -> ExtensionTableEntry {
        ExtensionTableEntry {
            canonical_id: CanonicalId::derive(&[7; 32], "acme_codec", "0.1.0"),
            human_label: "com.acme.codec@0.1.0".into(),
            kind: Kind::PlaneCodec,
            abi_version: 1,
            lifecycle: Lifecycle::Thread,
            flavor_hints: crate::extension::table::FLAVOR_NATIVE,
            capabilities: CapabilityMap::new(),
            attestation: Attestation::PgpSignature(vec![1, 2, 3]),
            install_hint: Some("pip install acme".into()),
            embedded_wasm_offset: None,
            embedded_wasm_length: None,
        }
    }

    #[test]
    fn every_builtin_round_trips_in_a_few_bytes() {
        let table = ExtensionTable {
            entries: builtin_entries(),
        };
        let bytes = compact_table_bytes(&table).unwrap();
        // Built-ins cost a tag and a short id or name.
        assert!(
            bytes.len() <= 4 * table.entries.len() + 30 * 5,
            "{} bytes for {} entries",
            bytes.len(),
            table.entries.len()
        );
        let back = read_compact_table(&bytes).unwrap();
        assert_eq!(back, table);
    }

    #[test]
    fn ops_and_codecs_cost_at_most_three_bytes() {
        let table = ExtensionTable {
            entries: builtin_entries()
                .into_iter()
                .filter(|e| matches!(e.kind, Kind::PlaneCodec | Kind::Transform))
                .collect(),
        };
        let bytes = compact_table_bytes(&table).unwrap();
        assert!(bytes.len() <= 1 + 3 * table.entries.len());
        assert_eq!(read_compact_table(&bytes).unwrap(), table);
    }

    #[test]
    fn third_party_entry_round_trips() {
        let mut entries = vec![builtin_registry().entry_by_name("zstd").unwrap().clone()];
        entries.push(third_party());
        let table = ExtensionTable { entries };
        let bytes = compact_table_bytes(&table).unwrap();
        assert_eq!(bytes[1], TAG_CODEC);
        assert_eq!(read_compact_table(&bytes).unwrap(), table);
    }

    #[test]
    fn writer_refuses_entries_it_cannot_store() {
        // A built-in id with other fields changed.
        let mut e = builtin_registry().entry_by_name("zstd").unwrap().clone();
        e.flavor_hints = 0;
        let t = ExtensionTable { entries: vec![e] };
        assert!(compact_table_bytes(&t).is_err());
        // A built-in of another release.
        let t = ExtensionTable {
            entries: vec![builtin_entry_at_version("zstd", "1.0.0").unwrap()],
        };
        let msg = compact_table_bytes(&t).unwrap_err().to_string();
        assert!(msg.contains("another release"), "{msg}");
        // Embedded WASM.
        let mut e = third_party();
        e.embedded_wasm_offset = Some(10);
        e.embedded_wasm_length = Some(10);
        assert!(compact_table_bytes(&ExtensionTable { entries: vec![e] }).is_err());
        // Duplicates.
        let z = builtin_registry().entry_by_name("zstd").unwrap().clone();
        let t = ExtensionTable {
            entries: vec![z.clone(), z],
        };
        assert!(compact_table_bytes(&t).is_err());
    }

    fn rejects(bytes: &[u8], needle: &str) {
        let msg = read_compact_table(bytes).unwrap_err().to_string();
        assert!(msg.contains(needle), "{needle:?} not in {msg:?}");
    }

    #[test]
    fn reader_rejects_non_canonical_and_unknown_entries() {
        let zstd = CodecId::Zstd.as_u16() as u8;
        // Overlong varint for the count and for an id.
        rejects(&[0x81, 0x00, TAG_CODEC, zstd], "non-minimal");
        rejects(&[1, TAG_CODEC, 0x80 | zstd, 0x00], "non-minimal");
        // Unknown tag, op id, codec id and name.
        rejects(&[1, 0x04, 0], "unknown entry tag");
        rejects(&[1, TAG_OP, 0x7F], "unknown op id");
        rejects(&[1, TAG_CODEC, 0x7F], "unknown codec id");
        rejects(&[1, TAG_NAMED, 3, b'f', b'o', b'o'], "unknown built-in name");
        // huffman_nibble has a codec id but no built-in entry.
        rejects(&[1, TAG_CODEC, CodecId::HuffmanNibble.as_u16() as u8], "no built-in entry");
        // A named entry for a built-in that has a wire id.
        rejects(&[1, TAG_NAMED, 4, b'z', b's', b't', b'd'], "non-canonical");
        // Duplicates.
        rejects(&[2, TAG_CODEC, zstd, TAG_CODEC, zstd], "duplicate");
        // Trailing bytes and a count that does not fit.
        rejects(&[1, TAG_CODEC, zstd, 0], "trailing");
        rejects(&[3, TAG_CODEC, zstd], "exceeds the table bytes");
        rejects(&[0x81, 0x80, 0x04], "exceeds 65536");
    }

    fn full_entry_bytes(e: &ExtensionTableEntry) -> Vec<u8> {
        let body = e.to_bytes().unwrap();
        let mut out = vec![1, TAG_FULL];
        write_varint(&mut out, body.len() as u64);
        out.extend_from_slice(&body);
        out
    }

    #[test]
    fn reader_rejects_builtins_stored_in_full() {
        let z = builtin_registry().entry_by_name("zstd").unwrap().clone();
        rejects(&full_entry_bytes(&z), "built-in id");
        let old = builtin_entry_at_version("zstd", "1.0.0").unwrap();
        rejects(&full_entry_bytes(&old), "built-in id");
        let mut wasm = third_party();
        wasm.embedded_wasm_offset = Some(1);
        wasm.embedded_wasm_length = Some(1);
        rejects(&full_entry_bytes(&wasm), "embedded WASM");
        // A full entry with bytes after it inside its length.
        let mut bytes = full_entry_bytes(&third_party());
        let body_len = bytes.len() - 3;
        bytes[2] = (body_len + 1) as u8;
        bytes.push(0);
        rejects(&bytes, "trailing");
    }
}
