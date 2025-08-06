//! On-disk Extension Table format.
//!
//! Wire layout (all little-endian):
//!
//! ```text
//!     ExtensionTable:
//!         count: u32
//!         entries: [ExtensionTableEntry; count]
//!
//!     ExtensionTableEntry:
//!         canonical_id:         [u8; 32]
//!         human_label_len:      u16
//!         human_label:          [u8; human_label_len]   (UTF-8)
//!         kind:                 u16
//!         abi_version:          u16
//!         lifecycle:            u8
//!         flavor_hints:         u8
//!         capabilities_len:     u32
//!         capabilities:         [u8; capabilities_len]   (canonical CBOR)
//!         attestation_kind:     u8                       (1 = pgp_signature)
//!         attestation_len:      u16
//!         attestation:          [u8; attestation_len]
//!         install_hint_len:     u16                      (0 if absent)
//!         install_hint:         [u8; install_hint_len]   (UTF-8)
//!         embedded_wasm_offset: u64                      (0 if absent)
//!         embedded_wasm_length: u64                      (0 if absent)
//! ```

use std::io::{Cursor, Read, Write};

use super::{CanonicalId, CapabilityMap, ExtensionError, Kind, Lifecycle};

pub const FLAVOR_WASM: u8 = 0b001;
pub const FLAVOR_NATIVE: u8 = 0b010;
pub const FLAVOR_HOST: u8 = 0b100;
pub const FLAVOR_KNOWN_MASK: u8 = FLAVOR_WASM | FLAVOR_NATIVE | FLAVOR_HOST;

/// Maximum allowed byte length for the serialized capabilities CBOR blob.
/// Capabilities are tiny in practice; 1 MiB is a conservative hard cap that
/// prevents a malicious file from triggering a large allocation before
/// `read_exact` has a chance to fail.
const MAX_CAPABILITIES_LEN: u32 = 1 << 20; // 1 MiB

/// Maximum number of entries an `ExtensionTable` may declare. Real files
/// have a handful; capping at 65 536 prevents a malicious `count` from
/// driving `Vec::with_capacity(count)` into an OOM before any entry bytes
/// are read. (libFuzzer caught this with `count = 0x2F000000` ≈ 750 M.)
const MAX_TABLE_ENTRIES: u32 = 1 << 16; // 65 536

pub const ATTESTATION_PGP: u8 = 1;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExtensionTable {
    pub entries: Vec<ExtensionTableEntry>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExtensionTableEntry {
    pub canonical_id: CanonicalId,
    pub human_label: String,
    pub kind: Kind,
    pub abi_version: u16,
    pub lifecycle: Lifecycle,
    pub flavor_hints: u8,
    pub capabilities: CapabilityMap,
    pub attestation: Attestation,
    pub install_hint: Option<String>,
    pub embedded_wasm_offset: Option<u64>,
    pub embedded_wasm_length: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Attestation {
    PgpSignature(Vec<u8>), // raw signature bytes
}

impl ExtensionTable {
    pub fn empty() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn write<W: Write>(&self, w: &mut W) -> Result<(), ExtensionError> {
        write_u32(w, self.entries.len() as u32)?;
        for entry in &self.entries {
            entry.write(w)?;
        }
        Ok(())
    }

    pub fn read<R: Read>(r: &mut R) -> Result<Self, ExtensionError> {
        let count = read_u32(r)?;
        if count > MAX_TABLE_ENTRIES {
            return Err(ExtensionError::TableParse(format!(
                "extension table count too large: {count} (max {MAX_TABLE_ENTRIES})"
            )));
        }
        let mut entries = Vec::with_capacity(count as usize);
        for _ in 0..count {
            entries.push(ExtensionTableEntry::read(r)?);
        }
        Ok(Self { entries })
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, ExtensionError> {
        let mut buf = Vec::new();
        self.write(&mut buf)?;
        Ok(buf)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ExtensionError> {
        let mut c = Cursor::new(bytes);
        Self::read(&mut c)
    }

    /// blake3 hash committed in the header to detect tampering / corruption.
    pub fn hash(bytes: &[u8]) -> [u8; 32] {
        let mut out = [0u8; 32];
        out.copy_from_slice(blake3::hash(bytes).as_bytes());
        out
    }
}

impl ExtensionTableEntry {
    fn write<W: Write>(&self, w: &mut W) -> Result<(), ExtensionError> {
        w.write_all(self.canonical_id.as_bytes())
            .map_err(io_err("write canonical_id"))?;
        write_lp_string_u16(w, &self.human_label)?;
        write_u16(w, self.kind.as_u16())?;
        write_u16(w, self.abi_version)?;
        w.write_all(&[self.lifecycle.as_u8()])
            .map_err(io_err("write lifecycle"))?;
        if self.flavor_hints & !FLAVOR_KNOWN_MASK != 0 {
            return Err(ExtensionError::UnknownFlavorBits(self.flavor_hints));
        }
        w.write_all(&[self.flavor_hints])
            .map_err(io_err("write flavor_hints"))?;

        let cap_bytes = self.capabilities.to_cbor()?;
        write_u32(w, cap_bytes.len() as u32)?;
        w.write_all(&cap_bytes)
            .map_err(io_err("write capabilities"))?;

        let Attestation::PgpSignature(sig) = &self.attestation;
        w.write_all(&[ATTESTATION_PGP])
            .map_err(io_err("write attestation_kind"))?;
        write_u16(w, sig.len() as u16)?;
        w.write_all(sig).map_err(io_err("write attestation"))?;

        match &self.install_hint {
            Some(s) => write_lp_string_u16(w, s)?,
            None => write_u16(w, 0)?,
        }

        write_u64(w, self.embedded_wasm_offset.unwrap_or(0))?;
        write_u64(w, self.embedded_wasm_length.unwrap_or(0))?;
        Ok(())
    }

    fn read<R: Read>(r: &mut R) -> Result<Self, ExtensionError> {
        let mut id_bytes = [0u8; 32];
        r.read_exact(&mut id_bytes)
            .map_err(io_err("read canonical_id"))?;
        let canonical_id = CanonicalId::from_bytes(id_bytes);

        let human_label = read_lp_string_u16(r)?;
        let kind = Kind::from_u16(read_u16(r)?)?;
        let abi_version = read_u16(r)?;
        let mut lifecycle_buf = [0u8; 1];
        r.read_exact(&mut lifecycle_buf)
            .map_err(io_err("read lifecycle"))?;
        let lifecycle = Lifecycle::from_u8(lifecycle_buf[0])?;

        let mut fh_buf = [0u8; 1];
        r.read_exact(&mut fh_buf)
            .map_err(io_err("read flavor_hints"))?;
        let flavor_hints = fh_buf[0];
        if flavor_hints & !FLAVOR_KNOWN_MASK != 0 {
            return Err(ExtensionError::UnknownFlavorBits(flavor_hints));
        }

        let cap_len = read_u32(r)?;
        if cap_len > MAX_CAPABILITIES_LEN {
            return Err(ExtensionError::TableParse(format!(
                "capability blob too large: {cap_len} bytes (max {MAX_CAPABILITIES_LEN})"
            )));
        }
        let mut cap_bytes = vec![0u8; cap_len as usize];
        r.read_exact(&mut cap_bytes)
            .map_err(io_err("read capabilities"))?;
        let capabilities = CapabilityMap::from_cbor(&cap_bytes)?;

        let mut at_kind = [0u8; 1];
        r.read_exact(&mut at_kind)
            .map_err(io_err("read attestation_kind"))?;
        if at_kind[0] != ATTESTATION_PGP {
            return Err(ExtensionError::AttestationParse(format!(
                "unknown attestation kind: {}",
                at_kind[0]
            )));
        }
        let at_len = read_u16(r)?;
        let mut at_bytes = vec![0u8; at_len as usize];
        r.read_exact(&mut at_bytes)
            .map_err(io_err("read attestation"))?;
        let attestation = Attestation::PgpSignature(at_bytes);

        let hint_len = read_u16(r)?;
        let install_hint = if hint_len == 0 {
            None
        } else {
            let mut buf = vec![0u8; hint_len as usize];
            r.read_exact(&mut buf)
                .map_err(io_err("read install_hint"))?;
            Some(String::from_utf8(buf).map_err(|e| ExtensionError::TableParse(e.to_string()))?)
        };

        let off = read_u64(r)?;
        let len = read_u64(r)?;
        let embedded_wasm_offset = if off == 0 { None } else { Some(off) };
        let embedded_wasm_length = if len == 0 { None } else { Some(len) };

        Ok(Self {
            canonical_id,
            human_label,
            kind,
            abi_version,
            lifecycle,
            flavor_hints,
            capabilities,
            attestation,
            install_hint,
            embedded_wasm_offset,
            embedded_wasm_length,
        })
    }
}

// ----- I/O helpers ---------------------------------------------------------

fn io_err(ctx: &'static str) -> impl Fn(std::io::Error) -> ExtensionError {
    move |e| ExtensionError::TableParse(format!("{ctx}: {e}"))
}

fn write_u16<W: Write>(w: &mut W, v: u16) -> Result<(), ExtensionError> {
    w.write_all(&v.to_le_bytes()).map_err(io_err("write u16"))
}
fn write_u32<W: Write>(w: &mut W, v: u32) -> Result<(), ExtensionError> {
    w.write_all(&v.to_le_bytes()).map_err(io_err("write u32"))
}
fn write_u64<W: Write>(w: &mut W, v: u64) -> Result<(), ExtensionError> {
    w.write_all(&v.to_le_bytes()).map_err(io_err("write u64"))
}
fn read_u16<R: Read>(r: &mut R) -> Result<u16, ExtensionError> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b).map_err(io_err("read u16"))?;
    Ok(u16::from_le_bytes(b))
}
fn read_u32<R: Read>(r: &mut R) -> Result<u32, ExtensionError> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(io_err("read u32"))?;
    Ok(u32::from_le_bytes(b))
}
fn read_u64<R: Read>(r: &mut R) -> Result<u64, ExtensionError> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).map_err(io_err("read u64"))?;
    Ok(u64::from_le_bytes(b))
}

fn write_lp_string_u16<W: Write>(w: &mut W, s: &str) -> Result<(), ExtensionError> {
    let bytes = s.as_bytes();
    if bytes.len() > u16::MAX as usize {
        return Err(ExtensionError::TableSer(format!(
            "string too long: {}",
            bytes.len()
        )));
    }
    write_u16(w, bytes.len() as u16)?;
    w.write_all(bytes).map_err(io_err("write lp_string"))
}

fn read_lp_string_u16<R: Read>(r: &mut R) -> Result<String, ExtensionError> {
    let len = read_u16(r)?;
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf).map_err(io_err("read lp_string"))?;
    String::from_utf8(buf).map_err(|e| ExtensionError::TableParse(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extension::capability::CapabilityValue;

    fn sample_entry() -> ExtensionTableEntry {
        let mut caps = CapabilityMap::new();
        caps.set("determinism", CapabilityValue::Bool(true));
        caps.set("hardware_class", CapabilityValue::Text("cpu".into()));
        ExtensionTableEntry {
            canonical_id: CanonicalId::derive(&[0x11; 32], "foo", "1.2.3"),
            human_label: "io.example.foo".into(),
            kind: Kind::Transform,
            abi_version: 1,
            lifecycle: Lifecycle::Thread,
            flavor_hints: FLAVOR_WASM | FLAVOR_NATIVE,
            capabilities: caps,
            attestation: Attestation::PgpSignature(vec![0xAB; 64]),
            install_hint: Some("https://example/foo.tar.zst".into()),
            embedded_wasm_offset: None,
            embedded_wasm_length: None,
        }
    }

    #[test]
    fn roundtrip_empty_table() {
        let t = ExtensionTable::empty();
        let bytes = t.to_bytes().unwrap();
        assert_eq!(bytes, vec![0, 0, 0, 0]);
        let back = ExtensionTable::from_bytes(&bytes).unwrap();
        assert_eq!(back, t);
    }

    #[test]
    fn roundtrip_one_entry() {
        let t = ExtensionTable {
            entries: vec![sample_entry()],
        };
        let bytes = t.to_bytes().unwrap();
        let back = ExtensionTable::from_bytes(&bytes).unwrap();
        assert_eq!(back, t);
    }

    #[test]
    fn rejects_unknown_flavor_bits() {
        let mut e = sample_entry();
        e.flavor_hints |= 0b1000_0000; // an undefined bit
        let mut buf = Vec::new();
        let res = e.write(&mut buf);
        assert!(matches!(res, Err(ExtensionError::UnknownFlavorBits(_))));
    }

    #[test]
    fn truncated_input_is_an_error_not_panic() {
        let t = ExtensionTable {
            entries: vec![sample_entry()],
        };
        let bytes = t.to_bytes().unwrap();
        for cut in 0..bytes.len() {
            let res = ExtensionTable::from_bytes(&bytes[..cut]);
            assert!(res.is_err(), "expected error at truncation {cut}");
        }
    }

    #[test]
    fn table_hash_is_stable() {
        let t = ExtensionTable {
            entries: vec![sample_entry()],
        };
        let bytes = t.to_bytes().unwrap();
        let h1 = ExtensionTable::hash(&bytes);
        let h2 = ExtensionTable::hash(&t.to_bytes().unwrap());
        assert_eq!(h1, h2);
    }

    #[test]
    fn oversize_entry_count_is_rejected() {
        // count = u32::MAX would otherwise drive Vec::with_capacity into an
        // OOM before we read any entry bytes. Make sure the cap rejects it.
        let mut buf = u32::MAX.to_le_bytes().to_vec();
        // Pad with a few bytes so the cursor has something to read if the
        // bound were missing.
        buf.extend_from_slice(&[0u8; 16]);
        let res = ExtensionTable::from_bytes(&buf);
        assert!(
            matches!(res, Err(ExtensionError::TableParse(_))),
            "oversize count should be rejected, got: {res:?}"
        );
    }

    /// Regression for the libFuzzer crash on the `parse_wire` target.
    /// The 24-byte input had `chain_ref` with the inline bit set, which
    /// routes into `ExtensionTable::from_bytes`; the embedded `count` was
    /// `0x2F000000` ≈ 750 M, triggering OOM via `Vec::with_capacity`.
    #[test]
    fn fuzz_regression_parse_wire_inline_chain_oom() {
        // Reconstruct just the inline-table slice: bytes 18..24 of the
        // 24-byte fuzz input were `0, 0, 0, 47, 0, 0` — the leading u32
        // count of the (would-be) ExtensionTable.
        let payload = [0u8, 0, 0, 47, 0, 0];
        let res = ExtensionTable::from_bytes(&payload);
        assert!(
            matches!(res, Err(ExtensionError::TableParse(_))),
            "fuzz crash count should be rejected, got: {res:?}"
        );
    }

    #[test]
    fn oversize_capabilities_len_is_rejected() {
        // Hand-build a malformed entry whose capabilities_len = u32::MAX.
        // Wire layout up through cap_len:
        //   [u8; 32]  canonical_id
        //   u16       human_label_len = 0
        //   u16       kind = 1 (Transform)
        //   u16       abi_version = 1
        //   u8        lifecycle = 1 (Thread)
        //   u8        flavor_hints = FLAVOR_WASM
        //   u32       capabilities_len = u32::MAX  ← the oversized field
        let mut entry_buf = Vec::new();
        entry_buf.extend_from_slice(&[0xAA; 32]); // canonical_id
        entry_buf.extend_from_slice(&0u16.to_le_bytes()); // human_label_len = 0
        entry_buf.extend_from_slice(&1u16.to_le_bytes()); // kind = 1 (Transform)
        entry_buf.extend_from_slice(&1u16.to_le_bytes()); // abi_version = 1
        entry_buf.push(1); // lifecycle = Thread
        entry_buf.push(FLAVOR_WASM); // flavor_hints
        entry_buf.extend_from_slice(&u32::MAX.to_le_bytes()); // capabilities_len = 4 GiB

        // Wrap in a table with count = 1.
        let mut table_buf = 1u32.to_le_bytes().to_vec();
        table_buf.extend_from_slice(&entry_buf);

        let res = ExtensionTable::from_bytes(&table_buf);
        assert!(
            matches!(res, Err(ExtensionError::TableParse(_))),
            "oversized cap_len should be rejected, got: {res:?}"
        );
    }
}
