//! Shared-state prelude.

use crate::codec::CodecId;
use crate::error::PtwmCoreError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreludeEntry {
    pub shared_state_id: u16,
    pub codec_id: CodecId,
    /// Per-codec state-blob version. Codecs like `Order1ScaleAC` use it
    /// to evolve their internal state layout; codecs with no state leave
    /// it at 0.
    pub state_format_version: u8,
    pub applies_to_mask: u8,
    /// xxhash64 of `state_bytes`, always present on disk.
    pub state_xxhash64: u64,
    /// Human-readable label for diagnostics (`weights info`). The empty
    /// string is valid and serializes as `name_len = 0` with no padding.
    pub name: String,
    pub state_bytes: Vec<u8>,
}

pub fn write_prelude(entries: &[PreludeEntry], out: &mut Vec<u8>) -> Result<(), PtwmCoreError> {
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for e in entries {
        out.extend_from_slice(&e.shared_state_id.to_le_bytes());
        out.extend_from_slice(&e.codec_id.as_u16().to_le_bytes());
        out.push(e.state_format_version);
        out.push(e.applies_to_mask);
        out.extend_from_slice(&0u16.to_le_bytes()); // reserved
        out.extend_from_slice(&e.state_xxhash64.to_le_bytes());
        // name_len (u16) + name_bytes + pad-to-4
        let name_bytes = e.name.as_bytes();
        let name_len = name_bytes.len();
        if name_len > u16::MAX as usize {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "PreludeEntry name length {name_len} exceeds u16::MAX"
            )));
        }
        if e.state_bytes.len() > u32::MAX as usize {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "PreludeEntry state_bytes length {} exceeds u32::MAX",
                e.state_bytes.len()
            )));
        }
        out.extend_from_slice(&(name_len as u16).to_le_bytes());
        out.extend_from_slice(name_bytes);
        // Pad to 4-byte alignment.
        let pad = (4usize.wrapping_sub(name_len % 4)) % 4;
        if pad > 0 {
            out.extend_from_slice(&vec![0u8; pad]);
        }
        out.extend_from_slice(&(e.state_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&e.state_bytes);
    }
    Ok(())
}

/// Minimum on-disk size of one entry; serves as a conservative upper
/// bound on the feasible per-buffer entry count: 8 fixed header + 8
/// state_xxhash64 + 2 name_len + 4 state_len = 22 bytes minimum. The
/// per-entry bounds checks downstream are authoritative for actual
/// layout validation.
const MIN_ENTRY_BYTES: usize = 22;

pub fn parse_prelude(buf: &[u8]) -> Result<(Vec<PreludeEntry>, usize), PtwmCoreError> {
    if buf.len() < 4 {
        return Err(PtwmCoreError::InvalidContainer("prelude too short".into()));
    }
    let count = u32::from_le_bytes(buf[..4].try_into().unwrap()) as usize;
    let max_count = (buf.len() - 4) / MIN_ENTRY_BYTES;
    if count > max_count {
        return Err(PtwmCoreError::InvalidContainer(format!(
            "prelude count {count} exceeds buffer capacity ({max_count} max)"
        )));
    }
    let mut pos = 4;
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        if pos + 12 > buf.len() {
            return Err(PtwmCoreError::InvalidContainer(
                "prelude entry header truncated".into(),
            ));
        }
        let shared_state_id = u16::from_le_bytes(buf[pos..pos + 2].try_into().unwrap());
        let codec_raw = u16::from_le_bytes(buf[pos + 2..pos + 4].try_into().unwrap());
        let codec_id = CodecId::from_u16(codec_raw).ok_or_else(|| {
            PtwmCoreError::InvalidContainer(format!("unknown codec_id 0x{:04x}", codec_raw))
        })?;
        let state_format_version = buf[pos + 4];
        let applies_to_mask = buf[pos + 5];
        // reserved u16 at pos+6..pos+8
        pos += 8;
        if pos + 8 > buf.len() {
            return Err(PtwmCoreError::InvalidContainer(
                "prelude entry state_xxhash64 truncated".into(),
            ));
        }
        let state_xxhash64 = u64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap());
        pos += 8;
        // name_len (u16) + name_bytes + pad-to-4
        if pos + 2 > buf.len() {
            return Err(PtwmCoreError::InvalidContainer(
                "prelude entry name_len truncated".into(),
            ));
        }
        let name_len = u16::from_le_bytes(buf[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;
        if pos + name_len > buf.len() {
            return Err(PtwmCoreError::InvalidContainer(
                "prelude entry name_bytes truncated".into(),
            ));
        }
        let name = String::from_utf8(buf[pos..pos + name_len].to_vec()).map_err(|e| {
            PtwmCoreError::InvalidContainer(format!("PreludeEntry name not valid UTF-8: {e}"))
        })?;
        pos += name_len;
        let pad = (4usize.wrapping_sub(name_len % 4)) % 4;
        if pos + pad > buf.len() {
            return Err(PtwmCoreError::InvalidContainer(
                "prelude entry name padding truncated".into(),
            ));
        }
        pos += pad;
        // state_len + state_bytes
        if pos + 4 > buf.len() {
            return Err(PtwmCoreError::InvalidContainer(
                "prelude entry state_len truncated".into(),
            ));
        }
        let state_len = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        if pos + state_len > buf.len() {
            return Err(PtwmCoreError::InvalidContainer(
                "prelude entry state_bytes truncated".into(),
            ));
        }
        let state_bytes = buf[pos..pos + state_len].to_vec();
        pos += state_len;
        entries.push(PreludeEntry {
            shared_state_id,
            codec_id,
            state_format_version,
            applies_to_mask,
            state_xxhash64,
            name,
            state_bytes,
        });
    }
    Ok((entries, pos))
}

/// Convenience wrapper for cursor-based readers (used in tests).
pub fn read_prelude<R: std::io::Read>(reader: &mut R) -> Result<Vec<PreludeEntry>, PtwmCoreError> {
    let mut buf = Vec::new();
    reader
        .read_to_end(&mut buf)
        .map_err(|e| PtwmCoreError::Io(e.to_string()))?;
    let (entries, _) = parse_prelude(&buf)?;
    Ok(entries)
}

/// Verify two `PreludeEntry` shared states with the same
/// `shared_state_id` carry identical bytes. A multi-shard reader merging
/// preludes across shards calls this to fast-fail on silently divergent
/// shared state.
pub fn shared_state_matches(a: &PreludeEntry, b: &PreludeEntry) -> bool {
    if a.shared_state_id != b.shared_state_id || a.codec_id != b.codec_id {
        return false;
    }
    if a.state_format_version != b.state_format_version {
        return false;
    }
    if a.state_xxhash64 != b.state_xxhash64 {
        return false;
    }
    a.state_bytes == b.state_bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::CodecId;

    #[test]
    fn prelude_roundtrip() {
        let entries = vec![
            PreludeEntry {
                shared_state_id: 0,
                codec_id: CodecId::Huffman,
                state_format_version: 0,
                applies_to_mask: 0b0001,
                state_xxhash64: 0,
                name: String::new(),
                state_bytes: vec![1, 2, 3, 4, 5],
            },
            PreludeEntry {
                shared_state_id: 1,
                codec_id: CodecId::PerGroupCodebook,
                state_format_version: 0,
                applies_to_mask: 0b0001,
                state_xxhash64: 0,
                name: String::new(),
                state_bytes: vec![10; 128],
            },
        ];
        let mut out = Vec::new();
        write_prelude(&entries, &mut out).unwrap();
        let (parsed, consumed) = parse_prelude(&out).unwrap();
        assert_eq!(parsed, entries);
        assert_eq!(consumed, out.len());
    }

    #[test]
    fn prelude_unknown_codec_rejected() {
        let mut out = Vec::new();
        out.extend_from_slice(&1u32.to_le_bytes()); // count = 1
        out.extend_from_slice(&0u16.to_le_bytes()); // shared_state_id
        out.extend_from_slice(&0x9999u16.to_le_bytes()); // invalid codec_id
        out.push(0); // state_format_version
        out.push(0); // applies_to_mask
        out.extend_from_slice(&0u16.to_le_bytes()); // reserved
        out.extend_from_slice(&0u64.to_le_bytes()); // state_xxhash64
        out.extend_from_slice(&0u16.to_le_bytes()); // name_len = 0
        out.extend_from_slice(&0u32.to_le_bytes()); // state_len = 0
        assert!(parse_prelude(&out).is_err());
    }

    #[test]
    fn prelude_entry_named_roundtrip() {
        let e = PreludeEntry {
            shared_state_id: 7,
            codec_id: CodecId::Order1ScaleAC,
            state_format_version: 0,
            applies_to_mask: 0b0010,
            state_xxhash64: 0,
            name: "order1_scale_ac/cluster_0_of_3".to_string(),
            state_bytes: vec![1, 2, 3],
        };
        let mut buf = Vec::new();
        write_prelude(std::slice::from_ref(&e), &mut buf).unwrap();
        let parsed = read_prelude(&mut std::io::Cursor::new(buf)).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].name, "order1_scale_ac/cluster_0_of_3");
        assert_eq!(parsed[0].shared_state_id, 7);
        assert_eq!(parsed[0].state_bytes, vec![1, 2, 3]);
    }

    #[test]
    fn write_rejects_overlong_name() {
        let e = PreludeEntry {
            shared_state_id: 0,
            codec_id: CodecId::Huffman,
            state_format_version: 0,
            applies_to_mask: 0b0001,
            state_xxhash64: 0,
            name: "x".repeat(u16::MAX as usize + 1),
            state_bytes: Vec::new(),
        };
        let mut buf = Vec::new();
        let err = write_prelude(&[e], &mut buf).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("u16::MAX"), "{msg}");
    }

    #[test]
    fn parse_rejects_invalid_utf8_name() {
        // Hand-craft a prelude with non-UTF-8 bytes in the name field.
        let mut buf = Vec::new();
        buf.extend_from_slice(&1u32.to_le_bytes()); // count
        buf.extend_from_slice(&0u16.to_le_bytes()); // shared_state_id
        buf.extend_from_slice(&CodecId::Huffman.as_u16().to_le_bytes());
        buf.push(0); // version
        buf.push(0b0001); // mask
        buf.extend_from_slice(&0u16.to_le_bytes()); // reserved
        buf.extend_from_slice(&0u64.to_le_bytes()); // state_xxhash64
        buf.extend_from_slice(&4u16.to_le_bytes()); // name_len = 4
        buf.extend_from_slice(&[0xFFu8, 0xFE, 0xFD, 0xFC]); // invalid UTF-8
        // pad: name_len % 4 == 0 → no pad
        buf.extend_from_slice(&0u32.to_le_bytes()); // state_len
        let err = parse_prelude(&buf).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("UTF-8"), "{msg}");
    }

    #[test]
    fn prelude_entry_multibyte_utf8_name_roundtrip() {
        // Multi-byte names surface char-vs-byte length regressions ASCII
        // tests can't catch. Test cases force every pad-to-4 residue.
        // "状態" = 3+3 = 6 bytes → pad=2.
        // "状態x" = 7 bytes → pad=1.
        // "状" = 3 bytes → pad=1 (pad arithmetic agnostic to name content).
        // "状態/0" = 3+3+1+1 = 8 bytes → pad=0.
        for (id, name) in [
            (10u16, "状態"),
            (11, "状態x"),
            (12, "状態/0"),
            (13, "🦀rust"), // 4-byte codepoint + 4 ASCII = 8 bytes → pad=0
        ] {
            let e = PreludeEntry {
                shared_state_id: id,
                codec_id: CodecId::Order1ScaleAC,
                state_format_version: 0,
                applies_to_mask: 0b0010,
                state_xxhash64: 0,
                name: name.to_string(),
                state_bytes: vec![1, 2, 3],
            };
            let mut buf = Vec::new();
            write_prelude(std::slice::from_ref(&e), &mut buf).unwrap();
            let parsed = read_prelude(&mut std::io::Cursor::new(buf)).unwrap();
            assert_eq!(parsed[0].name, name, "id={id}");
            assert_eq!(parsed[0].state_bytes, vec![1, 2, 3], "id={id}");
        }
    }

    #[test]
    fn prelude_entry_unnamed_roundtrip() {
        let e = PreludeEntry {
            shared_state_id: 0,
            codec_id: CodecId::PerGroupCodebook,
            state_format_version: 0,
            applies_to_mask: 0b0001,
            state_xxhash64: 0,
            name: String::new(),
            state_bytes: vec![0u8; 4],
        };
        let mut buf = Vec::new();
        write_prelude(std::slice::from_ref(&e), &mut buf).unwrap();
        let parsed = read_prelude(&mut std::io::Cursor::new(buf)).unwrap();
        assert_eq!(parsed[0].name, "");
        assert_eq!(parsed[0].state_bytes, vec![0u8; 4]);
    }

    #[test]
    fn prelude_entry_carries_state_xxhash64() {
        let e = PreludeEntry {
            shared_state_id: 3,
            codec_id: CodecId::Order1ScaleAC,
            state_format_version: 1,
            applies_to_mask: 0b0010,
            state_xxhash64: 0x0123456789ABCDEF,
            name: "shared/order1".to_string(),
            state_bytes: vec![9, 8, 7, 6, 5],
        };
        let mut buf = Vec::new();
        write_prelude(std::slice::from_ref(&e), &mut buf).unwrap();
        let parsed = read_prelude(&mut std::io::Cursor::new(buf)).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0], e);
    }

    #[test]
    fn prelude_entry_on_disk_size() {
        // Total entry size on disk:
        //   4 (count u32) + 8 (fixed prefix) + 8 (state_xxhash64) +
        //   2 (name_len) + 0 (name) + 0 (pad) + 4 (state_len) +
        //   5 (state_bytes) = 31 bytes.
        let e = PreludeEntry {
            shared_state_id: 0,
            codec_id: CodecId::Huffman,
            state_format_version: 0,
            applies_to_mask: 0b0001,
            state_xxhash64: 0,
            name: String::new(),
            state_bytes: vec![1, 2, 3, 4, 5],
        };
        let mut buf = Vec::new();
        write_prelude(std::slice::from_ref(&e), &mut buf).unwrap();
        assert_eq!(buf.len(), 4 + 8 + 8 + 2 + 4 + 5);
        let parsed = read_prelude(&mut std::io::Cursor::new(buf)).unwrap();
        assert_eq!(parsed[0].state_xxhash64, 0);
        assert_eq!(parsed[0].state_format_version, 0);
    }

    #[test]
    fn shared_state_matches_detects_divergence() {
        let a = PreludeEntry {
            shared_state_id: 0,
            codec_id: CodecId::Order1ScaleAC,
            state_format_version: 1,
            applies_to_mask: 0,
            state_xxhash64: 0xAAAA_AAAA_AAAA_AAAA,
            name: "x".into(),
            state_bytes: vec![1, 2, 3],
        };
        let mut b = a.clone();
        assert!(super::shared_state_matches(&a, &b));
        b.state_xxhash64 = 0xBBBB_BBBB_BBBB_BBBB;
        assert!(!super::shared_state_matches(&a, &b));
    }
}
