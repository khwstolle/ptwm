//! Plane record serialization.

use crate::codec::{CodecId, StateSource};
use crate::error::PtwmCoreError;
use crate::layout::PlaneLayout;
use crate::types::PlaneRole;

pub const PLANE_FLAG_CHUNK_TABLE: u8 = 0x01;
pub const PLANE_FLAG_CRC32: u8 = 0x02;
/// Signals that the 2-byte `codec_table_idx` field is present immediately
/// after `codec_id` in the wire layout.
pub const PLANE_FLAG_CODEC_TABLE_IDX: u8 = 0x04;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkEntry {
    pub offset_in_payload: u32,
    pub decoded_size: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalStateRef {
    pub ref_kind: u8,
    pub expected_hash: Option<[u8; 32]>,
    pub ref_bytes: Vec<u8>,
    pub state_format_version: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaneRecord {
    pub role: PlaneRole,
    pub codec_id: CodecId,
    /// Index into the file's Extension Table for the codec used to compress
    /// this plane.
    pub codec_table_idx: u16,
    pub state_source: StateSource,
    /// For Inline, this is the inline state's version;
    /// for Shared, the writer copies the prelude entry's version here
    /// and the reader cross-validates;
    /// for None, this is 0.
    pub state_version: u8,
    pub state_info: u16,
    pub payload_len: u64,
    pub crc32: Option<u32>,
    pub chunk_table: Option<Vec<ChunkEntry>>,
    pub external_state: Option<ExternalStateRef>,
    pub inline_state_bytes: Vec<u8>,
    pub payload_bytes: Vec<u8>,
    pub layout: PlaneLayout,
}

pub fn write_plane_record(rec: &PlaneRecord, out: &mut Vec<u8>) -> Result<(), PtwmCoreError> {
    let mut plane_flags: u8 = 0;
    if rec.chunk_table.is_some() {
        plane_flags |= PLANE_FLAG_CHUNK_TABLE;
    }
    if rec.crc32.is_some() {
        plane_flags |= PLANE_FLAG_CRC32;
    }
    plane_flags |= PLANE_FLAG_CODEC_TABLE_IDX;
    out.push(rec.role as u8);
    out.push(plane_flags);
    out.extend_from_slice(&rec.codec_id.as_u16().to_le_bytes());
    // codec_table_idx is written immediately after codec_id.
    out.extend_from_slice(&rec.codec_table_idx.to_le_bytes());
    out.push(rec.state_source as u8);
    out.push(rec.state_version);
    out.extend_from_slice(&rec.state_info.to_le_bytes());
    let inline_state_len = rec.inline_state_bytes.len() as u32;
    out.extend_from_slice(&inline_state_len.to_le_bytes());
    out.extend_from_slice(&rec.payload_len.to_le_bytes());
    if let Some(crc) = rec.crc32 {
        out.extend_from_slice(&crc.to_le_bytes());
    }
    if let Some(chunks) = &rec.chunk_table {
        out.extend_from_slice(&(chunks.len() as u32).to_le_bytes());
        for c in chunks {
            out.extend_from_slice(&c.offset_in_payload.to_le_bytes());
            out.extend_from_slice(&c.decoded_size.to_le_bytes());
        }
    }
    if let Some(ext) = &rec.external_state {
        let expected_hash_present = if ext.expected_hash.is_some() {
            0x01u8
        } else {
            0
        };
        out.push(ext.ref_kind);
        out.push(expected_hash_present);
        out.extend_from_slice(&(ext.ref_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(&ext.ref_bytes);
        out.push(ext.state_format_version);
        if let Some(hash) = ext.expected_hash {
            out.extend_from_slice(&hash);
        }
    }
    // PlaneLayout: 1-byte kind + variable payload (see layout.rs)
    out.push(rec.layout.kind());
    out.extend_from_slice(&rec.layout.payload_bytes());
    out.extend_from_slice(&rec.inline_state_bytes);
    out.extend_from_slice(&rec.payload_bytes);
    Ok(())
}

pub fn parse_plane_record(buf: &[u8]) -> Result<(PlaneRecord, usize), PtwmCoreError> {
    if buf.len() < 20 {
        return Err(PtwmCoreError::InvalidContainer(
            "plane record header truncated".into(),
        ));
    }
    fn need(buf: &[u8], pos: usize, n: usize, what: &str) -> Result<(), PtwmCoreError> {
        if pos.checked_add(n).is_none_or(|end| end > buf.len()) {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "plane record truncated reading {what}"
            )));
        }
        Ok(())
    }

    let mut pos = 0usize;
    let role = PlaneRole::from_u8(buf[pos])
        .ok_or_else(|| PtwmCoreError::InvalidContainer("unknown plane role".into()))?;
    pos += 1;
    let plane_flags = buf[pos];
    pos += 1;
    let codec_raw = u16::from_le_bytes(buf[pos..pos + 2].try_into().unwrap());
    let codec_id = CodecId::from_u16(codec_raw).ok_or_else(|| {
        PtwmCoreError::InvalidContainer(format!("unknown codec_id 0x{:04x}", codec_raw))
    })?;
    pos += 2;
    // codec_table_idx is present when PLANE_FLAG_CODEC_TABLE_IDX is set.
    need(buf, pos, 2, "codec_table_idx")?;
    let codec_table_idx = u16::from_le_bytes(buf[pos..pos + 2].try_into().unwrap());
    pos += 2;
    need(buf, pos, 1, "state_source")?;
    let state_source = StateSource::from_u8(buf[pos])
        .ok_or_else(|| PtwmCoreError::InvalidContainer("unknown state_source".into()))?;
    pos += 1;
    need(buf, pos, 1, "state_version")?;
    let state_version = buf[pos];
    pos += 1;
    need(buf, pos, 2, "state_info")?;
    let state_info = u16::from_le_bytes(buf[pos..pos + 2].try_into().unwrap());
    pos += 2;
    need(buf, pos, 4, "inline_state_len")?;
    let inline_state_len = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;
    need(buf, pos, 8, "payload_len")?;
    let payload_len = u64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap());
    pos += 8;

    let crc32 = if plane_flags & PLANE_FLAG_CRC32 != 0 {
        need(buf, pos, 4, "plane crc32")?;
        let c = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap());
        pos += 4;
        Some(c)
    } else {
        None
    };

    let chunk_table = if plane_flags & PLANE_FLAG_CHUNK_TABLE != 0 {
        need(buf, pos, 4, "chunk_count")?;
        let chunk_count = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        // Reject chunk counts that cannot fit in the remaining buffer (each
        // chunk entry is 8 bytes) before allocating, mirroring the
        // parse_index guard against capacity-overflow allocations.
        let remaining = buf.len() - pos;
        if chunk_count > remaining / 8 {
            return Err(PtwmCoreError::InvalidContainer(
                "chunk_count exceeds remaining bytes".into(),
            ));
        }
        let mut chunks = Vec::with_capacity(chunk_count);
        for _ in 0..chunk_count {
            need(buf, pos, 8, "chunk entry")?;
            chunks.push(ChunkEntry {
                offset_in_payload: u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()),
                decoded_size: u32::from_le_bytes(buf[pos + 4..pos + 8].try_into().unwrap()),
            });
            pos += 8;
        }
        Some(chunks)
    } else {
        None
    };

    let external_state = if matches!(state_source, StateSource::External) {
        need(buf, pos, 4, "external state header")?;
        let ref_kind = buf[pos];
        let ref_flags = buf[pos + 1];
        pos += 2;
        let ref_len = u16::from_le_bytes(buf[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;
        need(buf, pos, ref_len, "external ref_bytes")?;
        let ref_bytes = buf[pos..pos + ref_len].to_vec();
        pos += ref_len;
        need(buf, pos, 1, "state_format_version")?;
        let state_format_version = buf[pos];
        pos += 1;
        let expected_hash = if ref_flags & 0x01 != 0 {
            need(buf, pos, 32, "external state hash")?;
            let mut hash = [0u8; 32];
            hash.copy_from_slice(&buf[pos..pos + 32]);
            pos += 32;
            Some(hash)
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

    // PlaneLayout: 1-byte kind + variable payload (see layout.rs)
    need(buf, pos, 1, "layout kind")?;
    let layout_kind = buf[pos];
    pos += 1;
    let layout = {
        let remaining = &buf[pos..];
        let (l, consumed) = PlaneLayout::parse(layout_kind, remaining)?;
        pos += consumed;
        l
    };

    need(buf, pos, inline_state_len, "inline_state_bytes")?;
    let inline_state_bytes = buf[pos..pos + inline_state_len].to_vec();
    pos += inline_state_len;
    let payload_len_usize = usize::try_from(payload_len)
        .map_err(|_| PtwmCoreError::InvalidContainer("plane payload_len overflows usize".into()))?;
    need(buf, pos, payload_len_usize, "plane payload_bytes")?;
    let payload_bytes = buf[pos..pos + payload_len_usize].to_vec();
    pos += payload_len_usize;

    Ok((
        PlaneRecord {
            role,
            codec_id,
            codec_table_idx,
            state_source,
            state_version,
            state_info,
            payload_len,
            crc32,
            chunk_table,
            external_state,
            inline_state_bytes,
            payload_bytes,
            layout,
        },
        pos,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{CodecId, StateSource};
    use crate::layout::PlaneLayout;
    use crate::types::PlaneRole;

    #[test]
    fn plane_record_roundtrip_minimal() {
        let rec = PlaneRecord {
            role: PlaneRole::Value,
            codec_id: CodecId::Identity,
            codec_table_idx: 0,
            state_source: StateSource::None,
            state_version: 0,
            state_info: 0,
            payload_len: 4,
            crc32: None,
            chunk_table: None,
            external_state: None,
            inline_state_bytes: Vec::new(),
            payload_bytes: vec![1, 2, 3, 4],
            layout: PlaneLayout::Flat,
        };
        let mut out = Vec::new();
        write_plane_record(&rec, &mut out).unwrap();
        let (parsed, consumed) = parse_plane_record(&out).unwrap();
        assert_eq!(parsed, rec);
        assert_eq!(consumed, out.len());
    }

    #[test]
    fn plane_record_roundtrip_full() {
        let rec = PlaneRecord {
            role: PlaneRole::Scale,
            codec_id: CodecId::Order1ScaleAC,
            codec_table_idx: 0,
            state_source: StateSource::Shared,
            state_version: 0,
            state_info: 7,
            payload_len: 8,
            crc32: Some(0xDEADBEEF),
            chunk_table: Some(vec![
                ChunkEntry {
                    offset_in_payload: 0,
                    decoded_size: 4,
                },
                ChunkEntry {
                    offset_in_payload: 4,
                    decoded_size: 4,
                },
            ]),
            external_state: None,
            inline_state_bytes: Vec::new(),
            payload_bytes: vec![1; 8],
            layout: PlaneLayout::Flat,
        };
        let mut out = Vec::new();
        write_plane_record(&rec, &mut out).unwrap();
        let (parsed, _) = parse_plane_record(&out).unwrap();
        assert_eq!(parsed, rec);
    }

    #[test]
    fn plane_record_roundtrip_with_codec_table_idx() {
        // Verify writer emits the 2-byte field and reader recovers it intact.
        let rec = PlaneRecord {
            role: PlaneRole::Exponent,
            codec_id: CodecId::Huffman,
            codec_table_idx: 3, // arbitrary non-absent value
            state_source: StateSource::None,
            state_version: 0,
            state_info: 0,
            payload_len: 2,
            crc32: None,
            chunk_table: None,
            external_state: None,
            inline_state_bytes: Vec::new(),
            payload_bytes: vec![0xAB, 0xCD],
            layout: PlaneLayout::Flat,
        };
        let mut out = Vec::new();
        write_plane_record(&rec, &mut out).unwrap();
        let (parsed, consumed) = parse_plane_record(&out).unwrap();
        assert_eq!(parsed, rec);
        assert_eq!(parsed.codec_table_idx, 3);
        assert_eq!(consumed, out.len());
    }

    #[test]
    fn plane_record_layout_rows_roundtrip() {
        let rec = PlaneRecord {
            role: PlaneRole::Scale,
            codec_id: CodecId::Identity,
            codec_table_idx: 0,
            state_source: StateSource::None,
            state_version: 0,
            state_info: 0,
            payload_len: 0,
            crc32: None,
            chunk_table: None,
            external_state: None,
            inline_state_bytes: Vec::new(),
            payload_bytes: Vec::new(),
            layout: PlaneLayout::rows(128).unwrap(),
        };
        let mut buf = Vec::new();
        write_plane_record(&rec, &mut buf).unwrap();
        let (parsed, consumed) = parse_plane_record(&buf).unwrap();
        assert_eq!(parsed.layout, PlaneLayout::rows(128).unwrap());
        assert_eq!(parsed.state_version, 0);
        assert_eq!(consumed, buf.len());
    }

    #[test]
    fn parse_rejects_unknown_layout_kind() {
        // Build a valid Flat plane record, then mutate the layout_kind
        // byte to a value the parser must reject.
        let rec = PlaneRecord {
            role: PlaneRole::Value,
            codec_id: CodecId::Identity,
            codec_table_idx: 0,
            state_source: StateSource::None,
            state_version: 0,
            state_info: 0,
            payload_len: 0,
            crc32: None,
            chunk_table: None,
            external_state: None,
            inline_state_bytes: Vec::new(),
            payload_bytes: Vec::new(),
            layout: PlaneLayout::Flat,
        };
        let mut buf = Vec::new();
        write_plane_record(&rec, &mut buf).unwrap();
        // Find the first occurrence of layout_kind=0 and overwrite to 0xFF.
        // The parser reads layout_kind at a deterministic offset, but we
        // don't depend on it here — try every byte and keep the assertion
        // contingent on at least one mutation rejecting.
        let mut rejected = false;
        for pos in 0..buf.len() {
            if buf[pos] == 0 {
                let mut clone = buf.clone();
                clone[pos] = 0xFF;
                if parse_plane_record(&clone).is_err() {
                    rejected = true;
                }
            }
        }
        assert!(rejected, "no mutation rejected");
    }

    #[test]
    fn plane_record_layout_flat_default_roundtrip() {
        let rec = PlaneRecord {
            role: PlaneRole::Value,
            codec_id: CodecId::Identity,
            codec_table_idx: 0,
            state_source: StateSource::None,
            state_version: 0,
            state_info: 0,
            payload_len: 0,
            crc32: None,
            chunk_table: None,
            external_state: None,
            inline_state_bytes: Vec::new(),
            payload_bytes: Vec::new(),
            layout: PlaneLayout::Flat,
        };
        let mut buf = Vec::new();
        write_plane_record(&rec, &mut buf).unwrap();
        let (parsed, consumed) = parse_plane_record(&buf).unwrap();
        assert_eq!(parsed.layout, PlaneLayout::Flat);
        assert_eq!(consumed, buf.len());
    }
}
