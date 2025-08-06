//! EOF sentinel and tensor index.

use crate::error::PtwmCoreError;
use xxhash_rust::xxh64::xxh64;

pub const EOF_SENTINEL: [u8; 8] = [0x89, b'T', b'W', b'M', b'E', b'O', b'F', 0x1A];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexEntry {
    pub name_hash: u64,
    pub tensor_offset: u64,
    pub tensor_len: u64,
}

pub fn write_index(entries: &mut [IndexEntry], out: &mut Vec<u8>) {
    entries.sort_by_key(|e| e.name_hash);
    out.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    for e in entries {
        out.extend_from_slice(&e.name_hash.to_le_bytes());
        out.extend_from_slice(&e.tensor_offset.to_le_bytes());
        out.extend_from_slice(&e.tensor_len.to_le_bytes());
    }
}

pub fn parse_index(buf: &[u8]) -> Result<Vec<IndexEntry>, PtwmCoreError> {
    if buf.len() < 8 {
        return Err(PtwmCoreError::InvalidContainer(
            "index section too short".into(),
        ));
    }
    let count_u64 = u64::from_le_bytes(buf[..8].try_into().unwrap());
    // Reject counts that cannot fit in the remaining bytes before
    // allocating. Adversarial inputs (e.g. count = 0x06_06_..._06) would
    // otherwise wrap on `count * 24` in release mode, pass the length
    // check, then panic at `Vec::with_capacity` with a capacity overflow.
    let remaining = (buf.len() - 8) as u64;
    if count_u64 > remaining / 24 {
        return Err(PtwmCoreError::InvalidContainer(
            "index count exceeds remaining bytes".into(),
        ));
    }
    let count = count_u64 as usize;
    let expected_len = 8 + count * 24;
    if buf.len() < expected_len {
        return Err(PtwmCoreError::InvalidContainer(
            "index section truncated".into(),
        ));
    }
    let mut entries = Vec::with_capacity(count);
    let mut pos = 8;
    for _ in 0..count {
        entries.push(IndexEntry {
            name_hash: u64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap()),
            tensor_offset: u64::from_le_bytes(buf[pos + 8..pos + 16].try_into().unwrap()),
            tensor_len: u64::from_le_bytes(buf[pos + 16..pos + 24].try_into().unwrap()),
        });
        pos += 24;
    }
    Ok(entries)
}

pub fn name_hash(name: &str) -> u64 {
    xxh64(name.as_bytes(), 0)
}

pub fn verify_eof_sentinel(buf: &[u8]) -> Result<(), PtwmCoreError> {
    if buf.len() < 8 {
        return Err(PtwmCoreError::InvalidContainer("file too short".into()));
    }
    if buf[buf.len() - 8..] != EOF_SENTINEL {
        return Err(PtwmCoreError::InvalidContainer(
            "missing EOF sentinel (file may be truncated)".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_roundtrip_sorted() {
        let mut entries = vec![
            IndexEntry {
                name_hash: 5,
                tensor_offset: 100,
                tensor_len: 10,
            },
            IndexEntry {
                name_hash: 1,
                tensor_offset: 0,
                tensor_len: 20,
            },
            IndexEntry {
                name_hash: 3,
                tensor_offset: 50,
                tensor_len: 30,
            },
        ];
        let mut out = Vec::new();
        write_index(&mut entries, &mut out);
        let parsed = parse_index(&out).unwrap();
        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed[0].name_hash, 1);
        assert_eq!(parsed[1].name_hash, 3);
        assert_eq!(parsed[2].name_hash, 5);
    }

    #[test]
    fn eof_sentinel_round_trip() {
        let mut file = b"some content".to_vec();
        file.extend_from_slice(&EOF_SENTINEL);
        assert!(verify_eof_sentinel(&file).is_ok());
    }

    #[test]
    fn missing_sentinel_rejected() {
        let file = b"some content without sentinel at end".to_vec();
        assert!(verify_eof_sentinel(&file).is_err());
    }
}
