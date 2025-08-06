//! Shared framing helpers for the 4-stream interleaved entropy codecs
//! (`huffman::compress` / `rans::compress`). Both codecs split their input into 4
//! roughly-equal segments, encode each segment as an independent stream, and
//! record per-stream compressed lengths in a fixed-size jump table so the
//! decoder can locate each stream without scanning.
//!
//! The jump table stores all four lengths as little-endian `u32` (16 bytes
//! total).

/// Size of the jump table in bytes: 4 × u32 LE.
pub const JUMP_TABLE_BYTES: usize = 16;

/// Error returned when the jump-table payload is shorter than expected. The
/// shared framing module is codec-agnostic; each caller (`huffman::decompress`,
/// `rans::decompress`) wraps this into its own `PtwmCoreError` variant.
#[derive(Debug, Clone, Copy)]
pub struct TruncatedJumpTable;

impl core::fmt::Display for TruncatedJumpTable {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("truncated jump table")
    }
}

/// Split a buffer length into 4 roughly-equal contiguous segments.
/// The first `len % 4` segments are 1 byte larger than the remainder.
pub fn split_four(len: usize) -> [(usize, usize); 4] {
    let base = len / 4;
    let rem = len % 4;
    let mut start = 0usize;
    let mut ranges = [(0usize, 0usize); 4];
    for (i, range) in ranges.iter_mut().enumerate() {
        let seg_len = base + usize::from(i < rem);
        let end = start + seg_len;
        *range = (start, end);
        start = end;
    }
    ranges
}

/// Write `lens` into `dst` as 4 × u32 LE. `dst` must be at least
/// [`JUMP_TABLE_BYTES`] long; a debug assertion catches a too-small slice.
pub fn write_jump_table(dst: &mut [u8], lens: &[u32; 4]) {
    debug_assert!(dst.len() >= JUMP_TABLE_BYTES);
    for (i, &len) in lens.iter().enumerate() {
        dst[i * 4..i * 4 + 4].copy_from_slice(&len.to_le_bytes());
    }
}

/// Read 4 × u32 LE jump-table entries from the start of `src`. Returns the
/// four lengths and the number of bytes consumed. Errors if `src` is shorter
/// than [`JUMP_TABLE_BYTES`].
pub fn read_jump_table(src: &[u8]) -> Result<([u32; 4], usize), TruncatedJumpTable> {
    if src.len() < JUMP_TABLE_BYTES {
        return Err(TruncatedJumpTable);
    }
    let mut lens = [0u32; 4];
    for (i, slot) in lens.iter_mut().enumerate() {
        *slot = u32::from_le_bytes([src[i * 4], src[i * 4 + 1], src[i * 4 + 2], src[i * 4 + 3]]);
    }
    Ok((lens, JUMP_TABLE_BYTES))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_four_covers_all_bytes() {
        let ranges = split_four(11);
        assert_eq!(ranges[0], (0, 3));
        assert_eq!(ranges[1], (3, 6));
        assert_eq!(ranges[2], (6, 9));
        assert_eq!(ranges[3], (9, 11));
    }

    #[test]
    fn split_four_divisible_by_four() {
        let ranges = split_four(16);
        assert_eq!(ranges[0], (0, 4));
        assert_eq!(ranges[3], (12, 16));
    }

    #[test]
    fn split_four_zero_len() {
        let ranges = split_four(0);
        for r in ranges {
            assert_eq!(r, (0, 0));
        }
    }

    #[test]
    fn jump_table_roundtrip() {
        let lens = [0u32, 1, 70_000, u32::MAX];
        let mut buf = [0u8; JUMP_TABLE_BYTES];
        write_jump_table(&mut buf, &lens);
        let (out, consumed) = read_jump_table(&buf).unwrap();
        assert_eq!(consumed, JUMP_TABLE_BYTES);
        assert_eq!(out, lens);
    }

    #[test]
    fn jump_table_read_rejects_truncated_input() {
        let buf = [0u8; JUMP_TABLE_BYTES - 1];
        assert!(read_jump_table(&buf).is_err());
    }

    #[test]
    fn jump_table_accepts_lengths_above_u16() {
        let lens = [100_000u32, 200_000, 300_000, 1_000_000];
        let mut buf = [0u8; JUMP_TABLE_BYTES];
        write_jump_table(&mut buf, &lens);
        let (out, _) = read_jump_table(&buf).unwrap();
        assert_eq!(out, lens);
    }
}
