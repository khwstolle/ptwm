//! 64-bit integer split/combine.
//!
//! Handles 64-bit integer data by splitting bytes into 8 buffers using
//! round-robin distribution (mode 10). No bit reordering is applied —
//! integer types lack the exponent/mantissa structure of IEEE 754 floats.

const NUM_BUF: usize = 8;

// ---------------------------------------------------------------------------
// Buffer sizes
// ---------------------------------------------------------------------------

/// Compute buffer sizes for mode 10 splitting with 8 buffers.
///
/// Each buffer gets `len / 8` bytes, with the first `len % 8` buffers getting
/// one extra byte.
pub fn buffer_sizes(len: usize, bytes_mode: i32) -> Vec<usize> {
    assert_eq!(bytes_mode, 10, "dtype64 only supports bytes_mode 10");
    let base = len / NUM_BUF;
    let remainder = len % NUM_BUF;
    (0..NUM_BUF)
        .map(|b| if b < remainder { base + 1 } else { base })
        .collect()
}

// ---------------------------------------------------------------------------
// Split
// ---------------------------------------------------------------------------

/// Split `src` into 8 buffers using round-robin distribution (mode 10).
///
/// No bit reordering is supported for integer types; `bits_mode` must be 0.
pub fn split(src: &[u8], bits_mode: i32, bytes_mode: i32) -> Vec<Vec<u8>> {
    assert_eq!(bytes_mode, 10, "dtype64 only supports bytes_mode 10");
    assert_eq!(bits_mode, 0, "dtype64 does not support bit reordering");

    let sizes = buffer_sizes(src.len(), bytes_mode);
    let mut bufs: Vec<Vec<u8>> = sizes.iter().map(|&s| Vec::with_capacity(s)).collect();

    // Round-robin: byte i goes to buffer (i % 8)
    for (i, &byte) in src.iter().enumerate() {
        bufs[i % NUM_BUF].push(byte);
    }

    bufs
}

// ---------------------------------------------------------------------------
// Combine
// ---------------------------------------------------------------------------

/// Combine 8 buffers back into `out`, reversing the mode 10 split.
///
/// No bit reordering is applied for integer types.
pub fn combine_bufs(
    bufs: &[&[u8]],
    buf_lens: &[usize],
    out: &mut [u8],
    bits_mode: i32,
    bytes_mode: i32,
) {
    assert_eq!(bytes_mode, 10, "dtype64 only supports bytes_mode 10");
    assert_eq!(bits_mode, 0, "dtype64 does not support bit reordering");
    assert_eq!(bufs.len(), NUM_BUF);
    assert_eq!(buf_lens.len(), NUM_BUF);

    let total_len = out.len();
    let q_len = total_len / NUM_BUF;
    let remainder = total_len % NUM_BUF;

    let mut dst = 0;

    // Interleave complete rounds
    for i in 0..q_len {
        out[dst] = bufs[0][i];
        dst += 1;
        out[dst] = bufs[1][i];
        dst += 1;
        out[dst] = bufs[2][i];
        dst += 1;
        out[dst] = bufs[3][i];
        dst += 1;
        out[dst] = bufs[4][i];
        dst += 1;
        out[dst] = bufs[5][i];
        dst += 1;
        out[dst] = bufs[6][i];
        dst += 1;
        out[dst] = bufs[7][i];
        dst += 1;
    }

    // Handle remainder bytes
    for b in 0..NUM_BUF {
        if b < remainder {
            out[dst] = bufs[b][buf_lens[b] - 1];
            dst += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_mode10_aligned() {
        // 256 bytes, divisible by 8
        let src: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let buffers = split(&src, 0, 10);
        assert_eq!(buffers.len(), 8);
        for buf in &buffers {
            assert_eq!(buf.len(), 32);
        }

        let sizes = buffer_sizes(src.len(), 10);
        let buf_refs: Vec<&[u8]> = buffers.iter().map(|b| b.as_slice()).collect();
        let mut out = vec![0u8; src.len()];
        combine_bufs(&buf_refs, &sizes, &mut out, 0, 10);
        assert_eq!(src, out);
    }

    #[test]
    fn roundtrip_mode10_remainder() {
        // 253 bytes, remainder = 5
        let src: Vec<u8> = (0..253).map(|i| i as u8).collect();
        let buffers = split(&src, 0, 10);
        assert_eq!(buffers.len(), 8);

        let sizes = buffer_sizes(src.len(), 10);
        // First 5 buffers get 32 bytes, last 3 get 31
        assert_eq!(sizes, vec![32, 32, 32, 32, 32, 31, 31, 31]);
        for (i, buf) in buffers.iter().enumerate() {
            assert_eq!(buf.len(), sizes[i]);
        }

        let buf_refs: Vec<&[u8]> = buffers.iter().map(|b| b.as_slice()).collect();
        let mut out = vec![0u8; src.len()];
        combine_bufs(&buf_refs, &sizes, &mut out, 0, 10);
        assert_eq!(src, out);
    }

    #[test]
    fn roundtrip_small() {
        // Less than 8 bytes
        let src: Vec<u8> = vec![10, 20, 30];
        let buffers = split(&src, 0, 10);
        assert_eq!(buffers.len(), 8);

        let sizes = buffer_sizes(src.len(), 10);
        assert_eq!(sizes, vec![1, 1, 1, 0, 0, 0, 0, 0]);

        let buf_refs: Vec<&[u8]> = buffers.iter().map(|b| b.as_slice()).collect();
        let mut out = vec![0u8; src.len()];
        combine_bufs(&buf_refs, &sizes, &mut out, 0, 10);
        assert_eq!(src, out);
    }

    #[test]
    fn buffer_sizes_exact_multiple() {
        let sizes = buffer_sizes(80, 10);
        assert_eq!(sizes, vec![10, 10, 10, 10, 10, 10, 10, 10]);
    }
}
