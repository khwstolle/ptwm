//! FP32 split/combine.
//!
//! Handles 32-bit float data by optionally reordering float bits (exponent first)
//! for better compression, then splitting bytes into 4 buffers using round-robin
//! distribution (mode 220).

const NUM_BUF: usize = 4;

// ---------------------------------------------------------------------------
// Bit reordering (matching C implementation exactly)
// ---------------------------------------------------------------------------

fn reorder_float_bits_dtype32(val: u32) -> u32 {
    let sign = (val >> 8) & 0x0080_0000;
    let exponent = (val << 1) & 0xFF00_0000;
    let mantissa = val & 0x007F_FFFF;
    exponent | sign | mantissa
}

fn revert_float_bits_dtype32(val: u32) -> u32 {
    let sign = (val << 8) & 0x8000_0000;
    let exponent = (val >> 1) & 0x7F80_0000;
    let mantissa = val & 0x007F_FFFF;
    sign | exponent | mantissa
}

/// Reorder all 32-bit floats in `data` (in-place) for better compression.
/// Processes complete 4-byte words; any trailing bytes are left untouched.
pub fn reorder_all_floats_dtype32(data: &mut [u8]) {
    // Process complete 4-byte words via unaligned u32 reads/writes, avoiding
    // the from_le_bytes / to_le_bytes array round-trip per iteration.
    // SAFETY: we only touch the floor(len/4) complete words; read_unaligned /
    // write_unaligned stay sound even when the slice is not 4-aligned.
    let word_count = data.len() / 4;
    let ptr = data.as_mut_ptr();
    for i in 0..word_count {
        // SAFETY: i*4 + 3 < data.len() because i < word_count = len/4.
        let val = unsafe { (ptr.add(i * 4) as *const u32).read_unaligned() }.to_le();
        let reordered = reorder_float_bits_dtype32(val);
        unsafe { (ptr.add(i * 4) as *mut u32).write_unaligned(reordered.to_le()) };
    }
}

/// Revert the bit reordering applied by [`reorder_all_floats_dtype32`].
/// Processes complete 4-byte words; any trailing bytes are left untouched.
pub fn revert_all_floats_dtype32(data: &mut [u8]) {
    let word_count = data.len() / 4;
    let ptr = data.as_mut_ptr();
    for i in 0..word_count {
        // SAFETY: i*4 + 3 < data.len() because i < word_count = len/4.
        let val = unsafe { (ptr.add(i * 4) as *const u32).read_unaligned() }.to_le();
        let reverted = revert_float_bits_dtype32(val);
        unsafe { (ptr.add(i * 4) as *mut u32).write_unaligned(reverted.to_le()) };
    }
}

// ---------------------------------------------------------------------------
// Buffer sizes
// ---------------------------------------------------------------------------

/// Compute buffer sizes for mode 220 splitting.
///
/// Each buffer gets `len / 4` bytes, with the first `len % 4` buffers getting
/// one extra byte.
pub fn buffer_sizes(len: usize, bytes_mode: i32) -> Vec<usize> {
    if bytes_mode == 0 {
        return vec![len, 0, 0, 0];
    }
    assert_eq!(bytes_mode, 220, "dtype32 only supports bytes_mode 0 or 220");
    let base = len / NUM_BUF;
    let remainder = len % NUM_BUF;
    (0..NUM_BUF)
        .map(|b| if b < remainder { base + 1 } else { base })
        .collect()
}

// ---------------------------------------------------------------------------
// Split
// ---------------------------------------------------------------------------

/// Split `src` into 4 buffers using round-robin distribution (mode 220).
///
/// If `bits_mode == 1`, float bits are reordered before splitting.
pub fn split(src: &[u8], bits_mode: i32, bytes_mode: i32) -> Vec<Vec<u8>> {
    assert!(
        matches!(bytes_mode, 0 | 220),
        "dtype32 only supports bytes_mode 0 or 220"
    );

    let mut data;
    let input = if bits_mode == 1 {
        data = src.to_vec();
        reorder_all_floats_dtype32(&mut data);
        &data[..]
    } else {
        src
    };

    if bytes_mode == 0 {
        return vec![input.to_vec(), Vec::new(), Vec::new(), Vec::new()];
    }

    let sizes = buffer_sizes(input.len(), bytes_mode);
    let mut bufs: Vec<Vec<u8>> = sizes.iter().map(|&s| Vec::with_capacity(s)).collect();

    // Round-robin: byte i goes to buffer (i % 4)
    for (i, &byte) in input.iter().enumerate() {
        bufs[i % NUM_BUF].push(byte);
    }

    bufs
}

// ---------------------------------------------------------------------------
// Combine
// ---------------------------------------------------------------------------

/// Combine 4 buffers back into `out`, reversing the mode 220 split.
///
/// If `bits_mode == 1`, float bit reordering is reverted after combining.
///
/// The `bufs` parameter is a slice of slices, and `buf_lens` provides the
/// length of each buffer.
pub fn combine_bufs(
    bufs: &[&[u8]],
    buf_lens: &[usize],
    out: &mut [u8],
    bits_mode: i32,
    bytes_mode: i32,
) {
    assert!(
        matches!(bytes_mode, 0 | 220),
        "dtype32 only supports bytes_mode 0 or 220"
    );
    assert_eq!(bufs.len(), NUM_BUF);
    assert_eq!(buf_lens.len(), NUM_BUF);

    if bytes_mode == 0 {
        assert_eq!(
            buf_lens[0],
            out.len(),
            "buffer length must match output length in mode 0"
        );
        assert_eq!(buf_lens[1], 0, "buffer 1 must be empty in mode 0");
        assert_eq!(buf_lens[2], 0, "buffer 2 must be empty in mode 0");
        assert_eq!(buf_lens[3], 0, "buffer 3 must be empty in mode 0");
        out.copy_from_slice(bufs[0]);
    } else {
        let total_len = out.len();
        let q_len = total_len / NUM_BUF; // number of complete rounds
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
        }

        // Handle remainder bytes (matching C exactly)
        for b in 0..NUM_BUF {
            if b < remainder {
                out[dst] = bufs[b][buf_lens[b] - 1];
                dst += 1;
            }
        }
    }

    // Revert bit reordering if needed
    if bits_mode == 1 {
        revert_all_floats_dtype32(out);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_mode220_aligned() {
        // 256 bytes, divisible by 4
        let src: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let buffers = split(&src, 0, 220);
        assert_eq!(buffers.len(), 4);
        // Each buffer should be exactly 64 bytes
        for buf in &buffers {
            assert_eq!(buf.len(), 64);
        }

        let sizes = buffer_sizes(src.len(), 220);
        let buf_refs: Vec<&[u8]> = buffers.iter().map(|b| b.as_slice()).collect();
        let mut out = vec![0u8; src.len()];
        combine_bufs(&buf_refs, &sizes, &mut out, 0, 220);
        assert_eq!(src, out);
    }

    #[test]
    fn roundtrip_mode0_no_split() {
        let src: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        let buffers = split(&src, 0, 0);
        assert_eq!(buffers.len(), 4);
        assert_eq!(buffers[0], src);
        assert!(buffers[1].is_empty());
        assert!(buffers[2].is_empty());
        assert!(buffers[3].is_empty());
        let sizes = buffer_sizes(src.len(), 0);
        let buf_refs: Vec<&[u8]> = buffers.iter().map(|b| b.as_slice()).collect();
        let mut out = vec![0u8; src.len()];
        combine_bufs(&buf_refs, &sizes, &mut out, 0, 0);
        assert_eq!(src, out);
    }

    #[test]
    fn roundtrip_mode0_with_bit_reorder() {
        let src: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        let buffers = split(&src, 1, 0);
        let sizes = buffer_sizes(src.len(), 0);
        let buf_refs: Vec<&[u8]> = buffers.iter().map(|b| b.as_slice()).collect();
        let mut out = vec![0u8; src.len()];
        combine_bufs(&buf_refs, &sizes, &mut out, 1, 0);
        assert_eq!(src, out);
    }

    #[test]
    fn roundtrip_mode220_remainder() {
        // 255 bytes, remainder = 3
        let src: Vec<u8> = (0..255).map(|i| i as u8).collect();
        let buffers = split(&src, 0, 220);
        assert_eq!(buffers.len(), 4);

        let sizes = buffer_sizes(src.len(), 220);
        // First 3 buffers get 64 bytes, last gets 63
        assert_eq!(sizes, vec![64, 64, 64, 63]);
        for (i, buf) in buffers.iter().enumerate() {
            assert_eq!(buf.len(), sizes[i]);
        }

        let buf_refs: Vec<&[u8]> = buffers.iter().map(|b| b.as_slice()).collect();
        let mut out = vec![0u8; src.len()];
        combine_bufs(&buf_refs, &sizes, &mut out, 0, 220);
        assert_eq!(src, out);
    }

    #[test]
    fn roundtrip_mode220_with_bit_reorder() {
        // 1024 bytes, bits_mode=1
        let src: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        let buffers = split(&src, 1, 220);
        assert_eq!(buffers.len(), 4);

        let sizes = buffer_sizes(src.len(), 220);
        let buf_refs: Vec<&[u8]> = buffers.iter().map(|b| b.as_slice()).collect();
        let mut out = vec![0u8; src.len()];
        combine_bufs(&buf_refs, &sizes, &mut out, 1, 220);
        assert_eq!(src, out);
    }

    #[test]
    fn bit_reorder_revert_roundtrip() {
        // Verify that reorder followed by revert is identity
        let original: Vec<u8> = (0..64).map(|i| (i * 7 + 13) as u8).collect();
        let mut data = original.clone();
        reorder_all_floats_dtype32(&mut data);
        // Data should be different after reordering (for most inputs)
        revert_all_floats_dtype32(&mut data);
        assert_eq!(original, data);
    }
}
