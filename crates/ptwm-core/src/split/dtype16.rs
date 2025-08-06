//! BF16/FP16 split/combine.
//!
//! Handles 16-bit float data by optionally reordering float bits (exponent first)
//! for better compression, then splitting bytes into 2 buffers based on mode.

/// Reorder bits of two packed 16-bit floats so exponent comes first.
///
/// Layout: `[sign(1) exponent(8) mantissa(7)]` per 16-bit float.
/// After reorder: `[exponent(8) sign(1) mantissa(7)]` per 16-bit float.
#[inline]
fn reorder_float_bits_dtype16(val: u32) -> u32 {
    let sign = (val >> 8) & 0x0080_0080;
    let exponent = (val << 1) & 0xFF00_FF00;
    let mantissa = val & 0x007F_007F;
    exponent | sign | mantissa
}

/// Revert the bit reordering performed by [`reorder_float_bits_dtype16`].
#[inline]
fn revert_float_bits_dtype16(val: u32) -> u32 {
    let sign = (val << 8) & 0x8000_8000;
    let exponent = (val >> 1) & 0x7F80_7F80;
    let mantissa = val & 0x007F_007F;
    sign | exponent | mantissa
}

/// Apply float bit reordering to an entire buffer in-place.
///
/// The buffer is interpreted as `&mut [u32]` (little-endian). Any trailing bytes
/// that don't fill a complete `u32` are left untouched.
pub fn reorder_all_floats_dtype16(data: &mut [u8]) {
    // Reinterpret the byte slice as a u32 slice in-place (little-endian host).
    // This avoids the from_le_bytes / to_le_bytes round-trip per iteration —
    // on a little-endian host the byte layout is already u32-compatible.
    // SAFETY: u32 has align 4; we only process the floor(len/4) complete words.
    let word_count = data.len() / 4;
    // SAFETY: u8 and u32 have no aliasing constraints when the pointer is
    // sufficiently aligned. On aarch64/x86-64 unaligned reads are safe (the
    // compiler will emit the appropriate load). We use ptr::read_unaligned and
    // ptr::write_unaligned to stay sound even when the slice is not 4-aligned.
    let ptr = data.as_mut_ptr();
    for i in 0..word_count {
        // SAFETY: i*4 + 3 < data.len() because i < word_count = len/4.
        let val = unsafe { (ptr.add(i * 4) as *const u32).read_unaligned() }.to_le();
        let reordered = reorder_float_bits_dtype16(val);
        unsafe { (ptr.add(i * 4) as *mut u32).write_unaligned(reordered.to_le()) };
    }
}

/// Revert float bit reordering on an entire buffer in-place.
pub fn revert_all_floats_dtype16(data: &mut [u8]) {
    let word_count = data.len() / 4;
    let ptr = data.as_mut_ptr();
    for i in 0..word_count {
        let val = unsafe { (ptr.add(i * 4) as *const u32).read_unaligned() }.to_le();
        let reverted = revert_float_bits_dtype16(val);
        unsafe { (ptr.add(i * 4) as *mut u32).write_unaligned(reverted.to_le()) };
    }
}

/// Compute the sizes of the output buffers for a given input length and bytes mode.
pub fn buffer_sizes(len: usize, bytes_mode: i32) -> Vec<usize> {
    match bytes_mode {
        0 => vec![len, 0],
        10 => {
            // Even bytes go to buf0, odd bytes go to buf1.
            let buf0_len = len.div_ceil(2);
            let buf1_len = len / 2;
            vec![buf0_len, buf1_len]
        }
        8 => {
            // MSB bytes only (odd-indexed in little-endian) → buf0; buf1 empty.
            let buf0_len = len / 2;
            vec![buf0_len, 0]
        }
        1 => {
            // LSB bytes only (even-indexed) → buf0; buf1 empty.
            let buf0_len = len.div_ceil(2);
            vec![buf0_len, 0]
        }
        // Invariant: caller validated bytes_mode via `crate::validate_modes`.
        // Adversarial inputs are rejected at the pipeline/PyO3 boundary.
        _ => unreachable!("dtype16: unsupported bytes_mode {bytes_mode}"),
    }
}

/// Split `src` into 2 buffers according to `bits_mode` and `bytes_mode`.
///
/// When `bits_mode == 1`, the source data is copied and float bits are reordered
/// before splitting (since `src` is immutable).
pub fn split(src: &[u8], bits_mode: i32, bytes_mode: i32) -> Vec<Vec<u8>> {
    if bytes_mode == 0 {
        if bits_mode == 1 {
            let mut copy = src.to_vec();
            reorder_all_floats_dtype16(&mut copy);
            return vec![copy, Vec::new()];
        }
        return vec![src.to_vec(), Vec::new()];
    }

    // If bits_mode == 1, we need to reorder float bits before splitting.
    // Since src is &[u8], we must make a copy.
    let data: std::borrow::Cow<'_, [u8]> = if bits_mode == 1 {
        let mut copy = src.to_vec();
        reorder_all_floats_dtype16(&mut copy);
        std::borrow::Cow::Owned(copy)
    } else {
        std::borrow::Cow::Borrowed(src)
    };

    let sizes = buffer_sizes(data.len(), bytes_mode);

    match bytes_mode {
        10 => {
            let mut buf0 = Vec::with_capacity(sizes[0]);
            let mut buf1 = Vec::with_capacity(sizes[1]);
            for (i, &b) in data.iter().enumerate() {
                if i % 2 == 0 {
                    buf0.push(b);
                } else {
                    buf1.push(b);
                }
            }
            vec![buf0, buf1]
        }
        8 => {
            // Keep MSB bytes only (odd-indexed in little-endian).
            let mut buf0 = Vec::with_capacity(sizes[0]);
            for i in (1..data.len()).step_by(2) {
                buf0.push(data[i]);
            }
            vec![buf0, Vec::new()]
        }
        1 => {
            // Keep LSB bytes only (even-indexed).
            let mut buf0 = Vec::with_capacity(sizes[0]);
            for i in (0..data.len()).step_by(2) {
                buf0.push(data[i]);
            }
            vec![buf0, Vec::new()]
        }
        // Invariant: caller validated bytes_mode via `crate::validate_modes`.
        // Adversarial inputs are rejected at the pipeline/PyO3 boundary.
        _ => unreachable!("dtype16: unsupported bytes_mode {bytes_mode}"),
    }
}

/// Combine buffers back into `out`, reversing the split operation.
pub fn combine_bufs(buf0: &[u8], buf1: &[u8], out: &mut [u8], bits_mode: i32, bytes_mode: i32) {
    match bytes_mode {
        0 => {
            assert!(
                buf1.is_empty(),
                "dtype16 mode 0 expects second buffer to be empty"
            );
            assert_eq!(
                buf0.len(),
                out.len(),
                "buffer length must match output length in mode 0"
            );
            out.copy_from_slice(buf0);
        }
        10 => {
            // Interleave: buf0 → even indices, buf1 → odd indices.
            let mut i0 = 0usize;
            let mut i1 = 0usize;
            for i in 0..out.len() {
                if i % 2 == 0 {
                    out[i] = buf0[i0];
                    i0 += 1;
                } else {
                    out[i] = buf1[i1];
                    i1 += 1;
                }
            }
        }
        8 => {
            // MSB only: odd bytes from buf0, even bytes are zero.
            out.fill(0);
            for (j, &b) in buf0.iter().enumerate() {
                let idx = j * 2 + 1;
                if idx < out.len() {
                    out[idx] = b;
                }
            }
        }
        1 => {
            // LSB only: even bytes from buf0, odd bytes are zero.
            out.fill(0);
            for (j, &b) in buf0.iter().enumerate() {
                let idx = j * 2;
                if idx < out.len() {
                    out[idx] = b;
                }
            }
        }
        // Invariant: caller validated bytes_mode via `crate::validate_modes`.
        // Adversarial inputs are rejected at the pipeline/PyO3 boundary.
        _ => unreachable!("dtype16: unsupported bytes_mode {bytes_mode}"),
    }

    // If bits_mode == 1, revert the float bit reordering.
    if bits_mode == 1 {
        revert_all_floats_dtype16(out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_mode10_even() {
        let src: Vec<u8> = (0..=255).map(|i| i as u8).collect();
        assert_eq!(src.len(), 256);
        let bufs = split(&src, 0, 10);
        assert_eq!(bufs.len(), 2);
        let sizes = buffer_sizes(src.len(), 10);
        assert_eq!(bufs[0].len(), sizes[0]);
        assert_eq!(bufs[1].len(), sizes[1]);
        let mut out = vec![0u8; src.len()];
        combine_bufs(&bufs[0], &bufs[1], &mut out, 0, 10);
        assert_eq!(src, out);
    }

    #[test]
    fn roundtrip_mode0_no_split() {
        let src: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        let bufs = split(&src, 0, 0);
        assert_eq!(bufs.len(), 2);
        assert_eq!(bufs[0], src);
        assert!(bufs[1].is_empty());
        let mut out = vec![0u8; src.len()];
        combine_bufs(&bufs[0], &bufs[1], &mut out, 0, 0);
        assert_eq!(src, out);
    }

    #[test]
    fn roundtrip_mode0_with_bit_reorder() {
        let src: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        let bufs = split(&src, 1, 0);
        assert_eq!(bufs.len(), 2);
        assert!(bufs[1].is_empty());
        let mut out = vec![0u8; src.len()];
        combine_bufs(&bufs[0], &bufs[1], &mut out, 1, 0);
        assert_eq!(src, out);
    }

    #[test]
    fn roundtrip_mode10_odd() {
        let src: Vec<u8> = (0..255).map(|i| i as u8).collect();
        assert_eq!(src.len(), 255);
        let bufs = split(&src, 0, 10);
        assert_eq!(bufs.len(), 2);
        let sizes = buffer_sizes(src.len(), 10);
        assert_eq!(bufs[0].len(), sizes[0]); // 128
        assert_eq!(bufs[1].len(), sizes[1]); // 127
        let mut out = vec![0u8; src.len()];
        combine_bufs(&bufs[0], &bufs[1], &mut out, 0, 10);
        assert_eq!(src, out);
    }

    #[test]
    fn roundtrip_mode10_with_bit_reorder() {
        let src: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        assert_eq!(src.len(), 1024);
        let bufs = split(&src, 1, 10);
        assert_eq!(bufs.len(), 2);
        let mut out = vec![0u8; src.len()];
        combine_bufs(&bufs[0], &bufs[1], &mut out, 1, 10);
        assert_eq!(src, out);
    }

    #[test]
    fn roundtrip_mode8_msb_only() {
        let src: Vec<u8> = (0..=255).map(|i| i as u8).collect();
        let bufs = split(&src, 0, 8);
        assert_eq!(bufs.len(), 2);
        assert_eq!(bufs[0].len(), 128); // MSB bytes only
        assert!(bufs[1].is_empty());

        // Reconstruct: even bytes should be 0, odd bytes should match.
        let mut out = vec![0u8; src.len()];
        combine_bufs(&bufs[0], &bufs[1], &mut out, 0, 8);
        for i in 0..out.len() {
            if i % 2 == 0 {
                assert_eq!(out[i], 0, "even byte at index {i} should be 0");
            } else {
                assert_eq!(out[i], src[i], "odd byte at index {i} should match src");
            }
        }
    }

    #[test]
    fn roundtrip_mode1_lsb_only() {
        let src: Vec<u8> = (0..=255).map(|i| i as u8).collect();
        let bufs = split(&src, 0, 1);
        assert_eq!(bufs.len(), 2);
        assert_eq!(bufs[0].len(), 128); // LSB bytes only
        assert!(bufs[1].is_empty());

        // Reconstruct: even bytes should match, odd bytes should be 0.
        let mut out = vec![0u8; src.len()];
        combine_bufs(&bufs[0], &bufs[1], &mut out, 0, 1);
        for i in 0..out.len() {
            if i % 2 == 0 {
                assert_eq!(out[i], src[i], "even byte at index {i} should match src");
            } else {
                assert_eq!(out[i], 0, "odd byte at index {i} should be 0");
            }
        }
    }

    #[test]
    fn bit_reorder_revert_roundtrip() {
        let original: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        let mut data = original.clone();
        reorder_all_floats_dtype16(&mut data);
        // After reordering, data should differ (in general).
        assert_ne!(data, original, "reordered data should differ from original");
        revert_all_floats_dtype16(&mut data);
        assert_eq!(data, original, "revert should restore original data");
    }
}
