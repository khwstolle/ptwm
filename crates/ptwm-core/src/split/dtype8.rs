//! FP8 split and combine — supports passthrough, per-byte bit reorder, and nibble split.
//!
//! # bytes_mode values
//!
//! | bytes_mode | num_buf | Behaviour |
//! |-----------|---------|-----------|
//! | 10 | 1 | Passthrough (no transform). Used for raw bytes, integers, and blobs |
//! |    |   | compressed before this feature was introduced. |
//! | 11 | 1 | E4M3FN bit reorder: `[S EEEE MMM]` → `[EEEE SMMM]`. Single Huffman stream. |
//! | 12 | 1 | E5M2 bit reorder:  `[S EEEEE MM]` → `[EEEEE SMM]`. Single Huffman stream. |
//! | 20 | 2 | E4M3FN bit reorder + nibble split. buf0 = high nibbles (exponent), |
//! |    |   | buf1 = low nibbles (sign+mantissa). Two independent Huffman streams. |
//! | 22 | 2 | E5M2 bit reorder + nibble split. Same structure as mode 20. |
//!
//! # Why bit reorder?
//!
//! FP8-E4M3FN: `[S EEEE MMM]` — the sign bit straddles the exponent/mantissa region,
//! making the byte look near-random to an entropy coder. After reordering to
//! `[EEEE SMMM]`, the exponent occupies the high nibble: for typical trained weights,
//! this nibble has very low entropy (a few dominant exponent values), giving Huffman
//! significant leverage. Modes 20/22 go further by separating the high and low nibble
//! into independent streams.

// ---------------------------------------------------------------------------
// Bit-reorder helpers
// ---------------------------------------------------------------------------

/// Reorder bits of one FP8-E4M3FN byte: `[S EEEE MMM]` → `[EEEE SMMM]`.
#[inline]
pub(crate) fn reorder_byte_e4m3fn(val: u8) -> u8 {
    // exponent bits 6-3 → bits 7-4  : (val & 0x78) << 1
    // sign bit 7       → bit 3      : (val & 0x80) >> 4
    // mantissa bits 2-0 stay        : val & 0x07
    ((val & 0x78) << 1) | ((val & 0x80) >> 4) | (val & 0x07)
}

/// Inverse of [`reorder_byte_e4m3fn`]: `[EEEE SMMM]` → `[S EEEE MMM]`.
#[inline]
pub(crate) fn revert_byte_e4m3fn(rval: u8) -> u8 {
    // exponent bits 7-4 → bits 6-3  : (rval & 0xF0) >> 1
    // sign bit 3        → bit 7     : (rval & 0x08) << 4
    // mantissa bits 2-0 stay        : rval & 0x07
    ((rval & 0xF0) >> 1) | ((rval & 0x08) << 4) | (rval & 0x07)
}

/// Reorder bits of one FP8-E5M2 byte: `[S EEEEE MM]` → `[EEEEE SMM]`.
#[inline]
pub(crate) fn reorder_byte_e5m2(val: u8) -> u8 {
    // exponent bits 6-2 → bits 7-3  : (val & 0x7C) << 1
    // sign bit 7        → bit 2     : (val & 0x80) >> 5
    // mantissa bits 1-0 stay        : val & 0x03
    ((val & 0x7C) << 1) | ((val & 0x80) >> 5) | (val & 0x03)
}

/// Inverse of [`reorder_byte_e5m2`]: `[EEEEE SMM]` → `[S EEEEE MM]`.
#[inline]
pub(crate) fn revert_byte_e5m2(rval: u8) -> u8 {
    // exponent bits 7-3 → bits 6-2  : (rval & 0xF8) >> 1
    // sign bit 2        → bit 7     : (rval & 0x04) << 5
    // mantissa bits 1-0 stay        : rval & 0x03
    ((rval & 0xF8) >> 1) | ((rval & 0x04) << 5) | (rval & 0x03)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Sizes of each buffer after splitting.
///
/// Modes 10/11/12 → one buffer of `len` bytes.
/// Modes 20/22   → two buffers of `len` bytes each (nibble planes).
pub fn buffer_sizes(len: usize, bytes_mode: i32) -> Vec<usize> {
    match bytes_mode {
        10..=12 => vec![len],
        20 | 22 => vec![len, len],
        // Invariant: caller validated bytes_mode via `crate::validate_modes`.
        // Adversarial inputs are rejected at the pipeline/PyO3 boundary.
        _ => unreachable!("dtype8: unsupported bytes_mode {bytes_mode}"),
    }
}

/// Split `src` into 1 or 2 buffers according to `bytes_mode`.
pub fn split(src: &[u8], bytes_mode: i32) -> Vec<Vec<u8>> {
    match bytes_mode {
        10 => vec![src.to_vec()],

        11 => {
            let buf: Vec<u8> = src.iter().copied().map(reorder_byte_e4m3fn).collect();
            vec![buf]
        }

        12 => {
            let buf: Vec<u8> = src.iter().copied().map(reorder_byte_e5m2).collect();
            vec![buf]
        }

        20 => {
            // Apply E4M3FN bit reorder, then split into high/low nibble planes.
            let mut high = Vec::with_capacity(src.len());
            let mut low = Vec::with_capacity(src.len());
            for &b in src {
                let r = reorder_byte_e4m3fn(b);
                high.push(r >> 4);
                low.push(r & 0x0F);
            }
            vec![high, low]
        }

        22 => {
            // Apply E5M2 bit reorder, then split into high/low nibble planes.
            let mut high = Vec::with_capacity(src.len());
            let mut low = Vec::with_capacity(src.len());
            for &b in src {
                let r = reorder_byte_e5m2(b);
                high.push(r >> 4);
                low.push(r & 0x0F);
            }
            vec![high, low]
        }

        // Invariant: caller validated bytes_mode via `crate::validate_modes`.
        // Adversarial inputs are rejected at the pipeline/PyO3 boundary.
        _ => unreachable!("dtype8: unsupported bytes_mode {bytes_mode}"),
    }
}

/// Combine 1 or 2 buffers back into `out`, reversing the split operation.
pub fn combine(buffers: &[Vec<u8>], out: &mut [u8], bytes_mode: i32) {
    match bytes_mode {
        10 => {
            assert_eq!(buffers.len(), 1, "bytes_mode 10 expects 1 buffer");
            assert_eq!(buffers[0].len(), out.len());
            out.copy_from_slice(&buffers[0]);
        }

        11 => {
            assert_eq!(buffers.len(), 1, "bytes_mode 11 expects 1 buffer");
            assert_eq!(buffers[0].len(), out.len());
            for (o, &b) in out.iter_mut().zip(buffers[0].iter()) {
                *o = revert_byte_e4m3fn(b);
            }
        }

        12 => {
            assert_eq!(buffers.len(), 1, "bytes_mode 12 expects 1 buffer");
            assert_eq!(buffers[0].len(), out.len());
            for (o, &b) in out.iter_mut().zip(buffers[0].iter()) {
                *o = revert_byte_e5m2(b);
            }
        }

        20 => {
            assert_eq!(buffers.len(), 2, "bytes_mode 20 expects 2 buffers");
            assert_eq!(buffers[0].len(), out.len());
            assert_eq!(buffers[1].len(), out.len());
            for (i, o) in out.iter_mut().enumerate() {
                let reordered = (buffers[0][i] << 4) | (buffers[1][i] & 0x0F);
                *o = revert_byte_e4m3fn(reordered);
            }
        }

        22 => {
            assert_eq!(buffers.len(), 2, "bytes_mode 22 expects 2 buffers");
            assert_eq!(buffers[0].len(), out.len());
            assert_eq!(buffers[1].len(), out.len());
            for (i, o) in out.iter_mut().enumerate() {
                let reordered = (buffers[0][i] << 4) | (buffers[1][i] & 0x0F);
                *o = revert_byte_e5m2(reordered);
            }
        }

        // Invariant: caller validated bytes_mode via `crate::validate_modes`.
        // Adversarial inputs are rejected at the pipeline/PyO3 boundary.
        _ => unreachable!("dtype8: unsupported bytes_mode {bytes_mode}"),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Passthrough (mode 10) ---

    #[test]
    fn roundtrip_mode10_even() {
        let src: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let buffers = split(&src, 10);
        assert_eq!(buffers.len(), 1);
        assert_eq!(buffers[0].len(), src.len());
        let mut out = vec![0u8; src.len()];
        combine(&buffers, &mut out, 10);
        assert_eq!(src, out);
    }

    #[test]
    fn roundtrip_mode10_odd() {
        let src: Vec<u8> = (0..255).map(|i| i as u8).collect();
        let buffers = split(&src, 10);
        assert_eq!(buffers[0].len(), 255);
        let mut out = vec![0u8; 255];
        combine(&buffers, &mut out, 10);
        assert_eq!(src, out);
    }

    // --- Known-value bit reorder correctness ---

    #[test]
    fn bit_reorder_known_e4m3fn() {
        // FP8-E4M3FN representation of +1.0: sign=0, exp=0111, mantissa=000
        // Binary: 0_0111_000 = 0x38
        // After reorder [EEEE SMMM]: exponent=0111→bits7-4, sign=0→bit3, mantissa=000
        // = 0111_0_000 = 0x70
        assert_eq!(reorder_byte_e4m3fn(0x38), 0x70);
        assert_eq!(revert_byte_e4m3fn(0x70), 0x38);

        // FP8-E4M3FN max normal: sign=0, exp=1111, mantissa=110
        // Binary: 0_1111_110 = 0x7E
        // After reorder: exponent=1111→bits7-4, sign=0→bit3, mantissa=110
        // = 1111_0_110 = 0xF6
        assert_eq!(reorder_byte_e4m3fn(0x7E), 0xF6);
        assert_eq!(revert_byte_e4m3fn(0xF6), 0x7E);

        // Negative +1.0: 1_0111_000 = 0xB8
        // After reorder: exponent=0111, sign=1→bit3, mantissa=000
        // = 0111_1_000 = 0x78
        assert_eq!(reorder_byte_e4m3fn(0xB8), 0x78);
        assert_eq!(revert_byte_e4m3fn(0x78), 0xB8);
    }

    #[test]
    fn bit_reorder_known_e5m2() {
        // FP8-E5M2 representation of +1.0: sign=0, exp=01111, mantissa=00
        // Binary: 0_01111_00 = 0x3C
        // After reorder [EEEEE SMM]: exponent=01111→bits7-3, sign=0→bit2, mantissa=00
        // = 01111_0_00 = 0x78
        assert_eq!(reorder_byte_e5m2(0x3C), 0x78);
        assert_eq!(revert_byte_e5m2(0x78), 0x3C);

        // Negative value: sign=1, exp=01111, mantissa=00
        // Binary: 1_01111_00 = 0xBC
        // After reorder: exponent=01111→bits7-3, sign=1→bit2, mantissa=00
        // = 01111_1_00 = 0x7C
        assert_eq!(reorder_byte_e5m2(0xBC), 0x7C);
        assert_eq!(revert_byte_e5m2(0x7C), 0xBC);
    }

    #[test]
    fn bit_reorder_all_bytes_e4m3fn_invertible() {
        for val in 0u8..=255 {
            let reordered = reorder_byte_e4m3fn(val);
            assert_eq!(
                revert_byte_e4m3fn(reordered),
                val,
                "failed at val=0x{val:02X}"
            );
        }
    }

    #[test]
    fn bit_reorder_all_bytes_e5m2_invertible() {
        for val in 0u8..=255 {
            let reordered = reorder_byte_e5m2(val);
            assert_eq!(
                revert_byte_e5m2(reordered),
                val,
                "failed at val=0x{val:02X}"
            );
        }
    }

    // --- Single-plane bit reorder (modes 11, 12) ---

    #[test]
    fn roundtrip_e4m3fn_mode11_even() {
        let src: Vec<u8> = (0u8..=255).collect();
        let buffers = split(&src, 11);
        assert_eq!(buffers.len(), 1);
        assert_eq!(buffer_sizes(src.len(), 11), vec![src.len()]);
        let mut out = vec![0u8; src.len()];
        combine(&buffers, &mut out, 11);
        assert_eq!(src, out);
    }

    #[test]
    fn roundtrip_e4m3fn_mode11_odd() {
        let src: Vec<u8> = (0u8..255).collect();
        let buffers = split(&src, 11);
        assert_eq!(buffers.len(), 1);
        assert_eq!(buffers[0].len(), 255);
        let mut out = vec![0u8; 255];
        combine(&buffers, &mut out, 11);
        assert_eq!(src, out);
    }

    #[test]
    fn roundtrip_e5m2_mode12_even() {
        let src: Vec<u8> = (0u8..=255).collect();
        let buffers = split(&src, 12);
        assert_eq!(buffers.len(), 1);
        let mut out = vec![0u8; src.len()];
        combine(&buffers, &mut out, 12);
        assert_eq!(src, out);
    }

    #[test]
    fn roundtrip_e5m2_mode12_odd() {
        let src: Vec<u8> = (0u8..255).collect();
        let buffers = split(&src, 12);
        assert_eq!(buffers[0].len(), 255);
        let mut out = vec![0u8; 255];
        combine(&buffers, &mut out, 12);
        assert_eq!(src, out);
    }

    // --- Nibble split (modes 20, 22) ---

    #[test]
    fn roundtrip_e4m3fn_nibble_mode20_even() {
        let src: Vec<u8> = (0u8..=255).collect();
        let buffers = split(&src, 20);
        assert_eq!(buffers.len(), 2);
        assert_eq!(buffer_sizes(src.len(), 20), vec![256, 256]);
        assert_eq!(buffers[0].len(), 256);
        assert_eq!(buffers[1].len(), 256);
        // All nibble values must be in range [0, 15]
        for &b in &buffers[0] {
            assert!(b <= 0x0F, "high nibble out of range: 0x{b:02X}");
        }
        for &b in &buffers[1] {
            assert!(b <= 0x0F, "low nibble out of range: 0x{b:02X}");
        }
        let mut out = vec![0u8; src.len()];
        combine(&buffers, &mut out, 20);
        assert_eq!(src, out);
    }

    #[test]
    fn roundtrip_e4m3fn_nibble_mode20_odd() {
        let src: Vec<u8> = (0u8..255).collect();
        let buffers = split(&src, 20);
        assert_eq!(buffers.len(), 2);
        assert_eq!(buffers[0].len(), 255);
        assert_eq!(buffers[1].len(), 255);
        let mut out = vec![0u8; 255];
        combine(&buffers, &mut out, 20);
        assert_eq!(src, out);
    }

    #[test]
    fn roundtrip_e5m2_nibble_mode22_even() {
        let src: Vec<u8> = (0u8..=255).collect();
        let buffers = split(&src, 22);
        assert_eq!(buffers.len(), 2);
        assert_eq!(buffer_sizes(src.len(), 22), vec![256, 256]);
        for &b in &buffers[0] {
            assert!(b <= 0x0F);
        }
        for &b in &buffers[1] {
            assert!(b <= 0x0F);
        }
        let mut out = vec![0u8; src.len()];
        combine(&buffers, &mut out, 22);
        assert_eq!(src, out);
    }

    #[test]
    fn roundtrip_e5m2_nibble_mode22_odd() {
        let src: Vec<u8> = (0u8..255).collect();
        let buffers = split(&src, 22);
        assert_eq!(buffers[0].len(), 255);
        assert_eq!(buffers[1].len(), 255);
        let mut out = vec![0u8; 255];
        combine(&buffers, &mut out, 22);
        assert_eq!(src, out);
    }

    #[test]
    fn nibble_planes_separate_exponent_e4m3fn() {
        // For E4M3FN, after bit reorder the high nibble is the exponent.
        // Verify that two values with the same exponent but different mantissa
        // map to the same high-nibble value after split.
        // +1.0 = 0x38 → reordered 0x70 → high nibble 0x07
        // +1.5 = 0x3C → reordered 0x78... wait let me compute:
        //   0x3C = 0011_1100, E4M3: sign=0, exp=0111, mantissa=100
        //   reorder: exp=0111→bits7-4, sign=0→bit3, mantissa=100→bits2-0
        //   = 0111_0_100 = 0x74 → high nibble = 0x07
        // Both +1.0 (0x38) and +1.5 (0x3C) should have the same high nibble.
        let bufs_1p0 = split(&[0x38], 20);
        let bufs_1p5 = split(&[0x3C], 20);
        assert_eq!(
            bufs_1p0[0][0], bufs_1p5[0][0],
            "+1.0 and +1.5 should share the same exponent nibble"
        );
    }
}
