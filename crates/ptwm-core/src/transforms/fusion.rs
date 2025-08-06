//! Fused transform implementations for hot pairs
//!
//! Each fused op runs both transforms in a single loop, eliminating the
//! intermediate buffer. These are NOT `Op` implementors; they are internal
//! helpers used by the fusion picker in `chain::runtime`.
//!
//! # Fused pairs
//!
//! - [`FusedIeee16ByteSplit2`]  — `BitReorderIeee16` + `ByteSplit{n=2}`
//! - [`FusedIeee32ByteSplit4`]  — `BitReorderIeee32` + `ByteSplit{n=4}`
//! - [`FusedFp8E4M3NibbleSplit`] — `BitReorderFp8E4M3` + `NibbleSplit`
//! - [`FusedFp8E5M2NibbleSplit`] — `BitReorderFp8E5M2` + `NibbleSplit`

use std::sync::Arc;

use crate::error::PtwmCoreError;
use crate::split::dtype8;
use crate::transforms::op::Plane;
use crate::types::descriptor::{ElementWidth, Layout, PlaneDescriptor};
use crate::types::role::{NibbleKind, Role};

// ---------------------------------------------------------------------------
// FusedIeee16ByteSplit2
// ---------------------------------------------------------------------------

/// Fused `BitReorderIeee16` + `ByteSplit{n=2}`.
///
/// In `forward`: applies IEEE-754 16-bit float bit reordering in-place, then
/// splits even/odd bytes into two planes — all in a single pass.
///
/// In `inverse`: merges two planes and reverts the bit reordering.
pub struct FusedIeee16ByteSplit2;

impl FusedIeee16ByteSplit2 {
    pub fn forward(input: &Plane) -> Result<Vec<Plane>, PtwmCoreError> {
        let src = &input.bytes;
        if !src.len().is_multiple_of(2) {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "FusedIeee16ByteSplit2.forward: input length {} is not even",
                src.len()
            )));
        }
        let plane_len = src.len() / 2;

        let mut buf0 = Vec::with_capacity(plane_len);
        let mut buf1 = Vec::with_capacity(plane_len);

        // Process pairs of bytes as u32 words (two BF16/FP16 elements),
        // apply the bit reorder, then split into even/odd byte planes.
        let word_count = src.len() / 4;
        for i in 0..word_count {
            let offset = i * 4;
            let val = u32::from_le_bytes([
                src[offset],
                src[offset + 1],
                src[offset + 2],
                src[offset + 3],
            ]);
            // Apply the same bit reorder as BitReorderIeee16 (2×16-bit packed).
            let reordered = {
                let sign = (val >> 8) & 0x0080_0080;
                let exponent = (val << 1) & 0xFF00_FF00;
                let mantissa = val & 0x007F_007F;
                exponent | sign | mantissa
            };
            let bytes = reordered.to_le_bytes();
            // Even bytes → buf0; odd bytes → buf1
            buf0.push(bytes[0]);
            buf1.push(bytes[1]);
            buf0.push(bytes[2]);
            buf1.push(bytes[3]);
        }
        // Handle trailing 2-byte pair if src.len() % 4 == 2.
        if src.len() % 4 == 2 {
            let i = src.len() - 2;
            // Single BF16/FP16 element — reorder 2-byte slice.
            // Load as two separate bytes; the reorder for a single 16-bit element:
            // [sign(1) exponent(8) mantissa(7)] in LE → bytes [b0, b1]
            // reordered: [exponent(8) sign(1) mantissa(7)]
            // Treat as u16:
            let val16 = u16::from_le_bytes([src[i], src[i + 1]]);
            let sign = (val16 >> 8) & 0x0080;
            let exponent = (val16 << 1) & 0xFF00;
            let mantissa = val16 & 0x007F;
            let reordered16 = exponent | sign | mantissa;
            let bytes16 = reordered16.to_le_bytes();
            buf0.push(bytes16[0]);
            buf1.push(bytes16[1]);
        }

        let len_bytes = plane_len as u64;
        let base = &input.descriptor;

        let desc0 = PlaneDescriptor {
            role: Role::IntegerByte { index: 0, of: 2 },
            element_width: ElementWidth::Byte,
            length_bytes: len_bytes,
            layout: Layout::Flat,
            derives_from_tensor: base.derives_from_tensor,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        };
        let desc1 = PlaneDescriptor {
            role: Role::IntegerByte { index: 1, of: 2 },
            element_width: ElementWidth::Byte,
            length_bytes: len_bytes,
            layout: Layout::Flat,
            derives_from_tensor: base.derives_from_tensor,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        };

        Ok(vec![
            Plane {
                bytes: Arc::from(buf0.into_boxed_slice()),
                descriptor: desc0,
            },
            Plane {
                bytes: Arc::from(buf1.into_boxed_slice()),
                descriptor: desc1,
            },
        ])
    }

    pub fn inverse(planes: &[Plane]) -> Result<Plane, PtwmCoreError> {
        if planes.len() != 2 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "FusedIeee16ByteSplit2.inverse: expected 2 planes, got {}",
                planes.len()
            )));
        }
        let buf0 = &planes[0].bytes;
        let buf1 = &planes[1].bytes;
        if buf0.len() != buf1.len() {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "FusedIeee16ByteSplit2.inverse: plane lengths differ: {} vs {}",
                buf0.len(),
                buf1.len()
            )));
        }

        let total_len = buf0.len() + buf1.len();
        let mut out = vec![0u8; total_len];

        // Interleave and revert bit reordering in one pass.
        let word_count = buf0.len() / 2; // each word processes 2 bytes from each plane = 4 bytes total
        for i in 0..word_count {
            // Reconstruct the reordered u32.
            let reordered =
                u32::from_le_bytes([buf0[i * 2], buf1[i * 2], buf0[i * 2 + 1], buf1[i * 2 + 1]]);
            let reverted = {
                let sign = (reordered << 8) & 0x8000_8000;
                let exponent = (reordered >> 1) & 0x7F80_7F80;
                let mantissa = reordered & 0x007F_007F;
                sign | exponent | mantissa
            };
            let bytes = reverted.to_le_bytes();
            out[i * 4] = bytes[0];
            out[i * 4 + 1] = bytes[1];
            out[i * 4 + 2] = bytes[2];
            out[i * 4 + 3] = bytes[3];
        }
        // Handle trailing element if buf0.len() is odd.
        if buf0.len() % 2 == 1 {
            let i = buf0.len() - 1;
            let reordered16 = u16::from_le_bytes([buf0[i], buf1[i]]);
            let sign = (reordered16 << 8) & 0x8000;
            let exponent = (reordered16 >> 1) & 0x7F80;
            let mantissa = reordered16 & 0x007F;
            let reverted16 = sign | exponent | mantissa;
            let bytes16 = reverted16.to_le_bytes();
            out[i * 2] = bytes16[0];
            out[i * 2 + 1] = bytes16[1];
        }

        let len = out.len() as u64;
        let desc = PlaneDescriptor {
            role: Role::Raw,
            element_width: ElementWidth::Word2,
            length_bytes: len,
            layout: Layout::Flat,
            derives_from_tensor: planes[0].descriptor.derives_from_tensor,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        };
        Ok(Plane {
            bytes: Arc::from(out.into_boxed_slice()),
            descriptor: desc,
        })
    }
}

// ---------------------------------------------------------------------------
// FusedIeee32ByteSplit4
// ---------------------------------------------------------------------------

/// Fused `BitReorderIeee32` + `ByteSplit{n=4}`.
///
/// Applies IEEE-754 32-bit float bit reordering, then splits bytes into 4
/// planes by round-robin — all in one pass.
pub struct FusedIeee32ByteSplit4;

impl FusedIeee32ByteSplit4 {
    pub fn forward(input: &Plane) -> Result<Vec<Plane>, PtwmCoreError> {
        let src = &input.bytes;
        if !src.len().is_multiple_of(4) {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "FusedIeee32ByteSplit4.forward: input length {} is not divisible by 4",
                src.len()
            )));
        }
        let plane_len = src.len() / 4;
        let mut bufs: [Vec<u8>; 4] = [
            Vec::with_capacity(plane_len),
            Vec::with_capacity(plane_len),
            Vec::with_capacity(plane_len),
            Vec::with_capacity(plane_len),
        ];

        for chunk in src.chunks_exact(4) {
            let val = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let reordered = {
                let sign = (val >> 8) & 0x0080_0000;
                let exponent = (val << 1) & 0xFF00_0000;
                let mantissa = val & 0x007F_FFFF;
                exponent | sign | mantissa
            };
            let bytes = reordered.to_le_bytes();
            bufs[0].push(bytes[0]);
            bufs[1].push(bytes[1]);
            bufs[2].push(bytes[2]);
            bufs[3].push(bytes[3]);
        }

        let len_bytes = plane_len as u64;
        let base = &input.descriptor;

        let planes: Vec<Plane> = (0u8..4)
            .zip(bufs)
            .map(|(i, bytes)| Plane {
                bytes: Arc::from(bytes.into_boxed_slice()),
                descriptor: PlaneDescriptor {
                    role: Role::IntegerByte { index: i, of: 4 },
                    element_width: ElementWidth::Byte,
                    length_bytes: len_bytes,
                    layout: Layout::Flat,
                    derives_from_tensor: base.derives_from_tensor,
                    residual_of: None,
                    is_nibble_packed: false,
                    vendor_bytes: vec![],
                },
            })
            .collect();

        Ok(planes)
    }

    pub fn inverse(planes: &[Plane]) -> Result<Plane, PtwmCoreError> {
        if planes.len() != 4 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "FusedIeee32ByteSplit4.inverse: expected 4 planes, got {}",
                planes.len()
            )));
        }
        let plane_len = planes[0].bytes.len();
        for (i, p) in planes.iter().enumerate() {
            if p.bytes.len() != plane_len {
                return Err(PtwmCoreError::InvalidContainer(format!(
                    "FusedIeee32ByteSplit4.inverse: plane {i} length {} != plane 0 length {plane_len}",
                    p.bytes.len()
                )));
            }
        }

        let mut out = vec![0u8; plane_len * 4];
        for i in 0..plane_len {
            let reordered = u32::from_le_bytes([
                planes[0].bytes[i],
                planes[1].bytes[i],
                planes[2].bytes[i],
                planes[3].bytes[i],
            ]);
            let reverted = {
                let sign = (reordered << 8) & 0x8000_0000;
                let exponent = (reordered >> 1) & 0x7F80_0000;
                let mantissa = reordered & 0x007F_FFFF;
                sign | exponent | mantissa
            };
            let bytes = reverted.to_le_bytes();
            out[i * 4] = bytes[0];
            out[i * 4 + 1] = bytes[1];
            out[i * 4 + 2] = bytes[2];
            out[i * 4 + 3] = bytes[3];
        }

        let len = out.len() as u64;
        let desc = PlaneDescriptor {
            role: Role::Raw,
            element_width: ElementWidth::Word4,
            length_bytes: len,
            layout: Layout::Flat,
            derives_from_tensor: planes[0].descriptor.derives_from_tensor,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        };
        Ok(Plane {
            bytes: Arc::from(out.into_boxed_slice()),
            descriptor: desc,
        })
    }
}

// ---------------------------------------------------------------------------
// FusedFp8E4M3NibbleSplit
// ---------------------------------------------------------------------------

/// Fused `BitReorderFp8E4M3` + `NibbleSplit`.
///
/// For each FP8-E4M3FN byte: reorders bits so the exponent is in the high
/// nibble, then splits into high-nibble (exponent) and low-nibble (sign+mantissa)
/// planes — all in one pass.
pub struct FusedFp8E4M3NibbleSplit;

impl FusedFp8E4M3NibbleSplit {
    pub fn forward(input: &Plane) -> Result<Vec<Plane>, PtwmCoreError> {
        let src = &input.bytes;
        let mut high = Vec::with_capacity(src.len());
        let mut low = Vec::with_capacity(src.len());
        for &b in src.iter() {
            let r = dtype8::reorder_byte_e4m3fn(b);
            high.push(r >> 4);
            low.push(r & 0x0F);
        }

        let len_bytes = src.len() as u64;
        let base = &input.descriptor;

        Ok(vec![
            Plane {
                bytes: Arc::from(high.into_boxed_slice()),
                descriptor: PlaneDescriptor {
                    role: Role::Nibble {
                        kind: NibbleKind::Exponent,
                    },
                    element_width: ElementWidth::Nibble,
                    length_bytes: len_bytes,
                    layout: Layout::Flat,
                    derives_from_tensor: base.derives_from_tensor,
                    residual_of: None,
                    is_nibble_packed: false,
                    vendor_bytes: vec![],
                },
            },
            Plane {
                bytes: Arc::from(low.into_boxed_slice()),
                descriptor: PlaneDescriptor {
                    role: Role::Nibble {
                        kind: NibbleKind::SignMantissa,
                    },
                    element_width: ElementWidth::Nibble,
                    length_bytes: len_bytes,
                    layout: Layout::Flat,
                    derives_from_tensor: base.derives_from_tensor,
                    residual_of: None,
                    is_nibble_packed: false,
                    vendor_bytes: vec![],
                },
            },
        ])
    }

    pub fn inverse(planes: &[Plane]) -> Result<Plane, PtwmCoreError> {
        if planes.len() != 2 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "FusedFp8E4M3NibbleSplit.inverse: expected 2 planes, got {}",
                planes.len()
            )));
        }
        let high = &planes[0].bytes;
        let low = &planes[1].bytes;
        if high.len() != low.len() {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "FusedFp8E4M3NibbleSplit.inverse: high plane length {} != low plane length {}",
                high.len(),
                low.len()
            )));
        }
        let bytes: Vec<u8> = high
            .iter()
            .zip(low.iter())
            .map(|(&h, &l)| {
                let reordered = ((h & 0x0F) << 4) | (l & 0x0F);
                dtype8::revert_byte_e4m3fn(reordered)
            })
            .collect();

        let len = bytes.len() as u64;
        let desc = PlaneDescriptor {
            role: Role::Raw,
            element_width: ElementWidth::Byte,
            length_bytes: len,
            layout: Layout::Flat,
            derives_from_tensor: planes[0].descriptor.derives_from_tensor,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        };
        Ok(Plane {
            bytes: Arc::from(bytes.into_boxed_slice()),
            descriptor: desc,
        })
    }
}

// ---------------------------------------------------------------------------
// FusedFp8E5M2NibbleSplit
// ---------------------------------------------------------------------------

/// Fused `BitReorderFp8E5M2` + `NibbleSplit`.
///
/// For each FP8-E5M2 byte: reorders bits so the exponent is in the high
/// nibble, then splits into high-nibble (exponent) and low-nibble (sign+mantissa)
/// planes — all in one pass.
pub struct FusedFp8E5M2NibbleSplit;

impl FusedFp8E5M2NibbleSplit {
    pub fn forward(input: &Plane) -> Result<Vec<Plane>, PtwmCoreError> {
        let src = &input.bytes;
        let mut high = Vec::with_capacity(src.len());
        let mut low = Vec::with_capacity(src.len());
        for &b in src.iter() {
            let r = dtype8::reorder_byte_e5m2(b);
            high.push(r >> 4);
            low.push(r & 0x0F);
        }

        let len_bytes = src.len() as u64;
        let base = &input.descriptor;

        Ok(vec![
            Plane {
                bytes: Arc::from(high.into_boxed_slice()),
                descriptor: PlaneDescriptor {
                    role: Role::Nibble {
                        kind: NibbleKind::Exponent,
                    },
                    element_width: ElementWidth::Nibble,
                    length_bytes: len_bytes,
                    layout: Layout::Flat,
                    derives_from_tensor: base.derives_from_tensor,
                    residual_of: None,
                    is_nibble_packed: false,
                    vendor_bytes: vec![],
                },
            },
            Plane {
                bytes: Arc::from(low.into_boxed_slice()),
                descriptor: PlaneDescriptor {
                    role: Role::Nibble {
                        kind: NibbleKind::SignMantissa,
                    },
                    element_width: ElementWidth::Nibble,
                    length_bytes: len_bytes,
                    layout: Layout::Flat,
                    derives_from_tensor: base.derives_from_tensor,
                    residual_of: None,
                    is_nibble_packed: false,
                    vendor_bytes: vec![],
                },
            },
        ])
    }

    pub fn inverse(planes: &[Plane]) -> Result<Plane, PtwmCoreError> {
        if planes.len() != 2 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "FusedFp8E5M2NibbleSplit.inverse: expected 2 planes, got {}",
                planes.len()
            )));
        }
        let high = &planes[0].bytes;
        let low = &planes[1].bytes;
        if high.len() != low.len() {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "FusedFp8E5M2NibbleSplit.inverse: high plane length {} != low plane length {}",
                high.len(),
                low.len()
            )));
        }
        let bytes: Vec<u8> = high
            .iter()
            .zip(low.iter())
            .map(|(&h, &l)| {
                let reordered = ((h & 0x0F) << 4) | (l & 0x0F);
                dtype8::revert_byte_e5m2(reordered)
            })
            .collect();

        let len = bytes.len() as u64;
        let desc = PlaneDescriptor {
            role: Role::Raw,
            element_width: ElementWidth::Byte,
            length_bytes: len,
            layout: Layout::Flat,
            derives_from_tensor: planes[0].descriptor.derives_from_tensor,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        };
        Ok(Plane {
            bytes: Arc::from(bytes.into_boxed_slice()),
            descriptor: desc,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transforms::op::{Op, Plane};
    use crate::transforms::{
        BitReorderFp8E4M3, BitReorderFp8E5M2, BitReorderIeee16, BitReorderIeee32, ByteSplit,
        NibbleSplit,
    };
    use crate::types::descriptor::{ElementWidth, Layout, PlaneDescriptor};
    use crate::types::role::Role;

    fn byte_plane(data: Vec<u8>) -> Plane {
        Plane {
            bytes: Arc::from(&data[..]),
            descriptor: PlaneDescriptor {
                role: Role::Raw,
                element_width: ElementWidth::Byte,
                length_bytes: data.len() as u64,
                layout: Layout::Flat,
                derives_from_tensor: None,
                residual_of: None,
                is_nibble_packed: false,
                vendor_bytes: vec![],
            },
        }
    }

    fn word2_plane(data: Vec<u8>) -> Plane {
        Plane {
            bytes: Arc::from(&data[..]),
            descriptor: PlaneDescriptor {
                role: Role::Raw,
                element_width: ElementWidth::Word2,
                length_bytes: data.len() as u64,
                layout: Layout::Flat,
                derives_from_tensor: None,
                residual_of: None,
                is_nibble_packed: false,
                vendor_bytes: vec![],
            },
        }
    }

    fn word4_plane(data: Vec<u8>) -> Plane {
        Plane {
            bytes: Arc::from(&data[..]),
            descriptor: PlaneDescriptor {
                role: Role::Raw,
                element_width: ElementWidth::Word4,
                length_bytes: data.len() as u64,
                layout: Layout::Flat,
                derives_from_tensor: None,
                residual_of: None,
                is_nibble_packed: false,
                vendor_bytes: vec![],
            },
        }
    }

    // ── FusedIeee16ByteSplit2 ────────────────────────────────────────────────

    #[test]
    fn fused_ieee16_matches_unfused() {
        // Use 16 bytes (8 BF16 elements).
        let data: Vec<u8> = (0u8..16).map(|i| i.wrapping_mul(37)).collect();
        let input = word2_plane(data.clone());

        // Fused path.
        let fused_out = FusedIeee16ByteSplit2::forward(&input).unwrap();

        // Unfused path: BitReorderIeee16 then ByteSplit{n=2}.
        // Note: BitReorderIeee16 returns a Word2 descriptor, but ByteSplit
        // requires Byte. In the unfused path we must reinterpret the descriptor
        // as Byte before passing to ByteSplit (the bytes are identical, just
        // the element_width label differs). The fused path avoids this by
        // operating on raw bytes directly.
        let reordered_planes = BitReorderIeee16
            .forward(std::slice::from_ref(&input))
            .unwrap();
        let reordered_as_byte = Plane {
            bytes: reordered_planes[0].bytes.clone(),
            descriptor: PlaneDescriptor {
                element_width: ElementWidth::Byte,
                ..reordered_planes[0].descriptor.clone()
            },
        };
        let split_out = ByteSplit::new(2)
            .unwrap()
            .forward(&[reordered_as_byte])
            .unwrap();

        assert_eq!(fused_out.len(), 2, "fused should produce 2 planes");
        assert_eq!(split_out.len(), 2, "unfused should produce 2 planes");
        assert_eq!(
            fused_out[0].bytes.as_ref(),
            split_out[0].bytes.as_ref(),
            "plane 0 mismatch"
        );
        assert_eq!(
            fused_out[1].bytes.as_ref(),
            split_out[1].bytes.as_ref(),
            "plane 1 mismatch"
        );
    }

    #[test]
    fn fused_ieee16_inverse_roundtrip() {
        let data: Vec<u8> = (0u8..16).map(|i| i.wrapping_mul(53)).collect();
        let input = word2_plane(data.clone());
        let planes = FusedIeee16ByteSplit2::forward(&input).unwrap();
        let recovered = FusedIeee16ByteSplit2::inverse(&planes).unwrap();
        assert_eq!(recovered.bytes.as_ref(), data);
    }

    // ── FusedIeee32ByteSplit4 ────────────────────────────────────────────────

    #[test]
    fn fused_ieee32_matches_unfused() {
        let data: Vec<u8> = (0u8..32).map(|i| i.wrapping_mul(37)).collect();
        let input = word4_plane(data.clone());

        let fused_out = FusedIeee32ByteSplit4::forward(&input).unwrap();

        // Unfused: BitReorderIeee32 then ByteSplit{n=4}.
        // ByteSplit requires Byte element_width; reinterpret the reordered plane.
        let reordered_planes = BitReorderIeee32
            .forward(std::slice::from_ref(&input))
            .unwrap();
        let reordered_as_byte = Plane {
            bytes: reordered_planes[0].bytes.clone(),
            descriptor: PlaneDescriptor {
                element_width: ElementWidth::Byte,
                ..reordered_planes[0].descriptor.clone()
            },
        };
        let split_out = ByteSplit::new(4)
            .unwrap()
            .forward(&[reordered_as_byte])
            .unwrap();

        assert_eq!(fused_out.len(), 4);
        for i in 0..4 {
            assert_eq!(
                fused_out[i].bytes.as_ref(),
                split_out[i].bytes.as_ref(),
                "plane {i} mismatch"
            );
        }
    }

    #[test]
    fn fused_ieee32_inverse_roundtrip() {
        let data: Vec<u8> = (0u8..32).map(|i| i.wrapping_mul(53)).collect();
        let input = word4_plane(data.clone());
        let planes = FusedIeee32ByteSplit4::forward(&input).unwrap();
        let recovered = FusedIeee32ByteSplit4::inverse(&planes).unwrap();
        assert_eq!(recovered.bytes.as_ref(), data);
    }

    // ── FusedFp8E4M3NibbleSplit ──────────────────────────────────────────────

    #[test]
    fn fused_fp8e4m3_matches_unfused() {
        let data: Vec<u8> = (0u8..32).map(|i| i.wrapping_mul(37)).collect();
        let input = byte_plane(data.clone());

        let fused_out = FusedFp8E4M3NibbleSplit::forward(&input).unwrap();

        let reordered = BitReorderFp8E4M3
            .forward(std::slice::from_ref(&input))
            .unwrap();
        let split_out = NibbleSplit.forward(&reordered).unwrap();

        assert_eq!(fused_out.len(), 2);
        assert_eq!(
            fused_out[0].bytes, split_out[0].bytes,
            "high nibble mismatch"
        );
        assert_eq!(
            fused_out[1].bytes, split_out[1].bytes,
            "low nibble mismatch"
        );
    }

    #[test]
    fn fused_fp8e4m3_inverse_roundtrip() {
        let data: Vec<u8> = (0u8..32).map(|i| i.wrapping_mul(53)).collect();
        let input = byte_plane(data.clone());
        let planes = FusedFp8E4M3NibbleSplit::forward(&input).unwrap();
        let recovered = FusedFp8E4M3NibbleSplit::inverse(&planes).unwrap();
        assert_eq!(recovered.bytes.as_ref(), data);
    }

    // ── FusedFp8E5M2NibbleSplit ──────────────────────────────────────────────

    #[test]
    fn fused_fp8e5m2_matches_unfused() {
        let data: Vec<u8> = (0u8..32).map(|i| i.wrapping_mul(37)).collect();
        let input = byte_plane(data.clone());

        let fused_out = FusedFp8E5M2NibbleSplit::forward(&input).unwrap();

        let reordered = BitReorderFp8E5M2
            .forward(std::slice::from_ref(&input))
            .unwrap();
        let split_out = NibbleSplit.forward(&reordered).unwrap();

        assert_eq!(fused_out.len(), 2);
        assert_eq!(
            fused_out[0].bytes, split_out[0].bytes,
            "high nibble mismatch"
        );
        assert_eq!(
            fused_out[1].bytes, split_out[1].bytes,
            "low nibble mismatch"
        );
    }

    #[test]
    fn fused_fp8e5m2_inverse_roundtrip() {
        let data: Vec<u8> = (0u8..32).map(|i| i.wrapping_mul(53)).collect();
        let input = byte_plane(data.clone());
        let planes = FusedFp8E5M2NibbleSplit::forward(&input).unwrap();
        let recovered = FusedFp8E5M2NibbleSplit::inverse(&planes).unwrap();
        assert_eq!(recovered.bytes.as_ref(), data);
    }
}
