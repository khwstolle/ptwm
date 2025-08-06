//! `BlockMicroscalingRepack` op — re-express a 16-bit float plane as a
//! per-block shared scale (E8M0) plus a scale-relative element plane.
//!
//! The OCP microscaling formats (MXFP4 / NVFP4) store one shared block
//! scale alongside the elements, and PTWM already compresses that scale
//! stream well (the E8M0 sequence is smooth and autocorrelated). BF16
//! weights that *aren't* in a microscaled container leave that headroom on
//! the table. This op recovers it without any precision loss.
//!
//! ## Lossless construction
//!
//! Each element is a little-endian `u16` viewed in the BF16 layout
//! `[ sign : 1 | exponent : 8 | mantissa : 7 ]`. For a block of
//! `block_size` consecutive elements:
//!
//! - the **scale** is the maximum 8-bit exponent field in the block
//!   (an E8M0 value);
//! - each element's exponent field is replaced by the non-negative
//!   residual `scale - exponent`, with sign and mantissa untouched.
//!
//! Because the scale is the block maximum, every residual is in `0..=255`
//! and reconstruction (`exponent = scale - residual`) is exact. The op is
//! therefore a pure, reversible re-encoding — it is lossless for *any*
//! `u16` data, not only well-formed BF16 — and the compression benefit
//! comes from the residual exponent field concentrating near zero (so the
//! downstream `bit_reorder` + `byte_split` exponent plane is far more
//! compressible) while the small per-block scale plane carries the rest.
//!
//! ## Outputs
//!
//! 1. **element plane** — same shape / width (`Word2`) as the input, with
//!    exponent fields rebased to residuals.
//! 2. **scale plane** — one `E8M0` byte per block (`Role::Scale`).

use std::sync::Arc;

use crate::error::PtwmCoreError;
use crate::transforms::op::{Op, OpId, Plane};
use crate::types::descriptor::{ElementWidth, Layout, PlaneDescriptor};
use crate::types::role::{Role, ScaleFormat};

/// Bits of an element occupied by sign (bit 15) + mantissa (bits 0..=6);
/// the exponent lane is the complementary `0x7F80`.
const SIGN_MANTISSA_MASK: u16 = 0x807F;
const EXPONENT_SHIFT: u16 = 7;

/// Re-express a 16-bit-float plane as (scale-relative element plane, E8M0
/// per-block scale plane).
pub struct BlockMicroscalingRepack {
    /// Elements per block. Must be ≥ 1.
    pub block_size: u8,
}

impl BlockMicroscalingRepack {
    /// Construct the op, validating `block_size >= 1`.
    pub fn new(block_size: u8) -> Result<Self, PtwmCoreError> {
        if block_size == 0 {
            return Err(PtwmCoreError::InvalidContainer(
                "BlockMicroscalingRepack: block_size must be >= 1".into(),
            ));
        }
        Ok(Self { block_size })
    }

    #[inline]
    fn block_size(&self) -> usize {
        self.block_size as usize
    }
}

fn require_word2(d: &PlaneDescriptor, ctx: &str) -> Result<(), PtwmCoreError> {
    if d.element_width != ElementWidth::Word2 {
        return Err(PtwmCoreError::InvalidContainer(format!(
            "BlockMicroscalingRepack.{ctx}: requires Word2 (16-bit) elements, got {:?}",
            d.element_width
        )));
    }
    Ok(())
}

impl Op for BlockMicroscalingRepack {
    fn propagate_descriptors(
        &self,
        inputs: &[PlaneDescriptor],
    ) -> Result<Vec<PlaneDescriptor>, PtwmCoreError> {
        if inputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "BlockMicroscalingRepack.propagate_descriptors: expected 1 input, got {}",
                inputs.len()
            )));
        }
        let inp = &inputs[0];
        require_word2(inp, "propagate_descriptors")?;
        let n_elem = (inp.length_bytes / 2) as usize;
        let n_blocks = n_elem.div_ceil(self.block_size());

        // Element plane keeps the input's shape exactly — only exponent
        // fields change. The role is intentionally inherited from the input
        // (Raw in the production chain) so the element plane flows straight
        // into the downstream BitReorder/ByteSplit sub-chain unchanged;
        // unlike `mxfp4_deinterleave`, this op does not retag it. Scale
        // plane is a flat E8M0 byte sequence.
        let mut element = inp.clone();
        element.residual_of = None;
        let scale = PlaneDescriptor {
            role: Role::Scale {
                format: ScaleFormat::E8M0,
            },
            element_width: ElementWidth::Byte,
            length_bytes: n_blocks as u64,
            layout: Layout::Flat,
            derives_from_tensor: inp.derives_from_tensor,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        };
        Ok(vec![element, scale])
    }

    fn forward(&self, inputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        if inputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "BlockMicroscalingRepack.forward: expected 1 input, got {}",
                inputs.len()
            )));
        }
        let inp = &inputs[0];
        require_word2(&inp.descriptor, "forward")?;
        if !inp.bytes.len().is_multiple_of(2) {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "BlockMicroscalingRepack.forward: byte length {} is not even",
                inp.bytes.len()
            )));
        }
        let descriptors = self.propagate_descriptors(std::slice::from_ref(&inp.descriptor))?;

        let bs = self.block_size();
        let n_elem = inp.bytes.len() / 2;
        let n_blocks = n_elem.div_ceil(bs);
        let mut element = vec![0u8; inp.bytes.len()];
        let mut scales = vec![0u8; n_blocks];

        for b in 0..n_blocks {
            let start = b * bs;
            let end = ((b + 1) * bs).min(n_elem);
            // Pass 1: block-max exponent lane.
            let mut max_exp = 0u16;
            for e in start..end {
                let v = u16::from_le_bytes([inp.bytes[2 * e], inp.bytes[2 * e + 1]]);
                let exp = (v >> EXPONENT_SHIFT) & 0xFF;
                max_exp = max_exp.max(exp);
            }
            scales[b] = max_exp as u8;
            // Pass 2: rebase each exponent to a non-negative residual.
            for e in start..end {
                let v = u16::from_le_bytes([inp.bytes[2 * e], inp.bytes[2 * e + 1]]);
                let exp = (v >> EXPONENT_SHIFT) & 0xFF;
                let residual = max_exp - exp; // ≤ max_exp ≤ 255, fits 8 bits
                let out = (v & SIGN_MANTISSA_MASK) | (residual << EXPONENT_SHIFT);
                element[2 * e..2 * e + 2].copy_from_slice(&out.to_le_bytes());
            }
        }

        Ok(vec![
            Plane {
                bytes: Arc::from(element.into_boxed_slice()),
                descriptor: descriptors[0].clone(),
            },
            Plane {
                bytes: Arc::from(scales.into_boxed_slice()),
                descriptor: descriptors[1].clone(),
            },
        ])
    }

    fn inverse(&self, outputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        if outputs.len() != 2 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "BlockMicroscalingRepack.inverse: expected 2 outputs, got {}",
                outputs.len()
            )));
        }
        let element = &outputs[0];
        let scales = &outputs[1].bytes;
        require_word2(&element.descriptor, "inverse")?;
        if !element.bytes.len().is_multiple_of(2) {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "BlockMicroscalingRepack.inverse: element byte length {} is not even",
                element.bytes.len()
            )));
        }

        let bs = self.block_size();
        let n_elem = element.bytes.len() / 2;
        let n_blocks = n_elem.div_ceil(bs);
        if scales.len() != n_blocks {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "BlockMicroscalingRepack.inverse: scale plane has {} blocks, expected {}",
                scales.len(),
                n_blocks
            )));
        }

        let mut out = vec![0u8; element.bytes.len()];
        for b in 0..n_blocks {
            let start = b * bs;
            let end = ((b + 1) * bs).min(n_elem);
            let max_exp = scales[b] as u16;
            for e in start..end {
                let v = u16::from_le_bytes([element.bytes[2 * e], element.bytes[2 * e + 1]]);
                let residual = (v >> EXPONENT_SHIFT) & 0xFF;
                // Wrapping keeps decode panic-free on malformed input; for
                // well-formed planes `residual <= max_exp`, so this is exact.
                let exp = max_exp.wrapping_sub(residual) & 0xFF;
                let orig = (v & SIGN_MANTISSA_MASK) | (exp << EXPONENT_SHIFT);
                out[2 * e..2 * e + 2].copy_from_slice(&orig.to_le_bytes());
            }
        }

        let mut descriptor = element.descriptor.clone();
        descriptor.role = Role::Raw;
        descriptor.length_bytes = out.len() as u64;
        Ok(vec![Plane {
            bytes: Arc::from(out.into_boxed_slice()),
            descriptor,
        }])
    }

    fn id(&self) -> OpId {
        OpId::BlockMicroscalingRepack
    }

    fn write_params(&self, out: &mut Vec<u8>) {
        out.push(self.block_size);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::descriptor::Layout;
    use crate::types::role::Role;

    fn word2_plane(values: &[u16]) -> Plane {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Plane {
            descriptor: PlaneDescriptor {
                role: Role::Raw,
                element_width: ElementWidth::Word2,
                length_bytes: bytes.len() as u64,
                layout: Layout::Flat,
                derives_from_tensor: None,
                residual_of: None,
                is_nibble_packed: false,
                vendor_bytes: vec![],
            },
            bytes: Arc::from(bytes.into_boxed_slice()),
        }
    }

    fn round_trip(values: &[u16], block_size: u8) {
        let op = BlockMicroscalingRepack::new(block_size).unwrap();
        let fwd = op.forward(&[word2_plane(values)]).unwrap();
        assert_eq!(fwd.len(), 2);
        let inv = op.inverse(&fwd).unwrap();
        let got: Vec<u16> = inv[0]
            .bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(got, values, "round-trip mismatch (block_size={block_size})");
    }

    #[test]
    fn roundtrip_bf16_like() {
        // A spread of BF16 bit patterns: varied sign/exp/mantissa.
        let values: Vec<u16> = (0..256u16).map(|i| i.wrapping_mul(617)).collect();
        round_trip(&values, 32);
    }

    #[test]
    fn roundtrip_block_sizes() {
        let values: Vec<u16> = (0..200u16).map(|i| 0x3F00u16.wrapping_add(i)).collect();
        for &bs in &[1u8, 2, 7, 16, 32, 64, 255] {
            round_trip(&values, bs);
        }
    }

    #[test]
    fn roundtrip_partial_last_block() {
        // 70 elements with block_size 32 → blocks of 32, 32, 6.
        let values: Vec<u16> = (0..70u16).map(|i| i.wrapping_mul(0x0181)).collect();
        round_trip(&values, 32);
    }

    #[test]
    fn roundtrip_extremes() {
        // All-zero, all-0xFFFF (NaN-ish exp=0xFF), and mixed.
        round_trip(&[0u16; 64], 32);
        round_trip(&[0xFFFFu16; 64], 32);
        let mut v = vec![0u16; 32];
        v.extend([0x7F80u16, 0x8000, 0xFFFF, 0x0001].iter().cycle().take(32));
        round_trip(&v, 16);
    }

    #[test]
    fn roundtrip_empty() {
        round_trip(&[], 32);
    }

    #[test]
    fn scale_is_block_max_exponent() {
        // Block of two elements with exponents 0x40 and 0x42 → scale 0x42,
        // residuals 0x02 and 0x00.
        let op = BlockMicroscalingRepack::new(2).unwrap();
        let v0 = (0x40u16) << EXPONENT_SHIFT; // exp lane 0x40
        let v1 = (0x42u16) << EXPONENT_SHIFT; // exp lane 0x42
        let fwd = op.forward(&[word2_plane(&[v0, v1])]).unwrap();
        assert_eq!(fwd[1].bytes.as_ref(), vec![0x42]);
        let e0 = u16::from_le_bytes([fwd[0].bytes[0], fwd[0].bytes[1]]);
        let e1 = u16::from_le_bytes([fwd[0].bytes[2], fwd[0].bytes[3]]);
        assert_eq!((e0 >> EXPONENT_SHIFT) & 0xFF, 0x02); // 0x42 - 0x40
        assert_eq!((e1 >> EXPONENT_SHIFT) & 0xFF, 0x00); // 0x42 - 0x42
    }

    #[test]
    fn descriptors_have_expected_roles() {
        let op = BlockMicroscalingRepack::new(32).unwrap();
        let inp = word2_plane(&[0u16; 64]).descriptor;
        let outs = op.propagate_descriptors(&[inp]).unwrap();
        assert_eq!(outs[0].element_width, ElementWidth::Word2);
        assert_eq!(
            outs[1].role,
            Role::Scale {
                format: ScaleFormat::E8M0
            }
        );
        assert_eq!(outs[1].element_width, ElementWidth::Byte);
        assert_eq!(outs[1].length_bytes, 2); // 64 elems / 32 = 2 blocks
    }

    #[test]
    fn rejects_non_word2() {
        let op = BlockMicroscalingRepack::new(32).unwrap();
        let mut p = word2_plane(&[0u16; 4]);
        p.descriptor.element_width = ElementWidth::Byte;
        assert!(op.forward(&[p]).is_err());
    }

    #[test]
    fn rejects_zero_block_size() {
        assert!(BlockMicroscalingRepack::new(0).is_err());
    }

    #[test]
    fn write_params_roundtrip() {
        let op = BlockMicroscalingRepack::new(32).unwrap();
        let mut buf = Vec::new();
        op.write_params(&mut buf);
        assert_eq!(buf, vec![32u8]);
    }
}
