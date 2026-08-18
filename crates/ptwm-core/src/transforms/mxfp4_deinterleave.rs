//! `MxFp4Deinterleave` op — split OCP MXFP4 block-packed data into a nibble
//! VALUE plane and a byte SCALE plane.
//!
//! Wire format (OCP MXFP4, block_size=32):
//!   Each block is 17 bytes: 16 packed-nibble bytes (32 nibble values) + 1 E8M0 scale byte.
//!   The 16 packed-nibble bytes encode 32 values as nibbles, low nibble first per byte.
//!
//! Implements the OCP MXFP4 `split` / `combine` logic via the PPG `Op` trait.

use std::sync::Arc;

use crate::error::PtwmCoreError;
use crate::transforms::op::{Op, OpId, Plane};
use crate::types::descriptor::{ElementWidth, Layout, PlaneDescriptor};
use crate::types::role::{Role, ScaleFormat, ValueFormat};

/// Bytes per OCP MXFP4 block: 16 nibble-packed value bytes + 1 scale byte.
const BLOCK_BYTES: u64 = 17;
/// Values per block (32 nibbles stored in 16 bytes).
const BLOCK_VALUES: u64 = 32;

/// Deinterleave OCP MXFP4 block-packed data into a VALUE (nibble) plane
/// and a SCALE (E8M0 byte) plane.
///
/// Only `block_size = 32` is supported (OCP MXFP4 specification).
pub struct MxFp4Deinterleave {
    /// Block size in values; must be 32.
    pub block_size: u8,
}

impl MxFp4Deinterleave {
    /// Construct a `MxFp4Deinterleave` op, validating that `block_size == 32`.
    pub fn new(block_size: u8) -> Result<Self, PtwmCoreError> {
        if block_size != 32 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "MxFp4Deinterleave: only block_size=32 (OCP MXFP4) is supported, got {block_size}"
            )));
        }
        Ok(MxFp4Deinterleave { block_size })
    }
}

impl Op for MxFp4Deinterleave {
    fn propagate_descriptors(
        &self,
        inputs: &[PlaneDescriptor],
    ) -> Result<Vec<PlaneDescriptor>, PtwmCoreError> {
        if inputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "MxFp4Deinterleave.propagate_descriptors: expected 1 input, got {}",
                inputs.len()
            )));
        }
        let inp = &inputs[0];
        if inp.element_width != ElementWidth::Byte {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "MxFp4Deinterleave.propagate_descriptors: expected element_width Byte, got {:?}",
                inp.element_width
            )));
        }
        if !inp.length_bytes.is_multiple_of(BLOCK_BYTES) {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "MxFp4Deinterleave.propagate_descriptors: length_bytes {} is not a multiple \
                 of block size {} (17 bytes/block for OCP MXFP4)",
                inp.length_bytes, BLOCK_BYTES
            )));
        }
        let block_count = inp.length_bytes / BLOCK_BYTES;
        let value_len = block_count * BLOCK_VALUES; // 32 nibble-byte slots per block
        let scale_len = block_count; // 1 scale byte per block

        // Determine row layout for each plane based on input layout.
        let (value_layout, scale_layout) = match inp.layout {
            Layout::Rows { row_len } => {
                // row_len is in blocks; value plane row = block_count_per_row * 32 nibbles.
                let value_row = (row_len as u64) * BLOCK_VALUES;
                // Clamp to u32 — should always fit for realistic tensors.
                let value_row = value_row.min(u32::MAX as u64) as u32;
                (
                    Layout::Rows { row_len: value_row },
                    Layout::Rows { row_len },
                )
            }
            Layout::Flat => (Layout::Flat, Layout::Flat),
        };

        Ok(vec![
            // Plane 0: VALUE — nibble-typed, one slot per MXFP4 element.
            PlaneDescriptor {
                role: Role::Value {
                    format: ValueFormat::Fp4E2m1,
                },
                element_width: ElementWidth::Nibble,
                length_bytes: value_len,
                layout: value_layout,
                derives_from_tensor: inp.derives_from_tensor,
                residual_of: None,
                // One nibble per byte: `forward` pushes each input byte's
                // low and high halves as separate slots, so nothing shares
                // a byte here. The flag means "two values share a byte",
                // which describes this op's *input*; `ElementWidth::Nibble`
                // already says these are 4-bit values. Claiming both left
                // every consumer to choose between reading the plane as
                // packed, which drops half the data, and reading it as
                // expanded, which contradicted the flag.
                is_nibble_packed: false,
                vendor_bytes: vec![],
            },
            // Plane 1: SCALE — one E8M0 byte per block.
            PlaneDescriptor {
                role: Role::Scale {
                    format: ScaleFormat::E8M0,
                },
                element_width: ElementWidth::Byte,
                length_bytes: scale_len,
                layout: scale_layout,
                derives_from_tensor: inp.derives_from_tensor,
                residual_of: None,
                is_nibble_packed: false,
                vendor_bytes: vec![],
            },
        ])
    }

    fn forward(&self, inputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        if inputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "MxFp4Deinterleave.forward: expected 1 input, got {}",
                inputs.len()
            )));
        }
        let inp = &inputs[0];
        let descriptors = self.propagate_descriptors(std::slice::from_ref(&inp.descriptor))?;

        let raw = &inp.bytes;
        let n_blocks = raw.len() / BLOCK_BYTES as usize;
        let mut values = Vec::with_capacity(n_blocks * BLOCK_VALUES as usize);
        let mut scales = Vec::with_capacity(n_blocks);

        for b in 0..n_blocks {
            let block = &raw[b * BLOCK_BYTES as usize..(b + 1) * BLOCK_BYTES as usize];
            // 16 packed-nibble bytes → 32 nibble slots (low nibble first).
            for i in 0..16 {
                let byte = block[i];
                values.push(byte & 0x0F); // low nibble first
                values.push((byte >> 4) & 0x0F); // high nibble second
            }
            scales.push(block[16]);
        }

        Ok(vec![
            Plane {
                bytes: Arc::from(values.into_boxed_slice()),
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
                "MxFp4Deinterleave.inverse: expected 2 outputs, got {}",
                outputs.len()
            )));
        }
        let values = &outputs[0].bytes;
        let scales = &outputs[1].bytes;

        let n_blocks = scales.len();
        if values.len() != n_blocks * BLOCK_VALUES as usize {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "MxFp4Deinterleave.inverse: VALUE plane length {} != expected {}",
                values.len(),
                n_blocks * BLOCK_VALUES as usize,
            )));
        }

        let orig_len = n_blocks * BLOCK_BYTES as usize;
        let mut out = Vec::with_capacity(orig_len);

        for b in 0..n_blocks {
            for i in 0..16 {
                let lo = values[b * BLOCK_VALUES as usize + 2 * i] & 0x0F;
                let hi = values[b * BLOCK_VALUES as usize + 2 * i + 1] & 0x0F;
                out.push(lo | (hi << 4));
            }
            out.push(scales[b]);
        }

        let in_descriptor = PlaneDescriptor {
            role: Role::Raw,
            element_width: ElementWidth::Byte,
            length_bytes: orig_len as u64,
            layout: Layout::Flat,
            derives_from_tensor: outputs[0].descriptor.derives_from_tensor,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        };
        Ok(vec![Plane {
            bytes: Arc::from(out.into_boxed_slice()),
            descriptor: in_descriptor,
        }])
    }

    fn id(&self) -> OpId {
        OpId::MxFp4Deinterleave
    }

    fn write_params(&self, out: &mut Vec<u8>) {
        out.push(self.block_size);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::descriptor::{ElementWidth, Layout};
    use crate::types::role::{Role, ScaleFormat, ValueFormat};

    fn raw_plane(data: Vec<u8>) -> Plane {
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

    fn make_raw_3_blocks() -> Vec<u8> {
        // 3 blocks of 17 bytes each = 51 bytes.
        let mut raw = Vec::new();
        for b in 0u8..3 {
            for i in 0u8..16 {
                raw.push(((b + 1) << 4) | (i & 0x0F));
            }
            raw.push(0xA0 | b);
        }
        raw
    }

    #[test]
    fn mxfp4_deinterleave_roundtrip() {
        let raw = make_raw_3_blocks();
        assert_eq!(raw.len(), 51);
        let op = MxFp4Deinterleave::new(32).unwrap();
        let inp = raw_plane(raw.clone());
        let fwd = op.forward(&[inp]).unwrap();
        assert_eq!(fwd.len(), 2);
        assert_eq!(fwd[0].bytes.len(), 3 * 32); // 96 nibble slots
        assert_eq!(fwd[1].bytes.len(), 3); // 3 scale bytes
        let inv = op.inverse(&fwd).unwrap();
        assert_eq!(inv.len(), 1);
        assert_eq!(inv[0].bytes.as_ref(), raw);
    }

    #[test]
    fn propagate_descriptors_rejects_misaligned_length() {
        let op = MxFp4Deinterleave::new(32).unwrap();
        let desc = PlaneDescriptor {
            role: Role::Raw,
            element_width: ElementWidth::Byte,
            length_bytes: 16, // 16 % 17 != 0
            layout: Layout::Flat,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        };
        assert!(op.propagate_descriptors(&[desc]).is_err());
    }

    #[test]
    fn propagate_descriptors_rejects_invalid_block_size() {
        assert!(MxFp4Deinterleave::new(64).is_err());
        assert!(MxFp4Deinterleave::new(16).is_err());
    }

    #[test]
    fn output_value_plane_role_correct() {
        let op = MxFp4Deinterleave::new(32).unwrap();
        let desc = PlaneDescriptor {
            role: Role::Raw,
            element_width: ElementWidth::Byte,
            length_bytes: 17, // 1 block
            layout: Layout::Flat,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        };
        let outs = op.propagate_descriptors(&[desc]).unwrap();
        assert_eq!(
            outs[0].role,
            Role::Value {
                format: ValueFormat::Fp4E2m1
            }
        );
        assert_eq!(outs[0].element_width, ElementWidth::Nibble);
        // One nibble per byte: nothing shares a byte in this op's output.
        assert!(!outs[0].is_nibble_packed);
        assert_eq!(outs[0].length_bytes, 32); // BLOCK_VALUES
    }

    #[test]
    fn output_scale_plane_role_correct() {
        let op = MxFp4Deinterleave::new(32).unwrap();
        let desc = PlaneDescriptor {
            role: Role::Raw,
            element_width: ElementWidth::Byte,
            length_bytes: 17, // 1 block
            layout: Layout::Flat,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        };
        let outs = op.propagate_descriptors(&[desc]).unwrap();
        assert_eq!(
            outs[1].role,
            Role::Scale {
                format: ScaleFormat::E8M0
            }
        );
        assert_eq!(outs[1].element_width, ElementWidth::Byte);
        assert!(!outs[1].is_nibble_packed);
        assert_eq!(outs[1].length_bytes, 1); // 1 block → 1 scale byte
    }

    #[test]
    fn write_params_roundtrip() {
        let op = MxFp4Deinterleave::new(32).unwrap();
        let mut buf = Vec::new();
        op.write_params(&mut buf);
        assert_eq!(buf, vec![32u8]);
    }
}
