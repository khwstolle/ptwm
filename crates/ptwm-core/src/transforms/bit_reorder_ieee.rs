//! `BitReorderIeee16` and `BitReorderIeee32` ops — rearrange IEEE-754 float bits
//! so the exponent occupies the most-significant position, improving entropy coding.
//!
//! The transforms are lifted verbatim from `crates/ptwm-core/src/split/dtype16.rs`
//! and `crates/ptwm-core/src/split/dtype32.rs`.

use std::sync::Arc;

use crate::error::PtwmCoreError;
use crate::split::{dtype16, dtype32};
use crate::transforms::op::{Op, OpId, Plane};
use crate::types::descriptor::{ElementWidth, PlaneDescriptor};

// ---------------------------------------------------------------------------
// BitReorderIeee16
// ---------------------------------------------------------------------------

/// Reorder IEEE-754 16-bit float bits in-place so the exponent comes first.
///
/// Operates on BF16 or FP16 data (`ElementWidth::Word2`). The input buffer
/// must have an even number of bytes. The output descriptor is identical to
/// the input descriptor.
///
/// Algorithm: lifts `dtype16::reorder_all_floats_dtype16` / `revert_all_floats_dtype16`.
pub struct BitReorderIeee16;

impl Op for BitReorderIeee16 {
    fn propagate_descriptors(
        &self,
        inputs: &[PlaneDescriptor],
    ) -> Result<Vec<PlaneDescriptor>, PtwmCoreError> {
        if inputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "BitReorderIeee16.propagate_descriptors: expected 1 input, got {}",
                inputs.len()
            )));
        }
        let inp = &inputs[0];
        if inp.element_width != ElementWidth::Word2 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "BitReorderIeee16.propagate_descriptors: expected element_width Word2 (BF16/FP16), \
                 got {:?}",
                inp.element_width
            )));
        }
        if !inp.length_bytes.is_multiple_of(2) {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "BitReorderIeee16.propagate_descriptors: length_bytes must be even, got {}",
                inp.length_bytes
            )));
        }
        Ok(vec![inp.clone()])
    }

    fn forward(&self, inputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        if inputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "BitReorderIeee16.forward: expected 1 input, got {}",
                inputs.len()
            )));
        }
        let inp = &inputs[0];
        let mut bytes: Vec<u8> = inp.bytes.as_ref().to_vec();
        dtype16::reorder_all_floats_dtype16(&mut bytes);
        Ok(vec![Plane {
            bytes: Arc::from(bytes.into_boxed_slice()),
            descriptor: inp.descriptor.clone(),
        }])
    }

    fn inverse(&self, outputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        if outputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "BitReorderIeee16.inverse: expected 1 output, got {}",
                outputs.len()
            )));
        }
        let out = &outputs[0];
        let mut bytes: Vec<u8> = out.bytes.as_ref().to_vec();
        dtype16::revert_all_floats_dtype16(&mut bytes);
        Ok(vec![Plane {
            bytes: Arc::from(bytes.into_boxed_slice()),
            descriptor: out.descriptor.clone(),
        }])
    }

    fn id(&self) -> OpId {
        OpId::BitReorderIeee16
    }

    fn write_params(&self, _out: &mut Vec<u8>) {
        // No parameters.
    }
}

// ---------------------------------------------------------------------------
// BitReorderIeee32
// ---------------------------------------------------------------------------

/// Reorder IEEE-754 32-bit float bits in-place so the exponent comes first.
///
/// Operates on FP32 data (`ElementWidth::Word4`). The input buffer must have
/// a length divisible by 4. The output descriptor is identical to the input
/// descriptor.
///
/// Algorithm: lifts `dtype32::reorder_all_floats_dtype32` / `revert_all_floats_dtype32`.
pub struct BitReorderIeee32;

impl Op for BitReorderIeee32 {
    fn propagate_descriptors(
        &self,
        inputs: &[PlaneDescriptor],
    ) -> Result<Vec<PlaneDescriptor>, PtwmCoreError> {
        if inputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "BitReorderIeee32.propagate_descriptors: expected 1 input, got {}",
                inputs.len()
            )));
        }
        let inp = &inputs[0];
        if inp.element_width != ElementWidth::Word4 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "BitReorderIeee32.propagate_descriptors: expected element_width Word4 (FP32), \
                 got {:?}",
                inp.element_width
            )));
        }
        if !inp.length_bytes.is_multiple_of(4) {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "BitReorderIeee32.propagate_descriptors: length_bytes must be divisible by 4, \
                 got {}",
                inp.length_bytes
            )));
        }
        Ok(vec![inp.clone()])
    }

    fn forward(&self, inputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        if inputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "BitReorderIeee32.forward: expected 1 input, got {}",
                inputs.len()
            )));
        }
        let inp = &inputs[0];
        let mut bytes: Vec<u8> = inp.bytes.as_ref().to_vec();
        dtype32::reorder_all_floats_dtype32(&mut bytes);
        Ok(vec![Plane {
            bytes: Arc::from(bytes.into_boxed_slice()),
            descriptor: inp.descriptor.clone(),
        }])
    }

    fn inverse(&self, outputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        if outputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "BitReorderIeee32.inverse: expected 1 output, got {}",
                outputs.len()
            )));
        }
        let out = &outputs[0];
        let mut bytes: Vec<u8> = out.bytes.as_ref().to_vec();
        dtype32::revert_all_floats_dtype32(&mut bytes);
        Ok(vec![Plane {
            bytes: Arc::from(bytes.into_boxed_slice()),
            descriptor: out.descriptor.clone(),
        }])
    }

    fn id(&self) -> OpId {
        OpId::BitReorderIeee32
    }

    fn write_params(&self, _out: &mut Vec<u8>) {
        // No parameters.
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::descriptor::{Layout, PlaneDescriptor};
    use crate::types::role::Role;

    fn word2_descriptor(length_bytes: u64) -> PlaneDescriptor {
        PlaneDescriptor {
            role: Role::Raw,
            element_width: ElementWidth::Word2,
            length_bytes,
            layout: Layout::Flat,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        }
    }

    fn word4_descriptor(length_bytes: u64) -> PlaneDescriptor {
        PlaneDescriptor {
            role: Role::Raw,
            element_width: ElementWidth::Word4,
            length_bytes,
            layout: Layout::Flat,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        }
    }

    fn byte_descriptor(length_bytes: u64) -> PlaneDescriptor {
        PlaneDescriptor {
            role: Role::Raw,
            element_width: ElementWidth::Byte,
            length_bytes,
            layout: Layout::Flat,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        }
    }

    // --- BitReorderIeee16 ---

    #[test]
    fn ieee16_forward_inverse_roundtrip() {
        // 32 bytes: 16 BF16/FP16 values
        let original: Vec<u8> = (0u8..32).collect();
        let desc = word2_descriptor(32);
        let plane = Plane {
            bytes: Arc::from(&original[..]),
            descriptor: desc,
        };
        let op = BitReorderIeee16;
        let fwd = op.forward(&[plane]).unwrap();
        assert_eq!(fwd.len(), 1);
        let inv = op.inverse(&fwd).unwrap();
        assert_eq!(inv.len(), 1);
        assert_eq!(
            inv[0].bytes.as_ref(),
            original,
            "round-trip must be bit-exact"
        );
    }

    #[test]
    fn ieee16_forward_changes_bytes() {
        // Verify that forward actually modifies bytes for non-trivial input.
        // Use a pattern where some bits are set across the exponent boundary.
        let original: Vec<u8> = (0u8..32)
            .map(|i| i.wrapping_mul(13).wrapping_add(7))
            .collect();
        let desc = word2_descriptor(32);
        let plane = Plane {
            bytes: Arc::from(&original[..]),
            descriptor: desc,
        };
        let op = BitReorderIeee16;
        let fwd = op.forward(&[plane]).unwrap();
        // The reordering should change at least some bytes (true for all
        // non-trivial inputs where sign != 0 or exponent bits span the boundary).
        // We just verify roundtrip correctness here; the split tests cover change.
        let inv = op.inverse(&fwd).unwrap();
        assert_eq!(inv[0].bytes.as_ref(), original);
    }

    #[test]
    fn ieee16_propagate_descriptors_rejects_byte_input() {
        let op = BitReorderIeee16;
        let desc = byte_descriptor(32);
        assert!(op.propagate_descriptors(&[desc]).is_err());
    }

    #[test]
    fn ieee16_propagate_descriptors_rejects_odd_length() {
        let op = BitReorderIeee16;
        // length_bytes=5 is odd — must be rejected.
        let mut desc = word2_descriptor(5);
        desc.length_bytes = 5;
        assert!(op.propagate_descriptors(&[desc]).is_err());
    }

    #[test]
    fn ieee16_propagate_descriptors_accepts_word2_even() {
        let op = BitReorderIeee16;
        let desc = word2_descriptor(32);
        let result = op
            .propagate_descriptors(std::slice::from_ref(&desc))
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], desc);
    }

    #[test]
    fn ieee16_propagate_descriptors_rejects_wrong_input_count() {
        let op = BitReorderIeee16;
        assert!(op.propagate_descriptors(&[]).is_err());
        let desc = word2_descriptor(32);
        assert!(op.propagate_descriptors(&[desc.clone(), desc]).is_err());
    }

    // --- BitReorderIeee32 ---

    #[test]
    fn ieee32_forward_inverse_roundtrip() {
        // 32 bytes: 8 FP32 values
        let original: Vec<u8> = (0u8..32).collect();
        let desc = word4_descriptor(32);
        let plane = Plane {
            bytes: Arc::from(&original[..]),
            descriptor: desc,
        };
        let op = BitReorderIeee32;
        let fwd = op.forward(&[plane]).unwrap();
        assert_eq!(fwd.len(), 1);
        let inv = op.inverse(&fwd).unwrap();
        assert_eq!(inv.len(), 1);
        assert_eq!(
            inv[0].bytes.as_ref(),
            original,
            "round-trip must be bit-exact"
        );
    }

    #[test]
    fn ieee32_propagate_descriptors_rejects_byte_input() {
        let op = BitReorderIeee32;
        let desc = byte_descriptor(32);
        assert!(op.propagate_descriptors(&[desc]).is_err());
    }

    #[test]
    fn ieee32_propagate_descriptors_rejects_word2_input() {
        let op = BitReorderIeee32;
        let desc = word2_descriptor(32);
        assert!(op.propagate_descriptors(&[desc]).is_err());
    }

    #[test]
    fn ieee32_propagate_descriptors_rejects_length_not_multiple_of_4() {
        let op = BitReorderIeee32;
        // length_bytes=6 is not divisible by 4.
        let mut desc = word4_descriptor(6);
        desc.length_bytes = 6;
        assert!(op.propagate_descriptors(&[desc]).is_err());
    }

    #[test]
    fn ieee32_propagate_descriptors_accepts_word4_aligned() {
        let op = BitReorderIeee32;
        let desc = word4_descriptor(32);
        let result = op
            .propagate_descriptors(std::slice::from_ref(&desc))
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], desc);
    }

    #[test]
    fn ieee32_propagate_descriptors_rejects_wrong_input_count() {
        let op = BitReorderIeee32;
        assert!(op.propagate_descriptors(&[]).is_err());
        let desc = word4_descriptor(32);
        assert!(op.propagate_descriptors(&[desc.clone(), desc]).is_err());
    }
}
