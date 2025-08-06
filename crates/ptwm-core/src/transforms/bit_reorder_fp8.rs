//! `BitReorderFp8E4M3` and `BitReorderFp8E5M2` ops — per-byte bit shuffle for
//! FP8 formats that moves the exponent into the high nibble for better entropy coding.
//!
//! Both ops operate on `ElementWidth::Byte` planes (one byte per FP8 element).
//! The algorithms are lifted from `crates/ptwm-core/src/split/dtype8`.

use std::sync::Arc;

use crate::error::PtwmCoreError;
use crate::split::dtype8;
use crate::transforms::op::{Op, OpId, Plane};
use crate::types::descriptor::{ElementWidth, PlaneDescriptor};

// ---------------------------------------------------------------------------
// BitReorderFp8E4M3
// ---------------------------------------------------------------------------

/// Per-byte bit reorder for FP8-E4M3FN: `[S EEEE MMM]` → `[EEEE SMMM]`.
///
/// Moving the exponent to the high nibble makes the exponent plane highly
/// compressible by downstream entropy coders (few dominant exponent values in
/// typical trained weights).
///
/// Requires `ElementWidth::Byte`; output descriptor matches the input.
/// Algorithm: lifts `dtype8::reorder_byte_e4m3fn` / `dtype8::revert_byte_e4m3fn`.
pub struct BitReorderFp8E4M3;

impl Op for BitReorderFp8E4M3 {
    fn propagate_descriptors(
        &self,
        inputs: &[PlaneDescriptor],
    ) -> Result<Vec<PlaneDescriptor>, PtwmCoreError> {
        if inputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "BitReorderFp8E4M3.propagate_descriptors: expected 1 input, got {}",
                inputs.len()
            )));
        }
        let inp = &inputs[0];
        if inp.element_width != ElementWidth::Byte {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "BitReorderFp8E4M3.propagate_descriptors: expected element_width Byte (FP8), \
                 got {:?}",
                inp.element_width
            )));
        }
        Ok(vec![inp.clone()])
    }

    fn forward(&self, inputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        if inputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "BitReorderFp8E4M3.forward: expected 1 input, got {}",
                inputs.len()
            )));
        }
        let inp = &inputs[0];
        let bytes: Vec<u8> = inp
            .bytes
            .iter()
            .copied()
            .map(dtype8::reorder_byte_e4m3fn)
            .collect();
        Ok(vec![Plane {
            bytes: Arc::from(bytes.into_boxed_slice()),
            descriptor: inp.descriptor.clone(),
        }])
    }

    fn inverse(&self, outputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        if outputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "BitReorderFp8E4M3.inverse: expected 1 output, got {}",
                outputs.len()
            )));
        }
        let out = &outputs[0];
        let bytes: Vec<u8> = out
            .bytes
            .iter()
            .copied()
            .map(dtype8::revert_byte_e4m3fn)
            .collect();
        Ok(vec![Plane {
            bytes: Arc::from(bytes.into_boxed_slice()),
            descriptor: out.descriptor.clone(),
        }])
    }

    fn id(&self) -> OpId {
        OpId::BitReorderFp8E4M3
    }

    fn write_params(&self, _out: &mut Vec<u8>) {
        // No parameters.
    }
}

// ---------------------------------------------------------------------------
// BitReorderFp8E5M2
// ---------------------------------------------------------------------------

/// Per-byte bit reorder for FP8-E5M2: `[S EEEEE MM]` → `[EEEEE SMM]`.
///
/// Same motivation as [`BitReorderFp8E4M3`] but for the 5-exponent-bit variant.
///
/// Requires `ElementWidth::Byte`; output descriptor matches the input.
/// Algorithm: lifts `dtype8::reorder_byte_e5m2` / `dtype8::revert_byte_e5m2`.
pub struct BitReorderFp8E5M2;

impl Op for BitReorderFp8E5M2 {
    fn propagate_descriptors(
        &self,
        inputs: &[PlaneDescriptor],
    ) -> Result<Vec<PlaneDescriptor>, PtwmCoreError> {
        if inputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "BitReorderFp8E5M2.propagate_descriptors: expected 1 input, got {}",
                inputs.len()
            )));
        }
        let inp = &inputs[0];
        if inp.element_width != ElementWidth::Byte {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "BitReorderFp8E5M2.propagate_descriptors: expected element_width Byte (FP8), \
                 got {:?}",
                inp.element_width
            )));
        }
        Ok(vec![inp.clone()])
    }

    fn forward(&self, inputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        if inputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "BitReorderFp8E5M2.forward: expected 1 input, got {}",
                inputs.len()
            )));
        }
        let inp = &inputs[0];
        let bytes: Vec<u8> = inp
            .bytes
            .iter()
            .copied()
            .map(dtype8::reorder_byte_e5m2)
            .collect();
        Ok(vec![Plane {
            bytes: Arc::from(bytes.into_boxed_slice()),
            descriptor: inp.descriptor.clone(),
        }])
    }

    fn inverse(&self, outputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        if outputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "BitReorderFp8E5M2.inverse: expected 1 output, got {}",
                outputs.len()
            )));
        }
        let out = &outputs[0];
        let bytes: Vec<u8> = out
            .bytes
            .iter()
            .copied()
            .map(dtype8::revert_byte_e5m2)
            .collect();
        Ok(vec![Plane {
            bytes: Arc::from(bytes.into_boxed_slice()),
            descriptor: out.descriptor.clone(),
        }])
    }

    fn id(&self) -> OpId {
        OpId::BitReorderFp8E5M2
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

    // --- BitReorderFp8E4M3 ---

    #[test]
    fn e4m3_forward_inverse_roundtrip() {
        // All 256 possible byte values.
        let original: Vec<u8> = (0u8..=255).collect();
        let desc = byte_descriptor(256);
        let plane = Plane {
            bytes: Arc::from(&original[..]),
            descriptor: desc,
        };
        let op = BitReorderFp8E4M3;
        let fwd = op.forward(&[plane]).unwrap();
        assert_eq!(fwd.len(), 1);
        assert_eq!(fwd[0].bytes.len(), 256);
        let inv = op.inverse(&fwd).unwrap();
        assert_eq!(inv.len(), 1);
        assert_eq!(
            inv[0].bytes.as_ref(),
            original,
            "round-trip must be bit-exact"
        );
    }

    #[test]
    fn e4m3_propagate_descriptors_accepts_byte() {
        let op = BitReorderFp8E4M3;
        let desc = byte_descriptor(64);
        let result = op
            .propagate_descriptors(std::slice::from_ref(&desc))
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], desc);
    }

    #[test]
    fn e4m3_propagate_descriptors_rejects_word2_input() {
        let op = BitReorderFp8E4M3;
        let desc = word2_descriptor(64);
        assert!(op.propagate_descriptors(&[desc]).is_err());
    }

    #[test]
    fn e4m3_propagate_descriptors_rejects_wrong_input_count() {
        let op = BitReorderFp8E4M3;
        assert!(op.propagate_descriptors(&[]).is_err());
        let desc = byte_descriptor(64);
        assert!(op.propagate_descriptors(&[desc.clone(), desc]).is_err());
    }

    // --- BitReorderFp8E5M2 ---

    #[test]
    fn e5m2_forward_inverse_roundtrip() {
        // All 256 possible byte values.
        let original: Vec<u8> = (0u8..=255).collect();
        let desc = byte_descriptor(256);
        let plane = Plane {
            bytes: Arc::from(&original[..]),
            descriptor: desc,
        };
        let op = BitReorderFp8E5M2;
        let fwd = op.forward(&[plane]).unwrap();
        assert_eq!(fwd.len(), 1);
        assert_eq!(fwd[0].bytes.len(), 256);
        let inv = op.inverse(&fwd).unwrap();
        assert_eq!(inv.len(), 1);
        assert_eq!(
            inv[0].bytes.as_ref(),
            original,
            "round-trip must be bit-exact"
        );
    }

    #[test]
    fn e5m2_propagate_descriptors_accepts_byte() {
        let op = BitReorderFp8E5M2;
        let desc = byte_descriptor(64);
        let result = op
            .propagate_descriptors(std::slice::from_ref(&desc))
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], desc);
    }

    #[test]
    fn e5m2_propagate_descriptors_rejects_word2_input() {
        let op = BitReorderFp8E5M2;
        let desc = word2_descriptor(64);
        assert!(op.propagate_descriptors(&[desc]).is_err());
    }

    #[test]
    fn e5m2_propagate_descriptors_rejects_wrong_input_count() {
        let op = BitReorderFp8E5M2;
        assert!(op.propagate_descriptors(&[]).is_err());
        let desc = byte_descriptor(64);
        assert!(op.propagate_descriptors(&[desc.clone(), desc]).is_err());
    }
}
