//! `BytePassthrough` op — identity transform that passes one plane unchanged.

use crate::error::PtwmCoreError;
use crate::transforms::op::{Op, OpId, Plane};
use crate::types::descriptor::PlaneDescriptor;

/// `BytePassthrough` is the identity op. It requires exactly one input plane
/// and returns it unchanged on both `forward` and `inverse`. No parameters.
///
/// Chain authors use this op to explicitly signal "no transform here" while
/// still allowing a `role_override` on the outgoing edge to retag the plane's
/// role for downstream codec dispatch.
pub struct BytePassthrough;

impl Op for BytePassthrough {
    fn propagate_descriptors(
        &self,
        inputs: &[PlaneDescriptor],
    ) -> Result<Vec<PlaneDescriptor>, PtwmCoreError> {
        if inputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "BytePassthrough.propagate_descriptors: expected 1 input, got {}",
                inputs.len()
            )));
        }
        Ok(vec![inputs[0].clone()])
    }

    fn forward(&self, inputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        if inputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "BytePassthrough.forward: expected 1 input, got {}",
                inputs.len()
            )));
        }
        Ok(vec![inputs[0].clone()])
    }

    fn inverse(&self, outputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        if outputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "BytePassthrough.inverse: expected 1 output, got {}",
                outputs.len()
            )));
        }
        Ok(vec![outputs[0].clone()])
    }

    fn id(&self) -> OpId {
        OpId::BytePassthrough
    }

    fn write_params(&self, _out: &mut Vec<u8>) {
        // No parameters.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::descriptor::{ElementWidth, Layout};
    use crate::types::role::Role;

    fn sample_descriptor() -> PlaneDescriptor {
        PlaneDescriptor {
            role: Role::ExponentByte,
            element_width: ElementWidth::Byte,
            length_bytes: 128,
            layout: Layout::Rows { row_len: 16 },
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![0xAB],
        }
    }

    fn sample_plane() -> Plane {
        Plane {
            bytes: (0u8..=127u8).collect(),
            descriptor: sample_descriptor(),
        }
    }

    #[test]
    fn forward_inverse_roundtrip() {
        let op = BytePassthrough;
        let plane = sample_plane();
        let original_bytes = plane.bytes.clone();

        let fwd = op.forward(&[plane]).unwrap();
        assert_eq!(fwd.len(), 1);
        assert_eq!(fwd[0].bytes, original_bytes);

        let inv = op.inverse(&fwd).unwrap();
        assert_eq!(inv.len(), 1);
        assert_eq!(inv[0].bytes, original_bytes);
    }

    #[test]
    fn propagate_descriptors_passes_through() {
        let op = BytePassthrough;
        let desc = sample_descriptor();
        let result = op
            .propagate_descriptors(std::slice::from_ref(&desc))
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], desc);
    }

    #[test]
    fn propagate_descriptors_rejects_zero_inputs() {
        let op = BytePassthrough;
        assert!(op.propagate_descriptors(&[]).is_err());
    }
}
