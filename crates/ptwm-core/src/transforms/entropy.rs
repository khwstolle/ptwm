//! `EntropyEstimate` op — computes per-byte Shannon entropy and stores it in
//! the output plane's `vendor_bytes` as a Q8.24 fixed-point value.

use crate::error::PtwmCoreError;
use crate::transforms::op::{Op, OpId, Plane};
use crate::types::descriptor::PlaneDescriptor;

/// Compute per-byte Shannon entropy of `data` in bits/byte.
/// Returns the value as a Q8.24 fixed-point `u32` (integer part in the high 8
/// bits, 24-bit fractional part in the low 24 bits; range [0, 8]).
fn entropy_q8_24(data: &[u8]) -> u32 {
    if data.is_empty() {
        return 0;
    }

    let mut hist = [0u64; 256];
    for &b in data {
        hist[b as usize] += 1;
    }

    let n = data.len() as f64;
    let mut entropy = 0.0f64;
    for &count in &hist {
        if count > 0 {
            let p = count as f64 / n;
            entropy -= p * p.log2();
        }
    }

    // Clamp to [0, 8] bits/byte (floating-point noise can push slightly above 8)
    let entropy = entropy.clamp(0.0, 8.0);

    // Convert to Q8.24: multiply by 2^24, round to nearest integer
    (entropy * (1u64 << 24) as f64).round() as u32
}

/// `EntropyEstimate` is a non-modifying inspection op. It reads the input
/// plane's bytes, computes the per-byte Shannon entropy as a Q8.24 `u32`
/// (4 bytes, little-endian), and stores it in the output plane's
/// `vendor_bytes`. The plane bytes themselves are forwarded unchanged.
///
/// On `inverse`, the `vendor_bytes` entropy annotation is stripped (it is
/// metadata, not part of the encoded data).
pub struct EntropyEstimate;

impl Op for EntropyEstimate {
    fn propagate_descriptors(
        &self,
        inputs: &[PlaneDescriptor],
    ) -> Result<Vec<PlaneDescriptor>, PtwmCoreError> {
        if inputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "EntropyEstimate.propagate_descriptors: expected 1 input, got {}",
                inputs.len()
            )));
        }
        // Propagate input descriptor unchanged except vendor_bytes is a
        // 4-byte placeholder (the real value is filled at forward time).
        let mut out = inputs[0].clone();
        out.vendor_bytes = vec![0u8; 4];
        Ok(vec![out])
    }

    fn forward(&self, inputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        if inputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "EntropyEstimate.forward: expected 1 input, got {}",
                inputs.len()
            )));
        }
        let q = entropy_q8_24(&inputs[0].bytes);
        let mut out = inputs[0].clone();
        out.descriptor.vendor_bytes = q.to_le_bytes().to_vec();
        Ok(vec![out])
    }

    fn inverse(&self, outputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        if outputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "EntropyEstimate.inverse: expected 1 output, got {}",
                outputs.len()
            )));
        }
        // Strip the entropy annotation; return the plane bytes unchanged.
        let mut out = outputs[0].clone();
        out.descriptor.vendor_bytes = vec![];
        Ok(vec![out])
    }

    fn id(&self) -> OpId {
        OpId::EntropyEstimate
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

    fn make_plane(bytes: Vec<u8>) -> Plane {
        Plane {
            descriptor: PlaneDescriptor {
                role: Role::Raw,
                element_width: ElementWidth::Byte,
                length_bytes: bytes.len() as u64,
                layout: Layout::Flat,
                derives_from_tensor: None,
                residual_of: None,
                is_nibble_packed: false,
                vendor_bytes: vec![],
            },
            bytes: bytes.into(),
        }
    }

    #[test]
    fn forward_writes_entropy_to_vendor_bytes() {
        let op = EntropyEstimate;

        // All-same bytes → entropy = 0
        let plane_zero = make_plane(vec![0x42u8; 256]);
        let fwd_zero = op.forward(&[plane_zero]).unwrap();
        assert_eq!(fwd_zero[0].descriptor.vendor_bytes.len(), 4);
        let q_zero =
            u32::from_le_bytes(fwd_zero[0].descriptor.vendor_bytes[..4].try_into().unwrap());
        assert_eq!(q_zero, 0, "all-same bytes should have entropy 0");

        // Uniform distribution over all 256 values → entropy ≈ 8 bits/byte
        let uniform: Vec<u8> = (0u8..=255).collect();
        let plane_uniform = make_plane(uniform);
        let fwd_uniform = op.forward(&[plane_uniform]).unwrap();
        let q_uniform = u32::from_le_bytes(
            fwd_uniform[0].descriptor.vendor_bytes[..4]
                .try_into()
                .unwrap(),
        );
        // Q8.24 for 8 bits/byte = 8 * 2^24 = 134217728
        let expected_q8 = 8u32 * (1u32 << 24);
        // Allow 1 ULP of tolerance for floating-point rounding
        assert!(
            q_uniform.abs_diff(expected_q8) <= 1,
            "uniform entropy Q8.24 {q_uniform} should be ~{expected_q8}"
        );
    }

    #[test]
    fn inverse_strips_entropy() {
        let op = EntropyEstimate;
        let plane = make_plane(vec![1, 2, 3, 4, 5]);
        let fwd = op.forward(&[plane]).unwrap();

        // After forward, vendor_bytes has 4 bytes
        assert_eq!(fwd[0].descriptor.vendor_bytes.len(), 4);

        // After inverse, vendor_bytes is empty
        let inv = op.inverse(&fwd).unwrap();
        assert_eq!(inv.len(), 1);
        assert!(inv[0].descriptor.vendor_bytes.is_empty());

        // Bytes are unchanged
        assert_eq!(inv[0].bytes.as_ref(), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn propagate_descriptors_rejects_zero_inputs() {
        let op = EntropyEstimate;
        assert!(op.propagate_descriptors(&[]).is_err());
    }
}
