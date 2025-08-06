//! `XorDelta` and `FloatDelta` ops — cross-tensor residual encoding.
//!
//! ## Cross-tensor op asymmetry
//!
//! Both ops consume two inputs (current plane + reference plane) on `forward`
//! and two inputs (residual + reference plane) on `inverse`. The `Op` trait
//! was designed for single-tensor chains; cross-tensor ops bend this by
//! accepting the reference as the second element in the slice on both sides:
//!
//! - `forward(&[current, reference])` → `[residual]`
//! - `inverse(&[residual, reference])` → `[original]`
//!
//! The runtime is responsible for supplying the reference as `inputs[1]` /
//! `outputs[1]` respectively. This is documented here so chain authors know
//! the convention.
//!
//! ## XorDelta
//!
//! Byte-level XOR residual. Because XOR is involutive, `forward` and `inverse`
//! are the same operation: `residual[i] = a[i] ^ b[i]`. Round-trip is
//! guaranteed byte-exact for all inputs.
//!
//! ## FloatDelta (DONE_WITH_CONCERNS)
//!
//! Element-wise float subtract (`output = current - reference`). Currently
//! only `f32` is supported. The round-trip `inverse(forward(a, b), b)` computes
//! `(a - b) + b`, which IEEE 754 does NOT guarantee to equal `a` bit-exactly
//! when `a - b` introduces a rounding error.
//!
//! **This op is marked DONE_WITH_CONCERNS**: it violates the PPG byte-exact
//! round-trip invariant for arbitrary f32 inputs where `|a - b|` causes
//! sub-normal rounding. It exists for opportunistic delta encoding where near-
//! identical tensors make the residual very small and coding gain outweighs the
//! approximation. A future version may use integer delta on the raw bit patterns
//! (which is lossless) instead.

use std::sync::Arc;

use crate::error::PtwmCoreError;
use crate::transforms::op::{Op, OpId, Plane};
use crate::types::descriptor::{ElementWidth, Layout, PlaneDescriptor};
use crate::types::role::{ResidualFormat, Role};

// ---------------------------------------------------------------------------
// XorDelta
// ---------------------------------------------------------------------------

/// Byte-level XOR residual against a reference tensor plane.
///
/// `dep_idx` identifies which dependency tensor supplies the reference.
///
/// **Input convention** (both forward and inverse):
/// - `inputs[0]` / `outputs[0]`: the plane being encoded or the residual.
/// - `inputs[1]` / `outputs[1]`: the reference plane (same shape).
///
/// Both are consumed; one residual plane is produced.
pub struct XorDelta {
    /// Index into the dependency-tensor list for the reference tensor.
    pub dep_idx: u8,
}

impl XorDelta {
    pub fn new(dep_idx: u8) -> Self {
        XorDelta { dep_idx }
    }
}

impl Op for XorDelta {
    fn propagate_descriptors(
        &self,
        inputs: &[PlaneDescriptor],
    ) -> Result<Vec<PlaneDescriptor>, PtwmCoreError> {
        if inputs.len() != 2 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "XorDelta.propagate_descriptors: expected 2 inputs (current, reference), got {}",
                inputs.len()
            )));
        }
        let cur = &inputs[0];
        let ref_ = &inputs[1];
        if cur.length_bytes != ref_.length_bytes {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "XorDelta.propagate_descriptors: length_bytes mismatch: current={}, ref={}",
                cur.length_bytes, ref_.length_bytes
            )));
        }
        if cur.element_width != ref_.element_width {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "XorDelta.propagate_descriptors: element_width mismatch: current={:?}, ref={:?}",
                cur.element_width, ref_.element_width
            )));
        }
        Ok(vec![PlaneDescriptor {
            role: Role::Residual {
                format: ResidualFormat::Xor,
            },
            element_width: cur.element_width,
            length_bytes: cur.length_bytes,
            layout: Layout::Flat,
            derives_from_tensor: Some(crate::types::descriptor::TensorRef {
                dep_idx: self.dep_idx,
            }),
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        }])
    }

    fn forward(&self, inputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        if inputs.len() != 2 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "XorDelta.forward: expected 2 inputs (current, reference), got {}",
                inputs.len()
            )));
        }
        let cur = &inputs[0];
        let ref_ = &inputs[1];
        if cur.bytes.len() != ref_.bytes.len() {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "XorDelta.forward: byte length mismatch: current={}, ref={}",
                cur.bytes.len(),
                ref_.bytes.len()
            )));
        }
        let descriptors =
            self.propagate_descriptors(&[cur.descriptor.clone(), ref_.descriptor.clone()])?;
        let residual: Vec<u8> = cur
            .bytes
            .iter()
            .zip(ref_.bytes.iter())
            .map(|(&a, &b)| a ^ b)
            .collect();
        Ok(vec![Plane {
            bytes: Arc::from(residual.into_boxed_slice()),
            descriptor: descriptors[0].clone(),
        }])
    }

    /// Decode: given `outputs[0]` = residual and `outputs[1]` = reference,
    /// reconstruct the original via XOR (involutive).
    fn inverse(&self, outputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        if outputs.len() != 2 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "XorDelta.inverse: expected 2 inputs (residual, reference), got {}",
                outputs.len()
            )));
        }
        let residual = &outputs[0];
        let ref_ = &outputs[1];
        if residual.bytes.len() != ref_.bytes.len() {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "XorDelta.inverse: byte length mismatch: residual={}, ref={}",
                residual.bytes.len(),
                ref_.bytes.len()
            )));
        }
        let original: Vec<u8> = residual
            .bytes
            .iter()
            .zip(ref_.bytes.iter())
            .map(|(&r, &b)| r ^ b)
            .collect();
        let len = original.len() as u64;
        let in_descriptor = PlaneDescriptor {
            role: residual.descriptor.role.clone(),
            element_width: residual.descriptor.element_width,
            length_bytes: len,
            layout: Layout::Flat,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        };
        Ok(vec![Plane {
            bytes: Arc::from(original.into_boxed_slice()),
            descriptor: in_descriptor,
        }])
    }

    fn id(&self) -> OpId {
        OpId::XorDelta
    }

    fn write_params(&self, out: &mut Vec<u8>) {
        out.push(self.dep_idx);
    }
}

// ---------------------------------------------------------------------------
// FloatDelta (f32 only; DONE_WITH_CONCERNS — see module docstring)
//
// Gated behind the `lossy` feature flag because IEEE 754 rounding in
// `(a - b) + b` can break PPG's byte-exact round-trip invariant.
// ---------------------------------------------------------------------------

/// Element-wise float subtract residual against a reference tensor plane.
///
/// **DONE_WITH_CONCERNS**: IEEE 754 does not guarantee that `(a - b) + b == a`
/// bit-exactly for all inputs. This op may violate the PPG byte-exact round-
/// trip invariant. Use `XorDelta` instead when lossless is required.
///
/// Currently only `dtype_code` corresponding to `f32` is supported.
///
/// **Input convention** (both forward and inverse):
/// - `inputs[0]` / `outputs[0]`: the plane being encoded or the residual.
/// - `inputs[1]` / `outputs[1]`: the reference plane.
pub struct FloatDelta {
    /// Index into the dependency-tensor list for the reference tensor.
    pub dep_idx: u8,
    /// Dtype code identifying the floating-point type. Only f32 (0x000B) is supported.
    pub dtype_code: u16,
}

/// dtype_code for f32 (matches `Dtype::Float32.code()` = 1 in the crate's dtype registry).
pub const DTYPE_CODE_F32: u16 = 1;

impl FloatDelta {
    pub fn new(dep_idx: u8, dtype_code: u16) -> Result<Self, PtwmCoreError> {
        if dtype_code != DTYPE_CODE_F32 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "FloatDelta: only f32 (dtype_code=0x{DTYPE_CODE_F32:04X}) is supported, \
                 got 0x{dtype_code:04X}"
            )));
        }
        Ok(FloatDelta {
            dep_idx,
            dtype_code,
        })
    }
}

impl Op for FloatDelta {
    fn propagate_descriptors(
        &self,
        inputs: &[PlaneDescriptor],
    ) -> Result<Vec<PlaneDescriptor>, PtwmCoreError> {
        if inputs.len() != 2 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "FloatDelta.propagate_descriptors: expected 2 inputs (current, reference), got {}",
                inputs.len()
            )));
        }
        let cur = &inputs[0];
        let ref_ = &inputs[1];
        if cur.length_bytes != ref_.length_bytes {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "FloatDelta.propagate_descriptors: length_bytes mismatch: current={}, ref={}",
                cur.length_bytes, ref_.length_bytes
            )));
        }
        Ok(vec![PlaneDescriptor {
            role: Role::Residual {
                format: ResidualFormat::FloatDelta {
                    dtype: self.dtype_code,
                },
            },
            element_width: ElementWidth::Word4, // f32
            length_bytes: cur.length_bytes,
            layout: Layout::Flat,
            derives_from_tensor: Some(crate::types::descriptor::TensorRef {
                dep_idx: self.dep_idx,
            }),
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        }])
    }

    fn forward(&self, inputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        if inputs.len() != 2 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "FloatDelta.forward: expected 2 inputs (current, reference), got {}",
                inputs.len()
            )));
        }
        let cur = &inputs[0];
        let ref_ = &inputs[1];
        if !cur.bytes.len().is_multiple_of(4) {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "FloatDelta.forward: current byte length {} is not f32-aligned (% 4 != 0)",
                cur.bytes.len()
            )));
        }
        if cur.bytes.len() != ref_.bytes.len() {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "FloatDelta.forward: byte length mismatch: current={}, ref={}",
                cur.bytes.len(),
                ref_.bytes.len()
            )));
        }
        let descriptors =
            self.propagate_descriptors(&[cur.descriptor.clone(), ref_.descriptor.clone()])?;
        let mut out = Vec::with_capacity(cur.bytes.len());
        for (ac, bc) in cur.bytes.chunks_exact(4).zip(ref_.bytes.chunks_exact(4)) {
            let a = f32::from_le_bytes([ac[0], ac[1], ac[2], ac[3]]);
            let b = f32::from_le_bytes([bc[0], bc[1], bc[2], bc[3]]);
            out.extend_from_slice(&(a - b).to_le_bytes());
        }
        Ok(vec![Plane {
            bytes: Arc::from(out.into_boxed_slice()),
            descriptor: descriptors[0].clone(),
        }])
    }

    /// Decode: `outputs[0]` = residual, `outputs[1]` = reference.
    /// Returns `residual + reference` element-wise.
    ///
    /// **Note**: result may differ from the original by ~1 ULP due to IEEE 754
    /// rounding in `(a - b) + b`. This op does not provide byte-exact round-trip.
    fn inverse(&self, outputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        if outputs.len() != 2 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "FloatDelta.inverse: expected 2 inputs (residual, reference), got {}",
                outputs.len()
            )));
        }
        let residual = &outputs[0];
        let ref_ = &outputs[1];
        if !residual.bytes.len().is_multiple_of(4) {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "FloatDelta.inverse: residual byte length {} is not f32-aligned (% 4 != 0)",
                residual.bytes.len()
            )));
        }
        if residual.bytes.len() != ref_.bytes.len() {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "FloatDelta.inverse: byte length mismatch: residual={}, ref={}",
                residual.bytes.len(),
                ref_.bytes.len()
            )));
        }
        let mut out = Vec::with_capacity(residual.bytes.len());
        for (rc, bc) in residual
            .bytes
            .chunks_exact(4)
            .zip(ref_.bytes.chunks_exact(4))
        {
            let r = f32::from_le_bytes([rc[0], rc[1], rc[2], rc[3]]);
            let b = f32::from_le_bytes([bc[0], bc[1], bc[2], bc[3]]);
            out.extend_from_slice(&(r + b).to_le_bytes());
        }
        let len = out.len() as u64;
        let in_descriptor = PlaneDescriptor {
            role: Role::Raw,
            element_width: ElementWidth::Word4,
            length_bytes: len,
            layout: Layout::Flat,
            derives_from_tensor: None,
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
        OpId::FloatDelta
    }

    fn write_params(&self, out: &mut Vec<u8>) {
        out.push(self.dep_idx);
        out.extend_from_slice(&self.dtype_code.to_le_bytes());
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::descriptor::{ElementWidth, Layout};
    use crate::types::role::Role;

    fn byte_plane(data: Vec<u8>, ew: ElementWidth) -> Plane {
        Plane {
            bytes: Arc::from(&data[..]),
            descriptor: PlaneDescriptor {
                role: Role::Raw,
                element_width: ew,
                length_bytes: data.len() as u64,
                layout: Layout::Flat,
                derives_from_tensor: None,
                residual_of: None,
                is_nibble_packed: false,
                vendor_bytes: vec![],
            },
        }
    }

    // ---- XorDelta tests ----

    #[test]
    fn xor_delta_roundtrip() {
        let a: Vec<u8> = (0u8..32).map(|i| i.wrapping_mul(7)).collect();
        let b: Vec<u8> = (0u8..32).map(|i| i.wrapping_mul(13)).collect();
        let op = XorDelta::new(0);
        let plane_a = byte_plane(a.clone(), ElementWidth::Byte);
        let plane_b = byte_plane(b.clone(), ElementWidth::Byte);
        // Forward: residual = a XOR b
        let fwd = op.forward(&[plane_a, plane_b.clone()]).unwrap();
        assert_eq!(fwd.len(), 1);
        let residual_expected: Vec<u8> = a.iter().zip(b.iter()).map(|(&x, &y)| x ^ y).collect();
        assert_eq!(fwd[0].bytes.as_ref(), residual_expected);
        // Inverse: (residual, b) → a
        let inv = op.inverse(&[fwd[0].clone(), plane_b]).unwrap();
        assert_eq!(inv.len(), 1);
        assert_eq!(inv[0].bytes.as_ref(), a);
    }

    #[test]
    fn xor_delta_propagate_rejects_unequal_lengths() {
        let op = XorDelta::new(0);
        let d1 = PlaneDescriptor {
            role: Role::Raw,
            element_width: ElementWidth::Byte,
            length_bytes: 32,
            layout: Layout::Flat,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        };
        let mut d2 = d1.clone();
        d2.length_bytes = 64;
        assert!(op.propagate_descriptors(&[d1, d2]).is_err());
    }

    #[test]
    fn xor_delta_propagate_rejects_unequal_element_widths() {
        let op = XorDelta::new(0);
        let d1 = PlaneDescriptor {
            role: Role::Raw,
            element_width: ElementWidth::Byte,
            length_bytes: 32,
            layout: Layout::Flat,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        };
        let mut d2 = d1.clone();
        d2.element_width = ElementWidth::Word2;
        assert!(op.propagate_descriptors(&[d1, d2]).is_err());
    }

    #[test]
    fn xor_delta_write_params_roundtrip() {
        let op = XorDelta::new(7);
        let mut buf = Vec::new();
        op.write_params(&mut buf);
        assert_eq!(buf, vec![7u8]);
    }

    // ---- FloatDelta tests (gated behind `lossy` feature) ----

    #[cfg(feature = "lossy")]
    #[test]
    fn float_delta_f32_approx_roundtrip() {
        // Use values that are representable without rounding error.
        let a_vals: Vec<f32> = vec![1.0, 2.0, 3.5, -1.25, 0.0, 100.0, -50.5, 0.125];
        let b_vals: Vec<f32> = vec![0.5, 1.5, 3.0, -1.0, 0.5, 99.0, -50.0, 0.0];
        let a_bytes: Vec<u8> = a_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = b_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let op = FloatDelta::new(0, DTYPE_CODE_F32).unwrap();
        let plane_a = byte_plane(a_bytes.clone(), ElementWidth::Word4);
        let plane_b = byte_plane(b_bytes.clone(), ElementWidth::Word4);
        let fwd = op.forward(&[plane_a, plane_b.clone()]).unwrap();
        assert_eq!(fwd.len(), 1);
        let inv = op.inverse(&[fwd[0].clone(), plane_b]).unwrap();
        assert_eq!(inv.len(), 1);
        // Approximate check (~1 ULP); for these values (powers of 2 / halves) it is exact.
        let reconstructed: Vec<f32> = inv[0]
            .bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        for (orig, recon) in a_vals.iter().zip(reconstructed.iter()) {
            assert!(
                (orig - recon).abs() < 1e-5,
                "round-trip error too large: orig={orig}, recon={recon}"
            );
        }
    }

    #[cfg(feature = "lossy")]
    #[test]
    fn float_delta_propagate_rejects_unequal_lengths() {
        let op = FloatDelta::new(0, DTYPE_CODE_F32).unwrap();
        let d1 = PlaneDescriptor {
            role: Role::Raw,
            element_width: ElementWidth::Word4,
            length_bytes: 32,
            layout: Layout::Flat,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        };
        let mut d2 = d1.clone();
        d2.length_bytes = 64;
        assert!(op.propagate_descriptors(&[d1, d2]).is_err());
    }

    #[cfg(feature = "lossy")]
    #[test]
    fn float_delta_write_params_roundtrip() {
        let op = FloatDelta::new(3, DTYPE_CODE_F32).unwrap();
        let mut buf = Vec::new();
        op.write_params(&mut buf);
        // dep_idx (1 byte) + dtype_code (2 bytes LE)
        assert_eq!(buf.len(), 3);
        assert_eq!(buf[0], 3u8);
        assert_eq!(
            u16::from_le_bytes(buf[1..3].try_into().unwrap()),
            DTYPE_CODE_F32
        );
    }

    #[cfg(feature = "lossy")]
    #[test]
    fn float_delta_rejects_unsupported_dtype() {
        // dtype_code 4 is f16, which FloatDelta does not yet support.
        assert!(FloatDelta::new(0, 4).is_err());
        // dtype_code 6 is bf16, also unsupported.
        assert!(FloatDelta::new(0, 6).is_err());
    }
}
