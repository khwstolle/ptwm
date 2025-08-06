//! `MantissaZeroStrip` op — strip a uniform run of trailing zero bits
//! from every byte of a quantized-weight mantissa plane.
//!
//! ## Motivation
//!
//! GPTQ / AWQ-style quantized checkpoints store weights with rounded
//! mantissas: an INT4-quantized weight, after dequantization back into
//! BF16 / FP16, has a mantissa whose low bits are always exactly zero.
//! After PTWM's standard `bit_reorder_ieee` + `byte_split` chain, the
//! mantissa plane is a flat byte stream where every element shares
//! the same uniform run of trailing zero bits. Without exploiting this
//! the downstream entropy coder sees a value range that is 2^k times
//! larger than the true alphabet, which costs roughly k bits per
//! element of compression on GGUF / GPTQ-style checkpoints.
//!
//! ## Scope of this op
//!
//! The op handles the dominant case: a *uniform* `k` (1..=7) that holds
//! across every byte in the plane. The chain author (or the explorer)
//! picks `k` at chain-build time and the op stores it as a one-byte
//! parameter on the wire. At encode time the op validates that every
//! byte's low `k` bits are zero; if any byte fails the check the op
//! returns an error and the trial-encode dispatcher falls back to a
//! chain that does not include this op.
//!
//! A non-uniform per-element variant (the "run-length" reading of the
//! issue title) would need a side bitmap and a sparse value buffer; it
//! is deliberately out of scope here. The uniform case captures the
//! published gap and stays a simple invertible byte transform.

use std::sync::Arc;

use crate::error::PtwmCoreError;
use crate::transforms::op::{Op, OpId, Plane};
use crate::types::descriptor::{ElementWidth, PlaneDescriptor};

/// Right-shift every byte in a plane by `k` bits. Inverse left-shifts
/// by `k`. The op requires that every input byte has at least `k`
/// trailing zero bits — otherwise the right-shift would lose
/// information and the inverse could not recover the original byte.
///
/// `k` is the wire-format parameter (1..=7). `k == 0` is rejected at
/// construction because it would make the op a pure no-op and adds
/// only cost; `k >= 8` is rejected because shifting a `u8` by 8 or
/// more zeroes every byte and loses any high-bit information.
pub struct MantissaZeroStrip {
    /// Number of low bits to strip from each byte. Stored on the wire
    /// as a single byte after the OpId.
    pub k: u8,
}

impl MantissaZeroStrip {
    /// Construct a `MantissaZeroStrip` with shift width `k`. Returns
    /// an error for `k == 0` or `k > 7`.
    pub fn new(k: u8) -> Result<Self, PtwmCoreError> {
        if k == 0 {
            return Err(PtwmCoreError::InvalidContainer(
                "MantissaZeroStrip: k=0 is a no-op; omit the op instead".into(),
            ));
        }
        if k > 7 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "MantissaZeroStrip: k={k} is out of range 1..=7 (would clear the byte)"
            )));
        }
        Ok(MantissaZeroStrip { k })
    }
}

impl Op for MantissaZeroStrip {
    fn propagate_descriptors(
        &self,
        inputs: &[PlaneDescriptor],
    ) -> Result<Vec<PlaneDescriptor>, PtwmCoreError> {
        if inputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "MantissaZeroStrip.propagate_descriptors: expected 1 input, got {}",
                inputs.len()
            )));
        }
        let d = &inputs[0];
        if d.element_width != ElementWidth::Byte {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "MantissaZeroStrip.propagate_descriptors: expected element_width Byte, got {:?}",
                d.element_width
            )));
        }
        // Descriptor passes through unchanged: the shift does not change
        // the plane shape, only the byte values.
        Ok(vec![d.clone()])
    }

    fn forward(&self, inputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        if inputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "MantissaZeroStrip.forward: expected 1 input, got {}",
                inputs.len()
            )));
        }
        let inp = &inputs[0];
        if inp.descriptor.element_width != ElementWidth::Byte {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "MantissaZeroStrip.forward: expected element_width Byte, got {:?}",
                inp.descriptor.element_width
            )));
        }
        // The op is correct only when every byte has at least `k`
        // trailing zero bits. Validate that invariant first with a
        // short-circuiting scan, then shift on a separate pass: when
        // this op is in the trial-encode menu on non-quantized data
        // the validation fails almost immediately on the first
        // non-zero low-bit byte, so the failure path avoids any
        // output allocation. On the success path two cache-friendly
        // passes are noise next to one pass plus the inner-loop OR.
        let low_mask: u8 = (1u8 << self.k) - 1;
        if inp.bytes.iter().any(|&b| (b & low_mask) != 0) {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "MantissaZeroStrip.forward: plane has a byte with non-zero low {} bits; \
                 cannot strip without information loss",
                self.k
            )));
        }
        let out: Vec<u8> = inp.bytes.iter().map(|&b| b >> self.k).collect();
        Ok(vec![Plane {
            bytes: Arc::from(out.into_boxed_slice()),
            descriptor: inp.descriptor.clone(),
        }])
    }

    fn inverse(&self, outputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        if outputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "MantissaZeroStrip.inverse: expected 1 output, got {}",
                outputs.len()
            )));
        }
        let p = &outputs[0];
        if p.descriptor.element_width != ElementWidth::Byte {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "MantissaZeroStrip.inverse: expected element_width Byte, got {:?}",
                p.descriptor.element_width
            )));
        }
        // The inverse left-shifts by k. The high `k` bits of every byte
        // in the stripped plane are necessarily zero (the forward
        // produced bytes in the range 0..(1 << (8 - k))), so the shift
        // cannot overflow a u8. iter().map().collect() pre-sizes the
        // output via TrustedLen, matching with_capacity + push without
        // the manual length bookkeeping.
        let out: Vec<u8> = p.bytes.iter().map(|&b| b << self.k).collect();
        Ok(vec![Plane {
            bytes: Arc::from(out.into_boxed_slice()),
            descriptor: p.descriptor.clone(),
        }])
    }

    fn id(&self) -> OpId {
        OpId::MantissaZeroStrip
    }

    fn write_params(&self, out: &mut Vec<u8>) {
        out.push(self.k);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::descriptor::{Layout, PlaneDescriptor};
    use crate::types::role::Role;

    fn descriptor(len: u64) -> PlaneDescriptor {
        PlaneDescriptor {
            role: Role::Raw,
            element_width: ElementWidth::Byte,
            length_bytes: len,
            layout: Layout::Flat,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        }
    }

    fn plane(bytes: Vec<u8>) -> Plane {
        let d = descriptor(bytes.len() as u64);
        Plane {
            bytes: Arc::from(bytes.into_boxed_slice()),
            descriptor: d,
        }
    }

    #[test]
    fn new_rejects_zero_k() {
        assert!(MantissaZeroStrip::new(0).is_err());
    }

    #[test]
    fn new_rejects_k_over_seven() {
        assert!(MantissaZeroStrip::new(8).is_err());
        assert!(MantissaZeroStrip::new(255).is_err());
    }

    #[test]
    fn new_accepts_valid_k() {
        for k in 1u8..=7 {
            assert!(MantissaZeroStrip::new(k).is_ok());
        }
    }

    #[test]
    fn write_params_roundtrip() {
        let op = MantissaZeroStrip::new(3).unwrap();
        let mut buf = Vec::new();
        op.write_params(&mut buf);
        assert_eq!(buf, vec![3u8]);
    }

    #[test]
    fn descriptor_passes_through() {
        let op = MantissaZeroStrip::new(4).unwrap();
        let d = descriptor(64);
        let out = op.propagate_descriptors(std::slice::from_ref(&d)).unwrap();
        assert_eq!(out, vec![d]);
    }

    #[test]
    fn propagate_descriptors_rejects_nibble_width() {
        let op = MantissaZeroStrip::new(3).unwrap();
        let mut d = descriptor(8);
        d.element_width = ElementWidth::Nibble;
        d.is_nibble_packed = true;
        assert!(op.propagate_descriptors(std::slice::from_ref(&d)).is_err());
    }

    #[test]
    fn empty_plane_round_trips() {
        let op = MantissaZeroStrip::new(4).unwrap();
        let p = plane(vec![]);
        let fwd = op.forward(std::slice::from_ref(&p)).unwrap();
        assert!(fwd[0].bytes.is_empty());
        let inv = op.inverse(&fwd).unwrap();
        assert!(inv[0].bytes.is_empty());
    }

    #[test]
    fn int4_like_plane_round_trips_at_k4() {
        // Every byte has its low 4 bits zero — the INT4 dequantization
        // case. The shifted plane should be exactly the high nibble.
        let raw: Vec<u8> = (0u8..16).map(|n| n << 4).collect();
        let op = MantissaZeroStrip::new(4).unwrap();
        let fwd = op.forward(&[plane(raw.clone())]).unwrap();
        let expected: Vec<u8> = (0u8..16).collect();
        assert_eq!(fwd[0].bytes.as_ref(), expected);
        let inv = op.inverse(&fwd).unwrap();
        assert_eq!(inv[0].bytes.as_ref(), raw);
    }

    #[test]
    fn int5_like_plane_round_trips_at_k3() {
        // Every byte has its low 3 bits zero — INT5 dequantization case.
        let raw: Vec<u8> = (0u8..32).map(|n| n << 3).collect();
        let op = MantissaZeroStrip::new(3).unwrap();
        let fwd = op.forward(&[plane(raw.clone())]).unwrap();
        assert_eq!(fwd[0].bytes.as_ref(), (0u8..32).collect::<Vec<_>>());
        let inv = op.inverse(&fwd).unwrap();
        assert_eq!(inv[0].bytes.as_ref(), raw);
    }

    #[test]
    fn round_trips_at_every_valid_k() {
        for k in 1u8..=7 {
            // Build a plane where every byte is a multiple of (1 << k);
            // those bytes have at least k trailing zero bits.
            let step = 1u16 << k;
            let raw: Vec<u8> = (0u16..256)
                .step_by(step as usize)
                .map(|n| n as u8)
                .collect();
            let op = MantissaZeroStrip::new(k).unwrap();
            let fwd = op
                .forward(&[plane(raw.clone())])
                .unwrap_or_else(|e| panic!("forward failed for k={k}: {e:?}"));
            let inv = op.inverse(&fwd).unwrap();
            assert_eq!(inv[0].bytes.as_ref(), raw, "round-trip failed at k={k}");
        }
    }

    #[test]
    fn forward_rejects_byte_with_insufficient_trailing_zeros() {
        // 0x01 has zero trailing-zero bits; a k=1 strip would lose
        // information.
        let op = MantissaZeroStrip::new(1).unwrap();
        let res = op.forward(&[plane(vec![0x02, 0x04, 0x01])]);
        assert!(res.is_err(), "expected forward rejection, got {res:?}");
    }

    #[test]
    fn forward_rejects_byte_with_low_bit_set_at_k4() {
        // 0x18 (binary 00011000) has 3 trailing zeros — k=4 must reject.
        let op = MantissaZeroStrip::new(4).unwrap();
        let res = op.forward(&[plane(vec![0x10, 0x18, 0x20])]);
        assert!(res.is_err());
    }

    #[test]
    fn forward_accepts_all_zero_plane_at_any_k() {
        // All-zero bytes have all 8 trailing zeros so any k works.
        let raw = vec![0u8; 32];
        for k in 1u8..=7 {
            let op = MantissaZeroStrip::new(k).unwrap();
            let fwd = op.forward(&[plane(raw.clone())]).unwrap();
            assert!(fwd[0].bytes.iter().all(|&b| b == 0));
            let inv = op.inverse(&fwd).unwrap();
            assert_eq!(inv[0].bytes.as_ref(), raw);
        }
    }

    #[test]
    fn forward_rejects_wrong_input_count() {
        let op = MantissaZeroStrip::new(3).unwrap();
        assert!(op.forward(&[]).is_err());
        assert!(op.forward(&[plane(vec![0]), plane(vec![0])]).is_err());
    }

    #[test]
    fn inverse_rejects_wrong_output_count() {
        let op = MantissaZeroStrip::new(3).unwrap();
        assert!(op.inverse(&[]).is_err());
        assert!(op.inverse(&[plane(vec![0]), plane(vec![0])]).is_err());
    }

    #[test]
    fn id_matches_opid() {
        let op = MantissaZeroStrip::new(4).unwrap();
        assert_eq!(op.id(), OpId::MantissaZeroStrip);
    }
}
