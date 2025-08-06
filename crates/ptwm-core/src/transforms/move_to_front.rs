//! `MoveToFront` op — the move-to-front (MTF) byte transform.
//!
//! MTF maintains a list of the 256 byte values, initialised to the
//! identity order `0, 1, …, 255`. Each input byte is replaced by its
//! current index in the list, and that byte is then moved to the front.
//! Bytes that recur close together therefore map to small indices, which
//! is exactly what a downstream entropy coder (Huffman / rANS) codes
//! cheaply.
//!
//! ## Why it pairs with [`crate::transforms::BurrowsWheeler`]
//!
//! After the bit-reorder + byte-split chain, the exponent plane is mostly
//! long runs of similar bytes. A Burrows–Wheeler pass clusters identical
//! bytes into long runs; MTF then turns those runs into long runs of
//! zeros (the front of the list), which collapse under the entropy coder.
//! This is the classic bzip2 pipeline (BWT → MTF → entropy). The two ops
//! are kept separate so each is independently addressable in a chain —
//! MTF on its own still helps any plane with strong local byte locality.
//!
//! ## Element widths
//!
//! MTF is a pure byte remap; it is defined on the raw byte stream
//! regardless of the logical element width. Nibble-packed planes are
//! rejected because their half-byte storage has no unambiguous byte
//! interpretation. The descriptor (length, role, layout, width) passes
//! through unchanged — MTF is length-preserving.
//!
//! Round-trip is exact for every input: `inverse(forward(x)) == x`.

use std::sync::Arc;

use crate::error::PtwmCoreError;
use crate::transforms::op::{Op, OpId, Plane};
use crate::types::descriptor::PlaneDescriptor;

/// Move-to-front byte transform.
#[derive(Debug, Default, Clone, Copy)]
pub struct MoveToFront;

impl MoveToFront {
    pub const fn new() -> Self {
        Self
    }
}

impl Op for MoveToFront {
    fn propagate_descriptors(
        &self,
        inputs: &[PlaneDescriptor],
    ) -> Result<Vec<PlaneDescriptor>, PtwmCoreError> {
        if inputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "MoveToFront.propagate_descriptors: expected 1 input, got {}",
                inputs.len()
            )));
        }
        let d = &inputs[0];
        reject_nibble_packed(d, "propagate_descriptors")?;
        // Length-preserving byte remap: everything passes through.
        Ok(vec![d.clone()])
    }

    fn forward(&self, inputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        if inputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "MoveToFront.forward: expected 1 input, got {}",
                inputs.len()
            )));
        }
        let p = &inputs[0];
        reject_nibble_packed(&p.descriptor, "forward")?;
        Ok(vec![Plane {
            bytes: Arc::from(mtf_forward(&p.bytes).into_boxed_slice()),
            descriptor: p.descriptor.clone(),
        }])
    }

    fn inverse(&self, outputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        if outputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "MoveToFront.inverse: expected 1 input, got {}",
                outputs.len()
            )));
        }
        let p = &outputs[0];
        reject_nibble_packed(&p.descriptor, "inverse")?;
        Ok(vec![Plane {
            bytes: Arc::from(mtf_inverse(&p.bytes).into_boxed_slice()),
            descriptor: p.descriptor.clone(),
        }])
    }

    fn id(&self) -> OpId {
        OpId::MoveToFront
    }

    fn write_params(&self, _out: &mut Vec<u8>) {
        // No parameters: the alphabet is the fixed 256-byte identity table.
    }
}

/// Encode: each byte becomes its index in the MTF list, then is moved to
/// the front.
fn mtf_forward(input: &[u8]) -> Vec<u8> {
    let mut table: [u8; 256] = core::array::from_fn(|i| i as u8);
    let mut out = Vec::with_capacity(input.len());
    for &b in input {
        // `position` always succeeds: every byte value is in the table.
        let idx = table.iter().position(|&x| x == b).unwrap();
        out.push(idx as u8);
        // Move the byte at `idx` to the front, shifting the prefix right.
        table[..=idx].rotate_right(1);
    }
    out
}

/// Decode: each index recovers the byte currently at that position in the
/// MTF list, which is then moved to the front (mirroring the encoder).
fn mtf_inverse(input: &[u8]) -> Vec<u8> {
    let mut table: [u8; 256] = core::array::from_fn(|i| i as u8);
    let mut out = Vec::with_capacity(input.len());
    for &idx in input {
        let i = idx as usize;
        out.push(table[i]);
        table[..=i].rotate_right(1);
    }
    out
}

fn reject_nibble_packed(d: &PlaneDescriptor, ctx: &str) -> Result<(), PtwmCoreError> {
    if d.is_nibble_packed {
        return Err(PtwmCoreError::InvalidContainer(format!(
            "MoveToFront.{ctx}: nibble-packed planes are not supported"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::descriptor::{ElementWidth, Layout};
    use crate::types::role::Role;

    fn plane(bytes: Vec<u8>) -> Plane {
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
            bytes: Arc::from(bytes.into_boxed_slice()),
        }
    }

    #[test]
    fn known_vector() {
        // Worked example over the identity table for the bytes of
        // "bananaaa" (b=98, a=97, n=110):
        //   b → 98, a → 98 (a pushed up one after b moved to front),
        //   n → 110, a → 1, n → 1, a → 1, a → 0, a → 0.
        // The trailing run of identical 'a' collapses to index 0 once 'a'
        // reaches the front of the list.
        let raw = b"bananaaa".to_vec();
        let fwd = mtf_forward(&raw);
        assert_eq!(fwd, vec![98, 98, 110, 1, 1, 1, 0, 0]);
        assert_eq!(mtf_inverse(&fwd), raw);
    }

    #[test]
    fn repeated_byte_collapses_to_zeros() {
        let raw = vec![0x42u8; 64];
        let fwd = mtf_forward(&raw);
        // First byte is its own index; every subsequent identical byte is
        // already at the front → index 0.
        assert_eq!(fwd[0], 0x42);
        assert!(fwd[1..].iter().all(|&b| b == 0));
        assert_eq!(mtf_inverse(&fwd), raw);
    }

    #[test]
    fn round_trip_all_byte_values() {
        let raw: Vec<u8> = (0u8..=255).collect();
        let op = MoveToFront::new();
        let fwd = op.forward(&[plane(raw.clone())]).unwrap();
        let inv = op.inverse(&[fwd[0].clone()]).unwrap();
        assert_eq!(inv[0].bytes.as_ref(), raw);
    }

    #[test]
    fn round_trip_pseudo_random() {
        // Deterministic LCG stream — no locality, but round-trip must hold.
        let mut state = 0x1234_5678u32;
        let raw: Vec<u8> = (0..4096)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 24) as u8
            })
            .collect();
        let op = MoveToFront::new();
        let fwd = op.forward(&[plane(raw.clone())]).unwrap();
        let inv = op.inverse(&[fwd[0].clone()]).unwrap();
        assert_eq!(inv[0].bytes.as_ref(), raw);
    }

    #[test]
    fn empty_plane_round_trips() {
        let op = MoveToFront::new();
        let fwd = op.forward(&[plane(vec![])]).unwrap();
        assert!(fwd[0].bytes.is_empty());
        let inv = op.inverse(&[fwd[0].clone()]).unwrap();
        assert!(inv[0].bytes.is_empty());
    }

    #[test]
    fn rejects_nibble_packed() {
        let mut p = plane(vec![0u8; 4]);
        p.descriptor.element_width = ElementWidth::Nibble;
        p.descriptor.is_nibble_packed = true;
        let op = MoveToFront::new();
        assert!(op.forward(&[p.clone()]).is_err());
        assert!(op.inverse(&[p]).is_err());
    }

    #[test]
    fn rejects_wrong_input_count() {
        let op = MoveToFront::new();
        let p = plane(vec![1u8, 2, 3, 4]);
        assert!(op.forward(&[p.clone(), p.clone()]).is_err());
        assert!(op.inverse(&[]).is_err());
    }

    #[test]
    fn write_params_is_empty() {
        let op = MoveToFront::new();
        let mut buf = Vec::new();
        op.write_params(&mut buf);
        assert!(buf.is_empty());
    }

    #[test]
    fn descriptor_passes_through() {
        let d = PlaneDescriptor {
            role: Role::Raw,
            element_width: ElementWidth::Byte,
            length_bytes: 8,
            layout: Layout::Flat,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        };
        let op = MoveToFront::new();
        let out = op.propagate_descriptors(std::slice::from_ref(&d)).unwrap();
        assert_eq!(out, vec![d]);
    }
}
