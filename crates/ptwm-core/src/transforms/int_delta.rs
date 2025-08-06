//! `IntDelta` op — intra-plane integer-subtraction delta between consecutive
//! elements (ndzip / FPC-family preprocessing).
//!
//! Unlike `XorDelta` and `FloatDelta`, which take a *reference tensor* as a
//! second input and produce a cross-tensor residual, `IntDelta` is a
//! single-input intra-plane transform: each element becomes the integer
//! difference from its predecessor in the same plane.
//!
//! For smoothly-varying data (think attention-weight rows whose neighbouring
//! values are numerically close), the integer delta is concentrated near zero
//! and compresses harder under the downstream entropy coder than the raw
//! byte stream does. This is the same preprocessing used by ndzip and FPC
//! for scientific floating-point streams; see
//! [ndzip-gpu (ICS 2021)](https://dps.uibk.ac.at/~fabian/publications/2021-ndzip-gpu-efficient-lossless-compression-of-scientific-floating-point-data-on-gpus.pdf).
//!
//! ## Element-width awareness
//!
//! The delta is computed at the plane descriptor's `element_width`:
//!
//! - `Byte` (1 B) — byte-level wrapping subtraction.
//! - `Word2` (2 B) — pairs interpreted as little-endian u16, wrapping sub.
//! - `Word4` (4 B) — u32 LE, wrapping sub.
//! - `Word8` (8 B) — u64 LE, wrapping sub.
//! - `Nibble` — rejected; semantics are ambiguous for half-byte elements.
//!
//! Round-trip is exact for every input the encoder accepts. The op is
//! pure-byte (no `lossy` feature gate) because integer wrapping is
//! involutive in the same way XOR is.

use std::sync::Arc;

use crate::error::PtwmCoreError;
use crate::transforms::op::{Op, OpId, Plane};
use crate::types::descriptor::{ElementWidth, PlaneDescriptor};

/// Intra-plane element-wise integer-subtraction delta.
///
/// `forward`: `out[0] = in[0]`, `out[i] = in[i].wrapping_sub(in[i-1])`.
/// `inverse`: `in[0] = out[0]`, `in[i] = out[i].wrapping_add(in[i-1])`.
///
/// The stride is the plane's `element_width`. Wrapping arithmetic at that
/// width is exactly invertible.
#[derive(Debug, Default, Clone, Copy)]
pub struct IntDelta;

impl IntDelta {
    pub const fn new() -> Self {
        Self
    }
}

impl Op for IntDelta {
    fn propagate_descriptors(
        &self,
        inputs: &[PlaneDescriptor],
    ) -> Result<Vec<PlaneDescriptor>, PtwmCoreError> {
        if inputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "IntDelta.propagate_descriptors: expected 1 input, got {}",
                inputs.len()
            )));
        }
        let d = &inputs[0];
        reject_nibble(d, "propagate_descriptors")?;
        // Length, role, layout, etc. all pass through unchanged — the
        // delta lives in the same byte budget as the original plane.
        Ok(vec![d.clone()])
    }

    fn forward(&self, inputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        if inputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "IntDelta.forward: expected 1 input, got {}",
                inputs.len()
            )));
        }
        let p = &inputs[0];
        reject_nibble(&p.descriptor, "forward")?;
        let width = stride_bytes(p.descriptor.element_width);
        if !p.bytes.len().is_multiple_of(width) {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "IntDelta.forward: byte length {} is not a multiple of element width {width}",
                p.bytes.len()
            )));
        }
        let out = delta_forward(&p.bytes, width);
        Ok(vec![Plane {
            bytes: Arc::from(out.into_boxed_slice()),
            descriptor: p.descriptor.clone(),
        }])
    }

    fn inverse(&self, outputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        if outputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "IntDelta.inverse: expected 1 input, got {}",
                outputs.len()
            )));
        }
        let p = &outputs[0];
        reject_nibble(&p.descriptor, "inverse")?;
        let width = stride_bytes(p.descriptor.element_width);
        if !p.bytes.len().is_multiple_of(width) {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "IntDelta.inverse: byte length {} is not a multiple of element width {width}",
                p.bytes.len()
            )));
        }
        let out = delta_inverse(&p.bytes, width);
        Ok(vec![Plane {
            bytes: Arc::from(out.into_boxed_slice()),
            descriptor: p.descriptor.clone(),
        }])
    }

    fn id(&self) -> OpId {
        OpId::IntDelta
    }

    fn write_params(&self, _out: &mut Vec<u8>) {
        // No parameters: stride is determined by the descriptor.
    }
}

/// Wrapping per-element delta in little-endian. Dispatches to a
/// per-width specialization so the inner loop avoids the stack-copy /
/// from_le_bytes round-trip that a generic implementation would do per
/// element — important because PRODUCTION_CHAINS now routes the
/// exponent byte plane through this op on every BF16/FP32 compress.
fn delta_forward(input: &[u8], width: usize) -> Vec<u8> {
    match width {
        1 => delta_forward_u8(input),
        2 => delta_forward_u16(input),
        4 => delta_forward_u32(input),
        8 => delta_forward_u64(input),
        _ => unreachable!("stride_bytes only returns 1/2/4/8"),
    }
}

fn delta_inverse(input: &[u8], width: usize) -> Vec<u8> {
    match width {
        1 => delta_inverse_u8(input),
        2 => delta_inverse_u16(input),
        4 => delta_inverse_u32(input),
        8 => delta_inverse_u64(input),
        _ => unreachable!("stride_bytes only returns 1/2/4/8"),
    }
}

// ---------------------------------------------------------------------------
// Per-width specializations. The byte-plane case (width=1) is the
// hottest because byte_split-then-int_delta on the exponent plane is
// the production path; the wider cases are exercised by direct
// IntDelta-on-non-split planes.
// ---------------------------------------------------------------------------

fn delta_forward_u8(input: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len());
    let Some((&first, rest)) = input.split_first() else {
        return out;
    };
    out.push(first);
    let mut prev = first;
    for &cur in rest {
        out.push(cur.wrapping_sub(prev));
        prev = cur;
    }
    out
}

fn delta_inverse_u8(input: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len());
    let mut acc: u8 = 0;
    let mut first = true;
    for &d in input {
        let cur = if first {
            first = false;
            d
        } else {
            acc.wrapping_add(d)
        };
        out.push(cur);
        acc = cur;
    }
    out
}

macro_rules! delta_forward_wide {
    ($name:ident, $ty:ty, $width:literal) => {
        fn $name(input: &[u8]) -> Vec<u8> {
            let mut out = Vec::with_capacity(input.len());
            let mut prev: $ty = 0;
            let mut first = true;
            for chunk in input.chunks_exact($width) {
                let cur = <$ty>::from_le_bytes(chunk.try_into().unwrap());
                let delta = if first {
                    first = false;
                    cur
                } else {
                    cur.wrapping_sub(prev)
                };
                out.extend_from_slice(&delta.to_le_bytes());
                prev = cur;
            }
            out
        }
    };
}

macro_rules! delta_inverse_wide {
    ($name:ident, $ty:ty, $width:literal) => {
        fn $name(input: &[u8]) -> Vec<u8> {
            let mut out = Vec::with_capacity(input.len());
            let mut acc: $ty = 0;
            let mut first = true;
            for chunk in input.chunks_exact($width) {
                let d = <$ty>::from_le_bytes(chunk.try_into().unwrap());
                let cur = if first {
                    first = false;
                    d
                } else {
                    acc.wrapping_add(d)
                };
                out.extend_from_slice(&cur.to_le_bytes());
                acc = cur;
            }
            out
        }
    };
}

delta_forward_wide!(delta_forward_u16, u16, 2);
delta_forward_wide!(delta_forward_u32, u32, 4);
delta_forward_wide!(delta_forward_u64, u64, 8);
delta_inverse_wide!(delta_inverse_u16, u16, 2);
delta_inverse_wide!(delta_inverse_u32, u32, 4);
delta_inverse_wide!(delta_inverse_u64, u64, 8);

fn stride_bytes(w: ElementWidth) -> usize {
    match w {
        ElementWidth::Byte => 1,
        ElementWidth::Nibble => 1, // rejected upstream; never reached
        ElementWidth::Word2 => 2,
        ElementWidth::Word4 => 4,
        ElementWidth::Word8 => 8,
    }
}

fn reject_nibble(d: &PlaneDescriptor, ctx: &str) -> Result<(), PtwmCoreError> {
    if d.element_width == ElementWidth::Nibble {
        return Err(PtwmCoreError::InvalidContainer(format!(
            "IntDelta.{ctx}: nibble-width planes are not supported (ambiguous delta semantics)"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::descriptor::Layout;
    use crate::types::role::Role;

    fn plane(bytes: Vec<u8>, width: ElementWidth) -> Plane {
        Plane {
            bytes: Arc::from(&bytes[..]),
            descriptor: PlaneDescriptor {
                role: Role::Raw,
                element_width: width,
                length_bytes: bytes.len() as u64,
                layout: Layout::Flat,
                derives_from_tensor: None,
                residual_of: None,
                is_nibble_packed: false,
                vendor_bytes: vec![],
            },
        }
    }

    #[test]
    fn byte_width_round_trip() {
        let raw = vec![10u8, 11, 13, 16, 20, 25, 30, 30, 31, 35];
        let op = IntDelta::new();
        let fwd = op
            .forward(&[plane(raw.clone(), ElementWidth::Byte)])
            .unwrap();
        // Expected delta: [10, 1, 2, 3, 4, 5, 5, 0, 1, 4]
        assert_eq!(fwd[0].bytes.as_ref(), vec![10, 1, 2, 3, 4, 5, 5, 0, 1, 4]);
        let inv = op.inverse(&[fwd[0].clone()]).unwrap();
        assert_eq!(inv[0].bytes.as_ref(), raw);
    }

    #[test]
    fn byte_width_wrapping_round_trip() {
        // Includes a transition that wraps around 0/0xFF.
        let raw = vec![5u8, 200, 30, 250, 0, 128];
        let op = IntDelta::new();
        let fwd = op
            .forward(&[plane(raw.clone(), ElementWidth::Byte)])
            .unwrap();
        let inv = op.inverse(&[fwd[0].clone()]).unwrap();
        assert_eq!(inv[0].bytes.as_ref(), raw);
    }

    #[test]
    fn word2_round_trip() {
        // Three u16 little-endian values: 0x0001, 0x0005, 0x0FFF
        let raw = vec![0x01, 0x00, 0x05, 0x00, 0xFF, 0x0F];
        let op = IntDelta::new();
        let fwd = op
            .forward(&[plane(raw.clone(), ElementWidth::Word2)])
            .unwrap();
        // Deltas: 0x0001, 0x0004, 0x0FFA
        assert_eq!(
            fwd[0].bytes.as_ref(),
            vec![0x01, 0x00, 0x04, 0x00, 0xFA, 0x0F]
        );
        let inv = op.inverse(&[fwd[0].clone()]).unwrap();
        assert_eq!(inv[0].bytes.as_ref(), raw);
    }

    #[test]
    fn word4_round_trip() {
        // Three u32 values; pick neighbouring smoothly-changing values.
        let vals: Vec<u32> = vec![100_000, 100_010, 100_005, 200_000];
        let raw: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let op = IntDelta::new();
        let fwd = op
            .forward(&[plane(raw.clone(), ElementWidth::Word4)])
            .unwrap();
        let inv = op.inverse(&[fwd[0].clone()]).unwrap();
        assert_eq!(inv[0].bytes.as_ref(), raw);
    }

    #[test]
    fn word8_round_trip() {
        let vals: Vec<u64> = vec![1, 5, 100, u64::MAX - 3, u64::MAX, 0, 10];
        let raw: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let op = IntDelta::new();
        let fwd = op
            .forward(&[plane(raw.clone(), ElementWidth::Word8)])
            .unwrap();
        let inv = op.inverse(&[fwd[0].clone()]).unwrap();
        assert_eq!(inv[0].bytes.as_ref(), raw);
    }

    #[test]
    fn single_element_passes_through() {
        let raw = vec![0x42, 0x00];
        let op = IntDelta::new();
        let fwd = op
            .forward(&[plane(raw.clone(), ElementWidth::Word2)])
            .unwrap();
        // First element has no predecessor; it should be emitted verbatim.
        assert_eq!(fwd[0].bytes.as_ref(), raw);
        let inv = op.inverse(&[fwd[0].clone()]).unwrap();
        assert_eq!(inv[0].bytes.as_ref(), raw);
    }

    #[test]
    fn empty_plane_round_trips() {
        let raw: Vec<u8> = vec![];
        let op = IntDelta::new();
        let fwd = op
            .forward(&[plane(raw.clone(), ElementWidth::Word4)])
            .unwrap();
        assert!(fwd[0].bytes.is_empty());
        let inv = op.inverse(&[fwd[0].clone()]).unwrap();
        assert!(inv[0].bytes.is_empty());
    }

    #[test]
    fn rejects_nibble_width() {
        let mut p = plane(vec![0u8; 4], ElementWidth::Nibble);
        p.descriptor.is_nibble_packed = true;
        let op = IntDelta::new();
        assert!(op.forward(&[p.clone()]).is_err());
        assert!(op.inverse(&[p]).is_err());
    }

    #[test]
    fn rejects_misaligned_byte_length() {
        // 3 bytes can't be split into Word2 (2-byte) elements cleanly.
        let p = plane(vec![1u8, 2, 3], ElementWidth::Word2);
        let op = IntDelta::new();
        assert!(op.forward(std::slice::from_ref(&p)).is_err());
        assert!(op.inverse(&[p]).is_err());
    }

    #[test]
    fn rejects_wrong_input_count() {
        let op = IntDelta::new();
        let p = plane(vec![1u8, 2, 3, 4], ElementWidth::Byte);
        assert!(op.forward(&[p.clone(), p.clone()]).is_err());
        assert!(op.inverse(&[]).is_err());
    }

    #[test]
    fn write_params_is_empty() {
        let op = IntDelta::new();
        let mut buf = Vec::new();
        op.write_params(&mut buf);
        assert!(buf.is_empty());
    }

    #[test]
    fn descriptor_passes_through() {
        let d = PlaneDescriptor {
            role: Role::Raw,
            element_width: ElementWidth::Word4,
            length_bytes: 32,
            layout: Layout::Flat,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        };
        let op = IntDelta::new();
        let out = op.propagate_descriptors(std::slice::from_ref(&d)).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], d);
    }
}
