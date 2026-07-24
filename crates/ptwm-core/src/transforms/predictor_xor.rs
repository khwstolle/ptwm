//! `PredictorXor` — FPC-style intra-plane predictor + XOR residual.
//!
//! For each element in a plane, predicts its value from the *previous*
//! values using a small fixed-context-model hash table, then emits
//! `truth XOR prediction`. The output is concentrated around zero
//! (heavy leading-zero distribution) when the underlying values change
//! smoothly — exactly the regime where Huffman / [`crate::codecs::fpc::Fpc`]
//! both win.
//!
//! ## Predictor: fixed-context model (fcm)
//!
//! Maintains a hash table of `1 << HASH_BITS` entries indexed by a
//! rolling hash of recent values. On each step:
//!
//! ```text
//! prediction[i] = table[hash]
//! residual[i]   = value[i] XOR prediction[i]
//! table[hash]   = value[i]
//! hash          = ((hash << SHIFT) ^ value[i]) & ((1 << HASH_BITS) - 1)
//! ```
//!
//! The first element has an all-zero hash context, so the prediction
//! is the table's initial (zero) entry and the residual is the value
//! itself. From then on the predictor learns local patterns. The
//! decoder runs the same recurrence in lock-step to recover the
//! original.
//!
//! ## Single-predictor variant
//!
//! Burtscher's original FPC uses *two* predictors (fcm + dfcm) and
//! emits one bit per element selecting the better. PTWM ships only the
//! fcm half here — the dual-predictor variant adds a per-element side
//! channel that would need its own plane and complicates the chain
//! shape; the single-predictor version captures most of the gain on
//! trained-weight exponent streams and stays byte-exact-invertible
//! through a single byte plane.
//!
//! ## Element widths
//!
//! Same support matrix as [`crate::transforms::IntDelta`]: `Byte`,
//! `Word2`, `Word4`, `Word8`. `Nibble` is rejected.

use std::sync::Arc;

use crate::error::PtwmCoreError;
use crate::transforms::op::{Op, OpId, Plane};
use crate::types::descriptor::{ElementWidth, PlaneDescriptor};

const HASH_BITS: u32 = 16;
const HASH_SIZE: usize = 1 << HASH_BITS;
const HASH_MASK: u64 = (HASH_SIZE - 1) as u64;
const SHIFT: u32 = 6;

// The predictor's hash table is 64 K * 8 B = 512 KiB. Allocating and
// zeroing it on every plane is the dominant cost on planes shorter than
// a few MB, so reuse a per-worker-thread allocation across calls. The
// table is reset (zeroed) at the start of every encode/decode so each
// call sees the same initial state as a fresh allocation would.
thread_local! {
    static TABLE: std::cell::RefCell<Vec<u64>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn with_zeroed_table<R>(f: impl FnOnce(&mut [u64]) -> R) -> R {
    TABLE.with(|cell| {
        let mut t = cell.borrow_mut();
        if t.len() != HASH_SIZE {
            t.clear();
            t.resize(HASH_SIZE, 0);
        } else {
            // Already sized — just zero in place. fill is a memset
            // which is significantly cheaper than reallocation.
            t.fill(0);
        }
        f(&mut t[..])
    })
}

#[derive(Debug, Default, Clone, Copy)]
pub struct PredictorXor;

impl PredictorXor {
    pub const fn new() -> Self {
        Self
    }
}

impl Op for PredictorXor {
    fn propagate_descriptors(
        &self,
        inputs: &[PlaneDescriptor],
    ) -> Result<Vec<PlaneDescriptor>, PtwmCoreError> {
        if inputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "PredictorXor.propagate_descriptors: expected 1 input, got {}",
                inputs.len()
            )));
        }
        let d = &inputs[0];
        reject_nibble(d, "propagate_descriptors")?;
        Ok(vec![d.clone()])
    }

    fn forward(&self, inputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        if inputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "PredictorXor.forward: expected 1 input, got {}",
                inputs.len()
            )));
        }
        let p = &inputs[0];
        reject_nibble(&p.descriptor, "forward")?;
        let width = stride_bytes(p.descriptor.element_width);
        if !p.bytes.len().is_multiple_of(width) {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "PredictorXor.forward: byte length {} is not a multiple of element width {width}",
                p.bytes.len()
            )));
        }
        let out = encode(&p.bytes, width);
        Ok(vec![Plane {
            bytes: Arc::from(out.into_boxed_slice()),
            descriptor: p.descriptor.clone(),
        }])
    }

    fn inverse(&self, outputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        if outputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "PredictorXor.inverse: expected 1 input, got {}",
                outputs.len()
            )));
        }
        let p = &outputs[0];
        reject_nibble(&p.descriptor, "inverse")?;
        let width = stride_bytes(p.descriptor.element_width);
        if !p.bytes.len().is_multiple_of(width) {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "PredictorXor.inverse: byte length {} is not a multiple of element width {width}",
                p.bytes.len()
            )));
        }
        let out = decode(&p.bytes, width);
        Ok(vec![Plane {
            bytes: Arc::from(out.into_boxed_slice()),
            descriptor: p.descriptor.clone(),
        }])
    }

    fn id(&self) -> OpId {
        OpId::PredictorXor
    }

    fn write_params(&self, _out: &mut Vec<u8>) {
        // No parameters: stride from descriptor, hash params are fixed.
    }
}

// ---------------------------------------------------------------------------
// Core encode/decode — element-width-specialized for the same reason
// IntDelta is: this is a hot path used in PRODUCTION_CHAINS.
// ---------------------------------------------------------------------------

fn encode(input: &[u8], width: usize) -> Vec<u8> {
    match width {
        1 => encode_specialized::<1>(input),
        2 => encode_specialized::<2>(input),
        4 => encode_specialized::<4>(input),
        8 => encode_specialized::<8>(input),
        _ => unreachable!("stride_bytes only returns 1/2/4/8"),
    }
}

fn decode(input: &[u8], width: usize) -> Vec<u8> {
    match width {
        1 => decode_specialized::<1>(input),
        2 => decode_specialized::<2>(input),
        4 => decode_specialized::<4>(input),
        8 => decode_specialized::<8>(input),
        _ => unreachable!("stride_bytes only returns 1/2/4/8"),
    }
}

fn encode_specialized<const W: usize>(input: &[u8]) -> Vec<u8> {
    // Empty input is a valid case (occurs on zero-length terminal
    // planes). Short-circuit before the 512 KiB table prep.
    if input.is_empty() {
        return Vec::new();
    }
    // Pre-size the output exactly: residuals are W bytes per element,
    // same width as the input. Pre-sizing lets the inner loop write
    // through indexing instead of growing the Vec one element at a
    // time, which sheds the per-call capacity-check and length-update
    // overhead of extend_from_slice.
    let n = input.len() / W;
    let mut out = vec![0u8; n * W];
    with_zeroed_table(|table| {
        let mut hash: u64 = 0;
        for (i, chunk) in input.chunks_exact(W).enumerate() {
            let value = read_le_u64::<W>(chunk);
            let prediction = table[hash as usize];
            let residual = value ^ prediction;
            out[i * W..(i + 1) * W].copy_from_slice(&residual.to_le_bytes()[..W]);
            table[hash as usize] = value;
            hash = update_hash(hash, value);
        }
    });
    out
}

fn decode_specialized<const W: usize>(input: &[u8]) -> Vec<u8> {
    if input.is_empty() {
        return Vec::new();
    }
    let n = input.len() / W;
    let mut out = vec![0u8; n * W];
    with_zeroed_table(|table| {
        let mut hash: u64 = 0;
        for (i, chunk) in input.chunks_exact(W).enumerate() {
            let residual = read_le_u64::<W>(chunk);
            let prediction = table[hash as usize];
            let value = residual ^ prediction;
            out[i * W..(i + 1) * W].copy_from_slice(&value.to_le_bytes()[..W]);
            table[hash as usize] = value;
            hash = update_hash(hash, value);
        }
    });
    out
}

#[inline]
fn read_le_u64<const W: usize>(chunk: &[u8]) -> u64 {
    let mut buf = [0u8; 8];
    buf[..W].copy_from_slice(chunk);
    u64::from_le_bytes(buf)
}

#[inline]
fn update_hash(prev: u64, value: u64) -> u64 {
    ((prev << SHIFT) ^ value) & HASH_MASK
}

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
            "PredictorXor.{ctx}: nibble-width planes are not supported"
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
    fn byte_round_trip_smooth_data() {
        // Smoothly-increasing byte stream — the regime where the predictor
        // should produce mostly-zero residuals.
        let raw: Vec<u8> = (0..128u8).collect();
        let op = PredictorXor::new();
        let fwd = op
            .forward(&[plane(raw.clone(), ElementWidth::Byte)])
            .unwrap();
        let inv = op.inverse(&[fwd[0].clone()]).unwrap();
        assert_eq!(inv[0].bytes.as_ref(), raw);
    }

    #[test]
    fn byte_round_trip_random_data() {
        // Random byte stream — predictor won't help (residual ≈ value)
        // but round-trip must still be exact.
        let raw: Vec<u8> = (0u8..=255).map(|i| i.wrapping_mul(31)).collect();
        let op = PredictorXor::new();
        let fwd = op
            .forward(&[plane(raw.clone(), ElementWidth::Byte)])
            .unwrap();
        let inv = op.inverse(&[fwd[0].clone()]).unwrap();
        assert_eq!(inv[0].bytes.as_ref(), raw);
    }

    #[test]
    fn word2_round_trip() {
        let vals: Vec<u16> = (0..64).map(|i| i * 17).collect();
        let raw: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let op = PredictorXor::new();
        let fwd = op
            .forward(&[plane(raw.clone(), ElementWidth::Word2)])
            .unwrap();
        let inv = op.inverse(&[fwd[0].clone()]).unwrap();
        assert_eq!(inv[0].bytes.as_ref(), raw);
    }

    #[test]
    fn word4_round_trip() {
        let vals: Vec<u32> = (0..32)
            .map(|i| (i as u32).wrapping_mul(1_000_003))
            .collect();
        let raw: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let op = PredictorXor::new();
        let fwd = op
            .forward(&[plane(raw.clone(), ElementWidth::Word4)])
            .unwrap();
        let inv = op.inverse(&[fwd[0].clone()]).unwrap();
        assert_eq!(inv[0].bytes.as_ref(), raw);
    }

    #[test]
    fn word8_round_trip() {
        let vals: Vec<u64> = (0..16)
            .map(|i| (i as u64).wrapping_mul(0xDEADBEEF_12345678))
            .collect();
        let raw: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let op = PredictorXor::new();
        let fwd = op
            .forward(&[plane(raw.clone(), ElementWidth::Word8)])
            .unwrap();
        let inv = op.inverse(&[fwd[0].clone()]).unwrap();
        assert_eq!(inv[0].bytes.as_ref(), raw);
    }

    #[test]
    fn empty_plane_round_trips() {
        let op = PredictorXor::new();
        let fwd = op.forward(&[plane(vec![], ElementWidth::Word4)]).unwrap();
        assert!(fwd[0].bytes.is_empty());
        let inv = op.inverse(&[fwd[0].clone()]).unwrap();
        assert!(inv[0].bytes.is_empty());
    }

    #[test]
    fn repeated_short_pattern_eventually_predicts_correctly() {
        // The fcm predictor needs the *hash* to cycle into a state
        // where the stored value at that slot matches the next value.
        // A short repeating pattern produces a short hash cycle, so
        // after one full cycle of warmup the residual collapses to
        // zero. Use a length-4 pattern over many cycles so the
        // hash-table converges well within the test budget.
        let pattern = [0x10u8, 0x20, 0x30, 0x40];
        let raw: Vec<u8> = pattern.iter().copied().cycle().take(4096).collect();
        let op = PredictorXor::new();
        let fwd = op
            .forward(&[plane(raw.clone(), ElementWidth::Byte)])
            .unwrap();
        // After the predictor stabilises the residual tail must be
        // mostly zero. Don't enforce a hard threshold — only that the
        // residual is *better* than passing the input through.
        let tail = &fwd[0].bytes[2048..];
        let zero_ratio = tail.iter().filter(|&&b| b == 0).count() as f64 / tail.len() as f64;
        assert!(
            zero_ratio > 0.5,
            "expected >50% zero residuals in tail, got {zero_ratio:.2}",
        );
        let inv = op.inverse(&[fwd[0].clone()]).unwrap();
        assert_eq!(inv[0].bytes.as_ref(), raw);
    }

    #[test]
    fn rejects_nibble_width() {
        let mut p = plane(vec![0u8; 4], ElementWidth::Nibble);
        p.descriptor.is_nibble_packed = true;
        let op = PredictorXor::new();
        assert!(op.forward(&[p.clone()]).is_err());
        assert!(op.inverse(&[p]).is_err());
    }

    #[test]
    fn rejects_misaligned_byte_length() {
        let p = plane(vec![1u8, 2, 3], ElementWidth::Word2);
        let op = PredictorXor::new();
        assert!(op.forward(std::slice::from_ref(&p)).is_err());
        assert!(op.inverse(&[p]).is_err());
    }

    #[test]
    fn rejects_wrong_input_count() {
        let op = PredictorXor::new();
        let p = plane(vec![1u8, 2, 3, 4], ElementWidth::Byte);
        assert!(op.forward(&[p.clone(), p.clone()]).is_err());
        assert!(op.inverse(&[]).is_err());
    }

    #[test]
    fn write_params_is_empty() {
        let op = PredictorXor::new();
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
        let op = PredictorXor::new();
        let out = op.propagate_descriptors(std::slice::from_ref(&d)).unwrap();
        assert_eq!(out, vec![d]);
    }
}
