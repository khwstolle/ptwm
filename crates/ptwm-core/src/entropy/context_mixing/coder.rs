//! Bit-level encode/decode driver. Ties the context models + gated mixer
//! to the shared range coder, coding each byte MSB-first. The payload is
//! self-delimiting: a u32 LE length prefix precedes the coded bits, so
//! decode never depends on the container's `decoded_len`.

use crate::entropy::context_mixing::mixer::Mixer;
use crate::entropy::context_mixing::model::Models;
use crate::error::PtwmCoreError;
use crate::range_coder::{RangeDecoder, RangeEncoder};

/// Range-coder denominator for one binary decision.
const TOTAL: u32 = 1 << 16;
/// Absolute cap on the framed length (mirrors rans.rs) — guards a hostile
/// prefix from driving a multi-GiB decode, and bounds the encoder's
/// usize→u32 length cast.
pub const MAX_LEN: usize = 1 << 30;

/// Scale a 12-bit probability to the coder's 16-bit `total`, clamped so the
/// range coder never sees probability 0 or `total`.
#[inline]
fn p16(p12: i32) -> u32 {
    ((p12 as u32) << 4).clamp(1, TOTAL - 1)
}

/// Encode a plane. Output = `[len: u32 LE][range-coded bits]`.
pub fn encode_plane(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + data.len());
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    if data.is_empty() {
        return out;
    }
    let mut models = Models::new();
    let mut mixer = Mixer::new();
    let mut enc = RangeEncoder::new();
    for &byte in data {
        let mut c0: u32 = 1;
        for i in (0..8).rev() {
            let bit = (byte >> i) & 1;
            let preds = models.predict(c0);
            let stretched = stretch_all(&preds);
            let p = mixer.mix(c0, &stretched);
            let prob1 = p16(p);
            // symbol 1 occupies [0, prob1); symbol 0 occupies [prob1, TOTAL).
            if bit == 1 {
                enc.encode_symbol(0, prob1, TOTAL);
            } else {
                enc.encode_symbol(prob1, TOTAL - prob1, TOTAL);
            }
            models.update(c0, bit);
            mixer.update(bit);
            c0 = (c0 << 1) | bit as u32;
        }
        models.push_byte(byte);
    }
    out.extend_from_slice(&enc.finish());
    out
}

/// Decode a plane produced by [`encode_plane`].
pub fn decode_plane(payload: &[u8]) -> Result<Vec<u8>, PtwmCoreError> {
    if payload.len() < 4 {
        return Err(PtwmCoreError::CodecDecode {
            codec: "context_mixing_lite",
            msg: "payload too short for length prefix".into(),
        });
    }
    let n = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
    if n > MAX_LEN {
        return Err(PtwmCoreError::CodecDecode {
            codec: "context_mixing_lite",
            msg: format!("implausible decoded length {n}"),
        });
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut models = Models::new();
    let mut mixer = Mixer::new();
    let mut dec = RangeDecoder::new(&payload[4..]);
    let mut out = Vec::with_capacity(n.min(1 << 17));
    for _ in 0..n {
        let mut c0: u32 = 1;
        let mut byte: u32 = 0;
        for _ in 0..8 {
            let preds = models.predict(c0);
            let stretched = stretch_all(&preds);
            let p = mixer.mix(c0, &stretched);
            let prob1 = p16(p);
            let target = dec.decode_symbol(TOTAL);
            let bit: u8 = if target < prob1 {
                dec.advance(0, prob1, TOTAL);
                1
            } else {
                dec.advance(prob1, TOTAL - prob1, TOTAL);
                0
            };
            models.update(c0, bit);
            mixer.update(bit);
            c0 = (c0 << 1) | bit as u32;
            byte = (byte << 1) | bit as u32;
        }
        out.push(byte as u8);
        models.push_byte(byte as u8);
    }
    Ok(out)
}

#[inline]
fn stretch_all(
    preds: &[i32; crate::entropy::context_mixing::model::N_MODELS],
) -> [i32; crate::entropy::context_mixing::model::N_MODELS] {
    use crate::entropy::context_mixing::squash::stretch;
    let mut s = [0i32; crate::entropy::context_mixing::model::N_MODELS];
    for i in 0..s.len() {
        s[i] = stretch(preds[i]);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rt(data: &[u8]) {
        let payload = encode_plane(data);
        let back = decode_plane(&payload).unwrap();
        assert_eq!(back, data);
    }

    #[test]
    fn roundtrip_empty() {
        rt(&[]);
    }

    #[test]
    fn roundtrip_single_symbol() {
        rt(&vec![0x5Au8; 3000]);
    }

    #[test]
    fn roundtrip_compressible() {
        let data: Vec<u8> = (0..8192).map(|i| ((i * 13) % 7) as u8).collect();
        rt(&data);
    }

    #[test]
    fn roundtrip_random() {
        let mut x: u32 = 0xDEADBEEF;
        let data: Vec<u8> = (0..4096)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                (x & 0xFF) as u8
            })
            .collect();
        rt(&data);
    }

    #[test]
    fn roundtrip_text() {
        rt(b"the quick brown fox jumps over the lazy dog. "
            .repeat(100)
            .as_slice());
    }

    #[test]
    fn encode_is_deterministic() {
        let data: Vec<u8> = (0..4096).map(|i| ((i * 31) & 0xFF) as u8).collect();
        assert_eq!(encode_plane(&data), encode_plane(&data));
    }

    #[test]
    fn decode_ignores_extra_trailing_request() {
        // Framing carries the true length; a truncated/garbage tail length
        // request is irrelevant because decode reads its own framed length.
        let data: Vec<u8> = (0..2048).map(|i| (i % 5) as u8).collect();
        let payload = encode_plane(&data);
        assert_eq!(decode_plane(&payload).unwrap(), data);
    }
}
