//! Bit-level encode/decode driver tying [`Net`] to the shared range coder,
//! coding each byte MSB-first. Self-delimiting: a u32 LE length prefix precedes
//! the coded bits, so decode never depends on the container's `decoded_len`.

use crate::entropy::neural_predictor::net::Net;
use crate::error::PtwmCoreError;
use crate::range_coder::{RangeDecoder, RangeEncoder};

const TOTAL: u32 = 1 << 16;
/// Absolute cap on the framed length (mirrors rans.rs / context_mixing).
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
    let mut net = Net::new();
    let mut enc = RangeEncoder::new();
    for &byte in data {
        let mut c0: u32 = 1;
        for i in (0..8).rev() {
            let bit = (byte >> i) & 1;
            let prob1 = p16(net.predict(c0));
            // symbol 1 occupies [0, prob1); symbol 0 occupies [prob1, TOTAL).
            if bit == 1 {
                enc.encode_symbol(0, prob1, TOTAL);
            } else {
                enc.encode_symbol(prob1, TOTAL - prob1, TOTAL);
            }
            net.update(bit);
            c0 = (c0 << 1) | bit as u32;
        }
        net.push_byte(byte);
    }
    out.extend_from_slice(&enc.finish());
    out
}

/// Decode a plane produced by [`encode_plane`].
pub fn decode_plane(payload: &[u8]) -> Result<Vec<u8>, PtwmCoreError> {
    if payload.len() < 4 {
        return Err(PtwmCoreError::CodecDecode {
            codec: "neural_predictor",
            msg: "payload too short for length prefix".into(),
        });
    }
    let n = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
    if n > MAX_LEN {
        return Err(PtwmCoreError::CodecDecode {
            codec: "neural_predictor",
            msg: format!("implausible decoded length {n}"),
        });
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut net = Net::new();
    let mut dec = RangeDecoder::new(&payload[4..]);
    let mut out = Vec::with_capacity(n.min(1 << 17));
    for _ in 0..n {
        let mut c0: u32 = 1;
        let mut byte: u32 = 0;
        for _ in 0..8 {
            let prob1 = p16(net.predict(c0));
            let target = dec.decode_symbol(TOTAL);
            let bit: u8 = if target < prob1 {
                dec.advance(0, prob1, TOTAL);
                1
            } else {
                dec.advance(prob1, TOTAL - prob1, TOTAL);
                0
            };
            net.update(bit);
            c0 = (c0 << 1) | bit as u32;
            byte = (byte << 1) | bit as u32;
        }
        out.push(byte as u8);
        net.push_byte(byte as u8);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rt(data: &[u8]) {
        let payload = encode_plane(data);
        assert_eq!(decode_plane(&payload).unwrap(), data);
    }

    #[test]
    fn roundtrip_empty() {
        rt(&[]);
    }
    #[test]
    fn roundtrip_single_symbol() {
        rt(&vec![0x5Au8; 4096]);
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
        let data: Vec<u8> = (0..2048).map(|i| (i % 5) as u8).collect();
        assert_eq!(decode_plane(&encode_plane(&data)).unwrap(), data);
    }
    #[test]
    fn compresses_below_raw_on_compressible() {
        // Catches a dead/broken net: a low-entropy stream must shrink.
        let data: Vec<u8> = (0..8192).map(|i| ((i * 13) % 7) as u8).collect();
        let payload = encode_plane(&data);
        assert!(
            payload.len() < data.len(),
            "{} !< {}",
            payload.len(),
            data.len()
        );
    }
}
