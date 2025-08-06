//! Entropy: measurement and coding.
//!
//! This module gathers everything PTWM does at the *information-theoretic*
//! layer of compression:
//!
//! - [`shannon`] / [`histogram`] — measure per-plane entropy in bits/byte.
//!   The PPG layer uses these to decide whether a chain is worth running;
//!   the explorer uses them to score candidate plane factorisations.
//! - [`huffman`] and [`rans`] — two pure-Rust byte-oriented entropy coders
//!   with the same public interface (`compress` / `decompress` /
//!   `compress_outcome`) and a peer position in the dispatcher's per-plane
//!   trial encode. Both share the 4-stream interleaved framing in
//!   [`stream_frame`] and return the [`outcome::CompressOutcome`] enum so
//!   the plane-codec wrapper can distinguish "encoded N bytes" from
//!   "not beneficial" / "incompressible" / "dst too small" / "len overflow".
//!
//! Neither codec is the default. The dispatcher trial-encodes every plane
//! against both (plus Identity and Zstd) and keeps the smallest payload,
//! so the per-plane winner is determined by the data shape rather than a
//! configuration choice.

pub mod arithmetic;
pub mod context_mixing;
pub mod huff_llm;
pub mod huffman;
pub mod neural_predictor;
pub mod outcome;
pub mod rans;
pub mod stream_frame;
pub mod tans;

/// Byte-frequency histogram for `data`. Saturates at `u32::MAX` per bucket —
/// fine for any realistic tensor (`u32::MAX` bytes ≈ 4 GiB).
pub fn histogram(data: &[u8]) -> [u32; 256] {
    let mut h = [0u32; 256];
    for &b in data {
        h[b as usize] = h[b as usize].saturating_add(1);
    }
    h
}

/// Base-2 Shannon entropy of `data` in bits/byte. Empty input yields 0.0.
pub fn shannon(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let hist = histogram(data);
    let total = data.len() as f64;
    let mut entropy = 0.0f64;
    for count in hist {
        if count != 0 {
            let p = f64::from(count) / total;
            entropy -= p * p.log2();
        }
    }
    entropy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_is_zero() {
        assert_eq!(shannon(&[]), 0.0);
    }

    #[test]
    fn single_value_is_zero() {
        let data = vec![42u8; 1024];
        assert_eq!(shannon(&data), 0.0);
    }

    #[test]
    fn uniform_is_eight() {
        let data: Vec<u8> = (0..=255).cycle().take(256 * 16).collect();
        let h = shannon(&data);
        assert!(
            (h - 8.0).abs() < 1e-9,
            "uniform entropy should be 8.0, got {h}"
        );
    }

    #[test]
    fn two_equal_symbols_is_one() {
        let mut data = vec![0u8; 512];
        data.extend(std::iter::repeat_n(1u8, 512));
        let h = shannon(&data);
        assert!((h - 1.0).abs() < 1e-9, "expected 1.0, got {h}");
    }

    #[test]
    fn histogram_counts_correctly() {
        let data = vec![0u8, 0, 1, 2, 2, 2];
        let h = histogram(&data);
        assert_eq!(h[0], 2);
        assert_eq!(h[1], 1);
        assert_eq!(h[2], 3);
        assert_eq!(h[3], 0);
    }
}
