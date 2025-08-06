//! rANS (range Asymmetric Numeral Systems) entropy codec.
//!
//! Drop-in alternative to the Huffman codec with the same public interface.
//! Uses 4 interleaved 32-bit rANS streams with 11-bit probability precision
//! and byte-granularity renormalization.

mod compress;
mod decompress;

use crate::entropy::outcome::CompressOutcome;
use crate::error::PtwmCoreError;

/// Internal entry point returning the richer [`CompressOutcome`].
/// Callers that need to distinguish "not beneficial" from "incompressible"
/// (e.g. `codec.rs`, which warns on unexpected fallbacks) use this directly.
pub(crate) fn compress_outcome(
    dst: &mut [u8],
    src: &[u8],
) -> Result<CompressOutcome, PtwmCoreError> {
    compress::compress(dst, src)
}

/// Pipeline-style entrypoint: returns the encoded length, or 0 for any
/// non-encoded outcome. The caller (chunk pipeline) treats 0 as "store raw".
pub fn compress(dst: &mut [u8], src: &[u8]) -> Result<usize, PtwmCoreError> {
    Ok(compress_outcome(dst, src)?.encoded_len_or_zero())
}

pub fn decompress(dst: &mut [u8], src: &[u8]) -> Result<usize, PtwmCoreError> {
    let ret = decompress::decompress(dst, src)?;
    if ret != dst.len() {
        return Err(PtwmCoreError::RansDecompress(format!(
            "rANS decompress size mismatch: got {ret}, expected {}",
            dst.len()
        )));
    }
    Ok(ret)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compress_decompress_roundtrip() {
        let src: Vec<u8> = (0..4096).map(|i| (i % 64) as u8).collect();
        let mut compressed = vec![0u8; src.len()];
        let comp_size = compress(&mut compressed, &src).unwrap();
        assert!(comp_size > 0, "data should be compressible");
        assert!(comp_size < src.len());

        let mut decompressed = vec![0u8; src.len()];
        let decomp_size = decompress(&mut decompressed, &compressed[..comp_size]).unwrap();
        assert_eq!(decomp_size, src.len());
        assert_eq!(decompressed, src);
    }

    #[test]
    fn reciprocal_roundtrip_diverse_distributions() {
        // Exercise the encode reciprocal across many frequency shapes (and thus
        // many freq values, incl. 1, powers of two, and the max ~2047): uniform
        // full alphabet, skewed, pseudo-random, two-symbol, three-symbol. The
        // debug_assert in encode_stream_into runs during every compress (even
        // non-beneficial), verifying the reciprocal equals the reference
        // div/mod on every symbol; the roundtrip checks decode where the data
        // actually compressed.
        let cases: Vec<Vec<u8>> = vec![
            (0..=255u8).cycle().take(8192).collect(),
            (0..8192)
                .map(|i| if i % 17 == 0 { (i % 256) as u8 } else { 0 })
                .collect(),
            (0..8192u64)
                .map(|i| (i.wrapping_mul(2654435761) % 256) as u8)
                .collect(),
            std::iter::repeat_n(0u8, 4000)
                .chain(std::iter::repeat_n(255u8, 4192))
                .collect(),
            (0..8192).map(|i| (i % 3) as u8).collect(),
        ];
        for (k, src) in cases.iter().enumerate() {
            let mut compressed = vec![0u8; src.len() + 4096];
            let comp = compress(&mut compressed, src).unwrap();
            if comp > 0 {
                let mut out = vec![0u8; src.len()];
                let n = decompress(&mut out, &compressed[..comp]).unwrap();
                assert_eq!(n, src.len(), "case {k} length");
                assert_eq!(&out, src, "case {k} roundtrip mismatch");
            }
        }
    }

    #[test]
    fn compress_decompress_high_byte_values() {
        let mut src = vec![0u8; 3000];
        src.extend_from_slice(&[200u8; 600]);
        src.extend_from_slice(&[128u8; 200]);
        src.extend_from_slice(&[255u8; 100]);

        let mut compressed = vec![0u8; src.len()];
        let comp_size = compress(&mut compressed, &src).unwrap();
        assert!(comp_size > 0);
        assert!(comp_size < src.len());

        let mut decompressed = vec![0u8; src.len()];
        let decomp_size = decompress(&mut decompressed, &compressed[..comp_size]).unwrap();
        assert_eq!(decomp_size, src.len());
        assert_eq!(decompressed, src);
    }

    #[test]
    fn compress_incompressible_returns_zero() {
        // Near-uniform data should be incompressible.
        let src: Vec<u8> = (0..=255).cycle().take(256 * 4).map(|b| b as u8).collect();
        let mut dst = vec![0u8; src.len()];
        let ret = compress(&mut dst, &src).unwrap();
        assert_eq!(ret, 0);
    }

    #[test]
    fn compress_empty_returns_zero() {
        let mut dst = vec![0u8; 64];
        let ret = compress(&mut dst, &[]).unwrap();
        assert_eq!(ret, 0);
    }

    #[test]
    fn compress_single_symbol_roundtrip() {
        let src = vec![42u8; 1024];
        let mut dst = vec![0u8; src.len()];
        let ret = compress(&mut dst, &src).unwrap();
        assert!(ret > 0 && ret < src.len());

        let mut dec = vec![0u8; src.len()];
        let dec_ret = decompress(&mut dec, &dst[..ret]).unwrap();
        assert_eq!(dec_ret, src.len());
        assert_eq!(dec, src);
    }

    #[test]
    fn compress_two_symbols_roundtrip() {
        let mut src = vec![0u8; 3000];
        src.extend_from_slice(&[1u8; 1000]);
        let mut compressed = vec![0u8; src.len()];
        let comp_size = compress(&mut compressed, &src).unwrap();
        assert!(comp_size > 0);

        let mut decompressed = vec![0u8; src.len()];
        let decomp_size = decompress(&mut decompressed, &compressed[..comp_size]).unwrap();
        assert_eq!(decomp_size, src.len());
        assert_eq!(decompressed, src);
    }

    #[test]
    fn roundtrip_large_data() {
        let src: Vec<u8> = (0..65536).map(|i| (i % 32) as u8).collect();
        let mut compressed = vec![0u8; src.len()];
        let comp_size = compress(&mut compressed, &src).unwrap();
        assert!(comp_size > 0);

        let mut decompressed = vec![0u8; src.len()];
        let decomp_size = decompress(&mut decompressed, &compressed[..comp_size]).unwrap();
        assert_eq!(decomp_size, src.len());
        assert_eq!(decompressed, src);
    }

    #[test]
    fn roundtrip_odd_size_not_divisible_by_4() {
        let src: Vec<u8> = (0..4093).map(|i| (i % 50) as u8).collect();
        let mut compressed = vec![0u8; src.len()];
        let comp_size = compress(&mut compressed, &src).unwrap();
        assert!(comp_size > 0);

        let mut decompressed = vec![0u8; src.len()];
        decompress(&mut decompressed, &compressed[..comp_size]).unwrap();
        assert_eq!(decompressed, src);
    }

    #[test]
    fn roundtrip_small_compressible_data() {
        let src: Vec<u8> = (0..64).map(|_| 0u8).chain((0..64).map(|_| 1u8)).collect();
        let mut compressed = vec![0u8; src.len()];
        let comp_size = compress(&mut compressed, &src).unwrap();
        if comp_size > 0 {
            let mut decompressed = vec![0u8; src.len()];
            decompress(&mut decompressed, &compressed[..comp_size]).unwrap();
            assert_eq!(decompressed, src);
        }
    }

    #[test]
    fn decompress_empty_input_errors() {
        let mut dst = vec![0u8; 64];
        assert!(decompress(&mut dst, &[]).is_err());
    }

    #[test]
    fn decompress_random_garbage_does_not_panic() {
        // Pure noise must return Err; under no circumstances may the decoder panic.
        let src: Vec<u8> = (0..4096).map(|i| ((i * 31 + 7) & 0xff) as u8).collect();
        let mut dst = vec![0u8; 1024];
        let result = decompress(&mut dst, &src);
        assert!(result.is_err());
    }

    #[test]
    fn decompress_truncated_valid_stream_errors() {
        // Use skewed data that rans can actually compress (uniform input
        // would trigger the incompressible-fallback and return 0).
        let mut data = vec![0u8; 4096];
        for (i, b) in data.iter_mut().enumerate() {
            *b = if i % 17 == 0 { (i & 0xff) as u8 } else { 0 };
        }
        let mut enc = vec![0u8; 2 * data.len() + 4096];
        let n = compress(&mut enc, &data).unwrap();
        if n < 32 {
            // Incompressible input: skip — this is a decoder-robustness test,
            // not a compressor-efficiency test.
            return;
        }
        let truncated = &enc[..n / 2];
        let mut dst = vec![0u8; data.len()];
        let result = decompress(&mut dst, truncated);
        assert!(result.is_err());
    }

    #[test]
    fn decompress_all_zero_src_errors_without_panic() {
        let src = vec![0u8; 256];
        let mut dst = vec![0u8; 256];
        let result = decompress(&mut dst, &src);
        assert!(result.is_err());
    }

    /// A low-entropy plane >256 KB must compress (regression guard for the
    /// old u16 per-stream jump-table cap, which silently bailed via `Ok(0)`
    /// once a stream exceeded 65 535 bytes).
    #[test]
    fn roundtrip_2mb_low_entropy() {
        let mut src = vec![0u8; 2 * 1024 * 1024];
        for (i, b) in src.iter_mut().enumerate() {
            *b = if i % 10 == 0 {
                ((i >> 3) & 0x0F) as u8
            } else {
                0
            };
        }
        let mut compressed = vec![0u8; 2 * src.len() + 4096];
        let comp_size = compress(&mut compressed, &src).unwrap();
        assert!(
            comp_size > 0 && comp_size < src.len() / 2,
            "low-entropy 2 MB should compress well, got {comp_size}"
        );

        let mut decompressed = vec![0u8; src.len()];
        decompress(&mut decompressed, &compressed[..comp_size]).unwrap();
        assert_eq!(decompressed, src);
    }
}
