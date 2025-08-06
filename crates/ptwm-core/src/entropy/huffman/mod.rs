//! Pure-Rust canonical Huffman codec.
//!
//! Uses a canonical-code layout with a nibble-packed weight table and
//! four interleaved 32-bit backward bitstreams (with fused decode in
//! `decompress.rs`).

mod bitstream;
mod compress;
mod decompress;
mod tree;
mod weights;

pub use tree::MAX_TABLE_LOG;

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
        return Err(PtwmCoreError::HuffmanDecompress(format!(
            "HUFF0 decompress size mismatch: got {ret}, expected {}",
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
    fn compress_decompress_high_byte_values() {
        // Exercises inputs with byte values >= 127. Previously any input whose
        // max symbol index exceeded 127 hit an early cap and returned Ok(0).
        // Uses a skewed distribution so the data is actually compressible.
        let mut src = vec![0u8; 3000]; // dominant symbol
        src.extend_from_slice(&[200u8; 600]); // high byte value > 127
        src.extend_from_slice(&[128u8; 200]); // another high byte value
        src.extend_from_slice(&[255u8; 100]); // max byte value
        // max_symbol_index = 255 → weights_prefix().len() = 256 → extended header

        let mut compressed = vec![0u8; src.len()];
        let comp_size = compress(&mut compressed, &src).unwrap();
        assert!(
            comp_size > 0,
            "skewed high-byte data should be compressible"
        );
        assert!(comp_size < src.len());

        let mut decompressed = vec![0u8; src.len()];
        let decomp_size = decompress(&mut decompressed, &compressed[..comp_size]).unwrap();
        assert_eq!(decomp_size, src.len());
        assert_eq!(decompressed, src);
    }

    #[test]
    fn compress_incompressible_returns_zero() {
        let src: Vec<u8> = (0..32).map(|i| (i * 137 + 42) as u8).collect();
        let mut dst = vec![0u8; src.len()];
        let ret = compress(&mut dst, &src).unwrap();
        assert_eq!(ret, 0);
    }

    #[test]
    fn malformed_payload_errors() {
        let mut dst = vec![0u8; 64];
        let err = decompress(&mut dst, &[0x80]).unwrap_err();
        assert!(err.to_string().contains("weight") || err.to_string().contains("truncated"));
    }

    // --- Edge-case coverage below ---

    #[test]
    fn compress_single_symbol_roundtrip() {
        // 1-symbol input compresses via 1-bit Huffman code.
        let src = vec![42u8; 1024];
        let mut dst = vec![0u8; src.len()];
        let ret = compress(&mut dst, &src).unwrap();
        assert!(ret > 0 && ret < src.len(), "single-symbol should compress");

        let mut dec = vec![0u8; src.len()];
        let dec_ret = decompress(&mut dec, &dst[..ret]).unwrap();
        assert_eq!(dec_ret, src.len());
        assert_eq!(dec, src);
    }

    #[test]
    fn compress_two_symbols_roundtrip() {
        // Minimal compressible case: 2 distinct symbols, skewed.
        let mut src = vec![0u8; 3000];
        src.extend_from_slice(&[1u8; 1000]);
        let mut compressed = vec![0u8; src.len()];
        let comp_size = compress(&mut compressed, &src).unwrap();
        assert!(comp_size > 0, "2-symbol skewed data should compress");

        let mut decompressed = vec![0u8; src.len()];
        let decomp_size = decompress(&mut decompressed, &compressed[..comp_size]).unwrap();
        assert_eq!(decomp_size, src.len());
        assert_eq!(decompressed, src);
    }

    #[test]
    fn compress_uniform_256_returns_zero() {
        // All 256 byte values equally likely → max entropy, incompressible.
        let src: Vec<u8> = (0..256).cycle().take(256 * 4).map(|b| b as u8).collect();
        let mut dst = vec![0u8; src.len()];
        let ret = compress(&mut dst, &src).unwrap();
        assert_eq!(ret, 0, "uniform distribution should be incompressible");
    }

    #[test]
    fn compress_empty_returns_zero() {
        let mut dst = vec![0u8; 64];
        let ret = compress(&mut dst, &[]).unwrap();
        assert_eq!(ret, 0);
    }

    #[test]
    fn roundtrip_large_data_exercises_fast_path() {
        // 64 KB of compressible data ensures the 4-symbol unrolled fast
        // path runs for many iterations before handing off to the tail.
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
        // 4093 bytes: split_four gives segments of unequal length,
        // exercising the remainder handling.
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
        // Small data (64 bytes) just above the minimum where Huffman
        // overhead might exceed savings.
        let src: Vec<u8> = (0..64).map(|_| 0u8).chain((0..64).map(|_| 1u8)).collect();
        let mut compressed = vec![0u8; src.len()];
        let comp_size = compress(&mut compressed, &src).unwrap();
        // May or may not compress — just verify no panics and roundtrip if it does.
        if comp_size > 0 {
            let mut decompressed = vec![0u8; src.len()];
            decompress(&mut decompressed, &compressed[..comp_size]).unwrap();
            assert_eq!(decompressed, src);
        }
    }

    #[test]
    fn decompress_truncated_jump_table_errors() {
        // Valid weight header but not enough bytes for the 16-byte jump table.
        // 0x82 = 2 symbols, then 1 nibble byte, then only a few bytes (need 16).
        let mut payload = vec![0x82, 0x11];
        payload.extend_from_slice(&[0u8; 3]); // too short for 16-byte jump table
        let mut dst = vec![0u8; 64];
        let err = decompress(&mut dst, &payload).unwrap_err();
        assert!(
            err.to_string().contains("truncated") || err.to_string().contains("jump table"),
            "expected truncated jump table error, got: {err}"
        );
    }

    #[test]
    fn decompress_corrupted_stream_errors() {
        // Compress valid data, then corrupt the stream bytes.
        let src: Vec<u8> = (0..4096).map(|i| (i % 64) as u8).collect();
        let mut compressed = vec![0u8; src.len()];
        let comp_size = compress(&mut compressed, &src).unwrap();
        assert!(comp_size > 0);

        // Corrupt bytes near the end of the compressed payload.
        let mid = comp_size / 2;
        compressed[mid] ^= 0xFF;
        compressed[mid + 1] ^= 0xFF;

        let mut decompressed = vec![0u8; src.len()];
        // Should either error or produce wrong output — must not panic.
        let result = decompress(&mut decompressed, &compressed[..comp_size]);
        if result.is_ok() {
            // If it didn't error, the output must differ (corruption detected implicitly).
            assert_ne!(decompressed, src, "corrupted stream should not roundtrip");
        }
    }

    #[test]
    fn decompress_empty_input_errors() {
        let mut dst = vec![0u8; 64];
        assert!(decompress(&mut dst, &[]).is_err());
    }

    #[test]
    fn decompress_random_garbage_does_not_panic() {
        // A sweep of pseudo-random byte strings must never panic the decoder.
        for seed in 0u64..16 {
            let src: Vec<u8> = (0..2048)
                .map(|i| ((i as u64 * (seed + 17)).wrapping_mul(31) & 0xff) as u8)
                .collect();
            let mut dst = vec![0u8; 1024];
            let _ = decompress(&mut dst, &src);
        }
    }

    #[test]
    fn decompress_corrupt_weight_table_errors() {
        // First byte of a Huffman payload encodes the weight table's byte
        // length. A value of 0xff claims a huge table that won't fit — the
        // decoder must reject cleanly.
        let payload = [0xff, 0x00, 0x00, 0x00];
        let mut dst = vec![0u8; 64];
        let result = decompress(&mut dst, &payload);
        assert!(result.is_err());
    }

    #[test]
    fn decompress_truncated_valid_stream_errors() {
        // Compress a valid buffer, then cut the encoded stream in half.
        let src: Vec<u8> = (0..4096).map(|i| (i % 64) as u8).collect();
        let mut enc = vec![0u8; src.len() + 4096];
        let n = compress(&mut enc, &src).unwrap();
        assert!(n > 32);
        let mut dst = vec![0u8; src.len()];
        let result = decompress(&mut dst, &enc[..n / 2]);
        assert!(result.is_err());
    }

    /// A low-entropy plane >256 KB must compress (regression guard for the
    /// old u16 per-stream jump-table cap, which silently bailed via `Ok(0)`
    /// once a stream exceeded 65 535 bytes).
    #[test]
    fn roundtrip_2mb_low_entropy() {
        // 2 MB of skewed bytes: ~2.5 bits/byte entropy, plenty of compression
        // headroom. 2 MB / 4 streams = 512 KB per stream — well past the old
        // 65 535-byte u16 cap.
        let mut src = vec![0u8; 2 * 1024 * 1024];
        for (i, b) in src.iter_mut().enumerate() {
            // 90 % zeros, 10 % spread over {1..16}.
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
