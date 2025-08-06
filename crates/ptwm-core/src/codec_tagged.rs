//! Codec-agnostic per-plane encode / decode.
//!
//! Exposes the entropy coders (Huffman, rANS, ZSTD, and IDENTITY) behind a
//! uniform `encode(data) -> bytes` / `decode(blob, expected_len) -> bytes`
//! signature so callers can treat them as swappable implementations.
//!
//! Unlike the whole-tensor pipeline in `lib.rs`, these functions operate
//! on a single byte buffer: no header, no preprocessing, no chunking.

use crate::entropy::outcome::CompressOutcome;
use crate::entropy::{huffman, rans};
use crate::error::PtwmCoreError;

// --- Identity ---------------------------------------------------------------

pub fn identity_encode(data: &[u8]) -> Vec<u8> {
    data.to_vec()
}

pub fn identity_decode(blob: &[u8], expected_len: usize) -> Result<Vec<u8>, PtwmCoreError> {
    if blob.len() != expected_len {
        return Err(PtwmCoreError::BufferTooSmall {
            expected: expected_len,
            got: blob.len(),
        });
    }
    Ok(blob.to_vec())
}

// --- Tagged-blob helpers (Huffman / rANS) ----------------------------------
//
// Both `huffman::compress` and `rans::compress` return `Ok(0)` when entropy
// coding cannot produce a useful result (e.g. single-symbol input, tree
// construction failure, output-buffer overflow). The codec then stores the
// input verbatim. A one-byte tag keeps decode self-describing: 0 = raw
// passthrough, 1 = entropy-coded.
//
// This mirrors the `comp_type` tagging the full chunk pipeline uses, just
// at the per-plane granularity.

const TAG_RAW: u8 = 0;
const TAG_ENCODED: u8 = 1;

fn tagged_raw(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + data.len());
    out.push(TAG_RAW);
    out.extend_from_slice(data);
    out
}

fn tagged_encoded(encoded: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + encoded.len());
    out.push(TAG_ENCODED);
    out.extend_from_slice(encoded);
    out
}

/// Log when an entropy coder bails out of its normal encoded path. Silent
/// `Ok(0)` fallbacks previously masked codec regressions; logging the
/// reason makes future regressions visible.
///
/// Severity splits by cause: normal-but-unprofitable compression
/// (`NotBeneficial`, `Incompressible`) logs at `debug` — expected on
/// high-entropy or degenerate inputs — while buffer / capacity errors
/// (`DstTooSmall`, `LenOverflow`) log at `warn` because a caller that
/// sizes `dst` generously (as `codec.rs` does via `huffman_output_bound`)
/// should never hit them.
fn log_fallback(codec: &'static str, outcome: CompressOutcome, input_len: usize) {
    match outcome {
        CompressOutcome::Encoded(_) => {}
        CompressOutcome::NotBeneficial => tracing::debug!(
            codec,
            input_len,
            reason = "not beneficial",
            "entropy coder falling back to tagged-raw"
        ),
        CompressOutcome::Incompressible => tracing::debug!(
            codec,
            input_len,
            reason = "incompressible",
            "entropy coder falling back to tagged-raw"
        ),
        CompressOutcome::DstTooSmall => tracing::warn!(
            codec,
            input_len,
            reason = "dst buffer too small",
            "entropy coder falling back to tagged-raw"
        ),
        CompressOutcome::LenOverflow => tracing::warn!(
            codec,
            input_len,
            reason = "encoded length exceeds u32::MAX",
            "entropy coder falling back to tagged-raw"
        ),
    }
}

fn decode_tagged<F>(
    blob: &[u8],
    expected_len: usize,
    codec_name: &'static str,
    inner: F,
) -> Result<Vec<u8>, PtwmCoreError>
where
    F: FnOnce(&[u8]) -> Result<Vec<u8>, PtwmCoreError>,
{
    if expected_len == 0 {
        return Ok(Vec::new());
    }
    let Some((&tag, rest)) = blob.split_first() else {
        return Err(PtwmCoreError::CodecDecode {
            codec: codec_name,
            msg: "empty blob".to_string(),
        });
    };
    match tag {
        TAG_RAW => {
            if rest.len() != expected_len {
                return Err(PtwmCoreError::BufferTooSmall {
                    expected: expected_len,
                    got: rest.len(),
                });
            }
            Ok(rest.to_vec())
        }
        TAG_ENCODED => inner(rest),
        other => Err(PtwmCoreError::CodecDecode {
            codec: codec_name,
            msg: format!("unknown tag byte {other:#04x}"),
        }),
    }
}

// --- Huffman ---------------------------------------------------------------

/// Upper bound for Huffman output given the input length. The coder
/// prepends a per-stream header table (~1 KiB) and at worst expands
/// incompressible data by a small factor; a generous bound avoids
/// reallocations.
fn huffman_output_bound(input_len: usize) -> usize {
    input_len.saturating_mul(2).saturating_add(4096)
}

pub fn huffman_encode(data: &[u8]) -> Result<Vec<u8>, PtwmCoreError> {
    if data.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = vec![0u8; huffman_output_bound(data.len())];
    match huffman::compress_outcome(&mut out, data)? {
        CompressOutcome::Encoded(n) => Ok(tagged_encoded(&out[..n])),
        outcome => {
            log_fallback("huffman", outcome, data.len());
            Ok(tagged_raw(data))
        }
    }
}

pub fn huffman_decode(blob: &[u8], expected_len: usize) -> Result<Vec<u8>, PtwmCoreError> {
    decode_tagged(blob, expected_len, "huffman", |encoded| {
        let mut out = vec![0u8; expected_len];
        huffman::decompress(&mut out, encoded)?;
        Ok(out)
    })
}

// --- rANS ------------------------------------------------------------------

fn rans_output_bound(input_len: usize) -> usize {
    // rANS uses the same 4-stream interleaved framing as Huffman and has a
    // comparable worst-case expansion.
    input_len.saturating_mul(2).saturating_add(4096)
}

pub fn rans_encode(data: &[u8]) -> Result<Vec<u8>, PtwmCoreError> {
    if data.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = vec![0u8; rans_output_bound(data.len())];
    match rans::compress_outcome(&mut out, data)? {
        CompressOutcome::Encoded(n) => Ok(tagged_encoded(&out[..n])),
        outcome => {
            log_fallback("rans", outcome, data.len());
            Ok(tagged_raw(data))
        }
    }
}

pub fn rans_decode(blob: &[u8], expected_len: usize) -> Result<Vec<u8>, PtwmCoreError> {
    decode_tagged(blob, expected_len, "rans", |encoded| {
        let mut out = vec![0u8; expected_len];
        rans::decompress(&mut out, encoded)?;
        Ok(out)
    })
}

// --- ZSTD ------------------------------------------------------------------

#[cfg(feature = "codec-zstd")]
pub fn zstd_encode(data: &[u8], level: i32) -> Result<Vec<u8>, PtwmCoreError> {
    zstd::encode_all(data, level).map_err(|e| PtwmCoreError::CodecEncode {
        codec: "zstd",
        msg: e.to_string(),
    })
}

#[cfg(feature = "codec-zstd")]
pub fn zstd_decode(blob: &[u8]) -> Result<Vec<u8>, PtwmCoreError> {
    if blob.is_empty() {
        return Ok(Vec::new());
    }
    zstd::decode_all(blob).map_err(|e| PtwmCoreError::CodecDecode {
        codec: "zstd",
        msg: e.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_roundtrip() {
        let data: Vec<u8> = (0..256).map(|i| (i % 97) as u8).collect();
        let blob = identity_encode(&data);
        let back = identity_decode(&blob, data.len()).unwrap();
        assert_eq!(data, back);
    }

    #[test]
    fn identity_decode_size_mismatch() {
        let err = identity_decode(&[1, 2, 3], 5);
        assert!(matches!(err, Err(PtwmCoreError::BufferTooSmall { .. })));
    }

    #[test]
    fn huffman_roundtrip() {
        let data: Vec<u8> = (0..8192).map(|i| (i % 64) as u8).collect();
        let blob = huffman_encode(&data).unwrap();
        let back = huffman_decode(&blob, data.len()).unwrap();
        assert_eq!(data, back);
    }

    #[test]
    fn rans_roundtrip() {
        let data: Vec<u8> = (0..8192).map(|i| (i % 64) as u8).collect();
        let blob = rans_encode(&data).unwrap();
        let back = rans_decode(&blob, data.len()).unwrap();
        assert_eq!(data, back);
    }

    #[test]
    fn empty_input_all_codecs() {
        assert!(huffman_encode(&[]).unwrap().is_empty());
        assert!(rans_encode(&[]).unwrap().is_empty());
        assert!(identity_encode(&[]).is_empty());
        assert!(huffman_decode(&[], 0).unwrap().is_empty());
        assert!(rans_decode(&[], 0).unwrap().is_empty());
    }

    #[test]
    fn huffman_incompressible_falls_back_to_raw() {
        // Near-uniform data that defeats the coder — must round-trip via the
        // tagged-raw fallback path.
        let data: Vec<u8> = (0..2048).map(|i| ((i * 31 + 7) % 251) as u8).collect();
        let blob = huffman_encode(&data).unwrap();
        // Ensure we're exercising the raw-tag path (first byte is 0).
        assert_eq!(blob[0], TAG_RAW);
        let back = huffman_decode(&blob, data.len()).unwrap();
        assert_eq!(data, back);
    }

    #[test]
    fn rans_incompressible_falls_back_to_raw() {
        let data: Vec<u8> = (0..2048).map(|i| ((i * 31 + 7) % 251) as u8).collect();
        let blob = rans_encode(&data).unwrap();
        assert_eq!(blob[0], TAG_RAW);
        let back = rans_decode(&blob, data.len()).unwrap();
        assert_eq!(data, back);
    }

    #[test]
    fn decode_rejects_unknown_tag() {
        let blob = vec![0x7f, 0, 1, 2, 3];
        let err = huffman_decode(&blob, 5);
        assert!(matches!(err, Err(PtwmCoreError::CodecDecode { .. })));
    }

    #[test]
    fn rans_decode_rejects_unknown_tag() {
        let blob = vec![2, 0, 1, 2, 3];
        let err = rans_decode(&blob, 5);
        assert!(matches!(err, Err(PtwmCoreError::CodecDecode { .. })));
    }

    #[test]
    fn huffman_decode_empty_blob_with_nonzero_expected_errors() {
        let err = huffman_decode(&[], 4);
        assert!(matches!(err, Err(PtwmCoreError::CodecDecode { .. })));
    }

    #[test]
    fn rans_decode_empty_blob_with_nonzero_expected_errors() {
        let err = rans_decode(&[], 4);
        assert!(matches!(err, Err(PtwmCoreError::CodecDecode { .. })));
    }

    #[test]
    fn huffman_decode_raw_tag_with_wrong_length_errors() {
        // Tag 0 (raw) + 3 bytes, but expected_len = 5
        let blob = vec![0, 1, 2, 3];
        let err = huffman_decode(&blob, 5);
        assert!(matches!(err, Err(PtwmCoreError::BufferTooSmall { .. })));
    }

    #[test]
    fn huffman_decode_random_garbage_does_not_panic() {
        // Encoded-tag (1) followed by pure nonsense — decoder must return Err, not panic.
        let mut blob = vec![1u8];
        blob.extend((0..512).map(|i| ((i * 17 + 3) & 0xff) as u8));
        let result = huffman_decode(&blob, 256);
        assert!(result.is_err());
    }

    #[test]
    fn rans_decode_random_garbage_does_not_panic() {
        let mut blob = vec![1u8];
        blob.extend((0..512).map(|i| ((i * 31 + 7) & 0xff) as u8));
        let result = rans_decode(&blob, 256);
        assert!(result.is_err());
    }

    #[test]
    fn identity_decode_larger_than_expected_errors() {
        let err = identity_decode(&[1, 2, 3, 4, 5], 3);
        assert!(matches!(err, Err(PtwmCoreError::BufferTooSmall { .. })));
    }

    #[cfg(feature = "codec-zstd")]
    #[test]
    fn zstd_roundtrip() {
        let data: Vec<u8> = (0..8192).map(|i| (i % 64) as u8).collect();
        let blob = zstd_encode(&data, 3).unwrap();
        let back = zstd_decode(&blob).unwrap();
        assert_eq!(data, back);
    }

    #[cfg(feature = "codec-zstd")]
    #[test]
    fn zstd_empty_roundtrip() {
        let blob = zstd_encode(&[], 3).unwrap();
        assert!(zstd_decode(&blob).unwrap().is_empty());
    }
}
