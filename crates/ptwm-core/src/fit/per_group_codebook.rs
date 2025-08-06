//! Per-group codebook fitter.
//!
//! Returns one `SharedStateEntry` (or `None` if no VALUE planes were
//! supplied) carrying the K-means-fitted PGC codebook.

use super::SharedStateEntry;
use crate::codec::CodecId;
use crate::codecs::per_group_codebook as pgc;
use crate::error::PtwmCoreError;

/// Hard cap on histograms fed into k-means. k-means cost is
/// O(N · K · iters · ALPHABET); cap keeps fit time bounded without
/// measurably affecting centroid quality on well-clustered nibble PMFs.
const FIT_HIST_SAMPLE_CAP: usize = 200_000;

/// Take a deterministic stride-based sample of `cap` items from `src`.
/// Returns `src` directly if it already has ≤ `cap` items.
fn stride_sample<T: Copy>(src: &[T], cap: usize) -> Vec<T> {
    if src.len() <= cap {
        return src.to_vec();
    }
    // Stride = ceil(N / cap) gives at most `cap + 1` items; clamp.
    let stride = src.len().div_ceil(cap).max(1);
    let mut out = Vec::with_capacity(cap + 1);
    let mut i = 0;
    while i < src.len() && out.len() < cap {
        out.push(src[i]);
        i += stride;
    }
    out
}

pub fn fit(planes_value: &[&[u8]]) -> Result<Option<SharedStateEntry>, PtwmCoreError> {
    if planes_value.is_empty() {
        return Ok(None);
    }
    let mut all_hists: Vec<[u32; pgc::ALPHABET]> = Vec::new();
    for plane in planes_value {
        all_hists.extend(pgc::histograms(plane));
    }
    let fit_hists = stride_sample(&all_hists, FIT_HIST_SAMPLE_CAP);
    let cb = pgc::fit_codebook_multi_seed(
        &fit_hists,
        &[
            pgc::KMEANS_PP_SEED,
            0xC0FFEE,
            0xDEADBEEF,
            0x1337CAFE,
            0xF00DBABE,
        ],
    )?;
    let mut codebook = [[0u32; pgc::ALPHABET]; pgc::K];
    for k in 0..pgc::K {
        codebook[k] = pgc::quantize_centroid(&cb.centroids[k], pgc::QUANT_TOTAL);
    }
    // Byte layout must match `per_group_codebook::deserialize_state`.
    let mut state_bytes = Vec::with_capacity(pgc::K * pgc::ALPHABET);
    for k in 0..pgc::K {
        for v in 0..pgc::ALPHABET {
            state_bytes.push(codebook[k][v] as u8);
        }
    }
    Ok(Some(SharedStateEntry::new(
        CodecId::PerGroupCodebook,
        0,
        0b0001, // VALUE
        "per_group_codebook".to_string(),
        state_bytes,
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_returns_state_when_planes_nonempty() {
        let plane: Vec<u8> = (0..1024u32).map(|i| (i % 16) as u8).collect();
        let entry = fit(&[&plane]).unwrap();
        assert!(entry.is_some());
        let e = entry.unwrap();
        assert_eq!(e.codec_id, CodecId::PerGroupCodebook);
        assert_eq!(e.state_bytes.len(), pgc::K * pgc::ALPHABET);
        assert_eq!(e.name, "per_group_codebook");
    }

    #[test]
    fn empty_inputs_produce_no_entry() {
        let entry = fit(&[]).unwrap();
        assert!(entry.is_none());
    }

    #[test]
    fn stride_sample_returns_input_when_under_cap() {
        let src: Vec<u32> = (0..100).collect();
        let s = stride_sample(&src, 200);
        assert_eq!(s, src);
    }

    #[test]
    fn stride_sample_caps_above_threshold() {
        let src: Vec<u32> = (0..10_000).collect();
        let s = stride_sample(&src, 100);
        assert!(s.len() <= 100);
        // First element preserved, then stride steps.
        assert_eq!(s[0], 0);
        assert_eq!(s[1], 100); // stride == 10000.div_ceil(100) == 100
    }

    #[test]
    fn fit_handles_corpus_above_sample_cap() {
        // 250K groups of 32 nibbles each = 8M nibble bytes. Above the
        // 200K cap, so subsampling kicks in. Smoke test that fit still
        // returns a valid state and doesn't panic.
        let plane: Vec<u8> = (0..(250_000u32 * 32)).map(|i| (i % 16) as u8).collect();
        let entry = fit(&[&plane]).unwrap();
        assert!(entry.is_some());
        let e = entry.unwrap();
        assert_eq!(e.state_bytes.len(), pgc::K * pgc::ALPHABET);
    }
}
