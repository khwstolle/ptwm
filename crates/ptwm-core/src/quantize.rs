//! Shared PMF quantization + validation primitives. Used by
//! PerGroupCodebook (PGC, alphabet=16, total=256) and Order1ScaleAC
//! (alphabet=256, total=4096); each codec passes its own `total` constant.

use crate::error::PtwmCoreError;

pub const PGC_QUANT_TOTAL: u32 = 256;
pub const ORDER1_QUANT_TOTAL: u32 = 4096;

/// Quantize raw integer counts into u16 counts summing to exactly `total`.
/// Applies Laplace +1 smoothing first (every entry becomes ≥1, guaranteeing
/// no zero-prob symbol on encode/decode).
pub fn quantize_pmf(raw_counts: &[u32], total: u32) -> Vec<u16> {
    let n = raw_counts.len();
    if n == 0 {
        return Vec::new();
    }
    let smoothed: Vec<f64> = raw_counts.iter().map(|&c| (c + 1) as f64).collect();
    let sum: f64 = smoothed.iter().sum();
    let total_f = total as f64;

    let mut out: Vec<u16> = smoothed
        .iter()
        .map(|&s| ((s / sum) * total_f).round().max(1.0) as u16)
        .collect();
    let mut current: u32 = out.iter().map(|&x| x as u32).sum();

    while current > total {
        let (argmax, _) = out.iter().enumerate().max_by_key(|&(_, &v)| v).unwrap();
        if out[argmax] > 1 {
            out[argmax] -= 1;
            current -= 1;
        } else {
            break;
        }
    }
    while current < total {
        let (argmin, _) = out.iter().enumerate().min_by_key(|&(_, &v)| v).unwrap();
        out[argmin] += 1;
        current += 1;
    }
    out
}

/// Verify that quantized counts sum to exactly `expected_total` AND that
/// every entry is at least 1. Returns a `CodecDecode` error on mismatch
/// (state-load corruption tripwire). Zero-probability symbols are forbidden
/// because the range coder divides by `prob` on encode and would loop
/// forever on `prob == 0` during decode renormalization.
pub fn validate_pmf_sum(
    pmf: &[u16],
    expected_total: u32,
    ctx: &'static str,
) -> Result<(), PtwmCoreError> {
    for (i, &p) in pmf.iter().enumerate() {
        if p == 0 {
            return Err(PtwmCoreError::CodecDecode {
                codec: ctx,
                msg: format!("PMF entry {i} is zero (zero-prob symbols forbidden)"),
            });
        }
    }
    let sum: u32 = pmf.iter().map(|&x| x as u32).sum();
    if sum != expected_total {
        return Err(PtwmCoreError::CodecDecode {
            codec: ctx,
            msg: format!("PMF sum {sum} != expected {expected_total}"),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_uniform() {
        let counts = vec![10u32; 16];
        let pmf = quantize_pmf(&counts, 256);
        assert_eq!(pmf.iter().map(|&x| x as u32).sum::<u32>(), 256);
        for &x in &pmf {
            assert!(x >= 1);
        }
    }

    #[test]
    fn quantize_skewed_total_4096() {
        let counts = vec![1000u32, 50, 0, 0, 5];
        let pmf = quantize_pmf(&counts, 4096);
        assert_eq!(pmf.iter().map(|&x| x as u32).sum::<u32>(), 4096);
        for &x in &pmf {
            assert!(x >= 1);
        }
        assert!(pmf[0] > pmf[1]);
        assert!(pmf[1] > pmf[2]);
    }

    #[test]
    fn quantize_single_dominant() {
        let mut counts = vec![0u32; 256];
        counts[42] = 1_000_000;
        let pmf = quantize_pmf(&counts, 4096);
        assert_eq!(pmf.iter().map(|&x| x as u32).sum::<u32>(), 4096);
        assert!(pmf[42] > 3000);
        for (i, &x) in pmf.iter().enumerate() {
            if i != 42 {
                assert_eq!(x, 1);
            }
        }
    }

    #[test]
    fn quantize_empty_input() {
        let pmf = quantize_pmf(&[], 4096);
        assert!(pmf.is_empty());
    }

    #[test]
    fn validate_accepts_correct_sum() {
        let pmf = vec![100u16, 100, 56];
        assert!(validate_pmf_sum(&pmf, 256, "test").is_ok());
    }

    #[test]
    fn validate_rejects_off_by_one() {
        let pmf = vec![100u16, 100, 55];
        assert!(validate_pmf_sum(&pmf, 256, "test").is_err());
    }

    #[test]
    fn validate_rejects_zero_entry() {
        // Sum is correct (256) but entry [1] is zero.
        let pmf = vec![128u16, 0, 128];
        let err = validate_pmf_sum(&pmf, 256, "test").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("zero"), "{msg}");
    }
}
