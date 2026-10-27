//! Acceptance rule for the decoded length a plane payload declares.
//!
//! rANS, tANS, and the arithmetic coders prefix their payload with the
//! decoded length as a little-endian `u32`, and the decoder sizes its output
//! (or its decode loop) from that prefix before reading the body. A corrupted
//! or hostile prefix must not drive an arbitrarily large allocation, so the
//! decoder checks it against [`accepted`]. Encoders apply the same rule before
//! writing a prefix, so every plane they emit is one their decoder accepts.

/// Decoded length accepted at any compression ratio.
///
/// These coders reach very high ratios on near-degenerate distributions (an
/// all-zero 64 KiB plane encodes to about 30 bytes), so below this size the
/// rule bounds the length alone.
pub(crate) const RATIO_FREE_LEN: usize = 1 << 30; // 1 GiB

/// Largest decoded-to-encoded length ratio accepted above [`RATIO_FREE_LEN`].
///
/// A payload that declares more than [`RATIO_FREE_LEN`] bytes must carry at
/// least `1 / MAX_RATIO` of that length as encoded body, which bounds the
/// allocation a short payload can request. Weight planes this large (a BF16
/// embedding or output projection of a large-vocabulary model) compress far
/// less than this.
pub(crate) const MAX_RATIO: usize = 64;

/// Whether a payload declaring `decoded_len` bytes from an encoded body of
/// `body_len` bytes is accepted.
///
/// Lengths that do not fit the `u32` prefix are never accepted.
pub(crate) fn accepted(decoded_len: usize, body_len: usize) -> bool {
    u32::try_from(decoded_len).is_ok()
        && (decoded_len <= RATIO_FREE_LEN || decoded_len <= body_len.saturating_mul(MAX_RATIO))
}

/// Error when the `u32` length prefix at the start of `prefixed` declares
/// more than `max_len` decoded bytes.
///
/// Used by the bounded decode paths before the decode loop runs. A payload
/// shorter than the prefix passes; the codec's own decoder reports it.
pub(crate) fn check_within(
    prefixed: &[u8],
    max_len: usize,
    codec: &'static str,
) -> Result<(), crate::error::PtwmCoreError> {
    let Some(prefix) = prefixed.first_chunk::<4>() else {
        return Ok(());
    };
    let declared = u32::from_le_bytes(*prefix) as usize;
    if declared > max_len {
        return Err(crate::error::PtwmCoreError::CodecDecode {
            codec,
            msg: format!("payload declares {declared} decoded bytes, bound is {max_len}"),
        });
    }
    Ok(())
}

/// Bytes from a skewed distribution (about 2 bits per byte), long enough to
/// exercise lengths above [`RATIO_FREE_LEN`] while staying far below
/// [`MAX_RATIO`].
#[cfg(test)]
pub(crate) fn skewed_bytes(len: usize) -> Vec<u8> {
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x.trailing_zeros().min(15) as u8
        })
        .collect()
}

/// A payload body shorter than `1 / MAX_RATIO` of the 3 GiB it declares.
#[cfg(test)]
pub(crate) const SHORT_BODY_DECLARED_LEN: u32 = 3 << 30;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn any_ratio_is_accepted_up_to_the_ratio_free_length() {
        assert!(accepted(0, 0));
        assert!(accepted(RATIO_FREE_LEN, 1));
    }

    #[test]
    fn a_large_plane_at_a_weight_like_ratio_is_accepted() {
        // A BF16 embedding of a 128256-token, 8192-wide model, at the ratio
        // rANS reached on it.
        let decoded = 128_256 * 8192 * 2;
        assert!(decoded > RATIO_FREE_LEN);
        assert!(accepted(decoded, 1_611_325_931));
    }

    #[test]
    fn a_short_body_cannot_declare_a_large_plane() {
        let decoded = RATIO_FREE_LEN + 1;
        assert!(!accepted(decoded, 16));
        assert!(!accepted(decoded, decoded / MAX_RATIO - 1));
        assert!(accepted(decoded, decoded.div_ceil(MAX_RATIO)));
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn lengths_beyond_the_u32_prefix_are_rejected() {
        let too_long = u32::MAX as usize + 1;
        assert!(!accepted(too_long, too_long));
        assert!(accepted(u32::MAX as usize, u32::MAX as usize));
    }
}
