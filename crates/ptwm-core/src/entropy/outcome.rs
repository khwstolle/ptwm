//! Typed return for the [`huffman`][crate::entropy::huffman] and
//! [`rans`][crate::entropy::rans] compressors.
//!
//! The thin public wrappers (`huffman::compress`, `rans::compress`) return
//! `Result<usize, _>` for the pipeline layer and collapse every non-encoded
//! path to `Ok(0)`. The per-plane codec layer needs to distinguish those
//! cases so it can warn on unexpected fallbacks rather than silently
//! emitting an `Ok(0)`. The `compress_outcome` entry points return this
//! richer enum and the plane-codec wrappers inspect it before deciding
//! whether to log.

/// Result of an entropy-coder compress attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressOutcome {
    /// Successful compression: `.0` bytes of the destination hold the blob.
    Encoded(usize),
    /// Compression succeeded but did not shrink the input. The caller
    /// should store the input verbatim. Routine on high-entropy data;
    /// logged at `debug` level.
    NotBeneficial,
    /// The codec could not compress the input (empty, single-symbol
    /// frequency table, tree-construction failure, and so on). The caller
    /// should store the input verbatim. Routine on degenerate inputs;
    /// logged at `debug`.
    Incompressible,
    /// The destination buffer ran out of space somewhere in the codec:
    /// either it could not hold the header, or a stream overran the
    /// remaining capacity mid-encode. Callers either enlarge `dst` or
    /// fall back to raw. Logged at `warn` because a caller that sizes
    /// `dst` from `input_len * 2 + 4096` (as `codec.rs` does) should
    /// never hit it.
    DstTooSmall,
    /// A single stream's compressed length would exceed `u32::MAX` and
    /// would not fit in the jump-table slot. Unreachable from the
    /// pipeline (chunks are ≤ tens of MB) but checked defensively;
    /// logged at `warn`.
    LenOverflow,
}

impl CompressOutcome {
    /// Collapse to the pipeline-style `usize`: the encoded length, or 0 for
    /// any non-encoded outcome.
    #[inline]
    pub fn encoded_len_or_zero(self) -> usize {
        match self {
            CompressOutcome::Encoded(n) => n,
            _ => 0,
        }
    }
}
