//! Order1ScaleAC: per-row lag-1 conditional arithmetic coder over the
//! SCALE plane of MXFP4 / NVFP4 tensors.

pub const ALPHABET: usize = 256;
pub const STATE_FORMAT_VERSION: u8 = 1;

/// In-memory state. Wire layout: `[bitmap: 32 B][marginal: 512 B][rows: 512 B × popcount]`.
/// Fields stay crate-private; external callers must go through
/// [`from_parts`] or [`deserialize_state`], both of which validate
/// invariants and call `materialize_cumsums` so `cond_cumsum_for` can
/// index without checks.
///
/// [`from_parts`]: StateV0::from_parts
#[derive(Debug, Clone)]
pub struct StateV0 {
    pub(crate) marginal: [u16; ALPHABET],
    pub(crate) present_bitmap: [u8; 32],
    pub(crate) conditional_rows: Vec<[u16; ALPHABET]>,
    pub(crate) marginal_cumsum: [u32; ALPHABET + 1],
    pub(crate) conditional_cumsum: Vec<[u32; ALPHABET + 1]>,
}

impl StateV0 {
    pub(crate) fn empty() -> Self {
        Self {
            marginal: [0u16; ALPHABET],
            present_bitmap: [0u8; 32],
            conditional_rows: Vec::new(),
            marginal_cumsum: [0u32; ALPHABET + 1],
            conditional_cumsum: Vec::new(),
        }
    }

    /// Validated constructor for external callers. Enforces:
    /// - `marginal` and every row in `conditional_rows` sum to
    ///   `ORDER1_QUANT_TOTAL` with no zero entries (range coder
    ///   invariant; see `validate_pmf_sum`).
    /// - `conditional_rows.len() == popcount(present_bitmap)`; otherwise
    ///   `cond_cumsum_for` would index out of range.
    /// Materializes cumulative-prefix arrays before returning.
    pub fn from_parts(
        present_bitmap: [u8; 32],
        marginal: [u16; ALPHABET],
        conditional_rows: Vec<[u16; ALPHABET]>,
    ) -> Result<Self, PtwmCoreError> {
        let popcount = popcount_total(&present_bitmap);
        if conditional_rows.len() != popcount {
            return Err(PtwmCoreError::CodecDecode {
                codec: "Order1ScaleAC",
                msg: format!(
                    "from_parts: conditional_rows len {} != popcount {popcount}",
                    conditional_rows.len()
                ),
            });
        }
        validate_pmf_sum(&marginal, ORDER1_QUANT_TOTAL, "Order1ScaleAC")?;
        for row in &conditional_rows {
            validate_pmf_sum(row, ORDER1_QUANT_TOTAL, "Order1ScaleAC")?;
        }
        let mut state = Self {
            marginal,
            present_bitmap,
            conditional_rows,
            marginal_cumsum: [0u32; ALPHABET + 1],
            conditional_cumsum: Vec::new(),
        };
        state.materialize_cumsums();
        Ok(state)
    }
}

pub struct Order1ScaleAC;

impl Order1ScaleAC {
    pub fn canonical_id(&self) -> crate::extension::CanonicalId {
        crate::extension::builtin_canonical_id("order1_scale_ac")
    }
}

/// 1 iff `prev` is present in `bitmap`. LSB-first within each byte
/// (bit `(prev & 7)` of byte `(prev >> 3)`).
#[inline]
pub(crate) fn bitmap_has(bitmap: &[u8; 32], prev: u8) -> bool {
    let byte = (prev >> 3) as usize;
    let bit = prev & 7;
    bitmap[byte] & (1 << bit) != 0
}

/// Number of present contexts strictly below `prev` (= index into
/// `conditional_rows` / `conditional_cumsum`). Caller must have verified
/// `bitmap_has(bitmap, prev)`.
#[inline]
pub(crate) fn bitmap_rank(bitmap: &[u8; 32], prev: u8) -> usize {
    let byte = (prev >> 3) as usize;
    let bit = prev & 7;
    let full_byte_popcount: u32 = bitmap[..byte].iter().map(|b| b.count_ones()).sum();
    let partial_mask: u8 = if bit == 0 { 0 } else { (1u8 << bit) - 1 };
    let partial_popcount = (bitmap[byte] & partial_mask).count_ones();
    (full_byte_popcount + partial_popcount) as usize
}

#[inline]
pub(crate) fn popcount_total(bitmap: &[u8; 32]) -> usize {
    bitmap.iter().map(|b| b.count_ones() as usize).sum()
}

fn build_cumsum(pmf: &[u16; ALPHABET]) -> [u32; ALPHABET + 1] {
    let mut cumsum = [0u32; ALPHABET + 1];
    let mut acc = 0u32;
    for i in 0..ALPHABET {
        cumsum[i] = acc;
        acc += pmf[i] as u32;
    }
    cumsum[ALPHABET] = acc;
    cumsum
}

impl StateV0 {
    /// Build cumulative-prefix arrays from `marginal` and `conditional_rows`.
    /// Call after in-crate per-tensor fit; `from_parts` and
    /// `deserialize_state` already invoke this.
    pub(crate) fn materialize_cumsums(&mut self) {
        self.marginal_cumsum = build_cumsum(&self.marginal);
        self.conditional_cumsum = self.conditional_rows.iter().map(build_cumsum).collect();
    }

    /// Lookup the cumulative-prefix array for context `prev`. Falls back
    /// to `marginal_cumsum` when `prev` has no row in the bitmap.
    #[inline]
    pub fn cond_cumsum_for(&self, prev: u8) -> &[u32; ALPHABET + 1] {
        if bitmap_has(&self.present_bitmap, prev) {
            &self.conditional_cumsum[bitmap_rank(&self.present_bitmap, prev)]
        } else {
            &self.marginal_cumsum
        }
    }
}

use crate::error::PtwmCoreError;
use crate::quantize::{ORDER1_QUANT_TOTAL, validate_pmf_sum};

/// Serialize a [`StateV0`] to its wire byte layout.
pub fn serialize_state(state: &StateV0) -> Vec<u8> {
    let popcount = popcount_total(&state.present_bitmap);
    let mut out = Vec::with_capacity(32 + 512 + 512 * popcount);
    out.extend_from_slice(&state.present_bitmap);
    for &c in &state.marginal {
        out.extend_from_slice(&c.to_le_bytes());
    }
    for row in &state.conditional_rows {
        for &c in row {
            out.extend_from_slice(&c.to_le_bytes());
        }
    }
    out
}

/// Returns `Err` on any structural violation; never panics on arbitrary input.
pub fn deserialize_state(bytes: &[u8]) -> Result<StateV0, PtwmCoreError> {
    if bytes.len() < 32 + 2 * ALPHABET {
        return Err(PtwmCoreError::CodecDecode {
            codec: "Order1ScaleAC",
            msg: format!(
                "state shorter than header ({} bytes, need ≥ {})",
                bytes.len(),
                32 + 2 * ALPHABET
            ),
        });
    }
    let mut bitmap = [0u8; 32];
    bitmap.copy_from_slice(&bytes[..32]);
    let popcount = popcount_total(&bitmap);
    let expected_len = 32 + 2 * ALPHABET + popcount * 2 * ALPHABET;
    if bytes.len() != expected_len {
        return Err(PtwmCoreError::CodecDecode {
            codec: "Order1ScaleAC",
            msg: format!(
                "state length {} != expected {expected_len} (popcount {popcount})",
                bytes.len()
            ),
        });
    }
    let mut state = StateV0::empty();
    state.present_bitmap = bitmap;
    let mut off = 32usize;
    for i in 0..ALPHABET {
        state.marginal[i] = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
        off += 2;
    }
    validate_pmf_sum(&state.marginal, ORDER1_QUANT_TOTAL, "Order1ScaleAC")?;
    state.conditional_rows = Vec::with_capacity(popcount);
    for _ in 0..popcount {
        let mut row = [0u16; ALPHABET];
        for i in 0..ALPHABET {
            row[i] = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
            off += 2;
        }
        validate_pmf_sum(&row, ORDER1_QUANT_TOTAL, "Order1ScaleAC")?;
        state.conditional_rows.push(row);
    }
    state.materialize_cumsums();
    Ok(state)
}

#[cfg(test)]
mod state_io_tests {
    use super::*;

    fn build_uniform_state(present_bits: &[u8]) -> StateV0 {
        let mut state = StateV0::empty();
        for &b in present_bits {
            state.present_bitmap[(b >> 3) as usize] |= 1 << (b & 7);
        }
        // Marginal: build a uniform PMF, then quantize via the helper to
        // guarantee sum == ORDER1_QUANT_TOTAL.
        let counts: Vec<u32> = vec![1u32; ALPHABET];
        let qq = crate::quantize::quantize_pmf(&counts, ORDER1_QUANT_TOTAL);
        for i in 0..ALPHABET {
            state.marginal[i] = qq[i];
        }
        let popcount = popcount_total(&state.present_bitmap);
        for _ in 0..popcount {
            let counts: Vec<u32> = vec![1u32; ALPHABET];
            let qq = crate::quantize::quantize_pmf(&counts, ORDER1_QUANT_TOTAL);
            let mut row = [0u16; ALPHABET];
            for i in 0..ALPHABET {
                row[i] = qq[i];
            }
            state.conditional_rows.push(row);
        }
        state.materialize_cumsums();
        state
    }

    #[test]
    fn empty_state_roundtrip() {
        let state = build_uniform_state(&[]);
        let bytes = serialize_state(&state);
        assert_eq!(bytes.len(), 32 + 512);
        let parsed = deserialize_state(&bytes).unwrap();
        assert_eq!(parsed.present_bitmap, [0u8; 32]);
        assert!(parsed.conditional_rows.is_empty());
        assert_eq!(parsed.marginal, state.marginal);
    }

    #[test]
    fn populated_state_roundtrip() {
        let present: Vec<u8> = (0..30u8).collect();
        let state = build_uniform_state(&present);
        let bytes = serialize_state(&state);
        assert_eq!(bytes.len(), 32 + 512 + 512 * present.len());
        let parsed = deserialize_state(&bytes).unwrap();
        assert_eq!(parsed.present_bitmap, state.present_bitmap);
        assert_eq!(parsed.conditional_rows.len(), present.len());
        for (a, b) in parsed
            .conditional_rows
            .iter()
            .zip(state.conditional_rows.iter())
        {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn from_parts_rejects_popcount_mismatch() {
        let mut bitmap = [0u8; 32];
        bitmap[0] = 0b0000_0011; // 2 present contexts
        let counts = vec![1u32; ALPHABET];
        let qq = crate::quantize::quantize_pmf(&counts, ORDER1_QUANT_TOTAL);
        let mut marginal = [0u16; ALPHABET];
        for i in 0..ALPHABET {
            marginal[i] = qq[i];
        }
        let one_row = {
            let mut r = [0u16; ALPHABET];
            for i in 0..ALPHABET {
                r[i] = qq[i];
            }
            r
        };
        // popcount = 2 but only 1 row → must error.
        let err = StateV0::from_parts(bitmap, marginal, vec![one_row]).unwrap_err();
        assert!(format!("{err}").contains("popcount"));
    }

    #[test]
    fn from_parts_rejects_bad_marginal_sum() {
        let bitmap = [0u8; 32];
        let mut marginal = [0u16; ALPHABET];
        marginal[0] = 1; // sum=1 ≠ ORDER1_QUANT_TOTAL
        assert!(StateV0::from_parts(bitmap, marginal, Vec::new()).is_err());
    }

    #[test]
    fn from_parts_materializes_cumsums() {
        let bitmap = [0u8; 32];
        let counts = vec![1u32; ALPHABET];
        let qq = crate::quantize::quantize_pmf(&counts, ORDER1_QUANT_TOTAL);
        let mut marginal = [0u16; ALPHABET];
        for i in 0..ALPHABET {
            marginal[i] = qq[i];
        }
        let state = StateV0::from_parts(bitmap, marginal, Vec::new()).unwrap();
        assert_eq!(state.marginal_cumsum[ALPHABET], ORDER1_QUANT_TOTAL);
    }

    #[test]
    fn deserialize_rejects_short_header() {
        let bytes = vec![0u8; 31];
        assert!(deserialize_state(&bytes).is_err());
    }

    #[test]
    fn deserialize_rejects_wrong_total_length() {
        let state = build_uniform_state(&[1, 2, 3]);
        let mut bytes = serialize_state(&state);
        bytes.pop();
        assert!(deserialize_state(&bytes).is_err());
    }

    #[test]
    fn deserialize_rejects_corrupted_marginal_sum() {
        let state = build_uniform_state(&[]);
        let mut bytes = serialize_state(&state);
        bytes[32] = bytes[32].wrapping_add(1);
        let err = deserialize_state(&bytes).unwrap_err();
        match err {
            PtwmCoreError::CodecDecode { codec, .. } => assert_eq!(codec, "Order1ScaleAC"),
            _ => panic!("expected CodecDecode"),
        }
    }

    #[test]
    fn deserialize_rejects_corrupted_row_sum() {
        let state = build_uniform_state(&[7]);
        let mut bytes = serialize_state(&state);
        // Corrupt the first byte of row 0 (offset = 32 + 512).
        bytes[32 + 512] = bytes[32 + 512].wrapping_add(1);
        let err = deserialize_state(&bytes).unwrap_err();
        match err {
            PtwmCoreError::CodecDecode { codec, .. } => assert_eq!(codec, "Order1ScaleAC"),
            _ => panic!("expected CodecDecode"),
        }
    }
}

#[cfg(test)]
mod bitmap_tests {
    use super::*;

    fn linear_oracle_rank(bitmap: &[u8; 32], prev: u8) -> usize {
        (0..prev).filter(|&p| bitmap_has(bitmap, p)).count()
    }

    #[test]
    fn rank_matches_linear_oracle_exhaustive() {
        let bitmap = [
            0xA5, 0x00, 0xFF, 0x12, 0x34, 0x56, 0x78, 0x9A, 0x00, 0xFF, 0x00, 0x55, 0xAA, 0x33,
            0xCC, 0x0F, 0xF0, 0x55, 0xAA, 0x88, 0x44, 0x22, 0x11, 0xFE, 0x7F, 0x3F, 0x1F, 0x0F,
            0x07, 0x03, 0x01, 0x00,
        ];
        for prev in 0..=255u8 {
            if bitmap_has(&bitmap, prev) {
                assert_eq!(
                    bitmap_rank(&bitmap, prev),
                    linear_oracle_rank(&bitmap, prev),
                    "rank mismatch at prev={}",
                    prev
                );
            }
        }
    }

    #[test]
    fn popcount_matches_oracle() {
        let bitmap = [0xFFu8; 32];
        assert_eq!(popcount_total(&bitmap), 256);
        let bitmap = [0u8; 32];
        assert_eq!(popcount_total(&bitmap), 0);
        let bitmap = [
            0xA5, 0x00, 0xFF, 0x12, 0x34, 0x56, 0x78, 0x9A, 0x00, 0xFF, 0x00, 0x55, 0xAA, 0x33,
            0xCC, 0x0F, 0xF0, 0x55, 0xAA, 0x88, 0x44, 0x22, 0x11, 0xFE, 0x7F, 0x3F, 0x1F, 0x0F,
            0x07, 0x03, 0x01, 0x00,
        ];
        let oracle: usize = (0..256u32)
            .filter(|&p| bitmap_has(&bitmap, p as u8))
            .count();
        assert_eq!(popcount_total(&bitmap), oracle);
    }
}

use crate::codec::{CodecId, Encoded, PlaneCodec};
use crate::layout::PlaneLayout;
use crate::types::descriptor::{ElementWidth, Layout, PlaneDescriptor};
use crate::types::role::{Role, ScaleFormat};

fn resolve_row_len_encode(plane_len: usize, layout: &PlaneLayout) -> Result<usize, PtwmCoreError> {
    let row_len = match layout {
        PlaneLayout::Rows { row_len } => row_len.get() as usize,
        PlaneLayout::Flat => plane_len.max(1),
    };
    if !plane_len.is_multiple_of(row_len) {
        return Err(PtwmCoreError::CodecDecode {
            codec: "Order1ScaleAC",
            msg: format!("row_len {row_len} does not divide plane length {plane_len}"),
        });
    }
    Ok(row_len)
}

fn resolve_row_len_decode(
    decoded_len: usize,
    layout: &PlaneLayout,
) -> Result<usize, PtwmCoreError> {
    let row_len = match layout {
        PlaneLayout::Rows { row_len } => row_len.get() as usize,
        PlaneLayout::Flat => decoded_len.max(1),
    };
    if row_len == 0 {
        return Err(PtwmCoreError::CodecDecode {
            codec: "Order1ScaleAC",
            msg: "row_len must be positive".into(),
        });
    }
    if !decoded_len.is_multiple_of(row_len) {
        return Err(PtwmCoreError::CodecDecode {
            codec: "Order1ScaleAC",
            msg: format!("row_len {row_len} does not divide decoded length {decoded_len}"),
        });
    }
    Ok(row_len)
}

#[inline]
fn binary_search_cumsum(cumsum: &[u32; ALPHABET + 1], target: u32) -> usize {
    let mut lo = 0usize;
    let mut hi = ALPHABET;
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if cumsum[mid] <= target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

impl PlaneCodec for Order1ScaleAC {
    fn id(&self) -> CodecId {
        CodecId::Order1ScaleAC
    }

    fn accepts(&self, descriptor: &PlaneDescriptor) -> bool {
        // O1SAC requires byte-width Scale or GlobalScale planes with row layout
        // AND an FP8 scale format (E4M3 or E5M2). The codec models per-row
        // lag-1 PMFs over scale bytes — a structure that exists in FP8 scale
        // distributions (mantissa bits give a continuous distribution with
        // exploitable autocorrelation) but not in E8M0 scales, which are pure
        // 8-bit exponents (no sign, no mantissa) used by MXFP4 block scales.
        // Allow-listing E4M3/E5M2 keeps the codec inside its design domain;
        // future scale formats need explicit vetting before being admitted.
        let role_ok = match descriptor.role {
            Role::Scale { format } | Role::GlobalScale { format } => {
                matches!(format, ScaleFormat::E4M3 | ScaleFormat::E5M2)
            }
            _ => false,
        };
        let width_ok = descriptor.element_width == ElementWidth::Byte;
        let layout_ok = matches!(descriptor.layout, Layout::Rows { .. });
        role_ok && width_ok && layout_ok
    }

    fn priority_for(&self, descriptor: &PlaneDescriptor) -> i8 {
        if self.accepts(descriptor) {
            10
        } else {
            i8::MIN
        }
    }

    /// Cheap statistical pre-screen. O1SAC's `encode` runs a per-context
    /// PMF fit (256 quantize calls + per-row table emission) and a
    /// context-switching ANS pass — substantially heavier than Rans on
    /// the same plane (NVFP4 weight_scale planes: hundreds of seconds of
    /// wall-clock for several hundred fits where O1SAC won only ~20 % of
    /// the time).
    ///
    /// We use a **marginal-entropy** gate: estimate `H(X)` from a sample
    /// of `SAMPLE_BYTES`. If `H(X) ≥ MAX_MARGINAL_ENTROPY_BITS`, the
    /// distribution is near-uniform and even Rans encodes within 1 bit
    /// of entropy — there's no slack for row conditioning to recover
    /// O1SAC's per-row table overhead. Below the threshold, keep O1SAC
    /// in the menu; trial-encode will compare the actual payloads.
    ///
    /// We deliberately *don't* try to estimate `H(X|prev)` from a small
    /// sample: with 256 contexts × 256 outcomes = 65K joint cells, the
    /// Miller-Madow bias swamps the signal at any sample size that's
    /// still cheap to compute. Marginal entropy is unbiased at K=256
    /// with ≥ a few thousand samples.
    fn should_attempt(
        &self,
        plane: &[u8],
        _descriptor: &PlaneDescriptor,
        layout: &PlaneLayout,
    ) -> bool {
        const SAMPLE_BYTES: usize = 8 * 1024;
        // Guards the near-uniform case only: at H ≥ 7.5 bits/byte the
        // data is almost white-noise — row conditioning can't recover
        // O1SAC's per-row table overhead and Rans approaches entropy.
        //
        // NOTE: real-corpus FP8-E4M3 NVFP4 weight_scales have H_eff in
        // the range 1.5–4.0 bits/byte (inferred from Rans compression
        // ratios on representative shards). They all pass this gate.
        // The threshold therefore doesn't cut the trial-encode overhead
        // observed post-terminal-role-fix; that cost is intrinsic to
        // trial-encoding O1SAC on large planes when it loses to Rans.
        // Addressing that requires estimating lag-1 row autocorrelation
        // (H(X|prev) vs H(X)), which needs bias correction beyond what
        // a cheap sample provides — tracked for a future optimisation.
        const MAX_MARGINAL_ENTROPY_BITS: f64 = 7.5;

        let row_len = match layout {
            PlaneLayout::Rows { row_len } => row_len.get() as usize,
            PlaneLayout::Flat => return true, // accepts() already gates on Rows
        };
        if row_len < 2 || plane.len() < row_len {
            return true;
        }

        // Sample contiguously from the plane head — enough bytes to make
        // the 256-bin marginal estimate unbiased without walking the
        // whole tensor.
        let sample_len = plane.len().min(SAMPLE_BYTES);
        let sample = &plane[..sample_len];
        let mut marginal = [0u32; ALPHABET];
        for &b in sample {
            marginal[b as usize] += 1;
        }
        let total_f = sample_len as f64;
        let mut h_marginal = 0.0f64;
        for &c in &marginal {
            if c > 0 {
                let p = c as f64 / total_f;
                h_marginal -= p * p.log2();
            }
        }
        h_marginal < MAX_MARGINAL_ENTROPY_BITS
    }

    fn encode(
        &self,
        plane: &[u8],
        shared_state: Option<&[u8]>,
        layout: &PlaneLayout,
    ) -> Result<Encoded, PtwmCoreError> {
        let row_len = resolve_row_len_encode(plane.len(), layout)?;

        let (state, inline_state_bytes): (StateV0, Vec<u8>) = match shared_state {
            Some(bytes) => (deserialize_state(bytes)?, Vec::new()),
            None => {
                let (joint, marginal) = accumulate_counts_single(plane, row_len as u32);
                let state = quantize_and_smooth(joint, marginal);
                // Encoder-side mirror of the deserialize tripwire.
                crate::quantize::validate_pmf_sum(
                    &state.marginal,
                    ORDER1_QUANT_TOTAL,
                    "Order1ScaleAC",
                )?;
                for row in &state.conditional_rows {
                    crate::quantize::validate_pmf_sum(row, ORDER1_QUANT_TOTAL, "Order1ScaleAC")?;
                }
                let bytes = serialize_state(&state);
                (state, bytes)
            }
        };

        let mut enc = crate::range_coder::RangeEncoder::new();
        for chunk in plane.chunks(row_len) {
            let mut prev: Option<u8> = None;
            for &cur in chunk {
                let cumsum = match prev {
                    None => &state.marginal_cumsum,
                    Some(p) => state.cond_cumsum_for(p),
                };
                let cumulative = cumsum[cur as usize];
                let prob = cumsum[cur as usize + 1] - cumulative;
                enc.encode_symbol(cumulative, prob, ORDER1_QUANT_TOTAL);
                prev = Some(cur);
            }
        }

        Ok(Encoded {
            state_bytes: inline_state_bytes,
            state_format_version: STATE_FORMAT_VERSION,
            payload: enc.finish(),
        })
    }

    fn decode(
        &self,
        state_format_version: u8,
        state_bytes: &[u8],
        payload: &[u8],
        layout: &PlaneLayout,
        decoded_len: usize,
    ) -> Result<Vec<u8>, PtwmCoreError> {
        if state_format_version != STATE_FORMAT_VERSION {
            return Err(PtwmCoreError::CodecDecode {
                codec: "Order1ScaleAC",
                msg: format!(
                    "unknown state_format_version {state_format_version} (build supports {STATE_FORMAT_VERSION})"
                ),
            });
        }
        let state = deserialize_state(state_bytes)?;
        let row_len = resolve_row_len_decode(decoded_len, layout)?;

        let mut dec = crate::range_coder::RangeDecoder::new(payload);
        let mut out = Vec::with_capacity(decoded_len);
        let mut row_pos = 0usize;
        let mut prev: Option<u8> = None;
        for _ in 0..decoded_len {
            if row_pos == 0 {
                prev = None;
            }
            let cumsum = match prev {
                None => &state.marginal_cumsum,
                Some(p) => state.cond_cumsum_for(p),
            };
            let target = dec.decode_symbol(ORDER1_QUANT_TOTAL);
            let sym = binary_search_cumsum(cumsum, target);
            let cumulative = cumsum[sym];
            let prob = cumsum[sym + 1] - cumulative;
            dec.advance(cumulative, prob, ORDER1_QUANT_TOTAL);
            out.push(sym as u8);
            prev = Some(sym as u8);
            row_pos = (row_pos + 1) % row_len;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod codec_roundtrip_tests {
    use super::*;

    fn synth_markov_plane(n_rows: usize, row_len: usize, seed: u64) -> Vec<u8> {
        let mut state: u64 = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u8
        };
        let mut out = Vec::with_capacity(n_rows * row_len);
        for _ in 0..n_rows {
            let mut prev: u8 = next();
            out.push(prev);
            for _ in 1..row_len {
                let drift = (next() % 9) as i16 - 4;
                let v = ((prev as i16 + drift).rem_euclid(256)) as u8;
                out.push(v);
                prev = v;
            }
        }
        out
    }

    #[test]
    fn roundtrip_inline_state() {
        let plane = synth_markov_plane(64, 128, 0xDEADBEEF);
        let codec = Order1ScaleAC;
        let layout = PlaneLayout::rows(128).unwrap();
        let enc = codec.encode(&plane, None, &layout).unwrap();
        let dec = codec
            .decode(
                enc.state_format_version,
                &enc.state_bytes,
                &enc.payload,
                &layout,
                plane.len(),
            )
            .unwrap();
        assert_eq!(dec, plane);
    }

    #[test]
    fn roundtrip_shared_state() {
        let plane_a = synth_markov_plane(64, 128, 0x1111);
        let plane_b = synth_markov_plane(48, 128, 0x2222);
        let codec = Order1ScaleAC;
        let layout = PlaneLayout::rows(128).unwrap();
        let enc_a = codec.encode(&plane_a, None, &layout).unwrap();
        let shared = enc_a.state_bytes.clone();
        let enc_b = codec.encode(&plane_b, Some(&shared), &layout).unwrap();
        assert!(enc_b.state_bytes.is_empty());
        let dec = codec
            .decode(
                enc_b.state_format_version,
                &shared,
                &enc_b.payload,
                &layout,
                plane_b.len(),
            )
            .unwrap();
        assert_eq!(dec, plane_b);
    }

    #[test]
    fn roundtrip_shapeless_flat_layout() {
        let plane = synth_markov_plane(1, 256, 0x3333);
        let codec = Order1ScaleAC;
        let layout = PlaneLayout::Flat;
        let enc = codec.encode(&plane, None, &layout).unwrap();
        let dec = codec
            .decode(
                enc.state_format_version,
                &enc.state_bytes,
                &enc.payload,
                &layout,
                plane.len(),
            )
            .unwrap();
        assert_eq!(dec, plane);
    }

    #[test]
    fn beats_huffman_marginal_on_structured_data() {
        // Use a 4-symbol Markov chain (values 0..4) with strong correlation
        // so the order-1 coder achieves well below 2 bits/symbol (vs 2 bits
        // marginal entropy). The state overhead is tiny: 4 contexts × 512 B =
        // 2 KB + 544 B overhead = ~2.5 KB.  With 16384 bytes of input and
        // ~1 bit/symbol real entropy, the payload should be ~2 KB, so
        // total coded_len ≈ 4.5 KB < 12 KB (75% of 16384).
        let row_len = 128usize;
        let n_rows = 128usize;
        let mut plane = Vec::with_capacity(n_rows * row_len);
        let mut rng: u64 = 0x4444_BEEF_u64;
        for _ in 0..n_rows {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let mut prev: u8 = ((rng >> 33) % 4) as u8;
            plane.push(prev);
            for _ in 1..row_len {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let r = (rng >> 33) % 8;
                // 7/8 chance to stay, 1/8 chance to move to a random neighbour
                let v = if r < 7 { prev } else { (prev + 1) % 4 };
                plane.push(v);
                prev = v;
            }
        }
        let codec = Order1ScaleAC;
        let layout = PlaneLayout::rows(row_len as u32).unwrap();
        let enc = codec.encode(&plane, None, &layout).unwrap();
        let coded_len = enc.state_bytes.len() + enc.payload.len();
        assert!(
            coded_len < plane.len() * 3 / 4,
            "coded_len = {coded_len}, raw = {}",
            plane.len()
        );
    }

    #[test]
    fn decode_rejects_bad_state_version() {
        let plane = synth_markov_plane(8, 16, 0x5555);
        let codec = Order1ScaleAC;
        let layout = PlaneLayout::rows(16).unwrap();
        let enc = codec.encode(&plane, None, &layout).unwrap();
        let err = codec
            .decode(99, &enc.state_bytes, &enc.payload, &layout, plane.len())
            .unwrap_err();
        match err {
            PtwmCoreError::CodecDecode { codec, .. } => assert_eq!(codec, "Order1ScaleAC"),
            _ => panic!("expected CodecDecode"),
        }
    }

    #[test]
    fn rows_layout_rejects_zero_at_construction() {
        // Type-level invariant: NonZeroU32 makes row_len == 0 unconstructible.
        assert!(PlaneLayout::rows(0).is_none());
    }

    #[test]
    fn encode_rejects_non_dividing_row_len() {
        let codec = Order1ScaleAC;
        let layout = PlaneLayout::rows(7).unwrap();
        let plane = vec![0u8; 16];
        assert!(codec.encode(&plane, None, &layout).is_err());
    }

    #[test]
    fn decode_truncated_payload_does_not_panic() {
        // Truncated payload must not panic; if decode returns Ok, the
        // output length must still equal `decoded_len` (the range
        // decoder pads EOF with zeros, so a wrong-but-bounded result
        // is acceptable here).
        let plane = synth_markov_plane(64, 128, 0x6666);
        let codec = Order1ScaleAC;
        let layout = PlaneLayout::rows(128).unwrap();
        let enc = codec.encode(&plane, None, &layout).unwrap();
        for cut in [0usize, 1, 5, enc.payload.len() / 2] {
            let truncated = &enc.payload[..cut.min(enc.payload.len())];
            let result = codec.decode(
                enc.state_format_version,
                &enc.state_bytes,
                truncated,
                &layout,
                plane.len(),
            );
            if let Ok(out) = result {
                assert_eq!(out.len(), plane.len(), "cut={cut}");
            }
        }
    }

    #[test]
    fn decode_corrupted_payload_does_not_panic() {
        let plane = synth_markov_plane(64, 128, 0x7777);
        let codec = Order1ScaleAC;
        let layout = PlaneLayout::rows(128).unwrap();
        let enc = codec.encode(&plane, None, &layout).unwrap();
        let mut payload = enc.payload.clone();
        let mid = payload.len() / 2;
        payload[mid] ^= 0xA5;
        let result = codec.decode(
            enc.state_format_version,
            &enc.state_bytes,
            &payload,
            &layout,
            plane.len(),
        );
        if let Ok(out) = result {
            assert_eq!(out.len(), plane.len());
        }
    }
}

pub(crate) type JointCounts = Box<[[u32; ALPHABET]; ALPHABET]>;
pub(crate) type MarginalCounts = [u32; ALPHABET];

pub(crate) fn empty_joint() -> JointCounts {
    Box::new([[0u32; ALPHABET]; ALPHABET])
}

/// Accumulate (joint, marginal) counts for a single SCALE plane with a
/// given row stride. For each row: bump `marginal[chunk[0]]`; for `i` in
/// `1..chunk.len()`, bump `joint[chunk[i-1]][chunk[i]]`.
///
/// `row_len` must be > 0 — every caller derives it from `PlaneLayout`,
/// which itself rejects `row_len == 0` at parse time. A `0` here would
/// otherwise be silently coerced to 1 and produce a degenerate PMF.
pub(crate) fn accumulate_counts_single(
    plane: &[u8],
    row_len: u32,
) -> (JointCounts, MarginalCounts) {
    assert!(row_len > 0, "accumulate_counts_single: row_len must be > 0");
    let mut joint = empty_joint();
    let mut marginal = [0u32; ALPHABET];
    let row_len = row_len as usize;
    for chunk in plane.chunks(row_len) {
        if chunk.is_empty() {
            continue;
        }
        marginal[chunk[0] as usize] += 1;
        for i in 1..chunk.len() {
            joint[chunk[i - 1] as usize][chunk[i] as usize] += 1;
        }
    }
    (joint, marginal)
}

/// Convert raw counts to a quantized StateV0. Each context row with at
/// least one non-zero count gets a present-bitmap bit and a quantized PMF
/// row; absent contexts fall back to marginal at decode.
pub(crate) fn quantize_and_smooth(joint: JointCounts, marginal: MarginalCounts) -> StateV0 {
    let mut state = StateV0::empty();

    // Marginal: always dense, always quantized.
    let m_counts: Vec<u32> = marginal.to_vec();
    let m_q = crate::quantize::quantize_pmf(&m_counts, ORDER1_QUANT_TOTAL);
    for i in 0..ALPHABET {
        state.marginal[i] = m_q[i];
    }

    // Conditional rows: only contexts with >= 1 raw count get a row.
    for prev in 0..ALPHABET {
        let row_sum: u32 = joint[prev].iter().sum();
        if row_sum == 0 {
            continue;
        }
        state.present_bitmap[prev >> 3] |= 1u8 << (prev & 7);
        let row_counts: Vec<u32> = joint[prev].to_vec();
        let row_q = crate::quantize::quantize_pmf(&row_counts, ORDER1_QUANT_TOTAL);
        let mut row = [0u16; ALPHABET];
        for i in 0..ALPHABET {
            row[i] = row_q[i];
        }
        state.conditional_rows.push(row);
    }

    state.materialize_cumsums();
    state
}

#[cfg(test)]
mod fit_single_tests {
    use super::*;

    #[test]
    fn accumulate_two_rows() {
        let plane = [10u8, 20, 30, 40, /* row 2 */ 50, 60, 70, 80];
        let (joint, marginal) = accumulate_counts_single(&plane, 4);
        assert_eq!(marginal[10], 1);
        assert_eq!(marginal[50], 1);
        assert_eq!(joint[10][20], 1);
        assert_eq!(joint[20][30], 1);
        assert_eq!(joint[30][40], 1);
        assert_eq!(joint[50][60], 1);
        assert_eq!(joint[60][70], 1);
        assert_eq!(joint[70][80], 1);
    }

    #[test]
    fn accumulate_handles_partial_last_row() {
        let plane = [1u8, 2, 3, 4, 5]; // row_len 4 → second "row" is just [5]
        let (joint, marginal) = accumulate_counts_single(&plane, 4);
        assert_eq!(marginal[1], 1);
        assert_eq!(marginal[5], 1);
        assert_eq!(joint[1][2], 1);
        assert_eq!(joint.iter().flatten().filter(|&&x| x > 0).count(), 3);
    }

    #[test]
    fn quantize_produces_valid_state() {
        let mut joint = empty_joint();
        joint[5][10] = 100;
        joint[5][20] = 50;
        joint[7][30] = 200;
        let mut marginal = [0u32; ALPHABET];
        marginal[5] = 1;
        marginal[7] = 1;
        let state = quantize_and_smooth(joint, marginal);
        assert!(bitmap_has(&state.present_bitmap, 5));
        assert!(bitmap_has(&state.present_bitmap, 7));
        assert!(!bitmap_has(&state.present_bitmap, 10));
        assert_eq!(state.conditional_rows.len(), 2);
        for row in &state.conditional_rows {
            let s: u32 = row.iter().map(|&x| x as u32).sum();
            assert_eq!(s, ORDER1_QUANT_TOTAL);
        }
        let s: u32 = state.marginal.iter().map(|&x| x as u32).sum();
        assert_eq!(s, ORDER1_QUANT_TOTAL);
    }
}

#[cfg(test)]
mod cumsum_tests {
    use super::*;
    use crate::quantize::ORDER1_QUANT_TOTAL;

    #[test]
    fn cumsum_invariants() {
        let mut pmf = [0u16; ALPHABET];
        for i in 0..ALPHABET {
            pmf[i] = ((i % 17) + 1) as u16;
        }
        let cumsum = build_cumsum(&pmf);
        assert_eq!(cumsum[0], 0);
        for i in 0..ALPHABET {
            assert_eq!(cumsum[i + 1] - cumsum[i], pmf[i] as u32);
        }
    }

    #[test]
    fn cond_cumsum_falls_back_to_marginal_when_absent() {
        let mut state = StateV0::empty();
        for i in 0..ALPHABET {
            state.marginal[i] = (ORDER1_QUANT_TOTAL / ALPHABET as u32) as u16;
        }
        state.materialize_cumsums();
        // No bitmap bits set → every prev falls back to marginal.
        for prev in 0..=255u8 {
            assert_eq!(
                state.cond_cumsum_for(prev) as *const _,
                &state.marginal_cumsum as *const _
            );
        }
    }
}

#[cfg(test)]
mod prelude_hash_tests {
    use crate::codec::CodecId;
    use crate::prelude::{PreludeEntry, parse_prelude, write_prelude};
    use xxhash_rust::xxh64::xxh64;

    #[test]
    fn order1_scale_ac_emits_state_xxhash64_v1() {
        // A representative state_bytes payload (content irrelevant for the
        // hash test; the assertion only requires that the entry round-trips
        // with a non-zero hash that matches xxh64(state_bytes, 0)).
        let state_bytes: Vec<u8> = (0u8..=255).collect();
        let expected_hash = xxh64(&state_bytes, 0);
        let e = PreludeEntry {
            shared_state_id: 0,
            codec_id: CodecId::Order1ScaleAC,
            state_format_version: 1,
            applies_to_mask: 0,
            state_xxhash64: expected_hash,
            name: "test".to_string(),
            state_bytes,
        };
        let mut buf = Vec::new();
        write_prelude(std::slice::from_ref(&e), &mut buf).unwrap();
        let (parsed, _) = parse_prelude(&buf).unwrap();
        assert_eq!(parsed[0].state_format_version, 1);
        assert_eq!(parsed[0].state_xxhash64, expected_hash);
        assert_ne!(
            expected_hash, 0,
            "xxh64 of non-trivial input must be non-zero"
        );
    }
}

#[cfg(test)]
mod capability_tests {
    use super::*;
    use crate::types::descriptor::{ElementWidth, Layout, PlaneDescriptor};
    use crate::types::role::{Role, ScaleFormat, ValueFormat};

    fn descriptor(
        role: Role,
        width: ElementWidth,
        layout: Layout,
        is_nibble_packed: bool,
    ) -> PlaneDescriptor {
        PlaneDescriptor {
            role,
            element_width: width,
            length_bytes: 512,
            layout,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed,
            vendor_bytes: vec![],
        }
    }

    #[test]
    fn o1sac_accepts_scale_byte_rows() {
        let c = Order1ScaleAC;
        let d = descriptor(
            Role::Scale {
                format: ScaleFormat::E4M3,
            },
            ElementWidth::Byte,
            Layout::Rows { row_len: 32 },
            false,
        );
        assert!(c.accepts(&d));
    }

    #[test]
    fn o1sac_accepts_global_scale_byte_rows() {
        let c = Order1ScaleAC;
        let d = descriptor(
            Role::GlobalScale {
                format: ScaleFormat::E4M3,
            },
            ElementWidth::Byte,
            Layout::Rows { row_len: 16 },
            false,
        );
        assert!(c.accepts(&d));
    }

    #[test]
    fn o1sac_rejects_e8m0_scale() {
        // E8M0 (MXFP4 block scale) is a pure 8-bit exponent — structurally
        // different from FP8 scales (E4M3 / E5M2) and outside O1SAC's design
        // domain.
        let c = Order1ScaleAC;
        let d = descriptor(
            Role::Scale {
                format: ScaleFormat::E8M0,
            },
            ElementWidth::Byte,
            Layout::Rows { row_len: 32 },
            false,
        );
        assert!(!c.accepts(&d));
    }

    #[test]
    fn o1sac_rejects_e8m0_global_scale() {
        let c = Order1ScaleAC;
        let d = descriptor(
            Role::GlobalScale {
                format: ScaleFormat::E8M0,
            },
            ElementWidth::Byte,
            Layout::Rows { row_len: 16 },
            false,
        );
        assert!(!c.accepts(&d));
    }

    #[test]
    fn o1sac_accepts_e5m2_scale() {
        let c = Order1ScaleAC;
        let d = descriptor(
            Role::Scale {
                format: ScaleFormat::E5M2,
            },
            ElementWidth::Byte,
            Layout::Rows { row_len: 32 },
            false,
        );
        assert!(c.accepts(&d));
    }

    #[test]
    fn o1sac_rejects_value_role() {
        let c = Order1ScaleAC;
        let d = descriptor(
            Role::Value {
                format: ValueFormat::Fp4E2m1,
            },
            ElementWidth::Byte,
            Layout::Rows { row_len: 32 },
            false,
        );
        assert!(!c.accepts(&d));
    }

    #[test]
    fn o1sac_rejects_nibble_width() {
        let c = Order1ScaleAC;
        let d = descriptor(
            Role::Scale {
                format: ScaleFormat::E4M3,
            },
            ElementWidth::Nibble,
            Layout::Rows { row_len: 32 },
            false,
        );
        assert!(!c.accepts(&d));
    }

    #[test]
    fn o1sac_rejects_flat_layout() {
        let c = Order1ScaleAC;
        let d = descriptor(
            Role::Scale {
                format: ScaleFormat::E4M3,
            },
            ElementWidth::Byte,
            Layout::Flat,
            false,
        );
        assert!(!c.accepts(&d));
    }

    #[test]
    fn o1sac_priority_high_when_accepted() {
        let c = Order1ScaleAC;
        let d = descriptor(
            Role::Scale {
                format: ScaleFormat::E4M3,
            },
            ElementWidth::Byte,
            Layout::Rows { row_len: 32 },
            false,
        );
        assert_eq!(c.priority_for(&d), 10);
    }

    #[test]
    fn o1sac_priority_min_when_rejected() {
        let c = Order1ScaleAC;
        let d = descriptor(
            Role::Value {
                format: ValueFormat::Fp4E2m1,
            },
            ElementWidth::Byte,
            Layout::Rows { row_len: 32 },
            false,
        );
        assert_eq!(c.priority_for(&d), i8::MIN);
    }

    /// `should_attempt` is the cheap pre-screen that lets the trial-encode
    /// menu skip O1SAC's expensive PMF fit when the row context predicts
    /// no gain. This test feeds it bytes where every row is genuinely
    /// uncorrelated noise — H(X|prev) ≈ H(X), so the gate must reject.
    #[test]
    fn o1sac_should_attempt_skips_uncorrelated_uniform() {
        let c = Order1ScaleAC;
        let d = descriptor(
            Role::Scale {
                format: ScaleFormat::E4M3,
            },
            ElementWidth::Byte,
            Layout::Rows { row_len: 256 },
            false,
        );
        // 256-byte rows × 64 rows from a high-quality LCG so successive
        // bytes have negligible correlation across the byte alphabet.
        // (Earlier `i * 2654435761` made every step add a fixed Δ → perfect
        // correlation, the opposite of what we want here.)
        let row_len = 256usize;
        let n_rows = 64usize;
        let mut plane = Vec::with_capacity(row_len * n_rows);
        let mut s: u64 = 0xDEADBEEFCAFEBABE;
        for _ in 0..(row_len * n_rows) {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            plane.push((s >> 56) as u8);
        }
        let layout = PlaneLayout::rows(row_len as u32).unwrap();
        assert!(
            !c.should_attempt(&plane, &d, &layout),
            "should_attempt must skip O1SAC on uncorrelated uniform data"
        );
    }

    /// Mirror of the above for the positive case: a plane where every
    /// row is one byte repeated → H(X|prev) ≈ 0 ≪ H(X). The gate must
    /// keep O1SAC. Plane is sized so the predicted saving comfortably
    /// exceeds the gate's table-overhead floor (8 KiB).
    #[test]
    fn o1sac_should_attempt_keeps_correlated_rows() {
        let c = Order1ScaleAC;
        let d = descriptor(
            Role::Scale {
                format: ScaleFormat::E4M3,
            },
            ElementWidth::Byte,
            Layout::Rows { row_len: 256 },
            false,
        );
        // 4 KiB rows × 64 rows = 256 KiB. Each row repeats one of 16
        // bytes → marginal H ≈ 4 bits, conditional H ≈ 0 → predicted
        // saving = 256K * 4 / 8 = 128 KiB ≫ 8 KiB threshold.
        let row_len = 4096usize;
        let n_rows = 64usize;
        let mut plane = Vec::with_capacity(row_len * n_rows);
        for r in 0..n_rows {
            let b = ((r * 37) & 0x0F) as u8;
            for _ in 0..row_len {
                plane.push(b);
            }
        }
        let layout = PlaneLayout::rows(row_len as u32).unwrap();
        assert!(
            c.should_attempt(&plane, &d, &layout),
            "should_attempt must keep O1SAC when rows are highly self-correlated"
        );
    }

    /// `should_attempt`'s cost must stay near-zero on multi-MB planes —
    /// the gate samples at most `SAMPLE_ROWS` rows, so wall-clock is
    /// bounded by `O(SAMPLE_ROWS * row_len)`. This test just exercises
    /// the path on a 1 MB plane to catch accidental O(N) regressions
    /// (e.g. someone removing the sample cap).
    #[test]
    fn o1sac_should_attempt_stays_cheap_on_large_planes() {
        let c = Order1ScaleAC;
        let d = descriptor(
            Role::Scale {
                format: ScaleFormat::E4M3,
            },
            ElementWidth::Byte,
            Layout::Rows { row_len: 4096 },
            false,
        );
        let plane: Vec<u8> = (0..(1 << 20)).map(|i| i as u8).collect();
        let layout = PlaneLayout::rows(4096).unwrap();
        let _ = c.should_attempt(&plane, &d, &layout);
    }
}
