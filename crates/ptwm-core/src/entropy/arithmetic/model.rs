//! Byte models for the arithmetic plane codecs.

use crate::error::PtwmCoreError;
use crate::quantize::{ORDER1_QUANT_TOTAL, quantize_pmf, validate_pmf_sum};
use crate::range_coder::{RangeDecoder, RangeEncoder};

/// Fixed PMF denominator cap. Kept well under the range coder's 2^24
/// renorm floor so `range / total >= 1` always holds.
pub const TOTAL_CAP: u32 = 1 << 16; // 65536

/// A byte probability model the range-coding driver can run. Stateful:
/// adaptive models mutate via [`Self::update`]; static models no-op.
pub trait ByteModel {
    /// Current PMF denominator. Read before every symbol.
    fn total(&self) -> u32;
    /// Encode-side lookup: `(cumulative, probability)` for `sym` under the
    /// current PMF. `probability` must be > 0 for any byte that appears.
    fn encode_lookup(&self, sym: u8) -> (u32, u32);
    /// Decode-side lookup: given `target` in `[0, total())`, return the
    /// symbol whose cumulative range contains it plus its `(cum, prob)`.
    fn decode_lookup(&self, target: u32) -> (u8, u32, u32);
    /// Adapt after coding `sym`. Static models leave this empty.
    fn update(&mut self, _sym: u8) {}
}

/// Drive `model` over `data`, returning the range-coded payload.
pub fn encode_bytes<M: ByteModel>(model: &mut M, data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }
    let mut enc = RangeEncoder::new();
    for &b in data {
        let total = model.total();
        let (cum, prob) = model.encode_lookup(b);
        enc.encode_symbol(cum, prob, total);
        model.update(b);
    }
    enc.finish()
}

/// Inverse of [`encode_bytes`]; produces exactly `decoded_len` bytes.
pub fn decode_bytes<M: ByteModel>(model: &mut M, payload: &[u8], decoded_len: usize) -> Vec<u8> {
    if decoded_len == 0 {
        return Vec::new();
    }
    let mut dec = RangeDecoder::new(payload);
    // Cap the initial allocation: `decoded_len` comes from an untrusted
    // container header, so a malicious value must not force a huge upfront
    // allocation. The Vec still grows to the true length via `push`.
    let mut out = Vec::with_capacity(decoded_len.min(1 << 17));
    for _ in 0..decoded_len {
        let total = model.total();
        let target = dec.decode_symbol(total);
        let (sym, cum, prob) = model.decode_lookup(target);
        dec.advance(cum, prob, total);
        out.push(sym);
        model.update(sym);
    }
    out
}

// ── Order-0 static ──────────────────────────────────────────────────────

/// Order-0 model with a fixed, normalized 256-bin count table.
pub struct Order0Static {
    /// Normalized counts; `sum == self.total` and every entry that was
    /// nonzero in the source stays >= 1. Symbols absent from the source
    /// keep count 0 (never coded, zero cumulative width).
    pub counts: [u32; 256],
    total: u32,
    /// Prefix sums `cum[i] = sum(counts[0..i])`, `cum[256] == total`.
    cum: [u32; 257],
}

impl Order0Static {
    /// Fit a normalized table from `data`. Empty data yields a uniform
    /// degenerate table (never used — `encode_bytes` short-circuits empty).
    pub fn fit(data: &[u8]) -> Self {
        let mut raw = [0u64; 256];
        for &b in data {
            raw[b as usize] += 1;
        }
        let counts = normalize(&raw);
        Self::from_counts(counts)
    }

    fn from_counts(counts: [u32; 256]) -> Self {
        let mut cum = [0u32; 257];
        let mut acc = 0u32;
        for i in 0..256 {
            cum[i] = acc;
            acc += counts[i];
        }
        cum[256] = acc;
        Self {
            counts,
            total: acc.max(1),
            cum,
        }
    }

    /// Serialize the table: 256 little-endian u32 counts (1024 bytes).
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(1024);
        for c in self.counts {
            out.extend_from_slice(&c.to_le_bytes());
        }
        out
    }

    /// Parse a table previously produced by [`Self::serialize`].
    pub fn deserialize(bytes: &[u8]) -> Result<Self, PtwmCoreError> {
        if bytes.len() != 1024 {
            return Err(PtwmCoreError::CodecDecode {
                codec: "arithmetic_o0",
                msg: format!("expected 1024 state bytes, got {}", bytes.len()),
            });
        }
        let mut counts = [0u32; 256];
        let mut sum = 0u32;
        for i in 0..256 {
            counts[i] = u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
            sum = sum.saturating_add(counts[i]);
        }
        // A table emitted by `serialize` always sums to exactly TOTAL_CAP
        // (see `normalize`). Reject anything else: an untrusted container
        // with a zero (or otherwise malformed) table would drive a
        // degenerate decode (zero-width symbols / non-terminating renorm).
        if sum != TOTAL_CAP {
            return Err(PtwmCoreError::CodecDecode {
                codec: "arithmetic_o0",
                msg: format!("invalid count table: sum {sum} != TOTAL_CAP {TOTAL_CAP}"),
            });
        }
        Ok(Self::from_counts(counts))
    }
}

impl ByteModel for Order0Static {
    fn total(&self) -> u32 {
        self.total
    }
    fn encode_lookup(&self, sym: u8) -> (u32, u32) {
        (self.cum[sym as usize], self.counts[sym as usize])
    }
    fn decode_lookup(&self, target: u32) -> (u8, u32, u32) {
        // cum is non-decreasing; find the last index whose cum <= target.
        let sym = match self.cum.binary_search(&target) {
            Ok(i) => {
                // Land on a boundary: skip zero-width symbols forward.
                let mut s = i.min(255);
                while self.counts[s] == 0 && s < 255 {
                    s += 1;
                }
                s
            }
            Err(i) => i - 1, // first cum strictly greater → previous index
        };
        (sym as u8, self.cum[sym], self.counts[sym])
    }
}

/// Scale raw counts so the nonzero ones sum to exactly [`TOTAL_CAP`],
/// keeping every nonzero count >= 1. Drift from rounding is absorbed by
/// the largest bucket. Deterministic.
fn normalize(raw: &[u64; 256]) -> [u32; 256] {
    let sum: u64 = raw.iter().sum();
    if sum == 0 {
        return [0u32; 256];
    }
    let mut counts = [0u32; 256];
    let mut total = 0u32;
    let mut max_idx = 0usize;
    let mut max_val = 0u64;
    for i in 0..256 {
        if raw[i] == 0 {
            continue;
        }
        let scaled = ((raw[i] as u128 * TOTAL_CAP as u128) / sum as u128) as u32;
        let scaled = scaled.max(1);
        counts[i] = scaled;
        total += scaled;
        if raw[i] > max_val {
            max_val = raw[i];
            max_idx = i;
        }
    }
    // Correct drift so the table sums to exactly TOTAL_CAP.
    if total != TOTAL_CAP {
        let diff = TOTAL_CAP as i64 - total as i64;
        let adjusted = counts[max_idx] as i64 + diff;
        // The dominant bucket is large enough to absorb the (bounded) drift.
        counts[max_idx] = adjusted.max(1) as u32;
    }
    counts
}

// ── Order-0 adaptive ────────────────────────────────────────────────────

/// Order-0 model that starts uniform (every byte count = 1) and bumps the
/// observed symbol's count after coding it. Rescales when the total nears
/// [`TOTAL_CAP`]. No serialized state — encoder and decoder stay in lockstep.
pub struct Order0Adaptive {
    counts: [u32; 256],
    total: u32,
}

impl Order0Adaptive {
    pub fn new() -> Self {
        Self {
            counts: [1u32; 256],
            total: 256,
        }
    }
}

impl Default for Order0Adaptive {
    fn default() -> Self {
        Self::new()
    }
}

const ADAPT_INC: u32 = 24;

impl ByteModel for Order0Adaptive {
    fn total(&self) -> u32 {
        self.total
    }
    fn encode_lookup(&self, sym: u8) -> (u32, u32) {
        let mut cum = 0u32;
        for i in 0..sym as usize {
            cum += self.counts[i];
        }
        (cum, self.counts[sym as usize])
    }
    fn decode_lookup(&self, target: u32) -> (u8, u32, u32) {
        let mut cum = 0u32;
        for i in 0..256 {
            let c = self.counts[i];
            if target < cum + c {
                return (i as u8, cum, c);
            }
            cum += c;
        }
        // target < total guarantees we returned above; fall back to last.
        (255, self.total - self.counts[255], self.counts[255])
    }
    fn update(&mut self, sym: u8) {
        self.counts[sym as usize] += ADAPT_INC;
        self.total += ADAPT_INC;
        if self.total >= TOTAL_CAP {
            rescale(&mut self.counts, &mut self.total);
        }
    }
}

/// Halve every count (flooring at 1) and recompute the total. Identical on
/// both sides so the models stay synchronized.
fn rescale(counts: &mut [u32; 256], total: &mut u32) {
    let mut t = 0u32;
    for c in counts.iter_mut() {
        *c = (*c >> 1).max(1);
        t += *c;
    }
    *total = t;
}

// ── Order-1 adaptive ────────────────────────────────────────────────────

/// Order-1 model: 256 independent adaptive sub-models keyed on the
/// previous byte. Context starts at 0. Stateless on the wire.
pub struct Order1Adaptive {
    counts: Vec<[u32; 256]>, // 256 contexts
    totals: [u32; 256],
    ctx: u8,
}

impl Order1Adaptive {
    pub fn new() -> Self {
        Self {
            counts: vec![[1u32; 256]; 256],
            totals: [256u32; 256],
            ctx: 0,
        }
    }
}

impl Default for Order1Adaptive {
    fn default() -> Self {
        Self::new()
    }
}

impl ByteModel for Order1Adaptive {
    fn total(&self) -> u32 {
        self.totals[self.ctx as usize]
    }
    fn encode_lookup(&self, sym: u8) -> (u32, u32) {
        let row = &self.counts[self.ctx as usize];
        let mut cum = 0u32;
        for i in 0..sym as usize {
            cum += row[i];
        }
        (cum, row[sym as usize])
    }
    fn decode_lookup(&self, target: u32) -> (u8, u32, u32) {
        let row = &self.counts[self.ctx as usize];
        let mut cum = 0u32;
        for i in 0..256 {
            let c = row[i];
            if target < cum + c {
                return (i as u8, cum, c);
            }
            cum += c;
        }
        (255, self.totals[self.ctx as usize] - row[255], row[255])
    }
    fn update(&mut self, sym: u8) {
        let c = self.ctx as usize;
        self.counts[c][sym as usize] += ADAPT_INC;
        self.totals[c] += ADAPT_INC;
        if self.totals[c] >= TOTAL_CAP {
            let (row, total) = (&mut self.counts[c], &mut self.totals[c]);
            rescale(row, total);
        }
        self.ctx = sym;
    }
}

// ── Order-1 static ──────────────────────────────────────────────────────

const ALPHABET: usize = 256;
const ROW_BYTES: usize = 2 * ALPHABET; // 256 × u16

fn quantize_row(raw: &[u32; ALPHABET]) -> [u16; ALPHABET] {
    let q = quantize_pmf(raw, ORDER1_QUANT_TOTAL);
    let mut arr = [0u16; ALPHABET];
    arr.copy_from_slice(&q);
    arr
}

fn read_row(b: &[u8]) -> [u16; ALPHABET] {
    let mut row = [0u16; ALPHABET];
    for (i, slot) in row.iter_mut().enumerate() {
        *slot = u16::from_le_bytes([b[2 * i], b[2 * i + 1]]);
    }
    row
}

/// Prefix sums of a quantized row: `cum[i] = sum(row[0..i])`, `cum[ALPHABET]
/// == ORDER1_QUANT_TOTAL`. Every entry is ≥ 1 (Laplace +1), so `cum` is
/// strictly increasing — which makes `decode_lookup`'s binary search
/// unambiguous (no zero-width-symbol boundary ties).
fn compute_cum(row: &[u16; ALPHABET]) -> [u32; ALPHABET + 1] {
    let mut cum = [0u32; ALPHABET + 1];
    let mut acc = 0u32;
    for i in 0..ALPHABET {
        cum[i] = acc;
        acc += row[i] as u32;
    }
    cum[ALPHABET] = acc;
    cum
}

/// Static order-1 (lag-1) model. Counts are fit once on the plane and
/// frozen; [`update`] only advances the context. Contexts never observed as a
/// predecessor fall back to the `marginal` (order-0) table. Every row is
/// Laplace-smoothed and quantized to [`ORDER1_QUANT_TOTAL`], so no symbol is
/// zero-width and the range coder never divides by zero.
///
/// [`update`]: ByteModel::update
pub struct Order1Static {
    marginal: [u16; ALPHABET],
    cum_marginal: [u32; ALPHABET + 1],
    present_bitmap: [u8; 32],
    /// `rows` index for each context byte, or `u32::MAX` to use `marginal`.
    row_of_ctx: [u32; ALPHABET],
    rows: Vec<[u16; ALPHABET]>,
    /// Prefix sums of `rows`, materialized once so coding is O(1)/O(log).
    cum_rows: Vec<[u32; ALPHABET + 1]>,
    ctx: u8,
}

impl Order1Static {
    /// Fit marginal + per-present-context conditional tables from `data`.
    pub fn fit(data: &[u8]) -> Self {
        let mut marg_raw = [0u32; ALPHABET];
        for &b in data {
            marg_raw[b as usize] += 1;
        }
        let marginal = quantize_row(&marg_raw);
        let cum_marginal = compute_cum(&marginal);

        // Dense transient tally; a context is "present" iff it occurs with a
        // successor (i.e. as `w[0]` in some adjacent pair).
        let mut raw_rows: Vec<[u32; ALPHABET]> = vec![[0u32; ALPHABET]; ALPHABET];
        let mut present = [false; ALPHABET];
        for w in data.windows(2) {
            present[w[0] as usize] = true;
            raw_rows[w[0] as usize][w[1] as usize] += 1;
        }

        let mut present_bitmap = [0u8; 32];
        let mut row_of_ctx = [u32::MAX; ALPHABET];
        let mut rows = Vec::new();
        let mut cum_rows = Vec::new();
        for c in 0..ALPHABET {
            if present[c] {
                present_bitmap[c >> 3] |= 1 << (c & 7);
                row_of_ctx[c] = rows.len() as u32;
                let row = quantize_row(&raw_rows[c]);
                cum_rows.push(compute_cum(&row));
                rows.push(row);
            }
        }
        Self {
            marginal,
            cum_marginal,
            present_bitmap,
            row_of_ctx,
            rows,
            cum_rows,
            ctx: 0,
        }
    }

    fn active_cum(&self) -> &[u32; ALPHABET + 1] {
        let idx = self.row_of_ctx[self.ctx as usize];
        if idx == u32::MAX {
            &self.cum_marginal
        } else {
            &self.cum_rows[idx as usize]
        }
    }

    /// Wire layout: `[bitmap 32 B][marginal 512 B][present rows 512 B each]`.
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(32 + ROW_BYTES * (1 + self.rows.len()));
        out.extend_from_slice(&self.present_bitmap);
        for &v in &self.marginal {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for row in &self.rows {
            for &v in row {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }

    /// Parse + validate state produced by [`Self::serialize`]. Rejects a
    /// wrong length or any row that does not sum to [`ORDER1_QUANT_TOTAL`]
    /// with all entries ≥ 1 (a malformed table would hang the range coder).
    pub fn deserialize(bytes: &[u8]) -> Result<Self, PtwmCoreError> {
        const CODEC: &str = "order1_arithmetic";
        if bytes.len() < 32 + ROW_BYTES {
            return Err(PtwmCoreError::CodecDecode {
                codec: CODEC,
                msg: format!("state too short: {} bytes", bytes.len()),
            });
        }
        let mut present_bitmap = [0u8; 32];
        present_bitmap.copy_from_slice(&bytes[..32]);
        let popcount: usize = present_bitmap.iter().map(|b| b.count_ones() as usize).sum();
        let expected = 32 + ROW_BYTES * (1 + popcount);
        if bytes.len() != expected {
            return Err(PtwmCoreError::CodecDecode {
                codec: CODEC,
                msg: format!("state len {} != expected {expected}", bytes.len()),
            });
        }

        let marginal = read_row(&bytes[32..32 + ROW_BYTES]);
        validate_pmf_sum(&marginal, ORDER1_QUANT_TOTAL, CODEC)?;
        let cum_marginal = compute_cum(&marginal);

        let mut rows = Vec::with_capacity(popcount);
        let mut cum_rows = Vec::with_capacity(popcount);
        let mut row_of_ctx = [u32::MAX; ALPHABET];
        let mut off = 32 + ROW_BYTES;
        for c in 0..ALPHABET {
            if present_bitmap[c >> 3] & (1 << (c & 7)) != 0 {
                let row = read_row(&bytes[off..off + ROW_BYTES]);
                validate_pmf_sum(&row, ORDER1_QUANT_TOTAL, CODEC)?;
                row_of_ctx[c] = rows.len() as u32;
                cum_rows.push(compute_cum(&row));
                rows.push(row);
                off += ROW_BYTES;
            }
        }
        Ok(Self {
            marginal,
            cum_marginal,
            present_bitmap,
            row_of_ctx,
            rows,
            cum_rows,
            ctx: 0,
        })
    }
}

impl ByteModel for Order1Static {
    fn total(&self) -> u32 {
        ORDER1_QUANT_TOTAL
    }
    fn encode_lookup(&self, sym: u8) -> (u32, u32) {
        // O(1): `cum` is the materialized prefix-sum array for the active row.
        let cum = self.active_cum();
        let s = sym as usize;
        (cum[s], cum[s + 1] - cum[s])
    }
    fn decode_lookup(&self, target: u32) -> (u8, u32, u32) {
        // `cum` is strictly increasing (every count ≥ 1), so the symbol whose
        // half-open range `[cum[s], cum[s+1])` contains `target` is found by
        // binary search: an exact hit lands on a start boundary (symbol `i`);
        // otherwise the insertion point `i` means `cum[i-1] < target < cum[i]`,
        // i.e. symbol `i-1`. `target < total` keeps `s` in `0..ALPHABET`.
        let cum = self.active_cum();
        let s = match cum.binary_search(&target) {
            Ok(i) => i,
            Err(i) => i - 1,
        }
        .min(ALPHABET - 1);
        (s as u8, cum[s], cum[s + 1] - cum[s])
    }
    fn update(&mut self, sym: u8) {
        self.ctx = sym;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip<M: ByteModel>(make: impl Fn() -> M, data: &[u8]) {
        let payload = encode_bytes(&mut make(), data);
        let decoded = decode_bytes(&mut make(), &payload, data.len());
        assert_eq!(decoded, data, "roundtrip mismatch");
    }

    #[test]
    fn order0_static_roundtrip_skewed() {
        let data: Vec<u8> = (0..4096).map(|i| ((i * 13) % 7) as u8).collect();
        roundtrip(|| Order0Static::fit(&data), &data);
    }

    #[test]
    fn order0_static_roundtrip_single_symbol() {
        let data = vec![0xABu8; 500];
        roundtrip(|| Order0Static::fit(&data), &data);
    }

    #[test]
    fn order0_static_table_roundtrip() {
        let data: Vec<u8> = (0..1000).map(|i| (i % 5) as u8).collect();
        let m = Order0Static::fit(&data);
        let bytes = m.serialize();
        let m2 = Order0Static::deserialize(&bytes).unwrap();
        assert_eq!(m.counts, m2.counts);
    }

    #[test]
    fn order0_static_deserialize_rejects_malformed_table() {
        // Untrusted input: an all-zero table (sum 0) must be rejected rather
        // than driving a degenerate decode.
        assert!(Order0Static::deserialize(&[0u8; 1024]).is_err());
        // Wrong length is also rejected.
        assert!(Order0Static::deserialize(&[0u8; 16]).is_err());
    }

    #[test]
    fn order0_adaptive_roundtrip() {
        let data: Vec<u8> = (0..8192).map(|i| ((i * 31) & 0x3F) as u8).collect();
        roundtrip(Order0Adaptive::new, &data);
    }

    #[test]
    fn order1_adaptive_roundtrip() {
        // Markov-ish stream: each byte depends on the previous.
        let mut data = vec![0u8];
        for i in 1..8192 {
            data.push(data[i - 1].wrapping_add((i % 3) as u8));
        }
        roundtrip(Order1Adaptive::new, &data);
    }

    #[test]
    fn empty_roundtrips() {
        roundtrip(|| Order0Static::fit(&[]), &[]);
        roundtrip(Order0Adaptive::new, &[]);
        roundtrip(Order1Adaptive::new, &[]);
    }

    #[test]
    fn order0_static_compresses_skewed_below_one_byte_per_symbol() {
        let data: Vec<u8> = (0..16384)
            .map(|i| if i % 16 == 0 { 1u8 } else { 0u8 })
            .collect();
        let payload = encode_bytes(&mut Order0Static::fit(&data), &data);
        // Plus the 1 KiB table this still beats raw on a 16 KiB plane.
        assert!(
            payload.len() < data.len(),
            "expected compression, got {} >= {}",
            payload.len(),
            data.len()
        );
    }

    #[test]
    fn order1_static_roundtrip_markov() {
        let mut data = vec![0u8];
        for i in 1..16384 {
            data.push(data[i - 1].wrapping_add((i % 5) as u8));
        }
        roundtrip(|| Order1Static::fit(&data), &data);
    }

    #[test]
    fn order1_static_roundtrip_single_symbol() {
        let data = vec![0x42u8; 4096];
        roundtrip(|| Order1Static::fit(&data), &data);
    }

    #[test]
    fn order1_static_empty_roundtrips() {
        roundtrip(|| Order1Static::fit(&[]), &[]);
    }

    #[test]
    fn order1_static_state_roundtrip_and_validation() {
        let data: Vec<u8> = (0..8192).map(|i| ((i * 7) % 13) as u8).collect();
        let bytes = Order1Static::fit(&data).serialize();
        let reloaded = Order1Static::deserialize(&bytes).expect("valid state");
        // Decode through the reloaded model must reproduce the source.
        let payload = encode_bytes(&mut Order1Static::fit(&data), &data);
        let mut model = reloaded;
        let dec = decode_bytes(&mut model, &payload, data.len());
        assert_eq!(dec, data);
        // Truncated state is rejected.
        assert!(Order1Static::deserialize(&bytes[..bytes.len() - 1]).is_err());
        // Zeroing the marginal's first u16 breaks the row sum → rejected.
        let mut corrupt = bytes.clone();
        corrupt[32] = 0;
        corrupt[33] = 0;
        assert!(Order1Static::deserialize(&corrupt).is_err());
    }
}
