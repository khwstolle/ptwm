//! Minimal, correct 32-bit range coder for PerGroupCodebook / Order1ScaleAC.
//!
//! Encodes/decodes symbol streams under PMFs represented as quantized counts
//! out of a caller-supplied total.  Correctness over speed.
//!
//! ## Design
//!
//! Uses the LZMA-style carry-propagation range coder:
//! - `range` starts at `0xFFFF_FFFF` (`2^32 - 1`).
//! - `low` is tracked as a 64-bit integer; bit 32 signals a carry.
//! - The encoder emits a leading alignment byte (`0x00`) followed by data bytes.
//! - The decoder preloads 5 bytes at init (one is the alignment byte).
//! - Carry propagation is handled via a single-byte cache plus a pending-`0xFF` count.

/// Range encoder. Call [`Self::encode_symbol`] for each symbol, then [`Self::finish`].
pub struct RangeEncoder {
    low: u64,
    range: u64,
    /// Last byte that hasn't been written yet; might be incremented by a carry.
    cache: u8,
    /// Count of `0xFF` bytes waiting behind `cache`; they all become `0x00` on carry.
    cache_size: u32,
    out: Vec<u8>,
}

impl RangeEncoder {
    /// Create a new encoder.
    ///
    /// The first byte of the output is always `0x00` (alignment); the decoder
    /// accounts for it by preloading one extra byte at init.
    pub fn new() -> Self {
        Self {
            low: 0,
            range: 0xFFFF_FFFF,
            cache: 0, // leads to a 0x00 alignment byte being emitted first
            cache_size: 0,
            out: Vec::new(),
        }
    }

    /// Encode one symbol.
    ///
    /// # Parameters
    /// - `cumulative`: sum of counts for all symbols strictly below this one.
    /// - `probability`: this symbol's count.
    /// - `total`: PMF denominator.
    ///
    /// # Panics
    /// Panics if `probability == 0` or `cumulative + probability > total`.
    pub fn encode_symbol(&mut self, cumulative: u32, probability: u32, total: u32) {
        assert!(probability > 0, "probability must be > 0");
        assert!(
            cumulative + probability <= total,
            "cumulative + probability must be <= total"
        );

        let rpt = self.range / total as u64;
        self.low += cumulative as u64 * rpt;
        self.range = probability as u64 * rpt;

        // Renormalize while range is too small.
        while self.range < (1u64 << 24) {
            self.shift_low();
            self.range <<= 8;
        }
    }

    /// Flush remaining state and return the encoded bytes.
    ///
    /// The output always starts with the alignment byte `0x00` followed by
    /// data bytes and 5 flush bytes.
    pub fn finish(mut self) -> Vec<u8> {
        // Flush 5 bytes to commit the remaining `low` state.
        for _ in 0..5 {
            self.shift_low();
        }
        // Emit the final cached byte and any pending 0xFF chain.
        self.out.push(self.cache);
        for _ in 0..self.cache_size {
            self.out.push(0xFF);
        }
        self.out
    }

    /// Emit the pending byte with carry propagation.
    ///
    /// Carry is signalled by bit 32 of `self.low`.  After the call, `low`
    /// is shifted left by 8 with the top 24 bits kept.
    fn shift_low(&mut self) {
        let carry = (self.low >> 32) as u8; // 0 or 1
        let top = ((self.low >> 24) & 0xFF) as u8;

        if top == 0xFF && carry == 0 {
            // Current byte might be incremented by a later carry; defer.
            self.cache_size += 1;
        } else {
            // Emit the buffered byte (+ carry), then the pending 0xFF chain
            // (which becomes 0x00 on carry, or stays 0xFF without carry).
            self.out.push(self.cache.wrapping_add(carry));
            let fill: u8 = if carry != 0 { 0x00 } else { 0xFF };
            for _ in 0..self.cache_size {
                self.out.push(fill);
            }
            self.cache_size = 0;
            self.cache = top;
        }

        // Keep bottom 24 bits and shift up by one byte.
        self.low = (self.low & 0x00FF_FFFF) << 8;
    }
}

impl Default for RangeEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Range decoder.
pub struct RangeDecoder<'a> {
    low: u64,
    range: u64,
    /// 40-bit sliding window into the stream (alignment byte at MSB initially).
    code: u64,
    src: &'a [u8],
    pos: usize,
}

impl<'a> RangeDecoder<'a> {
    /// Initialise from a byte slice produced by [`RangeEncoder::finish`].
    ///
    /// Preloads 5 bytes (the first is the encoder's alignment `0x00`).
    pub fn new(src: &'a [u8]) -> Self {
        let mut code: u64 = 0;
        let mut pos = 0usize;

        // Load 5 bytes into the 40-bit `code` register (high byte first).
        while pos < 5 {
            let b = if pos < src.len() { src[pos] } else { 0 };
            code = (code << 8) | b as u64;
            pos += 1;
        }

        Self {
            low: 0,
            range: 0xFFFF_FFFF,
            code,
            src,
            pos,
        }
    }

    /// Return the cumulative target in `[0, total)` for the next symbol.
    ///
    /// The caller uses this to find which symbol's cumulative range contains
    /// the target, then calls [`Self::advance`] with that symbol's
    /// `(cumulative, probability, total)`.
    pub fn decode_symbol(&mut self, total: u32) -> u32 {
        let rpt = self.range / total as u64;
        // The effective 32-bit code value is the bottom 32 bits of the 40-bit register.
        let offset = (self.code & 0xFFFF_FFFF).wrapping_sub(self.low) & 0xFFFF_FFFF;
        (offset / rpt).min((total - 1) as u64) as u32
    }

    /// Consume the current symbol from the coder state.
    ///
    /// Must be called with the exact `(cumulative, probability, total)` of the
    /// symbol returned by the preceding [`Self::decode_symbol`] call.
    pub fn advance(&mut self, cumulative: u32, probability: u32, total: u32) {
        let rpt = self.range / total as u64;
        self.low += cumulative as u64 * rpt;
        self.range = probability as u64 * rpt;

        // Mirror the encoder's carry handling: mask low to 32 bits.
        // The stream already contains the carry-adjusted bytes.
        self.low &= 0xFFFF_FFFF;

        // Mirror the encoder's renormalization.
        while self.range < (1u64 << 24) {
            let b = self.next_byte() as u64;
            // Shift the 40-bit code register left by 8 and bring in a new byte.
            self.code = ((self.code & 0xFF_FFFF_FFFF) << 8) | b;
            // Shift low left by 8 (keep bottom 24 bits).
            self.low = (self.low & 0x00FF_FFFF) << 8;
            self.range <<= 8;
        }
    }

    fn next_byte(&mut self) -> u8 {
        let b = if self.pos < self.src.len() {
            self.src[self.pos]
        } else {
            0
        };
        self.pos += 1;
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Small PMF with known skew for round-trip test.
    const COUNTS: [u32; 10] = [100, 50, 20, 10, 5, 3, 2, 1, 1, 1];
    // sum == 193
    const TOTAL: u32 = 193;

    fn cumulative(idx: usize) -> u32 {
        COUNTS.iter().take(idx).sum()
    }

    fn sample_sequence(len: usize) -> Vec<u8> {
        // Deterministic "random" walk to produce skewed samples
        let mut rng_state: u64 = 0xBADC0FFEE0DDF00D;
        (0..len)
            .map(|_| {
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let mut t = (rng_state >> 33) as u32 % TOTAL;
                for (i, c) in COUNTS.iter().enumerate() {
                    if t < *c {
                        return i as u8;
                    }
                    t -= c;
                }
                9
            })
            .collect()
    }

    #[test]
    fn range_coder_roundtrip_10k_symbols() {
        let data = sample_sequence(10_000);
        let mut enc = RangeEncoder::new();
        for &s in &data {
            enc.encode_symbol(cumulative(s as usize), COUNTS[s as usize], TOTAL);
        }
        let coded = enc.finish();

        let mut dec = RangeDecoder::new(&coded);
        let mut decoded = Vec::with_capacity(data.len());
        for _ in 0..data.len() {
            let target = dec.decode_symbol(TOTAL);
            // Linear-scan the cumulative table to find the symbol
            let mut cum = 0u32;
            let mut sym = 0u8;
            for (i, c) in COUNTS.iter().enumerate() {
                let next = cum + c;
                if target < next {
                    sym = i as u8;
                    dec.advance(cum, *c, TOTAL);
                    break;
                }
                cum = next;
            }
            decoded.push(sym);
        }
        assert_eq!(decoded, data);
    }

    #[test]
    fn range_coder_roundtrip_single_symbol() {
        // Edge case: sequence of length 1
        let mut enc = RangeEncoder::new();
        enc.encode_symbol(cumulative(3), COUNTS[3], TOTAL);
        let coded = enc.finish();

        let mut dec = RangeDecoder::new(&coded);
        let target = dec.decode_symbol(TOTAL);
        let mut cum = 0u32;
        let mut sym = 0u8;
        for (i, c) in COUNTS.iter().enumerate() {
            let next = cum + c;
            if target < next {
                sym = i as u8;
                dec.advance(cum, *c, TOTAL);
                break;
            }
            cum = next;
        }
        assert_eq!(sym, 3);
    }
}
