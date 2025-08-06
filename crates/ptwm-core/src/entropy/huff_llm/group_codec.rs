//! Canonical, length-limited Huffman over a small symbol alphabet.
//!
//! Used by the `huff_llm_5bit` codec to code each bit-field group of a 16-bit
//! float independently. Alphabets are tiny (`n_symbols` ∈ {32, 128}), so the
//! decoder is canonical-counts based rather than a flat lookup table. In the
//! Huff-LLM hardware reference, each 5-bit group maps to a 32-entry CAM; this
//! software decoder reproduces the same code assignment bit-serially.

use crate::error::PtwmCoreError;

/// Maximum canonical code length. Bounds the per-symbol length byte in the
/// serialized table and the decoder's per-length walk. 15 keeps every code in a
/// `u16` and is far above the natural Huffman depth of a ≤128-symbol alphabet.
pub const MAX_GROUP_CODE_LEN: u8 = 15;

/// Packs bits MSB-first into a growable byte buffer.
#[derive(Default)]
pub struct BitWriter {
    bytes: Vec<u8>,
    cur: u8,
    nbits: u8,
}

impl BitWriter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Write the low `len` bits of `code`, most-significant bit first.
    pub fn write_bits(&mut self, code: u16, len: u8) {
        for i in (0..len).rev() {
            let bit = ((code >> i) & 1) as u8;
            self.cur = (self.cur << 1) | bit;
            self.nbits += 1;
            if self.nbits == 8 {
                self.bytes.push(self.cur);
                self.cur = 0;
                self.nbits = 0;
            }
        }
    }

    /// Flush any partial byte (left-justified) and return the buffer.
    pub fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            self.cur <<= 8 - self.nbits;
            self.bytes.push(self.cur);
        }
        self.bytes
    }
}

/// Reads bits MSB-first from a byte slice. Reads past the end yield 0 bits; the
/// caller bounds symbol count and code length independently, so an over-read
/// cannot corrupt output or loop unbounded.
pub struct BitReader<'a> {
    bytes: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
}

impl<'a> BitReader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    #[inline]
    pub fn read_bit(&mut self) -> u8 {
        if self.byte_pos >= self.bytes.len() {
            return 0;
        }
        let bit = (self.bytes[self.byte_pos] >> (7 - self.bit_pos)) & 1;
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
        bit
    }

    /// Bits not yet consumed. Used to bound the requested symbol count before
    /// allocating, so a corrupt header claiming a huge count against a tiny
    /// stream cannot drive an oversized allocation.
    #[inline]
    pub fn remaining_bits(&self) -> usize {
        (self.bytes.len() - self.byte_pos)
            .saturating_mul(8)
            .saturating_sub(self.bit_pos as usize)
    }
}

/// Build canonical code lengths from symbol frequencies. Returns a vector of
/// length `freqs.len()`; absent symbols get length 0. Lengths are clamped to
/// `MAX_GROUP_CODE_LEN` with a Kraft-inequality repair (same algorithm as
/// `entropy::huffman::tree::enforce_max_code_length`, generalized to a variable
/// alphabet).
fn build_lengths(freqs: &[u32]) -> Vec<u8> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let n = freqs.len();
    let mut lengths = vec![0u8; n];
    let present: Vec<usize> = (0..n).filter(|&i| freqs[i] > 0).collect();
    match present.len() {
        0 => return lengths,
        1 => {
            lengths[present[0]] = 1;
            return lengths;
        }
        _ => {}
    }

    let m = present.len();
    let max_nodes = 2 * m;
    let mut node_freq = vec![0u64; max_nodes];
    let mut left = vec![usize::MAX; max_nodes];
    let mut right = vec![usize::MAX; max_nodes];
    for (id, &sym) in present.iter().enumerate() {
        node_freq[id] = freqs[sym] as u64;
    }
    let mut heap: BinaryHeap<Reverse<(u64, usize)>> =
        (0..m).map(|id| Reverse((node_freq[id], id))).collect();
    let mut next = m;
    while heap.len() > 1 {
        let Reverse((f1, a)) = heap.pop().unwrap();
        let Reverse((f2, b)) = heap.pop().unwrap();
        node_freq[next] = f1 + f2;
        left[next] = a;
        right[next] = b;
        heap.push(Reverse((f1 + f2, next)));
        next += 1;
    }
    let root = heap.pop().unwrap().0.1;
    let mut stack = vec![(root, 0u8)];
    while let Some((id, depth)) = stack.pop() {
        if id < m {
            lengths[present[id]] = depth;
        } else {
            let d = depth.saturating_add(1);
            if left[id] != usize::MAX {
                stack.push((left[id], d));
            }
            if right[id] != usize::MAX {
                stack.push((right[id], d));
            }
        }
    }
    enforce_max_len(&mut lengths, MAX_GROUP_CODE_LEN);
    lengths
}

fn enforce_max_len(lengths: &mut [u8], max_len: u8) {
    let mut clamped = false;
    for l in lengths.iter_mut() {
        if *l > max_len {
            *l = max_len;
            clamped = true;
        }
    }
    if !clamped {
        return;
    }
    let capacity = 1u32 << max_len;
    let kraft = |ls: &[u8]| -> u32 {
        ls.iter()
            .filter(|&&l| l > 0)
            .map(|&l| 1u32 << (max_len - l))
            .sum()
    };
    while kraft(lengths) > capacity {
        let victim = (0..lengths.len())
            .filter(|&s| lengths[s] > 0 && lengths[s] < max_len)
            .min_by_key(|&s| lengths[s]);
        match victim {
            Some(s) => lengths[s] += 1,
            None => break,
        }
    }
}

/// Assign canonical codes (MSB-first values) from code lengths.
fn canonical_codes(lengths: &[u8]) -> Vec<u16> {
    let n = lengths.len();
    let mut codes = vec![0u16; n];
    let max_len = lengths.iter().copied().max().unwrap_or(0);
    if max_len == 0 {
        return codes;
    }
    let mut bl_count = vec![0u16; (max_len + 1) as usize];
    for &l in lengths {
        if l > 0 {
            bl_count[l as usize] += 1;
        }
    }
    let mut next_code = vec![0u16; (max_len + 1) as usize];
    let mut code = 0u16;
    for bits in 1..=max_len as usize {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }
    for (sym, &l) in lengths.iter().enumerate() {
        if l != 0 {
            codes[sym] = next_code[l as usize];
            next_code[l as usize] += 1;
        }
    }
    codes
}

/// A built canonical Huffman code for one group: encode via `codes`/`lengths`,
/// decode via the canonical counts tables.
pub struct GroupCode {
    pub lengths: Vec<u8>,
    codes: Vec<u16>,
    // Canonical decode tables, indexed by code length 1..=max_len.
    max_len: u8,
    count: Vec<u32>,       // count[len] = symbols of that length
    first_code: Vec<u16>,  // first canonical code of that length
    first_index: Vec<u32>, // offset of that length's block in `sorted`
    sorted: Vec<u16>,      // symbols ordered by (length, symbol value)
}

impl GroupCode {
    pub fn from_freqs(freqs: &[u32]) -> Self {
        Self::from_lengths_unchecked(build_lengths(freqs))
    }

    /// Reconstruct from a serialized length table. Validates each length and
    /// that the resulting code is a complete-or-undercomplete prefix code.
    pub fn from_lengths(lengths: Vec<u8>) -> Result<Self, PtwmCoreError> {
        for &l in &lengths {
            if l > MAX_GROUP_CODE_LEN {
                return Err(PtwmCoreError::CodecDecode {
                    codec: "huff_llm_5bit",
                    msg: format!("group code length {l} exceeds max {MAX_GROUP_CODE_LEN}"),
                });
            }
        }
        let max_len = lengths.iter().copied().max().unwrap_or(0);
        if max_len > 0 {
            let cap = 1u32 << max_len;
            let kraft: u32 = lengths
                .iter()
                .filter(|&&l| l > 0)
                .map(|&l| 1u32 << (max_len - l))
                .sum();
            if kraft > cap {
                return Err(PtwmCoreError::CodecDecode {
                    codec: "huff_llm_5bit",
                    msg: "group code table is over-complete (undecodable)".into(),
                });
            }
        }
        Ok(Self::from_lengths_unchecked(lengths))
    }

    fn from_lengths_unchecked(lengths: Vec<u8>) -> Self {
        let codes = canonical_codes(&lengths);
        let max_len = lengths.iter().copied().max().unwrap_or(0);
        let mut count = vec![0u32; (max_len as usize) + 1];
        for &l in &lengths {
            if l > 0 {
                count[l as usize] += 1;
            }
        }
        // sorted symbols by (length, symbol)
        let mut sorted = Vec::new();
        let mut first_index = vec![0u32; (max_len as usize) + 1];
        for len in 1..=max_len as usize {
            first_index[len] = sorted.len() as u32;
            for (sym, &l) in lengths.iter().enumerate() {
                if l as usize == len {
                    sorted.push(sym as u16);
                }
            }
        }
        // first canonical code per length (mirror canonical_codes accumulation)
        let mut first_code = vec![0u16; (max_len as usize) + 1];
        let mut code = 0u16;
        for len in 1..=max_len as usize {
            code = (code + count.get(len - 1).copied().unwrap_or(0) as u16) << 1;
            first_code[len] = code;
        }
        Self {
            lengths,
            codes,
            max_len,
            count,
            first_code,
            first_index,
            sorted,
        }
    }

    #[inline]
    pub fn encode_symbol(&self, w: &mut BitWriter, sym: usize) {
        w.write_bits(self.codes[sym], self.lengths[sym]);
    }

    /// Decode `n` symbols from `r`. Errors if a code does not resolve within
    /// `max_len` bits (corrupt stream / over-read past a too-short payload).
    pub fn decode_symbols(&self, r: &mut BitReader, n: usize) -> Result<Vec<u16>, PtwmCoreError> {
        if n == 0 {
            return Ok(Vec::new());
        }
        // Every symbol consumes at least one bit (the shortest canonical code
        // is length 1), so a stream of `B` bits can yield at most `B` symbols.
        // Reject before allocating so a corrupt count cannot trigger an
        // oversized `Vec` against a tiny stream (decompression-bomb guard).
        if n > r.remaining_bits() {
            return Err(PtwmCoreError::CodecDecode {
                codec: "huff_llm_5bit",
                msg: "symbol count exceeds available stream bits".into(),
            });
        }
        let mut out = Vec::with_capacity(n);
        // Single-symbol code: length-1 codes, only one present symbol.
        for _ in 0..n {
            let mut code = 0u16;
            let mut len = 0u8;
            loop {
                len += 1;
                if len > self.max_len {
                    return Err(PtwmCoreError::CodecDecode {
                        codec: "huff_llm_5bit",
                        msg: "group code did not resolve within max length".into(),
                    });
                }
                code = (code << 1) | r.read_bit() as u16;
                let li = len as usize;
                if self.count[li] > 0 {
                    let delta = code.wrapping_sub(self.first_code[li]);
                    if (delta as u32) < self.count[li] {
                        let idx = self.first_index[li] + delta as u32;
                        out.push(self.sorted[idx as usize]);
                        break;
                    }
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(symbols: &[u16], n_alphabet: usize) {
        let mut freqs = vec![0u32; n_alphabet];
        for &s in symbols {
            freqs[s as usize] += 1;
        }
        let code = GroupCode::from_freqs(&freqs);
        // serialize lengths → rebuild (mirrors wire path)
        let rebuilt = GroupCode::from_lengths(code.lengths.clone()).unwrap();
        let mut w = BitWriter::new();
        for &s in symbols {
            rebuilt.encode_symbol(&mut w, s as usize);
        }
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        let decoded = rebuilt.decode_symbols(&mut r, symbols.len()).unwrap();
        assert_eq!(decoded, symbols);
    }

    #[test]
    fn roundtrip_skewed_32() {
        let mut syms = Vec::new();
        for i in 0..2000u32 {
            syms.push((i % 32) as u16);
        }
        // make it skewed
        syms.extend(std::iter::repeat_n(0u16, 5000));
        roundtrip(&syms, 32);
    }

    #[test]
    fn roundtrip_128() {
        let syms: Vec<u16> = (0..4000u32).map(|i| (i % 128) as u16).collect();
        roundtrip(&syms, 128);
    }

    #[test]
    fn single_symbol() {
        let syms = vec![7u16; 500];
        roundtrip(&syms, 32);
    }

    #[test]
    fn two_symbols() {
        let mut syms = vec![3u16; 1000];
        syms.extend(std::iter::repeat_n(9u16, 200));
        roundtrip(&syms, 32);
    }

    #[test]
    fn uniform_max_entropy_128() {
        let syms: Vec<u16> = (0..128u16).cycle().take(128 * 8).collect();
        roundtrip(&syms, 128);
    }

    #[test]
    fn empty_stream() {
        let code = GroupCode::from_freqs(&[0u32; 32]);
        let mut r = BitReader::new(&[]);
        assert!(code.decode_symbols(&mut r, 0).unwrap().is_empty());
    }

    #[test]
    fn length_limit_holds_under_max_skew() {
        // Geometric-ish frequencies that would push natural Huffman depth past
        // the limit. All lengths must stay ≤ MAX_GROUP_CODE_LEN.
        let mut freqs = vec![1u32; 128];
        freqs[0] = 1 << 30;
        for i in 1..20 {
            freqs[i] = 1 << (30 - i);
        }
        let code = GroupCode::from_freqs(&freqs);
        assert!(code.lengths.iter().all(|&l| l <= MAX_GROUP_CODE_LEN));
        // and it still round-trips
        let syms: Vec<u16> = (0..128u16).collect();
        let mut w = BitWriter::new();
        for &s in &syms {
            code.encode_symbol(&mut w, s as usize);
        }
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        assert_eq!(code.decode_symbols(&mut r, syms.len()).unwrap(), syms);
    }

    #[test]
    fn from_lengths_rejects_overlong() {
        let mut lengths = vec![0u8; 32];
        lengths[0] = MAX_GROUP_CODE_LEN + 1;
        assert!(GroupCode::from_lengths(lengths).is_err());
    }

    #[test]
    fn bitwriter_byte_alignment() {
        let mut w = BitWriter::new();
        w.write_bits(0b101, 3);
        w.write_bits(0b11, 2);
        let bytes = w.finish();
        // 5 bits 10111 left-justified → 1011_1000
        assert_eq!(bytes, vec![0b1011_1000]);
    }
}
