//! Tabled ANS (tANS / Finite State Entropy) entropy codec.
//!
//! A third entropy coder peer alongside [`crate::entropy::huffman`] and
//! [`crate::entropy::rans`]. Where Huffman wastes the fractional bits that
//! integer code lengths can't represent, tANS — like rANS — codes at
//! (near-)entropy cost; the *tabled* variant does it with a single
//! table lookup per symbol on both encode and decode, the construction
//! Zstd uses internally (FSE). See the 2024 ANS survey
//! (<https://arxiv.org/abs/2408.07322>) for the modern treatment.
//!
//! ## Design
//!
//! Single-stream, byte-alphabet, with an 11-bit normalized frequency
//! table (`TABLE_LOG = 11`, `L = 2048`) to match rANS's precision. The
//! coder is the textbook tANS with an explicit fixed start state, which
//! keeps the encode/decode bit accounting exactly symmetric (`n` encode
//! transitions ↔ `n` decode transitions, one flush ↔ one init) and
//! sidesteps the asymmetric first-symbol initialization micro-optimization
//! that the Zstd/FSE reference uses. The decoder recovers the start state
//! and verifies the bitstream is fully consumed, so a corrupt payload
//! fails rather than producing garbage.
//!
//! 4-stream interleaving (the shared `stream_frame` framing the Huffman /
//! rANS coders use for parallel decode) is a deliberate follow-up — this
//! single-stream form is a complete, correct dispatcher peer.
//!
//! ## Blob layout (what [`compress`] writes / [`decompress`] reads)
//!
//! ```text
//! [ table_log : u8 ]
//! [ nsym      : u16 LE ]                 number of symbols with freq > 0
//! nsym × [ sym : u8 ][ freq : u16 LE ]   normalized frequencies (sum = 2^table_log)
//! [ total_bits : u64 LE ]                length of the FSE bitstream in bits
//! [ bitstream : ⌈total_bits / 8⌉ bytes ]
//! ```

use crate::error::PtwmCoreError;

/// Table-size log. `L = 1 << TABLE_LOG` is the normalized-frequency total
/// and the number of decode-table slots. 11 mirrors rANS's precision and
/// leaves ample headroom over the 256-symbol byte alphabet.
const TABLE_LOG: u32 = 11;
const L: usize = 1 << TABLE_LOG;

/// `floor(log2(x))` for `x >= 1`.
#[inline]
fn highbit(x: u32) -> u32 {
    debug_assert!(x >= 1);
    31 - x.leading_zeros()
}

// ---------------------------------------------------------------------------
// Bit I/O — LIFO (the decoder pops chunks in the reverse order the encoder
// pushed them, which is what an ANS bitstream requires).
// ---------------------------------------------------------------------------

struct BitWriter {
    buf: Vec<u8>,
    acc: u64,
    nbits: u32,
    total: u64,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            acc: 0,
            nbits: 0,
            total: 0,
        }
    }

    /// Append the low `nb` bits of `value` (LSB first). `nb` may be 0.
    #[inline]
    fn add_bits(&mut self, value: u64, nb: u32) {
        if nb == 0 {
            return;
        }
        let mask = if nb >= 64 { u64::MAX } else { (1u64 << nb) - 1 };
        self.acc |= (value & mask) << self.nbits;
        self.nbits += nb;
        self.total += nb as u64;
        while self.nbits >= 8 {
            self.buf.push(self.acc as u8);
            self.acc >>= 8;
            self.nbits -= 8;
        }
    }

    fn finish(mut self) -> (Vec<u8>, u64) {
        if self.nbits > 0 {
            self.buf.push(self.acc as u8);
        }
        (self.buf, self.total)
    }
}

struct BitReader<'a> {
    buf: &'a [u8],
    /// Index one past the highest unread bit (the LIFO top).
    pos: u64,
}

impl<'a> BitReader<'a> {
    fn new(buf: &'a [u8], total_bits: u64) -> Self {
        Self {
            buf,
            pos: total_bits,
        }
    }

    /// Pop the most-recently-written `nb` bits, returning them with the
    /// same LSB-first convention [`BitWriter::add_bits`] used.
    #[inline]
    fn read(&mut self, nb: u32) -> Result<u64, PtwmCoreError> {
        if nb == 0 {
            return Ok(0);
        }
        if self.pos < nb as u64 {
            return Err(PtwmCoreError::CodecDecode {
                codec: "tans",
                msg: "bitstream underrun".into(),
            });
        }
        // `nb` is always 1..=TABLE_LOG (11) and the start bit's offset within
        // its byte is 0..=7, so the requested run spans at most 3 bytes
        // (`bit_offset + nb - 1 <= 17 < 24`). Load those bytes LSB-first, shift
        // the run down to bit 0, and mask — bit-exact with the per-bit loop but
        // without the inner branch. `byte_idx` is always in bounds:
        // `start = pos - nb <= total_bits - 1 < buf.len() * 8`.
        let start = self.pos - nb as u64;
        let byte_idx = (start >> 3) as usize;
        let bit_offset = (start & 7) as u32;
        let mut window = self.buf[byte_idx] as u64;
        if byte_idx + 1 < self.buf.len() {
            window |= (self.buf[byte_idx + 1] as u64) << 8;
        }
        if byte_idx + 2 < self.buf.len() {
            window |= (self.buf[byte_idx + 2] as u64) << 16;
        }
        let v = (window >> bit_offset) & ((1u64 << nb) - 1);
        self.pos = start;
        Ok(v)
    }
}

// ---------------------------------------------------------------------------
// Frequency normalization: scale a histogram so it sums to exactly `L`,
// every present symbol keeping a count of at least 1.
// ---------------------------------------------------------------------------

fn normalize(counts: &[u32; 256], total: u64) -> [u32; 256] {
    let mut norm = [0u32; 256];
    let mut sum = 0u32;
    let mut max_sym = 0usize;
    let mut max_count = 0u32;
    for (s, &c) in counts.iter().enumerate() {
        if c == 0 {
            continue;
        }
        let mut v = ((c as u64 * L as u64) / total) as u32;
        if v == 0 {
            v = 1;
        }
        norm[s] = v;
        sum += v;
        if c > max_count {
            max_count = c;
            max_sym = s;
        }
    }
    // Correct the rounding drift to hit `L` exactly. Surplus is shaved off
    // the currently-largest counts (never below 1); deficit is added to the
    // most frequent symbol, whose share dwarfs a few-unit adjustment.
    while sum > L as u32 {
        let mut bi = usize::MAX;
        let mut bv = 1u32;
        for (s, &v) in norm.iter().enumerate() {
            if v > bv {
                bv = v;
                bi = s;
            }
        }
        // `bi` is always valid here: sum > L ≥ nsym means some norm > 1.
        norm[bi] -= 1;
        sum -= 1;
    }
    while sum < L as u32 {
        norm[max_sym] += 1;
        sum += 1;
    }
    norm
}

/// Spread symbols across the `L` table slots in the FSE step order. With
/// `sum(norm) == L` and an odd step coprime to the power-of-two `L`, every
/// slot is assigned exactly once.
fn spread(norm: &[u32; 256]) -> Vec<u8> {
    let mut table = vec![0u8; L];
    let step = (L >> 1) + (L >> 3) + 3;
    let mask = L - 1;
    let mut pos = 0usize;
    for (s, &count) in norm.iter().enumerate() {
        for _ in 0..count {
            table[pos] = s as u8;
            pos = (pos + step) & mask;
        }
    }
    table
}

// ---------------------------------------------------------------------------
// Encode
// ---------------------------------------------------------------------------

/// Per-symbol encode transform (the FSE `symbolTT`). `delta_nb_bits` uses
/// wrapping `u32` fixed-point arithmetic exactly as the reference does.
#[derive(Clone, Copy)]
struct SymbolTransform {
    delta_nb_bits: u32,
    delta_find_state: i32,
}

fn build_encode(norm: &[u32; 256], sym: &[u8]) -> (Vec<u16>, [SymbolTransform; 256]) {
    // State table: for each slot `u` (in spread order) of symbol `s`, the
    // next encoder state `L + u`, packed contiguously per symbol.
    let mut cumul = [0u32; 256];
    let mut running = 0u32;
    for (s, &n) in norm.iter().enumerate() {
        cumul[s] = running;
        running += n;
    }
    let mut fill = cumul;
    let mut state_table = vec![0u16; L];
    for (u, &s8) in sym.iter().enumerate() {
        let s = s8 as usize;
        state_table[fill[s] as usize] = (L + u) as u16;
        fill[s] += 1;
    }

    let mut tt = [SymbolTransform {
        delta_nb_bits: 0,
        delta_find_state: 0,
    }; 256];
    let mut total = 0i32;
    for (s, &n) in norm.iter().enumerate() {
        if n == 0 {
            continue;
        }
        if n == 1 {
            tt[s] = SymbolTransform {
                delta_nb_bits: (TABLE_LOG << 16).wrapping_sub(L as u32),
                delta_find_state: total - 1,
            };
            total += 1;
        } else {
            let max_bits = TABLE_LOG - highbit(n - 1);
            let min_state_plus = n << max_bits;
            tt[s] = SymbolTransform {
                delta_nb_bits: (max_bits << 16).wrapping_sub(min_state_plus),
                delta_find_state: total - n as i32,
            };
            total += n as i32;
        }
    }
    (state_table, tt)
}

/// tANS-encode `src`. Returns the encoded length written to `dst`, or 0 to
/// signal "store raw" (incompressible, degenerate alphabet, or the encoded
/// form would not be smaller than the input).
pub fn compress(dst: &mut [u8], src: &[u8]) -> Result<usize, PtwmCoreError> {
    let n = src.len();
    if n == 0 {
        return Ok(0);
    }
    let mut counts = [0u32; 256];
    for &b in src {
        counts[b as usize] += 1;
    }
    let nsym = counts.iter().filter(|&&c| c != 0).count();
    // A single-symbol (or empty-alphabet) plane has no entropy structure
    // for tANS to exploit; let the raw fallback / other codecs handle it.
    if nsym < 2 {
        return Ok(0);
    }

    let norm = normalize(&counts, n as u64);
    let sym = spread(&norm);
    let (state_table, tt) = build_encode(&norm, &sym);

    // Encode the symbols in reverse from a fixed start state; the decoder
    // walks them back out in forward order.
    let mut writer = BitWriter::new();
    let mut value: u32 = L as u32;
    for &b in src.iter().rev() {
        let t = tt[b as usize];
        let nb = value.wrapping_add(t.delta_nb_bits) >> 16;
        writer.add_bits(value as u64, nb);
        let idx = (value >> nb) as i32 + t.delta_find_state;
        value = state_table[idx as usize] as u32;
    }
    // Flush the final state (its low TABLE_LOG bits == state - L).
    writer.add_bits(value as u64, TABLE_LOG);
    let (bitstream, total_bits) = writer.finish();

    // Assemble the blob and bail to raw if it doesn't actually shrink.
    let header_len = 1 + 2 + nsym * 3 + 8;
    let blob_len = header_len + bitstream.len();
    if blob_len >= n || blob_len > dst.len() {
        return Ok(0);
    }

    let mut w = 0usize;
    dst[w] = TABLE_LOG as u8;
    w += 1;
    dst[w..w + 2].copy_from_slice(&(nsym as u16).to_le_bytes());
    w += 2;
    for (s, &freq) in norm.iter().enumerate() {
        if freq == 0 {
            continue;
        }
        dst[w] = s as u8;
        dst[w + 1..w + 3].copy_from_slice(&(freq as u16).to_le_bytes());
        w += 3;
    }
    dst[w..w + 8].copy_from_slice(&total_bits.to_le_bytes());
    w += 8;
    dst[w..w + bitstream.len()].copy_from_slice(&bitstream);
    w += bitstream.len();
    Ok(w)
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct DecodeEntry {
    symbol: u8,
    nb_bits: u8,
    baseline: u16,
}

fn build_decode(norm: &[u32; 256], sym: &[u8]) -> Vec<DecodeEntry> {
    let mut next = *norm;
    let mut dt = vec![
        DecodeEntry {
            symbol: 0,
            nb_bits: 0,
            baseline: 0,
        };
        L
    ];
    for (u, slot) in dt.iter_mut().enumerate() {
        let s = sym[u] as usize;
        let x = next[s];
        next[s] += 1;
        let nb = TABLE_LOG - highbit(x);
        let baseline = (x << nb) - L as u32;
        *slot = DecodeEntry {
            symbol: s as u8,
            nb_bits: nb as u8,
            baseline: baseline as u16,
        };
    }
    dt
}

/// Decode a [`compress`] blob into `dst`. `dst.len()` is the symbol count.
pub fn decompress(dst: &mut [u8], src: &[u8]) -> Result<usize, PtwmCoreError> {
    let n = dst.len();
    if n == 0 {
        return Ok(0);
    }
    let err = |msg: &'static str| PtwmCoreError::CodecDecode {
        codec: "tans",
        msg: msg.into(),
    };

    // ── header ──────────────────────────────────────────────────────────
    if src.len() < 3 {
        return Err(err("blob too short for header"));
    }
    let table_log = src[0] as u32;
    if table_log != TABLE_LOG {
        return Err(err("unsupported table_log"));
    }
    let nsym = u16::from_le_bytes([src[1], src[2]]) as usize;
    if !(2..=256).contains(&nsym) {
        return Err(err("bad symbol count"));
    }
    let mut pos = 3usize;
    let pairs_end = pos + nsym * 3;
    if pairs_end + 8 > src.len() {
        return Err(err("truncated frequency table"));
    }
    let mut norm = [0u32; 256];
    let mut sum = 0u32;
    for _ in 0..nsym {
        let s = src[pos] as usize;
        let freq = u16::from_le_bytes([src[pos + 1], src[pos + 2]]) as u32;
        if norm[s] != 0 {
            return Err(err("duplicate symbol in frequency table"));
        }
        if freq == 0 {
            return Err(err("zero frequency in table"));
        }
        norm[s] = freq;
        sum += freq;
        pos += 3;
    }
    if sum != L as u32 {
        return Err(err("frequencies do not sum to table size"));
    }
    let total_bits = u64::from_le_bytes(src[pos..pos + 8].try_into().unwrap());
    pos += 8;
    let bitstream = &src[pos..];
    if total_bits > (bitstream.len() as u64) * 8 {
        return Err(err("bitstream shorter than declared bit length"));
    }

    // ── decode ──────────────────────────────────────────────────────────
    let sym = spread(&norm);
    let dt = build_decode(&norm, &sym);

    let mut reader = BitReader::new(bitstream, total_bits);
    let mut state = reader.read(TABLE_LOG)? as usize;
    if state >= L {
        return Err(err("initial state out of range"));
    }
    for slot in dst.iter_mut() {
        let e = dt[state];
        *slot = e.symbol;
        let low = reader.read(e.nb_bits as u32)?;
        state = e.baseline as usize + low as usize;
        if state >= L {
            return Err(err("state out of range"));
        }
    }
    // Symmetric accounting: a well-formed stream is fully consumed and the
    // decoder returns to the encoder's fixed start state (L → decoder 0).
    if reader.pos != 0 {
        return Err(err("trailing bits after decode"));
    }
    if state != 0 {
        return Err(err("decode did not return to start state"));
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip `src` through compress/decompress, asserting exact
    /// recovery. Returns the encoded length (0 = stored raw).
    fn round_trip(src: &[u8]) -> usize {
        let mut enc = vec![0u8; src.len() + 1024];
        let n = compress(&mut enc, src).unwrap();
        if n == 0 {
            return 0;
        }
        let mut dec = vec![0u8; src.len()];
        let got = decompress(&mut dec, &enc[..n]).unwrap();
        assert_eq!(got, src.len());
        assert_eq!(dec, src, "round-trip mismatch");
        n
    }

    #[test]
    fn roundtrip_skewed() {
        // Heavily skewed → should compress well.
        let src: Vec<u8> = (0..4096).map(|i| ((i * 13) % 17) as u8).collect();
        let n = round_trip(&src);
        assert!(
            (1..src.len()).contains(&n),
            "skewed data should compress: {n}"
        );
    }

    #[test]
    fn roundtrip_two_symbols() {
        let mut src = vec![0u8; 3000];
        src.extend_from_slice(&[1u8; 1000]);
        let n = round_trip(&src);
        assert!((1..src.len()).contains(&n));
    }

    #[test]
    fn roundtrip_low_entropy_runs() {
        // Resembles a byte-split exponent plane: a few dominant values.
        let mut src = Vec::new();
        for block in 0..400u32 {
            let byte = (block % 6) as u8 + 0x7C;
            src.extend(std::iter::repeat_n(byte, 32));
        }
        let n = round_trip(&src);
        assert!(
            (1..src.len() / 2).contains(&n),
            "low-entropy should compress: {n}"
        );
    }

    #[test]
    fn roundtrip_full_alphabet_skewed() {
        // All 256 byte values present but with a skewed distribution.
        let mut src = Vec::new();
        for v in 0u32..256 {
            let reps = 1 + (v % 7);
            src.extend(std::iter::repeat_n(v as u8, reps as usize));
        }
        // Make it large enough that the header amortizes.
        let src: Vec<u8> = src.iter().cycle().take(16384).copied().collect();
        round_trip(&src);
    }

    #[test]
    fn roundtrip_pseudo_random() {
        // Near-uniform: likely stored raw (Ok(0)), but if encoded it must
        // round-trip exactly.
        let mut state = 0x1234_5678u32;
        let src: Vec<u8> = (0..8192)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 24) as u8
            })
            .collect();
        round_trip(&src);
    }

    #[test]
    fn empty_returns_zero() {
        let mut dst = vec![0u8; 16];
        assert_eq!(compress(&mut dst, &[]).unwrap(), 0);
    }

    #[test]
    fn single_symbol_returns_zero() {
        let mut dst = vec![0u8; 2048];
        assert_eq!(compress(&mut dst, &[7u8; 1024]).unwrap(), 0);
    }

    #[test]
    fn decompress_rejects_garbage() {
        let src: Vec<u8> = (0..512).map(|i| (i * 31 + 7) as u8).collect();
        let mut dst = vec![0u8; 256];
        assert!(decompress(&mut dst, &src).is_err());
    }

    #[test]
    fn decompress_rejects_truncated() {
        let src: Vec<u8> = (0..8192).map(|i| ((i * 13) % 11) as u8).collect();
        let mut enc = vec![0u8; src.len() + 1024];
        let n = compress(&mut enc, &src).unwrap();
        assert!(n > 32);
        let mut dst = vec![0u8; src.len()];
        assert!(decompress(&mut dst, &enc[..n / 2]).is_err());
    }

    #[test]
    fn normalize_sums_to_table_size() {
        // Pathological near-uniform full alphabet: the surplus-shaving path.
        let mut counts = [0u32; 256];
        for c in counts.iter_mut() {
            *c = 16;
        }
        let norm = normalize(&counts, 256 * 16);
        assert_eq!(norm.iter().sum::<u32>(), L as u32);
        assert!(norm.iter().all(|&v| v >= 1));
    }

    #[test]
    fn roundtrip_many_sizes() {
        for &len in &[2usize, 3, 5, 17, 63, 64, 255, 1000, 4095, 4096, 65537] {
            let src: Vec<u8> = (0..len).map(|i| ((i * 7) % 13) as u8).collect();
            round_trip(&src);
        }
    }
}
