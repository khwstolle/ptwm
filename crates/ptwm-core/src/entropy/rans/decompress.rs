//! rANS decompression: parse freq table → build decode table → 4-stream interleaved decode.

use crate::entropy::stream_frame::{read_jump_table, split_four};
use crate::error::PtwmCoreError;

/// Probability precision in bits (must match compress.rs).
const PROB_BITS: u32 = 11;
const PROB_SCALE: u32 = 1 << PROB_BITS;
const PROB_MASK: u32 = PROB_SCALE - 1;

/// Lower bound of the rANS state (must match compress.rs).
const RANS_L: u32 = 1 << 23;

/// Decode table entry: symbol + its frequency and cumulative start.
#[derive(Clone, Copy)]
struct DecEntry {
    symbol: u8,
    freq: u16,
    start: u16,
}

/// Parse the frequency table from the compressed stream.
/// Returns (freqs, bytes_consumed).
fn decode_freq_table(src: &[u8]) -> Result<([u16; 256], usize), PtwmCoreError> {
    if src.is_empty() {
        return Err(PtwmCoreError::RansDecompress("empty input".to_string()));
    }

    let max_sym = src[0] as usize;
    let needed = 1 + 2 * (max_sym + 1);
    if src.len() < needed {
        return Err(PtwmCoreError::RansDecompress(
            "truncated frequency table".to_string(),
        ));
    }

    let mut freqs = [0u16; 256];
    for i in 0..=max_sym {
        let offset = 1 + i * 2;
        freqs[i] = u16::from_le_bytes([src[offset], src[offset + 1]]);
    }

    // Validate frequencies sum to PROB_SCALE.
    let total: u32 = freqs.iter().map(|&f| f as u32).sum();
    if total != PROB_SCALE {
        return Err(PtwmCoreError::RansDecompress(format!(
            "frequency table sums to {total}, expected {PROB_SCALE}"
        )));
    }

    Ok((freqs, needed))
}

/// Build a flat decode table of PROB_SCALE entries.
fn build_decode_table(freqs: &[u16; 256]) -> Vec<DecEntry> {
    let mut table = vec![
        DecEntry {
            symbol: 0,
            freq: 0,
            start: 0,
        };
        PROB_SCALE as usize
    ];
    let mut pos = 0u16;
    for sym in 0..256u16 {
        let freq = freqs[sym as usize];
        for j in 0..freq {
            table[(pos + j) as usize] = DecEntry {
                symbol: sym as u8,
                freq,
                start: pos,
            };
        }
        pos += freq;
    }
    table
}

/// Decode a single stream. Writes `out_len` symbols to `out`.
fn decode_stream(stream: &[u8], table: &[DecEntry], out: &mut [u8]) -> Result<(), PtwmCoreError> {
    if stream.len() < 4 {
        return Err(PtwmCoreError::RansDecompress(
            "stream too short for initial state".to_string(),
        ));
    }

    let mut state = u32::from_le_bytes([stream[0], stream[1], stream[2], stream[3]]);
    let mut pos = 4usize;

    for slot in out.iter_mut() {
        let idx = state & PROB_MASK;
        let entry = table[idx as usize];
        *slot = entry.symbol;

        // Advance state: x' = freq * (x >> PROB_BITS) + (x & PROB_MASK) - start
        state = entry.freq as u32 * (state >> PROB_BITS) + idx - entry.start as u32;

        // Renormalize: read bytes while state < RANS_L.
        while state < RANS_L {
            if pos >= stream.len() {
                return Err(PtwmCoreError::RansDecompress(
                    "unexpected end of stream during renormalization".to_string(),
                ));
            }
            state = (state << 8) | stream[pos] as u32;
            pos += 1;
        }
    }

    Ok(())
}

pub(crate) fn decompress(dst: &mut [u8], src: &[u8]) -> Result<usize, PtwmCoreError> {
    // Parse frequency table.
    let (freqs, freq_len) = decode_freq_table(src)?;
    let mut src_pos = freq_len;

    // Parse jump table (4 x u32 LE).
    let (lens_u32, consumed) = read_jump_table(&src[src_pos..])
        .map_err(|e| PtwmCoreError::RansDecompress(e.to_string()))?;
    src_pos += consumed;
    let lens: [usize; 4] = [
        lens_u32[0] as usize,
        lens_u32[1] as usize,
        lens_u32[2] as usize,
        lens_u32[3] as usize,
    ];

    let tail = src.len() - src_pos;
    let total = lens
        .iter()
        .try_fold(0usize, |acc, &l| acc.checked_add(l))
        .ok_or_else(|| PtwmCoreError::RansDecompress("jump table overflow".to_string()))?;

    if total != tail {
        return Err(PtwmCoreError::RansDecompress(format!(
            "jump table sum {total} does not match payload tail {tail}"
        )));
    }

    let s0_end = src_pos + lens[0];
    let s1_end = s0_end + lens[1];
    let s2_end = s1_end + lens[2];
    let s3_end = s2_end + lens[3];
    let streams = [
        &src[src_pos..s0_end],
        &src[s0_end..s1_end],
        &src[s1_end..s2_end],
        &src[s2_end..s3_end],
    ];

    // Build decode table.
    let table = build_decode_table(&freqs);

    // Decode each stream into its segment of the output.
    let segments = split_four(dst.len());
    for i in 0..4 {
        let (start, end) = segments[i];
        if start == end {
            continue;
        }
        decode_stream(streams[i], &table, &mut dst[start..end])?;
    }

    Ok(dst.len())
}
