//! rANS compression: frequency counting → normalization → 4-stream interleaved encoding.

use crate::entropy::outcome::CompressOutcome;
use crate::entropy::stream_frame::{JUMP_TABLE_BYTES, split_four, write_jump_table};
use crate::error::PtwmCoreError;

/// Probability precision in bits. Frequencies are normalized to sum to 1 << PROB_BITS.
const PROB_BITS: u32 = 11;
const PROB_SCALE: u32 = 1 << PROB_BITS;

/// Lower bound of the rANS state. State is kept in [RANS_L, RANS_L << 8).
const RANS_L: u32 = 1 << 23;

/// Encode symbol table entry used during compression.
///
/// Carries a precomputed reciprocal of `freq` (Alverson / ryg_rans method) so
/// the encode hot loop replaces the per-symbol integer division `state / freq`
/// — the dominant cost — with a multiply + shift. The state update becomes
/// `state + bias + q * cmpl_freq` where `q = floor(state / freq)`, which is
/// algebraically identical to `(state / freq) * M + (state % freq) + start`.
struct EncSymbol {
    freq: u32,
    cmpl_freq: u32, // PROB_SCALE - freq
    bias: u32,
    rcp_freq: u32,
    rcp_shift: u32,
    #[cfg(debug_assertions)]
    start: u32, // kept only to verify the reciprocal update in debug builds
}

impl EncSymbol {
    /// Build a symbol entry with the reciprocal of `freq` for division-free
    /// encoding. `start` is the exclusive cumulative frequency. Exact for the
    /// rANS state range `[RANS_L, RANS_L << 8)` (< 2^31).
    fn new(start: u32, freq: u32) -> Self {
        let cmpl_freq = PROB_SCALE - freq;
        if freq < 2 {
            // freq == 1 (present symbols always have freq >= 1; the freq == 0
            // entries for absent symbols are never referenced by the encoder).
            EncSymbol {
                freq,
                cmpl_freq,
                bias: start + PROB_SCALE - 1,
                rcp_freq: !0u32,
                rcp_shift: 0,
                #[cfg(debug_assertions)]
                start,
            }
        } else {
            let mut shift = 0u32;
            while freq > (1u32 << shift) {
                shift += 1;
            }
            // rcp_freq = ceil(2^(shift+31) / freq); fits in u32 because
            // shift ≈ ceil(log2(freq)).
            let rcp_freq = (1u64 << (shift + 31)).div_ceil(freq as u64) as u32;
            EncSymbol {
                freq,
                cmpl_freq,
                bias: start,
                rcp_freq,
                rcp_shift: shift - 1,
                #[cfg(debug_assertions)]
                start,
            }
        }
    }
}

/// Normalize raw frequencies so they sum to exactly `PROB_SCALE`.
/// Returns None if no symbols have non-zero frequency.
fn normalize_freqs(raw: &[u32; 256]) -> Option<([u16; 256], [u16; 256])> {
    let total: u64 = raw.iter().map(|&f| f as u64).sum();
    if total == 0 {
        return None;
    }

    let mut freqs = [0u16; 256];
    let mut assigned: u32 = 0;
    let mut max_sym = 0usize;
    let mut max_freq_sym = 0usize;

    // Initial proportional assignment: each symbol gets at least 1 if it appeared.
    for (i, &raw_f) in raw.iter().enumerate() {
        if raw_f > 0 {
            let f = ((raw_f as u64 * PROB_SCALE as u64) / total).max(1) as u16;
            freqs[i] = f;
            assigned += f as u32;
            max_sym = i;
            if raw[i] > raw[max_freq_sym] {
                max_freq_sym = i;
            }
        }
    }

    // Adjust the most frequent symbol to make the total exact.
    if assigned > PROB_SCALE {
        let excess = assigned - PROB_SCALE;
        if freqs[max_freq_sym] as u32 > excess {
            freqs[max_freq_sym] -= excess as u16;
        } else {
            // Shouldn't happen with reasonable data, but handle gracefully.
            return None;
        }
    } else if assigned < PROB_SCALE {
        freqs[max_freq_sym] += (PROB_SCALE - assigned) as u16;
    }

    // Build cumulative frequency table.
    let mut cum = [0u16; 256];
    let mut running = 0u16;
    for i in 0..=max_sym {
        cum[i] = running;
        running += freqs[i];
    }
    debug_assert_eq!(
        running as u32, PROB_SCALE,
        "normalized freqs must sum to PROB_SCALE"
    );

    Some((freqs, cum))
}

/// Encode the frequency table into the output.
///
/// Format: [max_symbol_index: u8] [freq[0]: u16 LE] [freq[1]: u16 LE] ... [freq[max_sym]: u16 LE]
/// Total: 1 + 2*(max_sym+1) bytes.
fn encode_freq_table(freqs: &[u16; 256], out: &mut Vec<u8>) -> usize {
    let max_sym = freqs.iter().rposition(|&f| f > 0).unwrap_or(0);
    out.push(max_sym as u8);
    for &f in &freqs[..=max_sym] {
        out.extend_from_slice(&f.to_le_bytes());
    }
    1 + 2 * (max_sym + 1)
}

/// rANS encode a single stream (processes symbols backward) into `scratch`.
/// `scratch` is cleared before use; on return it holds the encoded bytes in
/// forward-read order (`[final_state_le: 4 B, renorm bytes reversed]`).
fn encode_stream_into(src: &[u8], symbols: &[EncSymbol], scratch: &mut Vec<u8>) {
    scratch.clear();

    let mut state: u32 = RANS_L;
    // Collect renormalization bytes in reverse order (we'll reverse at the end).
    let reverse_start = 4;
    // Reserve 4 bytes for the final state; we'll fill them at the end.
    scratch.extend_from_slice(&[0u8; 4]);

    // Process symbols backward.
    for &byte in src.iter().rev() {
        let sym = &symbols[byte as usize];
        let freq = sym.freq;

        // Renormalize: output bytes while state is too large for this symbol.
        let max_state = ((RANS_L >> PROB_BITS) << 8) * freq;
        while state >= max_state {
            scratch.push((state & 0xFF) as u8);
            state >>= 8;
        }

        // Encode via reciprocal: q = floor(state / freq) without a division.
        // state' = state + bias + q * cmpl_freq is identical to
        // (state / freq) * PROB_SCALE + (state % freq) + start.
        let q = (((state as u64 * sym.rcp_freq as u64) >> 32) as u32) >> sym.rcp_shift;
        let new_state = state + sym.bias + q * sym.cmpl_freq;
        // Verify the reciprocal against the reference div/mod in debug builds.
        // An explicit cfg block (not just debug_assert!) so the `sym.start`
        // reference is removed entirely in release, where the field is absent.
        #[cfg(debug_assertions)]
        {
            assert_eq!(
                new_state,
                (state / freq) * PROB_SCALE + (state % freq) + sym.start,
                "rANS reciprocal update must match reference div/mod formula"
            );
        }
        state = new_state;
    }

    // Write final state into the reserved 4 bytes.
    scratch[..4].copy_from_slice(&state.to_le_bytes());
    // Reverse the renormalization bytes (everything after the state prefix)
    // so they're in the correct order for forward decoding.
    scratch[reverse_start..].reverse();
}

pub(crate) fn compress(dst: &mut [u8], src: &[u8]) -> Result<CompressOutcome, PtwmCoreError> {
    if src.is_empty() {
        return Ok(CompressOutcome::Incompressible);
    }

    // Count frequencies.
    let mut raw_freqs = [0u32; 256];
    for &b in src {
        raw_freqs[b as usize] += 1;
    }

    let Some((freqs, cum)) = normalize_freqs(&raw_freqs) else {
        return Ok(CompressOutcome::Incompressible);
    };

    // Build encode symbol table.
    let symbols: Vec<EncSymbol> = (0..256)
        .map(|i| EncSymbol::new(cum[i] as u32, freqs[i] as u32))
        .collect();

    // Write frequency-table header + jump-table placeholder directly into dst.
    let mut header_buf = Vec::with_capacity(256 * 2 + 1 + JUMP_TABLE_BYTES);
    encode_freq_table(&freqs, &mut header_buf);
    let jump_table_pos = header_buf.len();
    header_buf.resize(header_buf.len() + JUMP_TABLE_BYTES, 0);
    let header_len = header_buf.len();

    if header_len >= dst.len() {
        return Ok(CompressOutcome::DstTooSmall);
    }
    dst[..header_len].copy_from_slice(&header_buf);

    // Encode each of the 4 streams, reusing a single scratch Vec and copying
    // the result straight into dst at the computed offset.
    let segments = split_four(src.len());
    let max_seg = segments.iter().map(|(s, e)| e - s).max().unwrap_or(0);
    let mut scratch: Vec<u8> = Vec::with_capacity(4 + max_seg.saturating_mul(2));

    let mut stream_lens = [0u32; 4];
    let mut off = header_len;

    for (i, &(start, end)) in segments.iter().enumerate() {
        let seg = &src[start..end];
        encode_stream_into(seg, &symbols, &mut scratch);
        let written = scratch.len();

        if off + written > dst.len() {
            return Ok(CompressOutcome::DstTooSmall);
        }
        // The jump table stores u32 lengths; a single stream exceeding
        // 4 GB would silently truncate on cast and corrupt the table. Not
        // reachable from the pipeline (chunks ≤ ~tens of MB) but cheap to
        // assert defensively — reject such inputs rather than emit a bad
        // blob.
        if written > u32::MAX as usize {
            return Ok(CompressOutcome::LenOverflow);
        }
        dst[off..off + written].copy_from_slice(&scratch);
        off += written;
        stream_lens[i] = written as u32;
    }

    let total = off;
    if total >= src.len() {
        return Ok(CompressOutcome::NotBeneficial);
    }

    write_jump_table(
        &mut dst[jump_table_pos..jump_table_pos + JUMP_TABLE_BYTES],
        &stream_lens,
    );

    Ok(CompressOutcome::Encoded(total))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_freqs_basic() {
        let mut raw = [0u32; 256];
        raw[0] = 3000;
        raw[1] = 1000;
        let (freqs, cum) = normalize_freqs(&raw).unwrap();
        let total: u32 = freqs.iter().map(|&f| f as u32).sum();
        assert_eq!(total, PROB_SCALE);
        assert_eq!(cum[0], 0);
        assert_eq!(cum[1], freqs[0]);
    }

    #[test]
    fn normalize_freqs_empty() {
        let raw = [0u32; 256];
        assert!(normalize_freqs(&raw).is_none());
    }

    #[test]
    fn compress_empty_returns_incompressible() {
        let mut dst = [0u8; 64];
        assert_eq!(
            compress(&mut dst, &[]).unwrap(),
            CompressOutcome::Incompressible
        );
    }
}
