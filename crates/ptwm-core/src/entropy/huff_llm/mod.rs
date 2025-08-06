//! Field-separated Huffman coding of 16-bit float planes (Huff-LLM).
//!
//! Each 16-bit word is partitioned MSB-first into `[sign:1][g0][g1][g2]`.
//! The sign bit is stored raw; each group is coded with an independent
//! canonical Huffman table (`group_codec`). Two layouts are supported and the
//! caller trials both, keeping the smaller:
//!   * FP16 `{1,5,5,5}` — sign · exp(5) · mantissa-hi(5) · mantissa-lo(5)
//!   * BF16 `{1,4,4,7}` — sign · exp-hi(4) · exp-lo(4) · mantissa(7)

mod group_codec;

use crate::error::PtwmCoreError;
use group_codec::{BitReader, BitWriter, GroupCode};

#[derive(Clone, Copy)]
struct LayoutDef {
    tag: u8,
    widths: [u8; 3],
}

const FP16: LayoutDef = LayoutDef {
    tag: 0,
    widths: [5, 5, 5],
};
const BF16: LayoutDef = LayoutDef {
    tag: 1,
    widths: [4, 4, 7],
};

fn layout_for_tag(tag: u8) -> Result<LayoutDef, PtwmCoreError> {
    match tag {
        0 => Ok(FP16),
        1 => Ok(BF16),
        other => Err(PtwmCoreError::CodecDecode {
            codec: "huff_llm_5bit",
            msg: format!("unknown layout tag {other}"),
        }),
    }
}

/// Split a word into (sign, three group symbols) MSB-first. `widths` sum to 15.
#[inline]
fn split_word(w: u16, widths: &[u8; 3]) -> (u8, [u16; 3]) {
    let sign = ((w >> 15) & 1) as u8;
    let rest = w & 0x7FFF;
    let mut total = 15u8;
    let mut groups = [0u16; 3];
    for i in 0..3 {
        total -= widths[i];
        groups[i] = (rest >> total) & ((1u16 << widths[i]) - 1);
    }
    (sign, groups)
}

#[inline]
fn join_word(sign: u8, groups: &[u16; 3], widths: &[u8; 3]) -> u16 {
    let mut w = (sign as u16) << 15;
    let mut total = 15u8;
    for i in 0..3 {
        total -= widths[i];
        w |= (groups[i] & ((1u16 << widths[i]) - 1)) << total;
    }
    w
}

/// Serialized result for one plane: `state_bytes` (layout tag + 3 length tables)
/// and `payload` (n_words + sign bits + 3 group bitstreams).
pub struct PlaneEncoding {
    pub state_bytes: Vec<u8>,
    pub payload: Vec<u8>,
}

fn encode_with_layout(words: &[u16], layout: &LayoutDef) -> Result<PlaneEncoding, PtwmCoreError> {
    // Per-group frequency tables.
    let mut freqs: [Vec<u32>; 3] = [
        vec![0u32; 1 << layout.widths[0]],
        vec![0u32; 1 << layout.widths[1]],
        vec![0u32; 1 << layout.widths[2]],
    ];
    let mut signs: Vec<u8> = vec![0u8; words.len().div_ceil(8)];
    let mut group_syms: [Vec<u16>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (i, &w) in words.iter().enumerate() {
        let (sign, groups) = split_word(w, &layout.widths);
        if sign != 0 {
            signs[i >> 3] |= 1 << (i & 7);
        }
        for g in 0..3 {
            freqs[g][groups[g] as usize] += 1;
            group_syms[g].push(groups[g]);
        }
    }

    let codes: [GroupCode; 3] = [
        GroupCode::from_freqs(&freqs[0]),
        GroupCode::from_freqs(&freqs[1]),
        GroupCode::from_freqs(&freqs[2]),
    ];

    // state: [tag] + 3 × [n_symbols u8][lengths...]
    let mut state = vec![layout.tag];
    for (g, code) in codes.iter().enumerate() {
        let n_syms = 1usize << layout.widths[g];
        debug_assert_eq!(code.lengths.len(), n_syms);
        state.push(n_syms as u8);
        state.extend_from_slice(&code.lengths);
    }

    // payload: [u32 n_words][signs][u32 len0][s0][u32 len1][s1][s2]
    let mut payload = Vec::new();
    let n_words = u32::try_from(words.len()).map_err(|_| PtwmCoreError::CodecEncode {
        codec: "huff_llm_5bit",
        msg: "word count overflows u32".into(),
    })?;
    payload.extend_from_slice(&n_words.to_le_bytes());
    payload.extend_from_slice(&signs);
    for (g, code) in codes.iter().enumerate() {
        let mut bw = BitWriter::new();
        for &s in &group_syms[g] {
            code.encode_symbol(&mut bw, s as usize);
        }
        let stream = bw.finish();
        if g < 2 {
            let stream_len =
                u32::try_from(stream.len()).map_err(|_| PtwmCoreError::CodecEncode {
                    codec: "huff_llm_5bit",
                    msg: "group stream length overflows u32".into(),
                })?;
            payload.extend_from_slice(&stream_len.to_le_bytes());
        }
        payload.extend_from_slice(&stream);
    }

    Ok(PlaneEncoding {
        state_bytes: state,
        payload,
    })
}

/// Encode a 16-bit word plane, trialing both layouts and keeping the smaller.
pub fn encode_plane(words: &[u16]) -> Result<PlaneEncoding, PtwmCoreError> {
    let a = encode_with_layout(words, &FP16)?;
    let b = encode_with_layout(words, &BF16)?;
    let sa = a.state_bytes.len() + a.payload.len();
    let sb = b.state_bytes.len() + b.payload.len();
    Ok(if sa <= sb { a } else { b })
}

fn read_u32(buf: &[u8], pos: &mut usize) -> Result<u32, PtwmCoreError> {
    if *pos + 4 > buf.len() {
        return Err(PtwmCoreError::CodecDecode {
            codec: "huff_llm_5bit",
            msg: "truncated u32".into(),
        });
    }
    let v = u32::from_le_bytes(buf[*pos..*pos + 4].try_into().unwrap());
    *pos += 4;
    Ok(v)
}

/// Decode a plane from its `state_bytes` and `payload`.
pub fn decode_plane(state: &[u8], payload: &[u8]) -> Result<Vec<u16>, PtwmCoreError> {
    let mut sp = 0usize;
    let tag = *state.first().ok_or_else(|| PtwmCoreError::CodecDecode {
        codec: "huff_llm_5bit",
        msg: "empty state".into(),
    })?;
    sp += 1;
    let layout = layout_for_tag(tag)?;
    let mut codes: Vec<GroupCode> = Vec::with_capacity(3);
    for g in 0..3 {
        let n_syms = *state.get(sp).ok_or_else(|| PtwmCoreError::CodecDecode {
            codec: "huff_llm_5bit",
            msg: "truncated state table header".into(),
        })? as usize;
        sp += 1;
        if n_syms != (1usize << layout.widths[g]) {
            return Err(PtwmCoreError::CodecDecode {
                codec: "huff_llm_5bit",
                msg: format!("table {g} symbol count {n_syms} mismatches layout"),
            });
        }
        let end = sp + n_syms;
        if end > state.len() {
            return Err(PtwmCoreError::CodecDecode {
                codec: "huff_llm_5bit",
                msg: "truncated state table body".into(),
            });
        }
        codes.push(GroupCode::from_lengths(state[sp..end].to_vec())?);
        sp = end;
    }

    let mut pp = 0usize;
    let n_words = read_u32(payload, &mut pp)? as usize;
    let sign_bytes = n_words.div_ceil(8);
    // Subtraction form (pp <= payload.len() always) avoids a usize add overflow
    // on 32-bit targets when n_words is large.
    if sign_bytes > payload.len() - pp {
        return Err(PtwmCoreError::CodecDecode {
            codec: "huff_llm_5bit",
            msg: "truncated sign bits".into(),
        });
    }
    let signs = &payload[pp..pp + sign_bytes];
    pp += sign_bytes;

    let mut group_out: Vec<Vec<u16>> = Vec::with_capacity(3);
    for g in 0..3 {
        let stream: &[u8] = if g < 2 {
            let len = read_u32(payload, &mut pp)? as usize;
            if len > payload.len() - pp {
                return Err(PtwmCoreError::CodecDecode {
                    codec: "huff_llm_5bit",
                    msg: "truncated group stream".into(),
                });
            }
            let s = &payload[pp..pp + len];
            pp += len;
            s
        } else {
            &payload[pp..]
        };
        let mut r = BitReader::new(stream);
        group_out.push(codes[g].decode_symbols(&mut r, n_words)?);
    }

    let mut words = Vec::with_capacity(n_words);
    for i in 0..n_words {
        let sign = (signs[i >> 3] >> (i & 7)) & 1;
        let groups = [group_out[0][i], group_out[1][i], group_out[2][i]];
        words.push(join_word(sign, &groups, &layout.widths));
    }
    Ok(words)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rt(words: &[u16]) {
        let enc = encode_plane(words).unwrap();
        let dec = decode_plane(&enc.state_bytes, &enc.payload).unwrap();
        assert_eq!(dec, words);
    }

    #[test]
    fn split_join_inverse_both_layouts() {
        for layout in [FP16, BF16] {
            for w in [0x0000u16, 0xFFFF, 0x8000, 0x7FFF, 0x1234, 0xABCD] {
                let (s, g) = split_word(w, &layout.widths);
                assert_eq!(join_word(s, &g, &layout.widths), w);
            }
        }
    }

    #[test]
    fn roundtrip_compressible() {
        // BF16-like: low exponent variation, many zeros.
        let words: Vec<u16> = (0..4096u32).map(|i| 0x3F00 | ((i % 8) as u16)).collect();
        rt(&words);
    }

    #[test]
    fn roundtrip_adversarial_bit_patterns() {
        // inf / nan / denormal / signed zero — codec is a pure bit-partition.
        let words = vec![
            0x0000, 0x8000, 0x7C00, 0xFC00, 0x7E00, 0x0001, 0x8001, 0x7FFF, 0xFFFF, 0x3C00,
        ];
        rt(&words);
    }

    #[test]
    fn roundtrip_all_same() {
        rt(&vec![0x4248u16; 1000]);
    }

    #[test]
    fn roundtrip_alternating() {
        let words: Vec<u16> = (0..2000)
            .map(|i| if i % 2 == 0 { 0x0000 } else { 0xFFFF })
            .collect();
        rt(&words);
    }

    #[test]
    fn roundtrip_empty() {
        rt(&[]);
    }

    #[test]
    fn trial_picks_smaller() {
        // A plane whose layout-specific entropy clearly favors one split.
        let words: Vec<u16> = (0..8192u32).map(|i| (i & 0x7FFF) as u16).collect();
        let enc = encode_plane(&words).unwrap();
        // Whichever was chosen must decode correctly.
        let dec = decode_plane(&enc.state_bytes, &enc.payload).unwrap();
        assert_eq!(dec, words);
    }
}
