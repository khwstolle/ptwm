pub mod arithmetic;
pub mod context_mixing;
pub mod fpc;
pub mod huff_llm;
pub mod huffman;
pub mod identity;
pub(crate) mod length_prefix;
pub mod neural_predictor;
pub mod order1_arithmetic;
pub mod order1_scale_ac;
pub mod per_group_codebook;
pub mod rans;
pub mod tans;
pub mod zstd;
pub mod zstd_dict;

/// A Zstd frame that decodes to `total` zero bytes, built from RLE blocks
/// of 128 KiB (4 bytes each), so it is about `total / 32768` bytes long and
/// is built without compressing anything.
#[cfg(test)]
pub(crate) fn zstd_zero_bomb(total: u64) -> Vec<u8> {
    const BLOCK: u64 = 128 * 1024;
    // Magic, then a frame header with no content size, no checksum and a
    // 128 KiB window (Window_Descriptor exponent 7, mantissa 0).
    let mut out = vec![0x28, 0xB5, 0x2F, 0xFD, 0x00, 0x38];
    let mut left = total;
    while left > 0 {
        let n = left.min(BLOCK);
        left -= n;
        let last = u32::from(left == 0);
        // Block header: Last_Block bit, Block_Type 1 (RLE), Block_Size.
        let header = ((n as u32) << 3) | (1 << 1) | last;
        out.extend_from_slice(&header.to_le_bytes()[..3]);
        out.push(0);
    }
    out
}

#[cfg(test)]
mod bounded_decode_tests {
    use crate::codec::{CodecId, PlaneCodec, codec_for};
    use crate::layout::PlaneLayout;

    /// Codecs that encode any byte plane with a flat layout.
    const FLAT_CODECS: &[CodecId] = &[
        CodecId::Identity,
        CodecId::Huffman,
        CodecId::Rans,
        CodecId::Tans,
        CodecId::Zstd,
        CodecId::ZstdDict,
        CodecId::ArithmeticO0,
        CodecId::ArithmeticO0Adaptive,
        CodecId::ArithmeticO1,
        CodecId::Order1Arithmetic,
        CodecId::ContextMixingLite,
        CodecId::NeuralPredictor,
    ];

    fn plane() -> Vec<u8> {
        (0..20_000u32).map(|i| ((i * 7) % 13) as u8).collect()
    }

    #[test]
    fn decode_bounded_accepts_the_exact_length_and_rejects_one_less() {
        let data = plane();
        for &id in FLAT_CODECS {
            let c = codec_for(id).unwrap();
            let enc = c.encode(&data, None, &PlaneLayout::Flat).unwrap();
            let dec = |max_len| {
                c.decode_bounded(
                    enc.state_format_version,
                    &enc.state_bytes,
                    &enc.payload,
                    &PlaneLayout::Flat,
                    data.len(),
                    max_len,
                )
            };
            assert_eq!(dec(data.len()).unwrap(), data, "{id:?}");
            assert_eq!(dec(usize::MAX).unwrap(), data, "{id:?}");
            assert!(dec(data.len() - 1).is_err(), "{id:?} exceeded its bound");
        }
    }

    /// Codecs whose payload starts with a `u32` decoded length, and the
    /// bytes before it.
    fn prefixed_codecs() -> Vec<(Box<dyn PlaneCodec>, Vec<u8>, u8)> {
        vec![
            (codec_for(CodecId::Rans).unwrap(), vec![1], 0),
            (codec_for(CodecId::Tans).unwrap(), vec![1], 0),
            (codec_for(CodecId::ArithmeticO0).unwrap(), vec![], 1),
            (codec_for(CodecId::ArithmeticO0Adaptive).unwrap(), vec![], 0),
            (codec_for(CodecId::Order1Arithmetic).unwrap(), vec![], 1),
            (codec_for(CodecId::ArithmeticO1).unwrap(), vec![], 0),
            (codec_for(CodecId::ContextMixingLite).unwrap(), vec![], 0),
            (codec_for(CodecId::NeuralPredictor).unwrap(), vec![], 0),
        ]
    }

    #[test]
    fn a_length_prefix_above_the_bound_fails_before_decoding() {
        // 1 GiB declared from a 64-byte body: within `length_prefix`'s
        // ratio-free length, so before the bound the decode loop ran for
        // 2^30 symbols.
        let t0 = std::time::Instant::now();
        for (c, head, version) in prefixed_codecs() {
            let mut payload = head;
            payload.extend_from_slice(&(1u32 << 30).to_le_bytes());
            payload.extend_from_slice(&[0u8; 64]);
            let err = c
                .decode_bounded(version, &[], &payload, &PlaneLayout::Flat, 4096, 4096)
                .unwrap_err()
                .to_string();
            assert!(err.contains("bound is 4096"), "{:?}: {err}", c.id());
        }
        assert!(t0.elapsed() < std::time::Duration::from_secs(2));
    }

    #[cfg(feature = "codec-zstd")]
    #[test]
    fn zstd_bomb_stops_at_the_bound() {
        let bomb = super::zstd_zero_bomb(3 << 30);
        assert!(bomb.len() < 100_000);
        let t0 = std::time::Instant::now();
        for id in [CodecId::Zstd, CodecId::ZstdDict] {
            let c = codec_for(id).unwrap();
            let err = c
                .decode_bounded(0, &[], &bomb, &PlaneLayout::Flat, 4096, 4096)
                .unwrap_err()
                .to_string();
            assert!(err.contains("more than 4096 bytes"), "{id:?}: {err}");
        }
        assert!(t0.elapsed() < std::time::Duration::from_secs(1));
    }

    #[cfg(feature = "codec-zstd")]
    #[test]
    fn zstd_reads_past_a_short_hint_up_to_the_bound() {
        // Legacy readers pass the descriptor length as the hint, which can
        // understate the plane; the bound is what limits it.
        let data = plane();
        let c = codec_for(CodecId::Zstd).unwrap();
        let enc = c.encode(&data, None, &PlaneLayout::Flat).unwrap();
        for hint in [0, 1, data.len() / 2, data.len(), data.len() * 3] {
            let got = c
                .decode_bounded(0, &[], &enc.payload, &PlaneLayout::Flat, hint, data.len())
                .unwrap();
            assert_eq!(got, data, "hint {hint}");
        }
        // A bomb frame followed by a skippable frame still decodes when it
        // fits: skippable frames are part of the format.
        let mut two = super::zstd_zero_bomb(1000);
        two.extend_from_slice(&[0x50, 0x2A, 0x4D, 0x18, 3, 0, 0, 0, 9, 9, 9]);
        let got = c
            .decode_bounded(0, &[], &two, &PlaneLayout::Flat, 10, 1000)
            .unwrap();
        assert_eq!(got, vec![0u8; 1000]);
    }
}
