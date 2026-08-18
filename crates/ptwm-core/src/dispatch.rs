//! PPG codec dispatch. Codecs declare capabilities via
//! [`PlaneCodec::accepts`] and [`PlaneCodec::priority_for`]; this module
//! filters and ranks them for a given [`PlaneDescriptor`].

use crate::codec::{CodecId, PlaneCodec, codec_for};
use crate::types::descriptor::PlaneDescriptor;

/// Return the ordered list of codec IDs that accept `descriptor`, sorted
/// by descending priority. The highest-priority codec sits at index 0.
///
/// Codecs that reject `descriptor` drop out. The list is never empty:
/// Identity always accepts and serves as the universal fallback.
pub fn dispatch(d: &PlaneDescriptor) -> Vec<CodecId> {
    const ALL: &[CodecId] = &[
        CodecId::Identity,
        CodecId::Huffman,
        CodecId::Rans,
        CodecId::Tans,
        CodecId::Zstd,
        CodecId::PerGroupCodebook,
        CodecId::Order1ScaleAC,
        CodecId::HuffLlm5Bit,
        // ArithmeticO0, ArithmeticO0Adaptive, ArithmeticO1, Order1Arithmetic,
        // ContextMixingLite, and NeuralPredictor are intentionally absent from
        // the default menu. The arithmetic coders approach the entropy limit
        // more closely than rANS but are 20-40× slower (sequential range
        // coder vs 4-stream ANS). The 0.7% ratio difference does not justify
        // the encode-speed penalty for production use. They remain available
        // via explicit codec_menu for research and archival workflows.
    ];

    let mut candidates: Vec<(CodecId, i8)> = ALL
        .iter()
        .filter_map(|&id| {
            let codec: Box<dyn PlaneCodec> = codec_for(id)?;
            if codec.accepts(d) {
                Some((id, codec.priority_for(d)))
            } else {
                None
            }
        })
        .collect();

    // Stable descending sort: ties preserve the original ALL order,
    // which lists lower-overhead codecs first (Identity → Huffman → Rans
    // → Tans → Zstd).
    candidates.sort_by_key(|&(_, p)| std::cmp::Reverse(p));
    candidates.into_iter().map(|(id, _)| id).collect()
}

#[cfg(test)]
mod tests {
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
    fn dispatch_scale_byte_rows_returns_o1sac_first() {
        let d = descriptor(
            Role::Scale {
                format: ScaleFormat::E4M3,
            },
            ElementWidth::Byte,
            Layout::Rows { row_len: 32 },
            false,
        );
        let result = dispatch(&d);
        assert!(
            !result.is_empty(),
            "dispatch must return at least one codec"
        );
        assert_eq!(
            result[0],
            CodecId::Order1ScaleAC,
            "O1SAC must be first for Scale + Byte + Rows; got {:?}",
            result
        );
        // Standard codecs still present
        assert!(result.contains(&CodecId::Huffman));
        assert!(result.contains(&CodecId::Identity));
    }

    #[test]
    fn dispatch_e8m0_scale_byte_rows_excludes_o1sac() {
        // uint8 weight_scale planes (MXFP4 E8M0 block scales) must not
        // surface Order1ScaleAC in the dispatch menu — the codec is FP8-only
        // by design.
        let d = descriptor(
            Role::Scale {
                format: ScaleFormat::E8M0,
            },
            ElementWidth::Byte,
            Layout::Rows { row_len: 32 },
            false,
        );
        let result = dispatch(&d);
        assert!(
            !result.contains(&CodecId::Order1ScaleAC),
            "O1SAC must not appear for E8M0 scales; got {:?}",
            result
        );
        // Standard codecs still present.
        assert!(result.contains(&CodecId::Huffman));
        assert!(result.contains(&CodecId::Rans));
        assert!(result.contains(&CodecId::Zstd));
        assert!(result.contains(&CodecId::Identity));
    }

    #[test]
    fn dispatch_expanded_nibble_value_returns_pgc_first() {
        let d = descriptor(
            Role::Value {
                format: ValueFormat::Fp4E2m1,
            },
            ElementWidth::Nibble,
            Layout::Flat,
            false, // expanded: one nibble per byte, which is what PGC models
        );
        let result = dispatch(&d);
        assert!(!result.is_empty());
        assert_eq!(
            result[0],
            CodecId::PerGroupCodebook,
            "PGC must be first for expanded Value + Nibble; got {:?}",
            result
        );
        assert!(result.contains(&CodecId::Huffman));
        assert!(result.contains(&CodecId::Identity));
    }

    #[test]
    fn dispatch_unknown_descriptor_falls_to_standard_codecs() {
        // Vendor role — not matched by O1SAC or PGC.
        let d = descriptor(
            Role::Vendor {
                tag: 0xFFFF,
                bytes: vec![],
            },
            ElementWidth::Byte,
            Layout::Flat,
            false,
        );
        let result = dispatch(&d);
        assert!(!result.is_empty());
        // Specialist codecs must NOT appear
        assert!(
            !result.contains(&CodecId::Order1ScaleAC),
            "O1SAC must not match vendor role"
        );
        assert!(
            !result.contains(&CodecId::PerGroupCodebook),
            "PGC must not match vendor role"
        );
        // Standard codecs must appear
        assert!(result.contains(&CodecId::Huffman));
        assert!(result.contains(&CodecId::Rans));
        assert!(result.contains(&CodecId::Zstd));
        assert!(result.contains(&CodecId::Identity));
        // Identity must be last (priority 0, lower than the rest at 1)
        assert_eq!(
            result.last(),
            Some(&CodecId::Identity),
            "Identity must be last (lowest priority)"
        );
    }

    #[test]
    fn dispatch_result_never_empty() {
        // Even for an exotic descriptor the universal Identity fallback is returned.
        let d = descriptor(Role::Index, ElementWidth::Word8, Layout::Flat, false);
        let result = dispatch(&d);
        assert!(!result.is_empty());
        assert!(result.contains(&CodecId::Identity));
    }
}
