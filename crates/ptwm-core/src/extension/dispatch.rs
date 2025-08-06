//! Static dispatch table for in-tree built-in ops and codecs.
//!
//! Resolves a [`CanonicalId`] back to the corresponding [`OpId`] or
//! [`CodecId`] for files that reference contributions via the Extension Table.

use crate::codec::CodecId;
use crate::extension::{CanonicalId, builtin_canonical_id};
use crate::transforms::op::OpId;

/// The kind of a resolved built-in contribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinKind {
    Op(OpId),
    Codec(CodecId),
}

/// Look up a [`CanonicalId`] in the static in-tree dispatch table.
///
/// Returns `Some(BuiltinKind)` for every built-in op and codec that ships
/// with ptwm-core, or `None` if the id is not recognized (e.g. a third-party
/// extension handled via dynamic loading instead).
pub fn dispatch_builtin(id: &CanonicalId) -> Option<BuiltinKind> {
    // Ops — 16 variants.
    if *id == OpId::Source.canonical_id() {
        return Some(BuiltinKind::Op(OpId::Source));
    }
    if *id == OpId::Terminal.canonical_id() {
        return Some(BuiltinKind::Op(OpId::Terminal));
    }
    if *id == OpId::BitReorderIeee16.canonical_id() {
        return Some(BuiltinKind::Op(OpId::BitReorderIeee16));
    }
    if *id == OpId::BitReorderIeee32.canonical_id() {
        return Some(BuiltinKind::Op(OpId::BitReorderIeee32));
    }
    if *id == OpId::BitReorderFp8E4M3.canonical_id() {
        return Some(BuiltinKind::Op(OpId::BitReorderFp8E4M3));
    }
    if *id == OpId::BitReorderFp8E5M2.canonical_id() {
        return Some(BuiltinKind::Op(OpId::BitReorderFp8E5M2));
    }
    if *id == OpId::ByteSplit.canonical_id() {
        return Some(BuiltinKind::Op(OpId::ByteSplit));
    }
    if *id == OpId::NibbleSplit.canonical_id() {
        return Some(BuiltinKind::Op(OpId::NibbleSplit));
    }
    if *id == OpId::BytePassthrough.canonical_id() {
        return Some(BuiltinKind::Op(OpId::BytePassthrough));
    }
    if *id == OpId::MxFp4Deinterleave.canonical_id() {
        return Some(BuiltinKind::Op(OpId::MxFp4Deinterleave));
    }
    if *id == OpId::BlockMicroscalingRepack.canonical_id() {
        return Some(BuiltinKind::Op(OpId::BlockMicroscalingRepack));
    }
    if *id == OpId::XorDelta.canonical_id() {
        return Some(BuiltinKind::Op(OpId::XorDelta));
    }
    if *id == OpId::FloatDelta.canonical_id() {
        return Some(BuiltinKind::Op(OpId::FloatDelta));
    }
    if *id == OpId::IntDelta.canonical_id() {
        return Some(BuiltinKind::Op(OpId::IntDelta));
    }
    if *id == OpId::PredictorXor.canonical_id() {
        return Some(BuiltinKind::Op(OpId::PredictorXor));
    }
    if *id == OpId::BurrowsWheeler.canonical_id() {
        return Some(BuiltinKind::Op(OpId::BurrowsWheeler));
    }
    if *id == OpId::MoveToFront.canonical_id() {
        return Some(BuiltinKind::Op(OpId::MoveToFront));
    }
    if *id == OpId::IndexBitwidthPack.canonical_id() {
        return Some(BuiltinKind::Op(OpId::IndexBitwidthPack));
    }
    if *id == OpId::EntropyEstimate.canonical_id() {
        return Some(BuiltinKind::Op(OpId::EntropyEstimate));
    }
    if *id == OpId::Concat.canonical_id() {
        return Some(BuiltinKind::Op(OpId::Concat));
    }
    if *id == OpId::Reshape.canonical_id() {
        return Some(BuiltinKind::Op(OpId::Reshape));
    }
    if *id == OpId::MantissaZeroStrip.canonical_id() {
        return Some(BuiltinKind::Op(OpId::MantissaZeroStrip));
    }
    if *id == OpId::SphericalNormalize.canonical_id() {
        return Some(BuiltinKind::Op(OpId::SphericalNormalize));
    }
    if *id == OpId::AlphaStableNormalize.canonical_id() {
        return Some(BuiltinKind::Op(OpId::AlphaStableNormalize));
    }

    // Codecs — 7 variants (including HuffmanNibble which has no struct,
    // represented only as CodecId for wire compatibility).
    if *id == builtin_canonical_id("identity") {
        return Some(BuiltinKind::Codec(CodecId::Identity));
    }
    if *id == builtin_canonical_id("huffman") {
        return Some(BuiltinKind::Codec(CodecId::Huffman));
    }
    if *id == builtin_canonical_id("huffman_nibble") {
        return Some(BuiltinKind::Codec(CodecId::HuffmanNibble));
    }
    if *id == builtin_canonical_id("rans") {
        return Some(BuiltinKind::Codec(CodecId::Rans));
    }
    if *id == builtin_canonical_id("zstd") {
        return Some(BuiltinKind::Codec(CodecId::Zstd));
    }
    if *id == builtin_canonical_id("zstd_dict") {
        return Some(BuiltinKind::Codec(CodecId::ZstdDict));
    }
    if *id == builtin_canonical_id("fpc") {
        return Some(BuiltinKind::Codec(CodecId::Fpc));
    }
    if *id == builtin_canonical_id("tans") {
        return Some(BuiltinKind::Codec(CodecId::Tans));
    }
    if *id == builtin_canonical_id("per_group_codebook") {
        return Some(BuiltinKind::Codec(CodecId::PerGroupCodebook));
    }
    if *id == builtin_canonical_id("order1_scale_ac") {
        return Some(BuiltinKind::Codec(CodecId::Order1ScaleAC));
    }
    if *id == builtin_canonical_id("arithmetic_o0") {
        return Some(BuiltinKind::Codec(CodecId::ArithmeticO0));
    }
    if *id == builtin_canonical_id("arithmetic_o0_adaptive") {
        return Some(BuiltinKind::Codec(CodecId::ArithmeticO0Adaptive));
    }
    if *id == builtin_canonical_id("arithmetic_o1") {
        return Some(BuiltinKind::Codec(CodecId::ArithmeticO1));
    }
    if *id == builtin_canonical_id("context_mixing_lite") {
        return Some(BuiltinKind::Codec(CodecId::ContextMixingLite));
    }
    if *id == builtin_canonical_id("huff_llm_5bit") {
        return Some(BuiltinKind::Codec(CodecId::HuffLlm5Bit));
    }
    if *id == builtin_canonical_id("order1_arithmetic") {
        return Some(BuiltinKind::Codec(CodecId::Order1Arithmetic));
    }
    if *id == builtin_canonical_id("neural_predictor") {
        return Some(BuiltinKind::Codec(CodecId::NeuralPredictor));
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_op_id_resolves() {
        for op in [
            OpId::Source,
            OpId::Terminal,
            OpId::BitReorderIeee16,
            OpId::BitReorderIeee32,
            OpId::BitReorderFp8E4M3,
            OpId::BitReorderFp8E5M2,
            OpId::ByteSplit,
            OpId::NibbleSplit,
            OpId::BytePassthrough,
            OpId::MxFp4Deinterleave,
            OpId::BlockMicroscalingRepack,
            OpId::XorDelta,
            OpId::FloatDelta,
            OpId::IntDelta,
            OpId::PredictorXor,
            OpId::BurrowsWheeler,
            OpId::MoveToFront,
            OpId::IndexBitwidthPack,
            OpId::EntropyEstimate,
            OpId::Concat,
            OpId::Reshape,
            OpId::MantissaZeroStrip,
            OpId::SphericalNormalize,
            OpId::AlphaStableNormalize,
        ] {
            let id = op.canonical_id();
            assert_eq!(
                dispatch_builtin(&id),
                Some(BuiltinKind::Op(op)),
                "dispatch_builtin failed for {op:?}"
            );
        }
    }

    #[test]
    fn every_codec_id_resolves() {
        // Enumerate the canonical codec REGISTRY so a newly-added codec that
        // forgets its `dispatch_builtin` arm fails here rather than at decode
        // time with "no codec registered for canonical id ...".
        for (name, raw) in crate::codec::REGISTRY {
            let expected =
                CodecId::from_u16(*raw).expect("REGISTRY id must map to a known CodecId");
            let id = builtin_canonical_id(name);
            assert_eq!(
                dispatch_builtin(&id),
                Some(BuiltinKind::Codec(expected)),
                "dispatch_builtin failed for codec {name}"
            );
        }
    }

    #[test]
    fn unknown_id_returns_none() {
        let id = CanonicalId::derive(&[0x00; 32], "unknown", "0.0.0");
        assert_eq!(dispatch_builtin(&id), None);
    }
}
