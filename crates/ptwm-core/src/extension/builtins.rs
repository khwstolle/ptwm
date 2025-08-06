//! Materialize ExtensionTableEntry values for in-tree built-ins.

use super::capability::CapabilityValue;
use super::table::FLAVOR_NATIVE;
use super::{
    Attestation, CanonicalId, CapabilityMap, ExtensionTableEntry, Kind, Lifecycle,
    builtin_canonical_id,
};

pub fn builtin_entry(name: &str, kind: Kind, deterministic: bool) -> ExtensionTableEntry {
    let mut caps = CapabilityMap::new();
    caps.set("determinism", CapabilityValue::Bool(deterministic));
    caps.set("hardware_class", CapabilityValue::Text("cpu".into()));

    ExtensionTableEntry {
        canonical_id: builtin_canonical_id(name),
        human_label: format!("io.ptwm.builtin.{name}@{}", env!("CARGO_PKG_VERSION")),
        kind,
        abi_version: 1,
        lifecycle: Lifecycle::Thread,
        flavor_hints: FLAVOR_NATIVE,
        capabilities: caps,
        // Built-ins are not attested by an external author key. A zero-byte
        // signature marks "no external attestation needed"; the verifier
        // short-circuits when the pubkey is BUILTIN_PUBKEY.
        attestation: Attestation::PgpSignature(Vec::new()),
        install_hint: None,
        embedded_wasm_offset: None,
        embedded_wasm_length: None,
    }
}

pub fn builtin_entries() -> Vec<ExtensionTableEntry> {
    vec![
        // Plane codecs
        builtin_entry("identity", Kind::PlaneCodec, true),
        builtin_entry("huffman", Kind::PlaneCodec, true),
        builtin_entry("rans", Kind::PlaneCodec, true),
        builtin_entry("zstd", Kind::PlaneCodec, true),
        builtin_entry("zstd_dict", Kind::PlaneCodec, true),
        builtin_entry("fpc", Kind::PlaneCodec, true),
        builtin_entry("tans", Kind::PlaneCodec, true),
        builtin_entry("order1_scale_ac", Kind::PlaneCodec, true),
        builtin_entry("arithmetic_o0", Kind::PlaneCodec, true),
        builtin_entry("arithmetic_o0_adaptive", Kind::PlaneCodec, true),
        builtin_entry("arithmetic_o1", Kind::PlaneCodec, true),
        builtin_entry("per_group_codebook", Kind::PlaneCodec, true),
        builtin_entry("context_mixing_lite", Kind::PlaneCodec, true),
        builtin_entry("huff_llm_5bit", Kind::PlaneCodec, true),
        builtin_entry("order1_arithmetic", Kind::PlaneCodec, true),
        builtin_entry("neural_predictor", Kind::PlaneCodec, true),
        // Transforms — names MUST match OpId::canonical_id() in transforms/op.rs
        builtin_entry("bit_reorder_ieee16", Kind::Transform, true),
        builtin_entry("bit_reorder_ieee32", Kind::Transform, true),
        builtin_entry("bit_reorder_fp8_e4m3", Kind::Transform, true),
        builtin_entry("bit_reorder_fp8_e5m2", Kind::Transform, true),
        builtin_entry("byte_split", Kind::Transform, true),
        builtin_entry("nibble_split", Kind::Transform, true),
        builtin_entry("byte_passthrough", Kind::Transform, true),
        builtin_entry("xor_delta", Kind::Transform, true),
        builtin_entry("int_delta", Kind::Transform, true),
        builtin_entry("predictor_xor", Kind::Transform, true),
        builtin_entry("burrows_wheeler", Kind::Transform, true),
        builtin_entry("move_to_front", Kind::Transform, true),
        builtin_entry("mxfp4_deinterleave", Kind::Transform, true),
        builtin_entry("block_microscaling_repack", Kind::Transform, true),
        builtin_entry("concat", Kind::Transform, true),
        builtin_entry("reshape", Kind::Transform, true),
        builtin_entry("index_bitwidth_pack", Kind::Transform, true),
        builtin_entry("mantissa_zero_strip", Kind::Transform, true),
        builtin_entry("source", Kind::Transform, true),
        builtin_entry("terminal", Kind::Transform, true),
        builtin_entry("spherical_normalize", Kind::Transform, true),
        builtin_entry("alpha_stable_normalize", Kind::Transform, true),
        builtin_entry("float_delta", Kind::Transform, true),
        builtin_entry("entropy_estimate", Kind::Transform, true),
        // Encode-only Python-resident contributions (chain builder + classifiers).
        {
            let mut e = builtin_entry("production_chains", Kind::ChainBuilder, true);
            e.flavor_hints = super::table::FLAVOR_HOST;
            e
        },
        {
            let mut e = builtin_entry("classifier_heuristic", Kind::Classifier, true);
            e.flavor_hints = super::table::FLAVOR_HOST;
            e
        },
        {
            let mut e = builtin_entry("classifier_hf_quant_config", Kind::Classifier, true);
            e.flavor_hints = super::table::FLAVOR_HOST;
            e
        },
        {
            let mut e = builtin_entry("classifier_ptwm_config", Kind::Classifier, true);
            e.flavor_hints = super::table::FLAVOR_HOST;
            e
        },
        {
            let mut e = builtin_entry("classifier_explicit_flags", Kind::Classifier, true);
            e.flavor_hints = super::table::FLAVOR_HOST;
            e
        },
    ]
}

pub fn is_builtin(id: &CanonicalId) -> bool {
    builtin_entries().iter().any(|e| &e.canonical_id == id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_builtin_has_unique_id() {
        let entries = builtin_entries();
        let mut ids: Vec<_> = entries.iter().map(|e| e.canonical_id).collect();
        ids.sort_by_key(|id| *id.as_bytes());
        ids.dedup();
        assert_eq!(ids.len(), entries.len(), "duplicate canonical ids");
    }

    #[test]
    fn is_builtin_detects_huffman() {
        let id = builtin_canonical_id("huffman");
        assert!(is_builtin(&id));
        let fake = CanonicalId::from_bytes([0xCC; 32]);
        assert!(!is_builtin(&fake));
    }

    #[test]
    fn every_op_has_a_matching_builtin_entry() {
        // Every OpId variant's canonical_id MUST match an entry in
        // builtin_entries(). Otherwise the writer's interning of a chain
        // referencing that op will succeed only because of the
        // placeholder helper — the *curated* table won't recognize it.
        use crate::transforms::op::OpId;
        for op in [
            OpId::BitReorderIeee16,
            OpId::BitReorderIeee32,
            OpId::BitReorderFp8E4M3,
            OpId::BitReorderFp8E5M2,
            OpId::ByteSplit,
            OpId::NibbleSplit,
            OpId::BytePassthrough,
            OpId::XorDelta,
            OpId::IntDelta,
            OpId::PredictorXor,
            OpId::BurrowsWheeler,
            OpId::MoveToFront,
            OpId::MxFp4Deinterleave,
            OpId::BlockMicroscalingRepack,
            OpId::Concat,
            OpId::Reshape,
            OpId::IndexBitwidthPack,
            OpId::MantissaZeroStrip,
            OpId::Source,
            OpId::Terminal,
            OpId::FloatDelta,
            OpId::EntropyEstimate,
            OpId::SphericalNormalize,
            OpId::AlphaStableNormalize,
        ] {
            let id = op.canonical_id();
            assert!(
                is_builtin(&id),
                "OpId {op:?} canonical_id has no matching builtin entry"
            );
        }
    }

    #[test]
    fn chain_builder_and_classifier_builtins_are_present() {
        let entries = builtin_entries();
        assert_eq!(
            entries
                .iter()
                .filter(|e| e.kind == Kind::ChainBuilder)
                .count(),
            1
        );
        assert_eq!(
            entries
                .iter()
                .filter(|e| e.kind == Kind::Classifier)
                .count(),
            4
        );
        // All host-flavored.
        for e in entries
            .iter()
            .filter(|e| matches!(e.kind, Kind::ChainBuilder | Kind::Classifier))
        {
            assert_eq!(e.flavor_hints, super::super::table::FLAVOR_HOST);
        }
    }

    #[test]
    fn every_plane_codec_has_a_matching_builtin_entry() {
        use crate::codecs::arithmetic::{ArithmeticO0, ArithmeticO0Adaptive, ArithmeticO1};
        use crate::codecs::context_mixing::ContextMixingLite;
        use crate::codecs::huff_llm::HuffLlm5Bit;
        use crate::codecs::huffman::Huffman;
        use crate::codecs::identity::Identity;
        use crate::codecs::order1_scale_ac::Order1ScaleAC;
        use crate::codecs::per_group_codebook::PerGroupCodebook;
        use crate::codecs::rans::Rans;
        use crate::codecs::tans::Tans;
        use crate::codecs::zstd::Zstd;
        for id in [
            Identity.canonical_id(),
            Huffman.canonical_id(),
            Rans.canonical_id(),
            Tans.canonical_id(),
            Zstd.canonical_id(),
            Order1ScaleAC.canonical_id(),
            ArithmeticO0.canonical_id(),
            ArithmeticO0Adaptive.canonical_id(),
            ArithmeticO1.canonical_id(),
            PerGroupCodebook.canonical_id(),
            ContextMixingLite.canonical_id(),
            HuffLlm5Bit.canonical_id(),
        ] {
            assert!(
                is_builtin(&id),
                "codec canonical_id has no matching builtin entry"
            );
        }
    }
}
