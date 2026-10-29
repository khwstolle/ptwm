//! Static dispatch table for in-tree built-in ops and codecs.
//!
//! Resolves a [`CanonicalId`] back to the corresponding [`OpId`] or
//! [`CodecId`] for files that reference contributions via the Extension Table.

use crate::codec::CodecId;
use crate::extension::CanonicalId;
use crate::extension::builtins::builtin_registry;
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
/// extension handled via dynamic loading instead). Only ids of the crate
/// version match; tables from other releases go through
/// [`crate::extension::normalize_builtin_ids`] first. The lookup is a single
/// hash-map probe.
pub fn dispatch_builtin(id: &CanonicalId) -> Option<BuiltinKind> {
    builtin_registry().get(id).and_then(|r| r.dispatch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extension::builtin_canonical_id;

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
