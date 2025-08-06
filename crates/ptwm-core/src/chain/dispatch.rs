//! Op dispatch: construct a `Box<dyn Op>` from an `OpId` and its raw
//! params.
//!
//! Inverse of `Op::write_params` — decodes the params bytes each op
//! wrote and instantiates the concrete struct.

use crate::error::PtwmCoreError;
#[cfg(feature = "lossy")]
use crate::transforms::FloatDelta;
use crate::transforms::op::{Op, OpId};
use crate::transforms::{
    BitReorderFp8E4M3, BitReorderFp8E5M2, BitReorderIeee16, BitReorderIeee32,
    BlockMicroscalingRepack, BurrowsWheeler, BytePassthrough, ByteSplit, EntropyEstimate,
    IndexBitwidthPack, IntDelta, MantissaZeroStrip, MoveToFront, MxFp4Deinterleave, NibbleSplit,
    PredictorXor, XorDelta,
};

/// Instantiate a `Box<dyn Op>` from `id` and the op's encoded `params` bytes.
///
/// `params` must hold exactly the bytes the op's `write_params` would
/// emit — no opcode prefix, no length prefix.
pub fn op_from_id(id: OpId, params: &[u8]) -> Result<Box<dyn Op>, PtwmCoreError> {
    match id {
        // ── zero-param ops ───────────────────────────────────────────────────
        OpId::BitReorderIeee16 => {
            expect_empty_params("BitReorderIeee16", params)?;
            Ok(Box::new(BitReorderIeee16))
        }
        OpId::BitReorderIeee32 => {
            expect_empty_params("BitReorderIeee32", params)?;
            Ok(Box::new(BitReorderIeee32))
        }
        OpId::BitReorderFp8E4M3 => {
            expect_empty_params("BitReorderFp8E4M3", params)?;
            Ok(Box::new(BitReorderFp8E4M3))
        }
        OpId::BitReorderFp8E5M2 => {
            expect_empty_params("BitReorderFp8E5M2", params)?;
            Ok(Box::new(BitReorderFp8E5M2))
        }
        OpId::NibbleSplit => {
            expect_empty_params("NibbleSplit", params)?;
            Ok(Box::new(NibbleSplit))
        }
        OpId::BytePassthrough => {
            expect_empty_params("BytePassthrough", params)?;
            Ok(Box::new(BytePassthrough))
        }
        OpId::EntropyEstimate => {
            expect_empty_params("EntropyEstimate", params)?;
            Ok(Box::new(EntropyEstimate))
        }

        // ── single-byte-param ops ────────────────────────────────────────────
        OpId::ByteSplit => {
            if params.is_empty() {
                return Err(PtwmCoreError::InvalidContainer(
                    "ByteSplit params: empty (need 1 byte for n)".into(),
                ));
            }
            let op = ByteSplit::new(params[0])?;
            Ok(Box::new(op))
        }
        OpId::MxFp4Deinterleave => {
            if params.is_empty() {
                return Err(PtwmCoreError::InvalidContainer(
                    "MxFp4Deinterleave params: empty (need 1 byte for block_size)".into(),
                ));
            }
            let op = MxFp4Deinterleave::new(params[0])?;
            Ok(Box::new(op))
        }
        OpId::BlockMicroscalingRepack => {
            if params.is_empty() {
                return Err(PtwmCoreError::InvalidContainer(
                    "BlockMicroscalingRepack params: empty (need 1 byte for block_size)".into(),
                ));
            }
            let op = BlockMicroscalingRepack::new(params[0])?;
            Ok(Box::new(op))
        }
        OpId::IndexBitwidthPack => {
            if params.is_empty() {
                return Err(PtwmCoreError::InvalidContainer(
                    "IndexBitwidthPack params: empty (need 1 byte for bits)".into(),
                ));
            }
            let bits = params[0];
            if bits == 0 || bits > 8 {
                return Err(PtwmCoreError::InvalidContainer(format!(
                    "IndexBitwidthPack: bits={bits} out of range 1..=8"
                )));
            }
            Ok(Box::new(IndexBitwidthPack { bits }))
        }
        OpId::MantissaZeroStrip => {
            if params.is_empty() {
                return Err(PtwmCoreError::InvalidContainer(
                    "MantissaZeroStrip params: empty (need 1 byte for k)".into(),
                ));
            }
            let op = MantissaZeroStrip::new(params[0])?;
            Ok(Box::new(op))
        }

        // ── delta ops ────────────────────────────────────────────────────────
        OpId::XorDelta => {
            if params.is_empty() {
                return Err(PtwmCoreError::InvalidContainer(
                    "XorDelta params: empty (need 1 byte for dep_idx)".into(),
                ));
            }
            Ok(Box::new(XorDelta::new(params[0])))
        }
        OpId::IntDelta => {
            expect_empty_params("IntDelta", params)?;
            Ok(Box::new(IntDelta::new()))
        }
        OpId::PredictorXor => {
            expect_empty_params("PredictorXor", params)?;
            Ok(Box::new(PredictorXor::new()))
        }
        OpId::BurrowsWheeler => {
            expect_empty_params("BurrowsWheeler", params)?;
            Ok(Box::new(BurrowsWheeler::new()))
        }
        OpId::MoveToFront => {
            expect_empty_params("MoveToFront", params)?;
            Ok(Box::new(MoveToFront::new()))
        }
        OpId::FloatDelta => {
            #[cfg(feature = "lossy")]
            {
                if params.len() < 3 {
                    return Err(PtwmCoreError::InvalidContainer(format!(
                        "FloatDelta params: need 3 bytes (dep_idx + dtype_code u16), got {}",
                        params.len()
                    )));
                }
                let dep_idx = params[0];
                let dtype_code = u16::from_le_bytes([params[1], params[2]]);
                tracing::warn!(
                    "FloatDelta op instantiated: this op may break byte-exact round-trip \
. Decoded tensor may differ from original by ~1 ULP."
                );
                Ok(Box::new(FloatDelta {
                    dep_idx,
                    dtype_code,
                }))
            }
            #[cfg(not(feature = "lossy"))]
            {
                let _ = params;
                Err(PtwmCoreError::InvalidContainer(
                    "FloatDelta op is disabled: rebuild ptwm-core with the `lossy` \
                     feature flag to enable approximate float-residual encoding."
                        .into(),
                ))
            }
        }

        // ── variable-param ops ───────────────────────────────────────────────
        OpId::Source => {
            let (op, _) = crate::transforms::source::read_source_params(params)?;
            Ok(Box::new(op))
        }
        OpId::Terminal => {
            let (op, _) = crate::transforms::terminal::read_terminal_params(params)?;
            Ok(Box::new(op))
        }
        OpId::Concat => {
            let (op, _) = crate::transforms::concat::read_concat_params(params)?;
            Ok(Box::new(op))
        }
        OpId::Reshape => {
            let (op, _) = crate::transforms::reshape::read_reshape_params(params)?;
            Ok(Box::new(op))
        }
        OpId::SphericalNormalize => {
            let (op, _) =
                crate::transforms::spherical_normalize::read_spherical_normalize_params(params)?;
            Ok(Box::new(op))
        }
        OpId::AlphaStableNormalize => {
            let (op, _) =
                crate::transforms::alpha_stable_normalize::read_alpha_stable_normalize_params(
                    params,
                )?;
            Ok(Box::new(op))
        }
    }
}

#[inline]
fn expect_empty_params(name: &str, params: &[u8]) -> Result<(), PtwmCoreError> {
    if !params.is_empty() {
        return Err(PtwmCoreError::InvalidContainer(format!(
            "{name} params: expected empty, got {} bytes",
            params.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::descriptor::{ElementWidth, Layout, PlaneDescriptor};
    use crate::types::role::Role;

    fn raw_descriptor(len: u64) -> PlaneDescriptor {
        PlaneDescriptor {
            role: Role::Raw,
            element_width: ElementWidth::Byte,
            length_bytes: len,
            layout: Layout::Flat,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        }
    }

    #[test]
    fn dispatch_source() {
        // Source params: dim_count=1, dims=[16], dtype=0x0002 (Float16, chain-internal).
        let mut params = vec![1u8];
        params.extend_from_slice(&16u32.to_le_bytes());
        params.extend_from_slice(&0x0002u16.to_le_bytes());

        let op = op_from_id(OpId::Source, &params).unwrap();
        let descs = op.propagate_descriptors(&[]).unwrap();
        assert_eq!(descs.len(), 1);
        // 16 elements × 2 bytes/elem = 32 bytes
        assert_eq!(descs[0].length_bytes, 32);
        assert_eq!(descs[0].role, Role::Raw);
    }

    #[test]
    fn dispatch_byte_split() {
        let op = op_from_id(OpId::ByteSplit, &[2u8]).unwrap();
        let inp = raw_descriptor(64);
        let descs = op.propagate_descriptors(&[inp]).unwrap();
        assert_eq!(descs.len(), 2);
        assert_eq!(descs[0].length_bytes, 32);
    }

    #[test]
    fn dispatch_unknown_op_rejected() {
        // OpId::from_u16 would already catch this; the dispatch doesn't see it.
        assert!(OpId::from_u16(0x9999).is_err());
    }

    #[test]
    fn spherical_normalize_dispatch_roundtrips_params() {
        let op = op_from_id(OpId::SphericalNormalize, &[0u8, 0, 0]).unwrap();
        assert_eq!(op.id(), OpId::SphericalNormalize);
        let mut buf = Vec::new();
        op.write_params(&mut buf);
        assert_eq!(buf, vec![0u8, 0, 0]);
        assert!(op_from_id(OpId::SphericalNormalize, &[]).is_err());
    }

    #[test]
    fn alpha_stable_normalize_dispatch_roundtrips_params() {
        let op = op_from_id(OpId::AlphaStableNormalize, &[0u8, 0]).unwrap();
        assert_eq!(op.id(), OpId::AlphaStableNormalize);
        let mut buf = Vec::new();
        op.write_params(&mut buf);
        assert_eq!(buf, vec![0u8, 0]);
        assert!(op_from_id(OpId::AlphaStableNormalize, &[]).is_err());
    }

    #[test]
    fn dispatch_truncated_params_rejected() {
        // ByteSplit requires 1 byte for n; empty params should error.
        assert!(op_from_id(OpId::ByteSplit, &[]).is_err());

        // FloatDelta requires 3 bytes; give only 1.
        assert!(op_from_id(OpId::FloatDelta, &[0u8]).is_err());

        // Source requires at least dim_count byte; empty params should error.
        assert!(op_from_id(OpId::Source, &[]).is_err());
    }
}
