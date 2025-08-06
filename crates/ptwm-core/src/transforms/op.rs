//! `Op` trait + `OpId` enum. The op catalogue's wire form lives here;
//! per-op behaviour lives in sibling modules under `transforms/`.

use std::sync::Arc;

use crate::error::PtwmCoreError;
use crate::types::descriptor::{ElementWidth, PlaneDescriptor};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum OpId {
    Source = 0x0000,
    Terminal = 0x0001,
    BitReorderIeee16 = 0x0010,
    BitReorderIeee32 = 0x0011,
    BitReorderFp8E4M3 = 0x0012,
    BitReorderFp8E5M2 = 0x0013,
    ByteSplit = 0x0020,
    NibbleSplit = 0x0021,
    BytePassthrough = 0x0030,
    MxFp4Deinterleave = 0x0040,
    BlockMicroscalingRepack = 0x0041,
    XorDelta = 0x0050,
    FloatDelta = 0x0051,
    IntDelta = 0x0052,
    PredictorXor = 0x0053,
    IndexBitwidthPack = 0x0060,
    EntropyEstimate = 0x0061,
    Concat = 0x0062,
    Reshape = 0x0063,
    MantissaZeroStrip = 0x0064,
    BurrowsWheeler = 0x0070,
    MoveToFront = 0x0071,
    SphericalNormalize = 0x0080,
    AlphaStableNormalize = 0x0081,
}

impl OpId {
    pub fn canonical_id(self) -> crate::extension::CanonicalId {
        let name = match self {
            OpId::Source => "source",
            OpId::Terminal => "terminal",
            OpId::BitReorderIeee16 => "bit_reorder_ieee16",
            OpId::BitReorderIeee32 => "bit_reorder_ieee32",
            OpId::BitReorderFp8E4M3 => "bit_reorder_fp8_e4m3",
            OpId::BitReorderFp8E5M2 => "bit_reorder_fp8_e5m2",
            OpId::ByteSplit => "byte_split",
            OpId::NibbleSplit => "nibble_split",
            OpId::BytePassthrough => "byte_passthrough",
            OpId::MxFp4Deinterleave => "mxfp4_deinterleave",
            OpId::BlockMicroscalingRepack => "block_microscaling_repack",
            OpId::XorDelta => "xor_delta",
            OpId::FloatDelta => "float_delta",
            OpId::IntDelta => "int_delta",
            OpId::PredictorXor => "predictor_xor",
            OpId::IndexBitwidthPack => "index_bitwidth_pack",
            OpId::EntropyEstimate => "entropy_estimate",
            OpId::Concat => "concat",
            OpId::Reshape => "reshape",
            OpId::MantissaZeroStrip => "mantissa_zero_strip",
            OpId::BurrowsWheeler => "burrows_wheeler",
            OpId::MoveToFront => "move_to_front",
            OpId::SphericalNormalize => "spherical_normalize",
            OpId::AlphaStableNormalize => "alpha_stable_normalize",
        };
        crate::extension::builtin_canonical_id(name)
    }

    pub fn as_u16(self) -> u16 {
        self as u16
    }

    pub fn from_u16(v: u16) -> Result<Self, PtwmCoreError> {
        let op = match v {
            0x0000 => OpId::Source,
            0x0001 => OpId::Terminal,
            0x0010 => OpId::BitReorderIeee16,
            0x0011 => OpId::BitReorderIeee32,
            0x0012 => OpId::BitReorderFp8E4M3,
            0x0013 => OpId::BitReorderFp8E5M2,
            0x0020 => OpId::ByteSplit,
            0x0021 => OpId::NibbleSplit,
            0x0030 => OpId::BytePassthrough,
            0x0040 => OpId::MxFp4Deinterleave,
            0x0041 => OpId::BlockMicroscalingRepack,
            0x0050 => OpId::XorDelta,
            0x0051 => OpId::FloatDelta,
            0x0052 => OpId::IntDelta,
            0x0053 => OpId::PredictorXor,
            0x0060 => OpId::IndexBitwidthPack,
            0x0061 => OpId::EntropyEstimate,
            0x0062 => OpId::Concat,
            0x0063 => OpId::Reshape,
            0x0064 => OpId::MantissaZeroStrip,
            0x0070 => OpId::BurrowsWheeler,
            0x0071 => OpId::MoveToFront,
            0x0080 => OpId::SphericalNormalize,
            0x0081 => OpId::AlphaStableNormalize,
            _ => {
                return Err(PtwmCoreError::InvalidContainer(format!(
                    "OpId: unknown 0x{v:04x}"
                )));
            }
        };
        Ok(op)
    }
}

#[derive(Debug, Clone)]
pub struct Plane {
    pub bytes: Arc<[u8]>,
    pub descriptor: PlaneDescriptor,
}

impl Plane {
    /// Construct a `Plane` after validating that `bytes.len()` matches
    /// `descriptor.length_bytes`. Expects one nibble-per-byte slot when
    /// `descriptor.element_width == Nibble` and `is_nibble_packed` is
    /// false; nibble-packed planes use half-byte storage.
    pub fn new(bytes: Vec<u8>, descriptor: PlaneDescriptor) -> Result<Self, PtwmCoreError> {
        let expected = expected_byte_len(&descriptor);
        if bytes.len() as u64 != expected {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "Plane::new: bytes.len()={} does not match descriptor (expected={expected}, \
                 length_bytes={}, element_width={:?}, is_nibble_packed={})",
                bytes.len(),
                descriptor.length_bytes,
                descriptor.element_width,
                descriptor.is_nibble_packed,
            )));
        }
        Ok(Plane {
            bytes: Arc::from(bytes.into_boxed_slice()),
            descriptor,
        })
    }

    /// Returns true when `bytes.len()` matches the descriptor.
    /// `debug_assert!` callers in the runtime use this; cheap enough for
    /// production calls, kept as a query so callers can decide.
    #[inline]
    pub fn is_consistent(&self) -> bool {
        self.bytes.len() as u64 == expected_byte_len(&self.descriptor)
    }
}

/// Expected byte length for a plane's storage given its descriptor.
fn expected_byte_len(descriptor: &PlaneDescriptor) -> u64 {
    if descriptor.is_nibble_packed && descriptor.element_width == ElementWidth::Nibble {
        // Two nibbles per byte; round up so an odd element count still fits.
        descriptor.length_bytes.div_ceil(2)
    } else {
        descriptor.length_bytes
    }
}

/// Pure-function trait every op implements.
pub trait Op {
    /// Return the descriptors of this op's outputs given its input
    /// descriptors. Pure, deterministic, side-effect-free.
    fn propagate_descriptors(
        &self,
        inputs: &[PlaneDescriptor],
    ) -> Result<Vec<PlaneDescriptor>, PtwmCoreError>;

    /// Encode-side: take input planes, produce output planes.
    fn forward(&self, inputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError>;

    /// Decode-side: take output planes, reproduce input planes. Must
    /// invert `forward` exactly.
    fn inverse(&self, outputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError>;

    /// This op's wire identifier.
    fn id(&self) -> OpId;

    /// Encode this op's parameters (after the OpId on the wire).
    fn write_params(&self, out: &mut Vec<u8>);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn op_id_roundtrip() {
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
            OpId::IndexBitwidthPack,
            OpId::EntropyEstimate,
            OpId::Concat,
            OpId::Reshape,
            OpId::MantissaZeroStrip,
            OpId::BurrowsWheeler,
            OpId::MoveToFront,
            OpId::SphericalNormalize,
            OpId::AlphaStableNormalize,
        ] {
            assert_eq!(OpId::from_u16(op.as_u16()).unwrap(), op);
        }
    }

    #[test]
    fn op_id_unknown_rejected() {
        assert!(OpId::from_u16(0x9999).is_err());
    }

    use crate::types::descriptor::{Layout, PlaneDescriptor};
    use crate::types::role::Role;

    #[test]
    fn plane_new_accepts_consistent_bytes() {
        let desc = PlaneDescriptor {
            role: Role::Raw,
            element_width: ElementWidth::Byte,
            length_bytes: 4,
            layout: Layout::Flat,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        };
        let p = Plane::new(vec![1, 2, 3, 4], desc).unwrap();
        assert!(p.is_consistent());
    }

    #[test]
    fn plane_new_rejects_inconsistent_bytes() {
        let desc = PlaneDescriptor {
            role: Role::Raw,
            element_width: ElementWidth::Byte,
            length_bytes: 4,
            layout: Layout::Flat,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        };
        assert!(Plane::new(vec![1, 2, 3], desc).is_err());
    }

    #[test]
    fn plane_new_handles_nibble_packed() {
        // 5 nibbles → 3 bytes packed.
        let desc = PlaneDescriptor {
            role: Role::Raw,
            element_width: ElementWidth::Nibble,
            length_bytes: 5,
            layout: Layout::Flat,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed: true,
            vendor_bytes: vec![],
        };
        assert!(Plane::new(vec![0; 3], desc.clone()).is_ok());
        assert!(Plane::new(vec![0; 2], desc).is_err());
    }
}
