//! On-disk plane role tag (single byte stored in each plane record).
//!
//! [`PlaneRole`] is the wire-format role enum — it identifies the semantic
//! purpose of a plane (value, scale, exponent, …) inside a `.ptwm`
//! container. The richer runtime [`crate::types::role::Role`] enum carries
//! per-variant metadata used during chain dispatch; this byte tag is its
//! canonical on-disk projection (see `legacy_plane_role` in the
//! compressor).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PlaneRole {
    Value = 0x01,
    Scale = 0x02,
    Exponent = 0x03,
    Sign = 0x04,
    Mantissa = 0x05,
    ZeroPoint = 0x06,
    Byte0 = 0x10,
    Byte1 = 0x11,
    Byte2 = 0x12,
    Byte3 = 0x13,
}

impl PlaneRole {
    pub fn from_u8(raw: u8) -> Option<Self> {
        match raw {
            0x01 => Some(Self::Value),
            0x02 => Some(Self::Scale),
            0x03 => Some(Self::Exponent),
            0x04 => Some(Self::Sign),
            0x05 => Some(Self::Mantissa),
            0x06 => Some(Self::ZeroPoint),
            0x10 => Some(Self::Byte0),
            0x11 => Some(Self::Byte1),
            0x12 => Some(Self::Byte2),
            0x13 => Some(Self::Byte3),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_role_roundtrip() {
        for role in [PlaneRole::Value, PlaneRole::Scale, PlaneRole::Byte0] {
            assert_eq!(PlaneRole::from_u8(role as u8), Some(role));
        }
    }
}
