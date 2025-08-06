//! Per-plane physical structure passed to codecs at encode/decode time.

use std::num::NonZeroU32;

use crate::error::PtwmCoreError;

#[non_exhaustive]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum PlaneLayout {
    /// No structure information. Default for every preprocessing spec.
    #[default]
    Flat,
    /// 1-D plane partitioned into fixed-width rows. `NonZeroU32` makes
    /// "row_len > 0" unrepresentable as illegal state.
    Rows { row_len: NonZeroU32 },
}

impl PlaneLayout {
    /// Construct a `Rows` layout, returning `None` if `row_len == 0`.
    pub fn rows(row_len: u32) -> Option<Self> {
        NonZeroU32::new(row_len).map(|row_len| Self::Rows { row_len })
    }

    /// Wire discriminant byte.
    pub fn kind(&self) -> u8 {
        match self {
            Self::Flat => 0,
            Self::Rows { .. } => 1,
        }
    }

    /// Wire payload: empty for `Flat`, 4 bytes (u32 LE) for `Rows`.
    pub fn payload_bytes(&self) -> Vec<u8> {
        match self {
            Self::Flat => Vec::new(),
            Self::Rows { row_len } => row_len.get().to_le_bytes().to_vec(),
        }
    }

    /// Parse from a discriminant byte and a slice positioned at the
    /// payload bytes. Returns the `PlaneLayout` and bytes consumed.
    pub fn parse(kind: u8, payload: &[u8]) -> Result<(Self, usize), PtwmCoreError> {
        match kind {
            0 => Ok((Self::Flat, 0)),
            1 => {
                if payload.len() < 4 {
                    return Err(PtwmCoreError::InvalidContainer(
                        "PlaneLayout::Rows requires 4-byte row_len".into(),
                    ));
                }
                let raw = u32::from_le_bytes(payload[..4].try_into().unwrap());
                let row_len = NonZeroU32::new(raw).ok_or_else(|| {
                    PtwmCoreError::InvalidContainer("PlaneLayout::Rows row_len must be > 0".into())
                })?;
                Ok((Self::Rows { row_len }, 4))
            }
            other => Err(PtwmCoreError::InvalidContainer(format!(
                "unknown PlaneLayout kind {other}"
            ))),
        }
    }

    /// The row stride if this layout carries one.
    pub fn row_len(&self) -> Option<u32> {
        match self {
            Self::Rows { row_len } => Some(row_len.get()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_roundtrip() {
        let l = PlaneLayout::Flat;
        let bytes = l.payload_bytes();
        let (parsed, consumed) = PlaneLayout::parse(l.kind(), &bytes).unwrap();
        assert_eq!(parsed, l);
        assert_eq!(consumed, 0);
    }

    #[test]
    fn rows_roundtrip() {
        let l = PlaneLayout::rows(12345).unwrap();
        let bytes = l.payload_bytes();
        let (parsed, consumed) = PlaneLayout::parse(l.kind(), &bytes).unwrap();
        assert_eq!(parsed, l);
        assert_eq!(consumed, 4);
    }

    #[test]
    fn rows_constructor_rejects_zero() {
        assert!(PlaneLayout::rows(0).is_none());
        assert!(PlaneLayout::rows(1).is_some());
    }

    #[test]
    fn parse_rejects_unknown_kind() {
        assert!(PlaneLayout::parse(2, &[]).is_err());
        assert!(PlaneLayout::parse(255, &[]).is_err());
    }

    #[test]
    fn parse_rejects_truncated_rows_payload() {
        assert!(PlaneLayout::parse(1, &[0u8; 3]).is_err());
    }

    #[test]
    fn parse_rejects_zero_row_len() {
        let bytes = 0u32.to_le_bytes();
        let err = PlaneLayout::parse(1, &bytes).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("row_len"), "{msg}");
    }

    #[test]
    fn rows_roundtrip_at_max_u32() {
        let l = PlaneLayout::rows(u32::MAX).unwrap();
        let bytes = l.payload_bytes();
        let (parsed, _) = PlaneLayout::parse(l.kind(), &bytes).unwrap();
        assert_eq!(parsed, l);
    }

    #[test]
    fn row_len_accessor() {
        assert_eq!(PlaneLayout::Flat.row_len(), None);
        assert_eq!(PlaneLayout::rows(7).unwrap().row_len(), Some(7));
    }
}
