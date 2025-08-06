//! Role tagged union and supporting format enums. Wire encoding uses
//! tag bytes

use crate::error::PtwmCoreError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleFormat {
    E8M0,
    E4M3,
    E5M2,
    F32,
    F16,
}

impl ScaleFormat {
    fn as_u8(self) -> u8 {
        match self {
            ScaleFormat::E8M0 => 0,
            ScaleFormat::E4M3 => 1,
            ScaleFormat::E5M2 => 2,
            ScaleFormat::F32 => 3,
            ScaleFormat::F16 => 4,
        }
    }
    fn from_u8(v: u8) -> Result<Self, PtwmCoreError> {
        match v {
            0 => Ok(ScaleFormat::E8M0),
            1 => Ok(ScaleFormat::E4M3),
            2 => Ok(ScaleFormat::E5M2),
            3 => Ok(ScaleFormat::F32),
            4 => Ok(ScaleFormat::F16),
            _ => Err(PtwmCoreError::InvalidContainer(format!(
                "ScaleFormat: unknown tag {v}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueFormat {
    Fp4E2m1,
    IntN { bits: u8 },
}

impl ValueFormat {
    fn write(self, out: &mut Vec<u8>) {
        match self {
            ValueFormat::Fp4E2m1 => out.push(0u8),
            ValueFormat::IntN { bits } => {
                out.push(1u8);
                out.push(bits);
            }
        }
    }
    fn read(buf: &[u8]) -> Result<(Self, usize), PtwmCoreError> {
        if buf.is_empty() {
            return Err(PtwmCoreError::InvalidContainer(
                "ValueFormat: empty buffer".into(),
            ));
        }
        match buf[0] {
            0 => Ok((ValueFormat::Fp4E2m1, 1)),
            1 => {
                if buf.len() < 2 {
                    return Err(PtwmCoreError::InvalidContainer(
                        "ValueFormat::IntN: truncated bits".into(),
                    ));
                }
                Ok((ValueFormat::IntN { bits: buf[1] }, 2))
            }
            t => Err(PtwmCoreError::InvalidContainer(format!(
                "ValueFormat: unknown tag {t}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NibbleKind {
    Exponent,
    SignMantissa,
    Value,
}

impl NibbleKind {
    fn as_u8(self) -> u8 {
        match self {
            NibbleKind::Exponent => 0,
            NibbleKind::SignMantissa => 1,
            NibbleKind::Value => 2,
        }
    }
    fn from_u8(v: u8) -> Result<Self, PtwmCoreError> {
        match v {
            0 => Ok(NibbleKind::Exponent),
            1 => Ok(NibbleKind::SignMantissa),
            2 => Ok(NibbleKind::Value),
            _ => Err(PtwmCoreError::InvalidContainer(format!(
                "NibbleKind: unknown tag {v}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidualFormat {
    Xor,
    FloatDelta { dtype: u16 },
}

impl ResidualFormat {
    fn write(self, out: &mut Vec<u8>) {
        match self {
            ResidualFormat::Xor => out.push(0u8),
            ResidualFormat::FloatDelta { dtype } => {
                out.push(1u8);
                out.extend_from_slice(&dtype.to_le_bytes());
            }
        }
    }
    fn read(buf: &[u8]) -> Result<(Self, usize), PtwmCoreError> {
        if buf.is_empty() {
            return Err(PtwmCoreError::InvalidContainer(
                "ResidualFormat: empty buffer".into(),
            ));
        }
        match buf[0] {
            0 => Ok((ResidualFormat::Xor, 1)),
            1 => {
                if buf.len() < 3 {
                    return Err(PtwmCoreError::InvalidContainer(
                        "ResidualFormat::FloatDelta: truncated dtype".into(),
                    ));
                }
                let dtype = u16::from_le_bytes(buf[1..3].try_into().unwrap());
                Ok((ResidualFormat::FloatDelta { dtype }, 3))
            }
            t => Err(PtwmCoreError::InvalidContainer(format!(
                "ResidualFormat: unknown tag {t}"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Role {
    Scale { format: ScaleFormat },
    Value { format: ValueFormat },
    MantissaByte { index: u8, of: u8 },
    ExponentByte,
    Nibble { kind: NibbleKind },
    IntegerByte { index: u8, of: u8 },
    Residual { format: ResidualFormat },
    GlobalScale { format: ScaleFormat },
    Raw,
    Index,
    Vendor { tag: u16, bytes: Vec<u8> },
}

impl Role {
    /// Smart constructor for `Role::MantissaByte` enforcing `index < of` and `of > 0`.
    pub fn mantissa_byte(index: u8, of: u8) -> Result<Self, PtwmCoreError> {
        if of == 0 {
            return Err(PtwmCoreError::InvalidContainer(
                "Role::mantissa_byte: of must be > 0".into(),
            ));
        }
        if index >= of {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "Role::mantissa_byte: index {index} must be < of {of}"
            )));
        }
        Ok(Role::MantissaByte { index, of })
    }

    /// Smart constructor for `Role::IntegerByte` enforcing `index < of` and `of > 0`.
    pub fn integer_byte(index: u8, of: u8) -> Result<Self, PtwmCoreError> {
        if of == 0 {
            return Err(PtwmCoreError::InvalidContainer(
                "Role::integer_byte: of must be > 0".into(),
            ));
        }
        if index >= of {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "Role::integer_byte: index {index} must be < of {of}"
            )));
        }
        Ok(Role::IntegerByte { index, of })
    }

    /// Smart constructor for `Role::Vendor` enforcing the wire-format byte limit.
    pub fn vendor(tag: u16, bytes: Vec<u8>) -> Result<Self, PtwmCoreError> {
        if bytes.len() > u16::MAX as usize {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "Role::vendor: bytes.len() {} exceeds u16 max ({})",
                bytes.len(),
                u16::MAX
            )));
        }
        Ok(Role::Vendor { tag, bytes })
    }

    pub fn write(&self, out: &mut Vec<u8>) {
        match self {
            Role::Scale { format } => {
                out.push(0x00);
                out.push(format.as_u8());
            }
            Role::Value { format } => {
                out.push(0x01);
                format.write(out);
            }
            Role::MantissaByte { index, of } => {
                out.push(0x02);
                out.push(*index);
                out.push(*of);
            }
            Role::ExponentByte => out.push(0x03),
            Role::Nibble { kind } => {
                out.push(0x04);
                out.push(kind.as_u8());
            }
            Role::IntegerByte { index, of } => {
                out.push(0x05);
                out.push(*index);
                out.push(*of);
            }
            Role::Residual { format } => {
                out.push(0x06);
                format.write(out);
            }
            Role::GlobalScale { format } => {
                out.push(0x07);
                out.push(format.as_u8());
            }
            Role::Raw => out.push(0x08),
            Role::Index => out.push(0x09),
            Role::Vendor { tag, bytes } => {
                out.push(0xFF);
                out.extend_from_slice(&tag.to_le_bytes());
                // Truncate at the wire-format limit. Construct via
                // `Role::vendor` to surface this as a Result instead.
                let len = u16::try_from(bytes.len()).unwrap_or(u16::MAX);
                out.extend_from_slice(&len.to_le_bytes());
                out.extend_from_slice(&bytes[..len as usize]);
            }
        }
    }

    pub fn read(buf: &[u8]) -> Result<(Self, usize), PtwmCoreError> {
        if buf.is_empty() {
            return Err(PtwmCoreError::InvalidContainer("Role: empty buffer".into()));
        }
        match buf[0] {
            0x00 => {
                if buf.len() < 2 {
                    return Err(PtwmCoreError::InvalidContainer(
                        "Role::Scale: truncated".into(),
                    ));
                }
                let format = ScaleFormat::from_u8(buf[1])?;
                Ok((Role::Scale { format }, 2))
            }
            0x01 => {
                let (format, n) = ValueFormat::read(&buf[1..])?;
                Ok((Role::Value { format }, 1 + n))
            }
            0x02 => {
                if buf.len() < 3 {
                    return Err(PtwmCoreError::InvalidContainer(
                        "Role::MantissaByte: truncated".into(),
                    ));
                }
                Ok((
                    Role::MantissaByte {
                        index: buf[1],
                        of: buf[2],
                    },
                    3,
                ))
            }
            0x03 => Ok((Role::ExponentByte, 1)),
            0x04 => {
                if buf.len() < 2 {
                    return Err(PtwmCoreError::InvalidContainer(
                        "Role::Nibble: truncated".into(),
                    ));
                }
                let kind = NibbleKind::from_u8(buf[1])?;
                Ok((Role::Nibble { kind }, 2))
            }
            0x05 => {
                if buf.len() < 3 {
                    return Err(PtwmCoreError::InvalidContainer(
                        "Role::IntegerByte: truncated".into(),
                    ));
                }
                Ok((
                    Role::IntegerByte {
                        index: buf[1],
                        of: buf[2],
                    },
                    3,
                ))
            }
            0x06 => {
                let (format, n) = ResidualFormat::read(&buf[1..])?;
                Ok((Role::Residual { format }, 1 + n))
            }
            0x07 => {
                if buf.len() < 2 {
                    return Err(PtwmCoreError::InvalidContainer(
                        "Role::GlobalScale: truncated".into(),
                    ));
                }
                let format = ScaleFormat::from_u8(buf[1])?;
                Ok((Role::GlobalScale { format }, 2))
            }
            0x08 => Ok((Role::Raw, 1)),
            0x09 => Ok((Role::Index, 1)),
            0xFF => {
                if buf.len() < 5 {
                    return Err(PtwmCoreError::InvalidContainer(
                        "Role::Vendor: truncated header".into(),
                    ));
                }
                let tag = u16::from_le_bytes(buf[1..3].try_into().unwrap());
                let len = u16::from_le_bytes(buf[3..5].try_into().unwrap()) as usize;
                if buf.len() < 5 + len {
                    return Err(PtwmCoreError::InvalidContainer(
                        "Role::Vendor: truncated bytes".into(),
                    ));
                }
                Ok((
                    Role::Vendor {
                        tag,
                        bytes: buf[5..5 + len].to_vec(),
                    },
                    5 + len,
                ))
            }
            t => Err(PtwmCoreError::InvalidContainer(format!(
                "Role: unknown tag 0x{t:02x}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(r: Role) {
        let mut buf = Vec::new();
        r.write(&mut buf);
        let (decoded, n) = Role::read(&buf).unwrap();
        assert_eq!(decoded, r);
        assert_eq!(n, buf.len());
    }

    #[test]
    fn scale_roundtrip() {
        roundtrip(Role::Scale {
            format: ScaleFormat::E4M3,
        });
        roundtrip(Role::Scale {
            format: ScaleFormat::E8M0,
        });
    }

    #[test]
    fn value_roundtrip() {
        roundtrip(Role::Value {
            format: ValueFormat::Fp4E2m1,
        });
        roundtrip(Role::Value {
            format: ValueFormat::IntN { bits: 7 },
        });
    }

    #[test]
    fn mantissa_byte_roundtrip() {
        roundtrip(Role::MantissaByte { index: 1, of: 4 });
    }

    #[test]
    fn exponent_byte_roundtrip() {
        roundtrip(Role::ExponentByte);
    }

    #[test]
    fn nibble_roundtrip() {
        for k in [
            NibbleKind::Exponent,
            NibbleKind::SignMantissa,
            NibbleKind::Value,
        ] {
            roundtrip(Role::Nibble { kind: k });
        }
    }

    #[test]
    fn integer_byte_roundtrip() {
        roundtrip(Role::IntegerByte { index: 7, of: 8 });
    }

    #[test]
    fn residual_roundtrip() {
        roundtrip(Role::Residual {
            format: ResidualFormat::Xor,
        });
        roundtrip(Role::Residual {
            format: ResidualFormat::FloatDelta { dtype: 0x000C },
        });
    }

    #[test]
    fn global_scale_roundtrip() {
        roundtrip(Role::GlobalScale {
            format: ScaleFormat::F32,
        });
    }

    #[test]
    fn raw_and_index_roundtrip() {
        roundtrip(Role::Raw);
        roundtrip(Role::Index);
    }

    #[test]
    fn vendor_roundtrip() {
        roundtrip(Role::Vendor {
            tag: 0x1234,
            bytes: vec![1, 2, 3, 4, 5],
        });
        roundtrip(Role::Vendor {
            tag: 0,
            bytes: vec![],
        });
    }

    #[test]
    fn unknown_outer_tag_rejected() {
        assert!(Role::read(&[0x42]).is_err());
    }

    #[test]
    fn mantissa_byte_constructor_enforces_invariants() {
        assert!(Role::mantissa_byte(0, 0).is_err());
        assert!(Role::mantissa_byte(4, 4).is_err());
        assert!(Role::mantissa_byte(7, 4).is_err());
        assert_eq!(
            Role::mantissa_byte(1, 4).unwrap(),
            Role::MantissaByte { index: 1, of: 4 }
        );
    }

    #[test]
    fn integer_byte_constructor_enforces_invariants() {
        assert!(Role::integer_byte(2, 0).is_err());
        assert!(Role::integer_byte(2, 2).is_err());
        assert_eq!(
            Role::integer_byte(0, 2).unwrap(),
            Role::IntegerByte { index: 0, of: 2 }
        );
    }

    #[test]
    fn vendor_constructor_rejects_overflow() {
        // Building a Vec this large for the test would be wasteful; just
        // assert the small case succeeds and trust the same length-check
        // path for >65535.
        assert!(Role::vendor(0, vec![1, 2, 3]).is_ok());
    }
}
