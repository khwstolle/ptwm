//! Plane descriptors and supporting type-level values.

use crate::error::PtwmCoreError;
use crate::types::role::Role;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementWidth {
    Nibble,
    Byte,
    Word2,
    Word4,
    Word8,
}

impl ElementWidth {
    pub fn as_u8(self) -> u8 {
        match self {
            ElementWidth::Nibble => 0,
            ElementWidth::Byte => 1,
            ElementWidth::Word2 => 2,
            ElementWidth::Word4 => 3,
            ElementWidth::Word8 => 4,
        }
    }

    pub fn from_u8(v: u8) -> Result<Self, PtwmCoreError> {
        match v {
            0 => Ok(ElementWidth::Nibble),
            1 => Ok(ElementWidth::Byte),
            2 => Ok(ElementWidth::Word2),
            3 => Ok(ElementWidth::Word4),
            4 => Ok(ElementWidth::Word8),
            _ => Err(PtwmCoreError::InvalidContainer(format!(
                "ElementWidth: unknown tag {v}"
            ))),
        }
    }

    pub fn bits_per_element(self) -> u8 {
        match self {
            ElementWidth::Nibble => 4,
            ElementWidth::Byte => 8,
            ElementWidth::Word2 => 16,
            ElementWidth::Word4 => 32,
            ElementWidth::Word8 => 64,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    Flat,
    Rows { row_len: u32 },
}

impl Layout {
    /// Construct a `Layout::Rows` after validating `row_len > 0`.
    pub fn rows(row_len: u32) -> Result<Self, PtwmCoreError> {
        if row_len == 0 {
            return Err(PtwmCoreError::InvalidContainer(
                "Layout::rows: row_len must be non-zero".into(),
            ));
        }
        Ok(Layout::Rows { row_len })
    }

    pub fn write(self, out: &mut Vec<u8>) {
        match self {
            Layout::Flat => out.push(0u8),
            Layout::Rows { row_len } => {
                out.push(1u8);
                out.extend_from_slice(&row_len.to_le_bytes());
            }
        }
    }

    pub fn read(buf: &[u8]) -> Result<(Self, usize), PtwmCoreError> {
        if buf.is_empty() {
            return Err(PtwmCoreError::InvalidContainer(
                "Layout: empty buffer".into(),
            ));
        }
        match buf[0] {
            0 => Ok((Layout::Flat, 1)),
            1 => {
                if buf.len() < 5 {
                    return Err(PtwmCoreError::InvalidContainer(
                        "Layout::Rows: truncated row_len".into(),
                    ));
                }
                let row_len = u32::from_le_bytes([buf[1], buf[2], buf[3], buf[4]]);
                if row_len == 0 {
                    return Err(PtwmCoreError::InvalidContainer(
                        "Layout::Rows: row_len must be non-zero".into(),
                    ));
                }
                Ok((Layout::Rows { row_len }, 5))
            }
            t => Err(PtwmCoreError::InvalidContainer(format!(
                "Layout: unknown tag {t}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorRef {
    pub dep_idx: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaneRef {
    pub chain_node_idx: u8,
    pub output_idx: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaneDescriptor {
    pub role: Role,
    pub element_width: ElementWidth,
    pub length_bytes: u64,
    pub layout: Layout,
    pub derives_from_tensor: Option<TensorRef>,
    pub residual_of: Option<PlaneRef>,
    pub is_nibble_packed: bool,
    pub vendor_bytes: Vec<u8>,
}

impl PlaneDescriptor {
    pub fn write(&self, out: &mut Vec<u8>) {
        self.role.write(out);
        out.push(self.element_width.as_u8());
        out.extend_from_slice(&self.length_bytes.to_le_bytes());
        self.layout.write(out);
        match self.derives_from_tensor {
            Some(t) => {
                out.push(1u8);
                out.push(t.dep_idx);
            }
            None => out.push(0u8),
        }
        match self.residual_of {
            Some(p) => {
                out.push(1u8);
                out.push(p.chain_node_idx);
                out.push(p.output_idx);
            }
            None => out.push(0u8),
        }
        out.push(self.is_nibble_packed as u8);
        let vlen = u16::try_from(self.vendor_bytes.len()).expect("vendor bytes ≤ 65535");
        out.extend_from_slice(&vlen.to_le_bytes());
        out.extend_from_slice(&self.vendor_bytes);
    }

    pub fn read(buf: &[u8]) -> Result<(Self, usize), PtwmCoreError> {
        let mut pos = 0;
        let (role, n) = Role::read(buf)?;
        pos += n;
        if buf.len() < pos + 1 {
            return Err(PtwmCoreError::InvalidContainer(
                "PlaneDescriptor: truncated element_width".into(),
            ));
        }
        let element_width = ElementWidth::from_u8(buf[pos])?;
        pos += 1;
        if buf.len() < pos + 8 {
            return Err(PtwmCoreError::InvalidContainer(
                "PlaneDescriptor: truncated length_bytes".into(),
            ));
        }
        let length_bytes = u64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap());
        pos += 8;
        let (layout, n) = Layout::read(&buf[pos..])?;
        pos += n;
        if buf.len() < pos + 1 {
            return Err(PtwmCoreError::InvalidContainer(
                "PlaneDescriptor: truncated derives_from_tensor flag".into(),
            ));
        }
        let derives_from_tensor = match buf[pos] {
            0 => {
                pos += 1;
                None
            }
            1 => {
                if buf.len() < pos + 2 {
                    return Err(PtwmCoreError::InvalidContainer(
                        "PlaneDescriptor::derives_from_tensor: truncated".into(),
                    ));
                }
                let dep_idx = buf[pos + 1];
                pos += 2;
                Some(TensorRef { dep_idx })
            }
            t => {
                return Err(PtwmCoreError::InvalidContainer(format!(
                    "PlaneDescriptor::derives_from_tensor: bad flag {t}"
                )));
            }
        };
        if buf.len() < pos + 1 {
            return Err(PtwmCoreError::InvalidContainer(
                "PlaneDescriptor: truncated residual_of flag".into(),
            ));
        }
        let residual_of = match buf[pos] {
            0 => {
                pos += 1;
                None
            }
            1 => {
                if buf.len() < pos + 3 {
                    return Err(PtwmCoreError::InvalidContainer(
                        "PlaneDescriptor::residual_of: truncated".into(),
                    ));
                }
                let chain_node_idx = buf[pos + 1];
                let output_idx = buf[pos + 2];
                pos += 3;
                Some(PlaneRef {
                    chain_node_idx,
                    output_idx,
                })
            }
            t => {
                return Err(PtwmCoreError::InvalidContainer(format!(
                    "PlaneDescriptor::residual_of: bad flag {t}"
                )));
            }
        };
        if buf.len() < pos + 1 {
            return Err(PtwmCoreError::InvalidContainer(
                "PlaneDescriptor: truncated is_nibble_packed".into(),
            ));
        }
        let is_nibble_packed = buf[pos] != 0;
        pos += 1;
        if buf.len() < pos + 2 {
            return Err(PtwmCoreError::InvalidContainer(
                "PlaneDescriptor: truncated vendor_bytes_len".into(),
            ));
        }
        let vlen = u16::from_le_bytes(buf[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;
        if buf.len() < pos + vlen {
            return Err(PtwmCoreError::InvalidContainer(
                "PlaneDescriptor: truncated vendor_bytes".into(),
            ));
        }
        let vendor_bytes = buf[pos..pos + vlen].to_vec();
        pos += vlen;
        Ok((
            PlaneDescriptor {
                role,
                element_width,
                length_bytes,
                layout,
                derives_from_tensor,
                residual_of,
                is_nibble_packed,
                vendor_bytes,
            },
            pos,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn element_width_roundtrip() {
        for ew in [
            ElementWidth::Nibble,
            ElementWidth::Byte,
            ElementWidth::Word2,
            ElementWidth::Word4,
            ElementWidth::Word8,
        ] {
            assert_eq!(ElementWidth::from_u8(ew.as_u8()).unwrap(), ew);
        }
    }

    #[test]
    fn element_width_unknown_tag_rejected() {
        assert!(ElementWidth::from_u8(99).is_err());
    }

    #[test]
    fn layout_flat_roundtrip() {
        let mut buf = Vec::new();
        Layout::Flat.write(&mut buf);
        assert_eq!(buf.len(), 1);
        let (l, n) = Layout::read(&buf).unwrap();
        assert_eq!(l, Layout::Flat);
        assert_eq!(n, 1);
    }

    #[test]
    fn layout_rows_roundtrip() {
        let mut buf = Vec::new();
        Layout::Rows {
            row_len: 0xCAFEBABE,
        }
        .write(&mut buf);
        assert_eq!(buf.len(), 5);
        let (l, n) = Layout::read(&buf).unwrap();
        assert_eq!(
            l,
            Layout::Rows {
                row_len: 0xCAFEBABE
            }
        );
        assert_eq!(n, 5);
    }

    #[test]
    fn layout_unknown_tag_rejected() {
        assert!(Layout::read(&[42u8]).is_err());
    }

    #[test]
    fn layout_rows_zero_row_len_rejected() {
        let buf = [1u8, 0, 0, 0, 0];
        assert!(Layout::read(&buf).is_err());
    }

    #[test]
    fn layout_rows_constructor_rejects_zero() {
        assert!(Layout::rows(0).is_err());
        assert!(Layout::rows(1).is_ok());
    }

    #[test]
    fn layout_truncated_rejected() {
        assert!(Layout::read(&[1u8, 0, 0]).is_err());
    }

    use crate::types::role::{Role, ScaleFormat};

    fn sample_descriptor() -> PlaneDescriptor {
        PlaneDescriptor {
            role: Role::Scale {
                format: ScaleFormat::E4M3,
            },
            element_width: ElementWidth::Byte,
            length_bytes: 1024,
            layout: Layout::Rows { row_len: 16 },
            derives_from_tensor: Some(TensorRef { dep_idx: 3 }),
            residual_of: Some(PlaneRef {
                chain_node_idx: 5,
                output_idx: 1,
            }),
            is_nibble_packed: true,
            vendor_bytes: vec![0xAB, 0xCD, 0xEF],
        }
    }

    #[test]
    fn descriptor_full_roundtrip() {
        let d = sample_descriptor();
        let mut buf = Vec::new();
        d.write(&mut buf);
        let (decoded, n) = PlaneDescriptor::read(&buf).unwrap();
        assert_eq!(decoded, d);
        assert_eq!(n, buf.len());
    }

    #[test]
    fn descriptor_no_optionals_roundtrip() {
        let d = PlaneDescriptor {
            role: Role::Raw,
            element_width: ElementWidth::Word4,
            length_bytes: 0,
            layout: Layout::Flat,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        };
        let mut buf = Vec::new();
        d.write(&mut buf);
        let (decoded, _) = PlaneDescriptor::read(&buf).unwrap();
        assert_eq!(decoded, d);
    }

    #[test]
    fn descriptor_truncated_rejected() {
        let d = sample_descriptor();
        let mut buf = Vec::new();
        d.write(&mut buf);
        for cut in 1..buf.len() {
            assert!(
                PlaneDescriptor::read(&buf[..cut]).is_err(),
                "expected error for truncation at {cut}"
            );
        }
    }
}
