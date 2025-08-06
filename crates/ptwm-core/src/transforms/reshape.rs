//! `Reshape` op — re-tags a plane's layout without modifying its bytes.

use crate::error::PtwmCoreError;
use crate::transforms::op::{Op, OpId, Plane};
use crate::types::descriptor::{Layout, PlaneDescriptor};

/// `Reshape` re-tags a plane's `Layout` without altering its byte content.
/// Used to convert `Flat` → `Rows{row_len}` before row-aware codecs, or
/// `Rows` → `Flat` to collapse structure.
///
/// When the target layout is `Layout::Rows{row_len}`, the plane's
/// `length_bytes` must be divisible by `row_len`.
pub struct Reshape {
    pub target_layout: Layout,
}

impl Reshape {
    fn validate_layout(&self, length_bytes: u64) -> Result<(), PtwmCoreError> {
        if let Layout::Rows { row_len } = self.target_layout {
            if row_len == 0 {
                return Err(PtwmCoreError::InvalidContainer(
                    "Reshape: row_len must be > 0".into(),
                ));
            }
            if !length_bytes.is_multiple_of(row_len as u64) {
                return Err(PtwmCoreError::InvalidContainer(format!(
                    "Reshape: length_bytes {length_bytes} is not divisible by row_len {row_len}"
                )));
            }
        }
        Ok(())
    }
}

impl Op for Reshape {
    fn propagate_descriptors(
        &self,
        inputs: &[PlaneDescriptor],
    ) -> Result<Vec<PlaneDescriptor>, PtwmCoreError> {
        if inputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "Reshape.propagate_descriptors: expected 1 input, got {}",
                inputs.len()
            )));
        }
        self.validate_layout(inputs[0].length_bytes)?;
        let mut out = inputs[0].clone();
        out.layout = self.target_layout;
        Ok(vec![out])
    }

    fn forward(&self, inputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        if inputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "Reshape.forward: expected 1 input, got {}",
                inputs.len()
            )));
        }
        self.validate_layout(inputs[0].descriptor.length_bytes)?;
        let mut out = inputs[0].clone();
        out.descriptor.layout = self.target_layout;
        Ok(vec![out])
    }

    fn inverse(&self, outputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        // The inverse of Reshape is just returning the plane; the original
        // layout is restored by the preceding op in the decode chain.
        // For a standalone inverse call (outside a full chain), we simply
        // return the output unchanged — the bytes are what matter.
        if outputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "Reshape.inverse: expected 1 output, got {}",
                outputs.len()
            )));
        }
        Ok(vec![outputs[0].clone()])
    }

    fn id(&self) -> OpId {
        OpId::Reshape
    }

    fn write_params(&self, out: &mut Vec<u8>) {
        self.target_layout.write(out);
    }
}

/// Parse `Reshape` params from a raw byte buffer.
pub fn read_reshape_params(buf: &[u8]) -> Result<(Reshape, usize), PtwmCoreError> {
    let (layout, n) = Layout::read(buf)?;
    Ok((
        Reshape {
            target_layout: layout,
        },
        n,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::descriptor::ElementWidth;
    use crate::types::role::Role;

    fn flat_plane(length_bytes: u64) -> Plane {
        Plane {
            bytes: vec![0xAB; length_bytes as usize].into(),
            descriptor: PlaneDescriptor {
                role: Role::Raw,
                element_width: ElementWidth::Byte,
                length_bytes,
                layout: Layout::Flat,
                derives_from_tensor: None,
                residual_of: None,
                is_nibble_packed: false,
                vendor_bytes: vec![],
            },
        }
    }

    #[test]
    fn flat_to_rows_validates_divisibility() {
        // 128 bytes, row_len=16 → 8 rows → OK
        let op = Reshape {
            target_layout: Layout::Rows { row_len: 16 },
        };
        let plane = flat_plane(128);
        assert!(op.forward(std::slice::from_ref(&plane)).is_ok());
        assert!(
            op.propagate_descriptors(std::slice::from_ref(&plane.descriptor))
                .is_ok()
        );

        // 100 bytes, row_len=16 → not divisible → error
        let bad_plane = flat_plane(100);
        assert!(op.forward(std::slice::from_ref(&bad_plane)).is_err());
        assert!(
            op.propagate_descriptors(std::slice::from_ref(&bad_plane.descriptor))
                .is_err()
        );
    }

    #[test]
    fn rows_to_flat_always_succeeds() {
        let op = Reshape {
            target_layout: Layout::Flat,
        };
        // Any length works for Flat
        for len in [0u64, 1, 7, 128, 999] {
            let plane = flat_plane(len);
            assert!(op.forward(&[plane]).is_ok());
        }
    }

    #[test]
    fn forward_inverse_roundtrip() {
        let op = Reshape {
            target_layout: Layout::Rows { row_len: 32 },
        };
        let plane = flat_plane(128);
        let original_bytes = plane.bytes.clone();

        let fwd = op.forward(&[plane]).unwrap();
        assert_eq!(fwd.len(), 1);
        assert_eq!(fwd[0].descriptor.layout, Layout::Rows { row_len: 32 });
        assert_eq!(fwd[0].bytes.as_ref(), original_bytes.as_ref());

        let inv = op.inverse(&fwd).unwrap();
        assert_eq!(inv.len(), 1);
        assert_eq!(inv[0].bytes.as_ref(), original_bytes.as_ref());
    }

    #[test]
    fn write_params_roundtrip() {
        let op = Reshape {
            target_layout: Layout::Rows { row_len: 64 },
        };
        let mut buf = Vec::new();
        op.write_params(&mut buf);
        let (decoded, n) = read_reshape_params(&buf).unwrap();
        assert_eq!(n, buf.len());
        assert_eq!(decoded.target_layout, op.target_layout);
    }
}
