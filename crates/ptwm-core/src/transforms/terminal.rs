//! `Terminal` op — graph leaf node that consumes one plane and feeds a codec.

use crate::error::PtwmCoreError;
use crate::transforms::op::{Op, OpId, Plane};
use crate::types::descriptor::PlaneDescriptor;
use crate::types::role::Role;

/// `Terminal` is the mandatory leaf node of every PPG chain. It consumes
/// exactly one plane (which flows to the codec) and produces no output planes.
/// The `role` stored on this node is the authoritative codec-dispatch key.
pub struct Terminal {
    pub role: Role,
}

impl Op for Terminal {
    fn propagate_descriptors(
        &self,
        inputs: &[PlaneDescriptor],
    ) -> Result<Vec<PlaneDescriptor>, PtwmCoreError> {
        if inputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "Terminal.propagate_descriptors: expected 1 input, got {}",
                inputs.len()
            )));
        }
        // Terminals consume their input and produce no output descriptors.
        Ok(vec![])
    }

    fn forward(&self, _inputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        // Terminal is a no-op data-transform; the runtime routes the plane to
        // the codec directly.
        Ok(vec![])
    }

    fn inverse(&self, _outputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        Ok(vec![])
    }

    fn id(&self) -> OpId {
        OpId::Terminal
    }

    fn write_params(&self, out: &mut Vec<u8>) {
        self.role.write(out);
    }
}

/// Parse `Terminal` params from a raw byte buffer. Returns `(Terminal, bytes_consumed)`.
pub fn read_terminal_params(buf: &[u8]) -> Result<(Terminal, usize), PtwmCoreError> {
    let (role, n) = Role::read(buf)?;
    Ok((Terminal { role }, n))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::descriptor::{ElementWidth, Layout};
    use crate::types::role::ScaleFormat;

    fn dummy_descriptor() -> PlaneDescriptor {
        PlaneDescriptor {
            role: Role::Raw,
            element_width: ElementWidth::Byte,
            length_bytes: 256,
            layout: Layout::Flat,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        }
    }

    #[test]
    fn propagate_descriptors_consumes_one_plane() {
        let t = Terminal {
            role: Role::Scale {
                format: ScaleFormat::E8M0,
            },
        };
        let result = t.propagate_descriptors(&[dummy_descriptor()]).unwrap();
        assert!(
            result.is_empty(),
            "Terminal should produce no output descriptors"
        );
    }

    #[test]
    fn propagate_descriptors_rejects_zero_inputs() {
        let t = Terminal { role: Role::Raw };
        assert!(t.propagate_descriptors(&[]).is_err());
    }

    #[test]
    fn propagate_descriptors_rejects_two_inputs() {
        let t = Terminal { role: Role::Raw };
        assert!(
            t.propagate_descriptors(&[dummy_descriptor(), dummy_descriptor()])
                .is_err()
        );
    }

    #[test]
    fn write_params_roundtrip() {
        let role = Role::Scale {
            format: ScaleFormat::F32,
        };
        let t = Terminal { role: role.clone() };
        let mut buf = Vec::new();
        t.write_params(&mut buf);
        let (decoded, n) = read_terminal_params(&buf).unwrap();
        assert_eq!(n, buf.len());
        assert_eq!(decoded.role, role);
    }
}
