use crate::codec::{CodecId, Encoded, PlaneCodec};
use crate::error::PtwmCoreError;
use crate::layout::PlaneLayout;
use crate::types::descriptor::PlaneDescriptor;

pub struct Identity;

impl Identity {
    pub fn canonical_id(&self) -> crate::extension::CanonicalId {
        crate::extension::builtin_canonical_id("identity")
    }
}

impl PlaneCodec for Identity {
    fn id(&self) -> CodecId {
        CodecId::Identity
    }

    fn accepts(&self, _descriptor: &PlaneDescriptor) -> bool {
        true
    }

    fn priority_for(&self, _descriptor: &PlaneDescriptor) -> i8 {
        0
    }

    fn encode(
        &self,
        plane: &[u8],
        _shared_state: Option<&[u8]>,
        _layout: &PlaneLayout,
    ) -> Result<Encoded, PtwmCoreError> {
        Ok(Encoded {
            state_bytes: Vec::new(),
            state_format_version: 0,
            payload: plane.to_vec(),
        })
    }

    fn decode(
        &self,
        _state_format_version: u8,
        _state_bytes: &[u8],
        payload: &[u8],
        _layout: &PlaneLayout,
        _decoded_len: usize,
    ) -> Result<Vec<u8>, PtwmCoreError> {
        Ok(payload.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::PlaneLayout;
    use crate::types::descriptor::{ElementWidth, Layout, PlaneDescriptor};
    use crate::types::role::{Role, ScaleFormat, ValueFormat};

    fn make_descriptor(role: Role, width: ElementWidth, layout: Layout) -> PlaneDescriptor {
        PlaneDescriptor {
            role,
            element_width: width,
            length_bytes: 1024,
            layout,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        }
    }

    #[test]
    fn identity_roundtrip() {
        let c = Identity;
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let enc = c.encode(&data, None, &PlaneLayout::Flat).unwrap();
        assert_eq!(enc.state_bytes.len(), 0);
        let dec = c
            .decode(
                enc.state_format_version,
                &enc.state_bytes,
                &enc.payload,
                &PlaneLayout::Flat,
                data.len(),
            )
            .unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn identity_accepts_any_descriptor() {
        let c = Identity;
        let descriptors = [
            make_descriptor(
                Role::Scale {
                    format: ScaleFormat::E4M3,
                },
                ElementWidth::Byte,
                Layout::Rows { row_len: 16 },
            ),
            make_descriptor(
                Role::Value {
                    format: ValueFormat::Fp4E2m1,
                },
                ElementWidth::Nibble,
                Layout::Flat,
            ),
            make_descriptor(Role::Raw, ElementWidth::Word4, Layout::Flat),
            make_descriptor(
                Role::GlobalScale {
                    format: ScaleFormat::F32,
                },
                ElementWidth::Byte,
                Layout::Flat,
            ),
        ];
        for d in &descriptors {
            assert!(c.accepts(d), "Identity must accept {:?}", d.role);
        }
    }

    #[test]
    fn identity_priority_is_zero() {
        let c = Identity;
        let d = make_descriptor(
            Role::Scale {
                format: ScaleFormat::E4M3,
            },
            ElementWidth::Byte,
            Layout::Flat,
        );
        assert_eq!(c.priority_for(&d), 0);
    }
}
