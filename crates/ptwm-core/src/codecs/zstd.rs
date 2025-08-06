//! Zstd plane codec. Wraps the existing
//! `crate::codec_tagged::zstd_{encode,decode}` helpers so the container
//! can use it as a generic meta-compressor without new dependencies.
//!
//! `Method::ZSTD` selects this codec for the whole tensor. The
//! upstream byte-split preprocessing op already supplies the SHUFFLE-
//! style reordering Zstd would otherwise need, so the plane codec runs
//! plain Zstd.
//!
//! The default compression level is 3 — a good balance between speed
//! and ratio.

use crate::codec::{CodecId, Encoded, PlaneCodec};
use crate::error::PtwmCoreError;
use crate::layout::PlaneLayout;
use crate::types::descriptor::PlaneDescriptor;

const DEFAULT_LEVEL: i32 = 3;

pub struct Zstd;

impl Zstd {
    pub fn canonical_id(&self) -> crate::extension::CanonicalId {
        crate::extension::builtin_canonical_id("zstd")
    }
}

impl PlaneCodec for Zstd {
    fn id(&self) -> CodecId {
        CodecId::Zstd
    }

    fn accepts(&self, _descriptor: &PlaneDescriptor) -> bool {
        true
    }

    fn priority_for(&self, _descriptor: &PlaneDescriptor) -> i8 {
        1
    }

    fn encode(
        &self,
        plane: &[u8],
        _shared_state: Option<&[u8]>,
        _layout: &PlaneLayout,
    ) -> Result<Encoded, PtwmCoreError> {
        #[cfg(feature = "codec-zstd")]
        {
            let payload = crate::codec_tagged::zstd_encode(plane, DEFAULT_LEVEL)?;
            Ok(Encoded {
                state_bytes: Vec::new(),
                state_format_version: 0,
                payload,
            })
        }
        #[cfg(not(feature = "codec-zstd"))]
        {
            let _ = plane;
            Err(PtwmCoreError::InvalidContainer(
                "codec-zstd feature is disabled".into(),
            ))
        }
    }

    fn decode(
        &self,
        _state_format_version: u8,
        _state_bytes: &[u8],
        payload: &[u8],
        _layout: &PlaneLayout,
        _decoded_len: usize,
    ) -> Result<Vec<u8>, PtwmCoreError> {
        #[cfg(feature = "codec-zstd")]
        {
            crate::codec_tagged::zstd_decode(payload)
        }
        #[cfg(not(feature = "codec-zstd"))]
        {
            let _ = payload;
            Err(PtwmCoreError::InvalidContainer(
                "codec-zstd feature is disabled".into(),
            ))
        }
    }
}

#[cfg(test)]
mod capability_tests {
    use super::*;
    use crate::types::descriptor::{ElementWidth, Layout, PlaneDescriptor};
    use crate::types::role::{Role, ScaleFormat};

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
    fn zstd_accepts_any_descriptor() {
        let c = Zstd;
        let descriptors = [
            make_descriptor(
                Role::Scale {
                    format: ScaleFormat::E4M3,
                },
                ElementWidth::Byte,
                Layout::Rows { row_len: 16 },
            ),
            make_descriptor(Role::Raw, ElementWidth::Word4, Layout::Flat),
            make_descriptor(Role::ExponentByte, ElementWidth::Byte, Layout::Flat),
        ];
        for d in &descriptors {
            assert!(c.accepts(d), "Zstd must accept {:?}", d.role);
        }
    }

    #[test]
    fn zstd_priority_is_one() {
        let c = Zstd;
        let d = make_descriptor(
            Role::Scale {
                format: ScaleFormat::E4M3,
            },
            ElementWidth::Byte,
            Layout::Flat,
        );
        assert_eq!(c.priority_for(&d), 1);
    }
}

#[cfg(all(test, feature = "codec-zstd"))]
mod tests {
    use super::*;
    use crate::layout::PlaneLayout;

    #[test]
    fn zstd_roundtrip() {
        let c = Zstd;
        // Compressible-ish data: repetitive but not trivially constant.
        let data: Vec<u8> = (0..1024).map(|i| ((i * 13) & 0xFF) as u8).collect();
        let enc = c.encode(&data, None, &PlaneLayout::Flat).unwrap();
        // ZSTD of 1 KB should not exceed 1 KB significantly.
        assert!(enc.payload.len() < 2048);
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
    fn zstd_empty_roundtrip() {
        let c = Zstd;
        let enc = c.encode(&[], None, &PlaneLayout::Flat).unwrap();
        let dec = c
            .decode(
                enc.state_format_version,
                &enc.state_bytes,
                &enc.payload,
                &PlaneLayout::Flat,
                0,
            )
            .unwrap();
        assert_eq!(dec, Vec::<u8>::new());
    }
}
