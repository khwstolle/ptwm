//! Opt-in context-mixing plane codec. Stateless; self-delimiting (the
//! coder frames the plane length, so `decode` ignores the container's
//! `decoded_len`). NOT in the default dispatch menu — selected only when
//! named in `codec_menu`.

use crate::codec::{CodecId, Encoded, PlaneCodec};
use crate::entropy::context_mixing::{decode_plane, encode_plane};
use crate::error::PtwmCoreError;
use crate::layout::PlaneLayout;
use crate::types::descriptor::PlaneDescriptor;

/// Below this size CM cannot amortize its cost; skip the (slow) attempt.
const CM_MIN_PLANE_BYTES: usize = 1 << 12;

pub struct ContextMixingLite;

impl ContextMixingLite {
    pub fn canonical_id(&self) -> crate::extension::CanonicalId {
        crate::extension::builtin_canonical_id("context_mixing_lite")
    }
}

impl PlaneCodec for ContextMixingLite {
    fn id(&self) -> CodecId {
        CodecId::ContextMixingLite
    }

    fn should_attempt(
        &self,
        plane: &[u8],
        _descriptor: &PlaneDescriptor,
        _layout: &PlaneLayout,
    ) -> bool {
        plane.len() >= CM_MIN_PLANE_BYTES
    }

    fn encode(
        &self,
        plane: &[u8],
        shared_state: Option<&[u8]>,
        _layout: &PlaneLayout,
    ) -> Result<Encoded, PtwmCoreError> {
        if shared_state.is_some() {
            return Err(PtwmCoreError::CodecEncode {
                codec: "context_mixing_lite",
                msg: "shared state not supported".into(),
            });
        }
        // The payload frames the plane length as u32; reject anything the
        // decoder would refuse (MAX_LEN) so the usize→u32 cast can't silently
        // truncate.
        if plane.len() > crate::entropy::context_mixing::coder::MAX_LEN {
            return Err(PtwmCoreError::CodecEncode {
                codec: "context_mixing_lite",
                msg: format!(
                    "plane length {} exceeds the {}-byte framing limit",
                    plane.len(),
                    crate::entropy::context_mixing::coder::MAX_LEN
                ),
            });
        }
        Ok(Encoded {
            state_bytes: Vec::new(),
            state_format_version: 0,
            payload: encode_plane(plane),
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
        decode_plane(payload)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rt(data: &[u8]) {
        let enc = ContextMixingLite
            .encode(data, None, &PlaneLayout::Flat)
            .unwrap();
        // Pass a deliberately-wrong decoded_len: self-delimiting decode must
        // ignore it (regression guard for wrong-length decode).
        let dec = ContextMixingLite
            .decode(
                0,
                &enc.state_bytes,
                &enc.payload,
                &PlaneLayout::Flat,
                data.len() * 4 + 7,
            )
            .unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn roundtrip_compressible() {
        let data: Vec<u8> = (0..8192).map(|i| ((i * 13) % 7) as u8).collect();
        rt(&data);
    }

    #[test]
    fn roundtrip_random() {
        let data: Vec<u8> = (0..4096).map(|i| ((i * 211) & 0xFF) as u8).collect();
        rt(&data);
    }

    #[test]
    fn roundtrip_empty_and_single() {
        rt(&[]);
        rt(&vec![0x33u8; 5000]);
    }

    #[test]
    fn id_and_no_state() {
        let c = ContextMixingLite;
        assert_eq!(c.id(), CodecId::ContextMixingLite);
        let enc = c.encode(&[1u8; 64], None, &PlaneLayout::Flat).unwrap();
        assert!(enc.state_bytes.is_empty());
    }

    #[test]
    fn should_attempt_size_gate() {
        let d = PlaneDescriptor {
            role: crate::types::role::Role::Raw,
            element_width: crate::types::descriptor::ElementWidth::Byte,
            length_bytes: 64,
            layout: crate::types::descriptor::Layout::Flat,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        };
        assert!(!ContextMixingLite.should_attempt(&[0u8; 64], &d, &PlaneLayout::Flat));
        assert!(ContextMixingLite.should_attempt(&[0u8; 8192], &d, &PlaneLayout::Flat));
    }
}
