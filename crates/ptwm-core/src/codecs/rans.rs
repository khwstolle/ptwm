//! rANS plane codec.
//!
//! Mirrors the Huffman wrapper: TAG byte + optional length prefix +
//! payload. Delegates to the `crate::codec_tagged::rans_{encode,decode}`
//! helpers, which already apply the raw-fallback-on-expansion guard.

use crate::codec::{CodecId, Encoded, PlaneCodec};
use crate::codec_tagged::{rans_decode, rans_encode};
use crate::error::PtwmCoreError;
use crate::layout::PlaneLayout;
use crate::types::descriptor::PlaneDescriptor;

const TAG_RAW: u8 = 0;
const TAG_RANS: u8 = 1;

pub struct Rans;

impl Rans {
    pub fn canonical_id(&self) -> crate::extension::CanonicalId {
        crate::extension::builtin_canonical_id("rans")
    }
}

impl PlaneCodec for Rans {
    fn id(&self) -> CodecId {
        CodecId::Rans
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
        shared_state: Option<&[u8]>,
        _layout: &PlaneLayout,
    ) -> Result<Encoded, PtwmCoreError> {
        if shared_state.is_some() {
            return Err(PtwmCoreError::CodecEncode {
                codec: "rans",
                msg: "shared state not supported".into(),
            });
        }
        if plane.is_empty() {
            return Ok(Encoded {
                state_bytes: Vec::new(),
                state_format_version: 0,
                payload: vec![TAG_RAW],
            });
        }

        // rans_encode already emits a tagged blob; peek the tag to
        // decide whether to wrap as TAG_RANS (with length prefix) or
        // TAG_RAW.
        let tagged = rans_encode(plane)?;
        // crate::codec_tagged's format: first byte 0 = raw, 1 = encoded.
        // Re-frame here to include the orig_len prefix the plane-codec
        // decode entry point needs, since PlaneCodec.decode has no
        // expected_len.
        let inner_tag = *tagged.first().unwrap_or(&0u8);
        if inner_tag == 0 {
            // raw fallback: drop the inner tagged framing and re-wrap.
            let mut payload = Vec::with_capacity(1 + plane.len());
            payload.push(TAG_RAW);
            payload.extend_from_slice(plane);
            return Ok(Encoded {
                state_bytes: Vec::new(),
                state_format_version: 0,
                payload,
            });
        }
        // tagged[1..] is the rANS-encoded bytes.
        let encoded = &tagged[1..];
        let orig_len = plane.len() as u32;
        let mut payload = Vec::with_capacity(1 + 4 + encoded.len());
        payload.push(TAG_RANS);
        payload.extend_from_slice(&orig_len.to_le_bytes());
        payload.extend_from_slice(encoded);
        Ok(Encoded {
            state_bytes: Vec::new(),
            state_format_version: 0,
            payload,
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
        let Some((&tag, rest)) = payload.split_first() else {
            return Err(PtwmCoreError::CodecDecode {
                codec: "rans",
                msg: "empty payload".into(),
            });
        };
        match tag {
            TAG_RAW => Ok(rest.to_vec()),
            TAG_RANS => {
                if rest.len() < 4 {
                    return Err(PtwmCoreError::CodecDecode {
                        codec: "rans",
                        msg: "payload too short for length prefix".into(),
                    });
                }
                let orig_len = u32::from_le_bytes([rest[0], rest[1], rest[2], rest[3]]) as usize;
                let encoded = &rest[4..];
                // Absolute cap to defend against adversarial headers that
                // would drive a multi-GiB allocation. Bounded by size, not
                // compression ratio: rANS reaches arbitrarily high ratios
                // on near-degenerate distributions (an all-zero 64 KiB
                // byte plane encodes to ~30 bytes), so a ratio cap rejects
                // legitimate inputs.
                const MAX_ORIG_LEN: usize = 1 << 30; // 1 GiB
                if orig_len > MAX_ORIG_LEN {
                    return Err(PtwmCoreError::CodecDecode {
                        codec: "rans",
                        msg: format!(
                            "implausible orig_len {orig_len} for {} encoded bytes",
                            encoded.len()
                        ),
                    });
                }
                // Re-emit the inner tagged format so rans_decode accepts it.
                let mut tagged = Vec::with_capacity(1 + encoded.len());
                tagged.push(1u8); // TAG_ENCODED in codec_tagged framing
                tagged.extend_from_slice(encoded);
                rans_decode(&tagged, orig_len)
            }
            other => Err(PtwmCoreError::CodecDecode {
                codec: "rans",
                msg: format!("bad rans tag: {other:#04x}"),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::PlaneLayout;
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
    fn rans_accepts_any_descriptor() {
        let c = Rans;
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
            assert!(c.accepts(d), "Rans must accept {:?}", d.role);
        }
    }

    #[test]
    fn rans_priority_is_one() {
        let c = Rans;
        let d = make_descriptor(
            Role::Scale {
                format: ScaleFormat::E4M3,
            },
            ElementWidth::Byte,
            Layout::Flat,
        );
        assert_eq!(c.priority_for(&d), 1);
    }

    #[test]
    fn rans_roundtrip_compressible() {
        let c = Rans;
        let data: Vec<u8> = (0..2048).map(|i| ((i * 13) % 17) as u8).collect();
        let enc = c.encode(&data, None, &PlaneLayout::Flat).unwrap();
        let dec = c
            .decode(
                0,
                &enc.state_bytes,
                &enc.payload,
                &PlaneLayout::Flat,
                data.len(),
            )
            .unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn rans_roundtrip_random() {
        let c = Rans;
        let data: Vec<u8> = (0..1024).map(|i| ((i * 97) & 0xFF) as u8).collect();
        let enc = c.encode(&data, None, &PlaneLayout::Flat).unwrap();
        let dec = c
            .decode(
                0,
                &enc.state_bytes,
                &enc.payload,
                &PlaneLayout::Flat,
                data.len(),
            )
            .unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn rans_empty_roundtrip() {
        let c = Rans;
        let enc = c.encode(&[], None, &PlaneLayout::Flat).unwrap();
        let dec = c
            .decode(0, &enc.state_bytes, &enc.payload, &PlaneLayout::Flat, 0)
            .unwrap();
        assert_eq!(dec, Vec::<u8>::new());
    }

    #[test]
    fn rans_shared_state_rejected() {
        let c = Rans;
        let result = c.encode(&[0u8; 32], Some(&[1, 2, 3]), &PlaneLayout::Flat);
        assert!(result.is_err());
    }
}
