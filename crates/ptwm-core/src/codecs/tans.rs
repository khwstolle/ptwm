//! tANS (tabled-ANS / Finite State Entropy) plane codec.
//!
//! A peer of [`crate::codecs::rans::Rans`] in the dispatcher's per-plane
//! trial encode. Wraps [`crate::entropy::tans`] with the same
//! `TAG_RAW` / length-prefix framing the rANS codec uses, so a plane that
//! tANS can't shrink is stored verbatim and decode stays self-describing.

use crate::codec::{CodecId, Encoded, PlaneCodec};
use crate::entropy::tans;
use crate::error::PtwmCoreError;
use crate::layout::PlaneLayout;
use crate::types::descriptor::PlaneDescriptor;

const TAG_RAW: u8 = 0;
const TAG_TANS: u8 = 1;

pub struct Tans;

impl Tans {
    pub fn canonical_id(&self) -> crate::extension::CanonicalId {
        crate::extension::builtin_canonical_id("tans")
    }
}

impl PlaneCodec for Tans {
    fn id(&self) -> CodecId {
        CodecId::Tans
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
                codec: "tans",
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

        // `tans::compress` returns 0 when the encoded form would not be
        // smaller (incompressible, degenerate alphabet, or expansion); in
        // that case store the plane verbatim under TAG_RAW.
        let mut scratch = vec![0u8; plane.len()];
        let encoded_len = tans::compress(&mut scratch, plane)?;
        if encoded_len == 0 {
            let mut payload = Vec::with_capacity(1 + plane.len());
            payload.push(TAG_RAW);
            payload.extend_from_slice(plane);
            return Ok(Encoded {
                state_bytes: Vec::new(),
                state_format_version: 0,
                payload,
            });
        }

        let orig_len = u32::try_from(plane.len()).map_err(|_| PtwmCoreError::CodecEncode {
            codec: "tans",
            msg: "plane length exceeds u32::MAX".into(),
        })?;
        let mut payload = Vec::with_capacity(1 + 4 + encoded_len);
        payload.push(TAG_TANS);
        payload.extend_from_slice(&orig_len.to_le_bytes());
        payload.extend_from_slice(&scratch[..encoded_len]);
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
                codec: "tans",
                msg: "empty payload".into(),
            });
        };
        match tag {
            TAG_RAW => Ok(rest.to_vec()),
            TAG_TANS => {
                if rest.len() < 4 {
                    return Err(PtwmCoreError::CodecDecode {
                        codec: "tans",
                        msg: "payload too short for length prefix".into(),
                    });
                }
                let orig_len = u32::from_le_bytes([rest[0], rest[1], rest[2], rest[3]]) as usize;
                let encoded = &rest[4..];
                // Absolute size cap (not a ratio cap): tANS reaches very
                // high ratios on near-degenerate distributions, so bound the
                // allocation by an implausible-size guard rather than ratio.
                const MAX_ORIG_LEN: usize = 1 << 30; // 1 GiB
                if orig_len > MAX_ORIG_LEN {
                    return Err(PtwmCoreError::CodecDecode {
                        codec: "tans",
                        msg: format!(
                            "implausible orig_len {orig_len} for {} encoded bytes",
                            encoded.len()
                        ),
                    });
                }
                let mut out = vec![0u8; orig_len];
                tans::decompress(&mut out, encoded)?;
                Ok(out)
            }
            other => Err(PtwmCoreError::CodecDecode {
                codec: "tans",
                msg: format!("bad tans tag: {other:#04x}"),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::PlaneLayout;
    use crate::types::descriptor::{ElementWidth, Layout, PlaneDescriptor};
    use crate::types::role::Role;

    fn descriptor() -> PlaneDescriptor {
        PlaneDescriptor {
            role: Role::Raw,
            element_width: ElementWidth::Byte,
            length_bytes: 1024,
            layout: Layout::Flat,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        }
    }

    fn round_trip(data: &[u8]) {
        let c = Tans;
        let enc = c.encode(data, None, &PlaneLayout::Flat).unwrap();
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
    fn id_is_tans() {
        assert_eq!(Tans.id(), CodecId::Tans);
    }

    #[test]
    fn accepts_any_descriptor() {
        assert!(Tans.accepts(&descriptor()));
    }

    #[test]
    fn roundtrip_compressible() {
        let data: Vec<u8> = (0..4096).map(|i| ((i * 13) % 17) as u8).collect();
        round_trip(&data);
        // Compressible data must take the TAG_TANS path.
        let enc = Tans.encode(&data, None, &PlaneLayout::Flat).unwrap();
        assert_eq!(enc.payload[0], TAG_TANS);
        assert!(enc.payload.len() < data.len());
    }

    #[test]
    fn roundtrip_incompressible_uses_raw() {
        let mut state = 0xABCD_1234u32;
        let data: Vec<u8> = (0..2048)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 24) as u8
            })
            .collect();
        round_trip(&data);
    }

    #[test]
    fn roundtrip_single_symbol_uses_raw() {
        let data = vec![0x55u8; 512];
        let enc = Tans.encode(&data, None, &PlaneLayout::Flat).unwrap();
        assert_eq!(enc.payload[0], TAG_RAW);
        round_trip(&data);
    }

    #[test]
    fn empty_roundtrip() {
        round_trip(&[]);
    }

    #[test]
    fn shared_state_rejected() {
        let r = Tans.encode(&[0u8; 32], Some(&[1, 2, 3]), &PlaneLayout::Flat);
        assert!(r.is_err());
    }

    #[test]
    fn decode_rejects_bad_tag() {
        let r = Tans.decode(0, &[], &[0x42, 1, 2, 3], &PlaneLayout::Flat, 3);
        assert!(r.is_err());
    }

    #[test]
    fn decode_rejects_empty_payload() {
        let r = Tans.decode(0, &[], &[], &PlaneLayout::Flat, 0);
        assert!(r.is_err());
    }
}
