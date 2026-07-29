//! Arithmetic-coding plane codecs: three peer models over the shared
//! range coder. Each frames the true plane length into its payload (like
//! `rans.rs`), so `decode` is self-delimiting and ignores the container's
//! `decoded_len` (which is source-derived and unreliable for byte-split
//! planes).

use crate::codec::{CodecId, Encoded, PlaneCodec};
use crate::entropy::arithmetic::{
    Order0Adaptive, Order0Static, Order1Adaptive, decode_bytes, encode_bytes,
};
use crate::error::PtwmCoreError;
use crate::layout::PlaneLayout;
use crate::types::descriptor::PlaneDescriptor;

/// Absolute cap on the framed decoded length, mirroring `rans.rs`. Guards a
/// hostile length prefix from driving a multi-GiB decode loop.
const MAX_DECODED_LEN: usize = 1 << 30; // 1 GiB

/// Prepend the true decoded length (u32 LE) so decode is self-delimiting.
///
/// The container's `decode` receives a `decoded_len` derived from the
/// *source-tensor* descriptor, which for byte-split planes is larger than the
/// actual plane length. Self-delimiting codecs (Huffman, rANS, …) ignore that
/// argument and recover the length from their own payload; the arithmetic
/// coders do the same here rather than trusting `decoded_len`.
pub(crate) fn frame_payload(decoded_len: usize, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + payload.len());
    out.extend_from_slice(&(decoded_len as u32).to_le_bytes());
    out.extend_from_slice(payload);
    out
}

/// Inverse of [`frame_payload`]: read the length prefix and return the rest.
pub(crate) fn unframe_payload<'a>(
    payload: &'a [u8],
    codec: &'static str,
) -> Result<(usize, &'a [u8]), PtwmCoreError> {
    if payload.len() < 4 {
        return Err(PtwmCoreError::CodecDecode {
            codec,
            msg: "payload too short for length prefix".into(),
        });
    }
    let n = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
    if n > MAX_DECODED_LEN {
        return Err(PtwmCoreError::CodecDecode {
            codec,
            msg: format!("implausible decoded length {n}"),
        });
    }
    Ok((n, &payload[4..]))
}

/// Order-0 arithmetic coder with a serialized normalized table.
pub struct ArithmeticO0;

impl ArithmeticO0 {
    pub fn canonical_id(&self) -> crate::extension::CanonicalId {
        crate::extension::builtin_canonical_id("arithmetic_o0")
    }
}

impl PlaneCodec for ArithmeticO0 {
    fn id(&self) -> CodecId {
        CodecId::ArithmeticO0
    }

    fn encode(
        &self,
        plane: &[u8],
        shared_state: Option<&[u8]>,
        _layout: &PlaneLayout,
    ) -> Result<Encoded, PtwmCoreError> {
        if shared_state.is_some() {
            return Err(PtwmCoreError::CodecEncode {
                codec: "arithmetic_o0",
                msg: "shared state not supported".into(),
            });
        }
        if plane.is_empty() {
            return Ok(Encoded {
                state_bytes: Vec::new(),
                state_format_version: 1,
                payload: Vec::new(),
            });
        }
        let mut model = Order0Static::fit(plane);
        let payload = frame_payload(plane.len(), &encode_bytes(&mut model, plane));
        Ok(Encoded {
            state_bytes: model.serialize(),
            state_format_version: 1,
            payload,
        })
    }

    fn decode(
        &self,
        state_format_version: u8,
        state_bytes: &[u8],
        payload: &[u8],
        _layout: &PlaneLayout,
        _decoded_len: usize,
    ) -> Result<Vec<u8>, PtwmCoreError> {
        if state_format_version != 1 {
            return Err(PtwmCoreError::CodecDecode {
                codec: "arithmetic_o0",
                msg: format!(
                    "unsupported state_format_version {state_format_version} (expected 1)"
                ),
            });
        }
        // Self-delimiting: the true length is framed into the payload; the
        // container's `decoded_len` is source-derived and not reliable here.
        if payload.is_empty() {
            return Ok(Vec::new());
        }
        let (n, body) = unframe_payload(payload, "arithmetic_o0")?;
        let mut model = Order0Static::deserialize(state_bytes)?;
        Ok(decode_bytes(&mut model, body, n))
    }
}

/// Order-0 adaptive arithmetic coder. No inline state.
pub struct ArithmeticO0Adaptive;

impl ArithmeticO0Adaptive {
    pub fn canonical_id(&self) -> crate::extension::CanonicalId {
        crate::extension::builtin_canonical_id("arithmetic_o0_adaptive")
    }
}

impl PlaneCodec for ArithmeticO0Adaptive {
    fn id(&self) -> CodecId {
        CodecId::ArithmeticO0Adaptive
    }

    fn encode(
        &self,
        plane: &[u8],
        shared_state: Option<&[u8]>,
        _layout: &PlaneLayout,
    ) -> Result<Encoded, PtwmCoreError> {
        if shared_state.is_some() {
            return Err(PtwmCoreError::CodecEncode {
                codec: "arithmetic_o0_adaptive",
                msg: "shared state not supported".into(),
            });
        }
        if plane.is_empty() {
            return Ok(Encoded {
                state_bytes: Vec::new(),
                state_format_version: 0,
                payload: Vec::new(),
            });
        }
        let payload = frame_payload(
            plane.len(),
            &encode_bytes(&mut Order0Adaptive::new(), plane),
        );
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
        if payload.is_empty() {
            return Ok(Vec::new());
        }
        let (n, body) = unframe_payload(payload, "arithmetic_o0_adaptive")?;
        Ok(decode_bytes(&mut Order0Adaptive::new(), body, n))
    }
}

/// Order-1 adaptive arithmetic coder (prev-byte context). No inline state.
pub struct ArithmeticO1;

/// Below this plane size the order-1 model's 256 contexts cannot warm up
/// enough to beat order-0; skip the expensive attempt.
const O1_MIN_PLANE_BYTES: usize = 4096;

impl ArithmeticO1 {
    pub fn canonical_id(&self) -> crate::extension::CanonicalId {
        crate::extension::builtin_canonical_id("arithmetic_o1")
    }
}

impl PlaneCodec for ArithmeticO1 {
    fn id(&self) -> CodecId {
        CodecId::ArithmeticO1
    }

    fn should_attempt(
        &self,
        plane: &[u8],
        _descriptor: &PlaneDescriptor,
        _layout: &PlaneLayout,
    ) -> bool {
        plane.len() >= O1_MIN_PLANE_BYTES
    }

    fn encode(
        &self,
        plane: &[u8],
        shared_state: Option<&[u8]>,
        _layout: &PlaneLayout,
    ) -> Result<Encoded, PtwmCoreError> {
        if shared_state.is_some() {
            return Err(PtwmCoreError::CodecEncode {
                codec: "arithmetic_o1",
                msg: "shared state not supported".into(),
            });
        }
        if plane.is_empty() {
            return Ok(Encoded {
                state_bytes: Vec::new(),
                state_format_version: 0,
                payload: Vec::new(),
            });
        }
        let payload = frame_payload(
            plane.len(),
            &encode_bytes(&mut Order1Adaptive::new(), plane),
        );
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
        if payload.is_empty() {
            return Ok(Vec::new());
        }
        let (n, body) = unframe_payload(payload, "arithmetic_o1")?;
        Ok(decode_bytes(&mut Order1Adaptive::new(), body, n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::descriptor::{ElementWidth, Layout, PlaneDescriptor};
    use crate::types::role::Role;

    fn desc() -> PlaneDescriptor {
        PlaneDescriptor {
            role: Role::ExponentByte,
            element_width: ElementWidth::Byte,
            length_bytes: 1024,
            layout: Layout::Flat,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        }
    }

    fn rt(codec: &dyn PlaneCodec, data: &[u8]) {
        let enc = codec.encode(data, None, &PlaneLayout::Flat).unwrap();
        let dec = codec
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
    fn o0_roundtrip_compressible() {
        let data: Vec<u8> = (0..4096).map(|i| ((i * 13) % 11) as u8).collect();
        rt(&ArithmeticO0, &data);
    }

    #[test]
    fn decode_ignores_wrong_decoded_len() {
        // Regression: the container passes a source-derived decoded_len that,
        // for byte-split planes, is larger than the actual plane (e.g. 4× for
        // a float32 plane split into 4 byte planes). Decode is self-delimiting
        // and must reconstruct the true plane regardless.
        let data: Vec<u8> = (0..2048).map(|i| (i % 7) as u8).collect();
        for codec in [
            &ArithmeticO0 as &dyn PlaneCodec,
            &ArithmeticO0Adaptive,
            &ArithmeticO1,
        ] {
            let enc = codec.encode(&data, None, &PlaneLayout::Flat).unwrap();
            let dec = codec
                .decode(
                    enc.state_format_version,
                    &enc.state_bytes,
                    &enc.payload,
                    &PlaneLayout::Flat,
                    data.len() * 4, // deliberately wrong, as the container does
                )
                .unwrap();
            assert_eq!(
                dec,
                data,
                "codec {:?} must ignore wrong decoded_len",
                codec.id()
            );
        }
    }

    #[test]
    fn o0_roundtrip_random() {
        let data: Vec<u8> = (0..2048).map(|i| ((i * 97) & 0xFF) as u8).collect();
        rt(&ArithmeticO0, &data);
    }

    #[test]
    fn o0_roundtrip_single_symbol() {
        rt(&ArithmeticO0, &vec![0x7Eu8; 777]);
    }

    #[test]
    fn o0_empty() {
        rt(&ArithmeticO0, &[]);
    }

    #[test]
    fn o0_id_and_accepts() {
        let c = ArithmeticO0;
        assert_eq!(c.id(), CodecId::ArithmeticO0);
        assert!(c.accepts(&desc()));
        assert_eq!(c.priority_for(&desc()), 1);
    }

    #[test]
    fn o0_rejects_shared_state() {
        assert!(
            ArithmeticO0
                .encode(&[0u8; 16], Some(&[1, 2]), &PlaneLayout::Flat)
                .is_err()
        );
    }

    #[test]
    fn o0a_roundtrip_compressible() {
        let data: Vec<u8> = (0..8192).map(|i| ((i * 7) % 5) as u8).collect();
        rt(&ArithmeticO0Adaptive, &data);
    }

    #[test]
    fn o0a_roundtrip_random() {
        let data: Vec<u8> = (0..2048).map(|i| ((i * 211) & 0xFF) as u8).collect();
        rt(&ArithmeticO0Adaptive, &data);
    }

    #[test]
    fn o0a_empty() {
        rt(&ArithmeticO0Adaptive, &[]);
    }

    #[test]
    fn o0a_no_state() {
        let enc = ArithmeticO0Adaptive
            .encode(&[1u8; 64], None, &PlaneLayout::Flat)
            .unwrap();
        assert!(
            enc.state_bytes.is_empty(),
            "adaptive codec must emit no inline state"
        );
    }

    #[test]
    fn o1_roundtrip_markov() {
        let mut data = vec![0u8];
        for i in 1..8192 {
            data.push(data[i - 1].wrapping_add((i % 3) as u8));
        }
        rt(&ArithmeticO1, &data);
    }

    #[test]
    fn o1_roundtrip_random() {
        let data: Vec<u8> = (0..2048).map(|i| ((i * 151) & 0xFF) as u8).collect();
        rt(&ArithmeticO1, &data);
    }

    #[test]
    fn o1_empty() {
        rt(&ArithmeticO1, &[]);
    }

    #[test]
    fn o1_should_attempt_skips_tiny_planes() {
        // 64-byte plane: per-context tables can't amortize; skip.
        assert!(!ArithmeticO1.should_attempt(&[0u8; 64], &desc(), &PlaneLayout::Flat));
        // Large plane: attempt.
        assert!(ArithmeticO1.should_attempt(&[0u8; 65536], &desc(), &PlaneLayout::Flat));
    }

    #[test]
    fn o0_static_beats_or_matches_huffman_on_structured_plane() {
        use crate::codecs::huffman::Huffman;
        // Skewed exponent-like plane: a few dominant byte values.
        let data: Vec<u8> = (0..262144)
            .map(|i| match i % 16 {
                0..=9 => 0x40u8,
                10..=12 => 0x41,
                13..=14 => 0x3F,
                _ => 0x42,
            })
            .collect();
        let ac = ArithmeticO0
            .encode(&data, None, &PlaneLayout::Flat)
            .unwrap();
        let huff = Huffman.encode(&data, None, &PlaneLayout::Flat).unwrap();
        let ac_total = ac.state_bytes.len() + ac.payload.len();
        let huff_total = huff.state_bytes.len() + huff.payload.len();
        assert!(
            ac_total <= huff_total,
            "arithmetic O0 ({ac_total}) must not exceed Huffman ({huff_total}) on a structured plane",
        );
    }
}
