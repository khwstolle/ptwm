//! `huff_llm_5bit` — field-separated Huffman codec for 16-bit float planes.
//!
//! See `entropy::huff_llm` for the bit-field decomposition. The codec gates on
//! `Role::Raw` terminals (see `accepts` for why role, not `element_width`),
//! interprets the plane as little-endian `u16` words, trials both the FP16
//! {1,5,5,5} and BF16 {1,4,4,7} layouts, and keeps the smaller (the chosen
//! layout tag travels in `state_bytes`). Odd-length or incompressible planes
//! fall back to a raw store, so the codec is always total. It is
//! self-delimiting (the word count is framed in the payload) and ignores the
//! container's `decoded_len`.

use crate::codec::{CodecId, Encoded, PlaneCodec};
use crate::entropy::huff_llm;
use crate::error::PtwmCoreError;
use crate::layout::PlaneLayout;
use crate::types::descriptor::PlaneDescriptor;
use crate::types::role::Role;

pub struct HuffLlm5Bit;

impl HuffLlm5Bit {
    pub fn canonical_id(&self) -> crate::extension::CanonicalId {
        crate::extension::builtin_canonical_id("huff_llm_5bit")
    }
}

const TAG_RAW: u8 = 0;
const TAG_HUFFLLM: u8 = 1;

/// Planes below this size rarely amortize three table headers.
const MIN_WORDS: usize = 128;

impl PlaneCodec for HuffLlm5Bit {
    fn id(&self) -> CodecId {
        CodecId::HuffLlm5Bit
    }

    fn accepts(&self, descriptor: &PlaneDescriptor) -> bool {
        // Gate on the *role*, not `element_width`. `source_descriptor_for`
        // derives `element_width` from a chain-internal dtype namespace, but
        // production passes *external* dtype codes, so a raw fp16/bf16 source
        // plane reaches a codec mislabeled (bf16 -> Byte, fp16 -> Word8 via a
        // namespace collision). `Role::Raw` is stamped reliably by
        // `source_descriptor_for` and preserved through the terminal re-tag,
        // and the codec reads raw 16-bit LE words directly (it never relied on
        // `element_width` for correctness). Only terminal planes are
        // trial-encoded. The huff_llm raw chain is the only one terminating on
        // the *un-transformed* source plane, but the spherical_normalize and
        // alpha_stable_normalize chains also expose `Role::Raw` terminals
        // (radius / direction / residual), so the codec competes there too —
        // harmless, since it is lossless on any byte stream and loses on size
        // when field-separated coding does not help.
        matches!(descriptor.role, Role::Raw)
    }

    fn priority_for(&self, _descriptor: &PlaneDescriptor) -> i8 {
        1
    }

    fn should_attempt(
        &self,
        plane: &[u8],
        _descriptor: &PlaneDescriptor,
        _layout: &PlaneLayout,
    ) -> bool {
        // Skip tiny planes (three table headers don't amortize) and odd lengths.
        plane.len() >= MIN_WORDS * 2 && plane.len().is_multiple_of(2)
    }

    fn encode(
        &self,
        plane: &[u8],
        shared_state: Option<&[u8]>,
        _layout: &PlaneLayout,
    ) -> Result<Encoded, PtwmCoreError> {
        if shared_state.is_some() {
            return Err(PtwmCoreError::CodecEncode {
                codec: "huff_llm_5bit",
                msg: "shared state not supported".into(),
            });
        }
        // Empty or odd-length → raw store (still total). Short-circuit empty
        // before allocating `words`, since trial-encode hits this path on every
        // candidate.
        if plane.is_empty() || !plane.len().is_multiple_of(2) {
            return Ok(raw_store(plane));
        }
        let words: Vec<u16> = plane
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        // `encode_plane` errors if the word/stream count overflows the u32 wire
        // frame (unreachable via the u32-bounded chunk path, but the entropy
        // module is reusable). Propagate rather than truncate silently.
        let enc = huff_llm::encode_plane(&words)?;
        // Compare against an Identity store, which emits the raw bytes with no
        // tag (size == plane.len()). The TAG_HUFFLLM record carries a 1-byte
        // tag, so its true size is `huff_total + 1`; emit it only when that is
        // *strictly* smaller than Identity. Otherwise fall back to raw_store so
        // a size tie defers to the simpler Identity codec (HuffLlm sorts ahead
        // of Identity in the priority-ordered menu and would otherwise win the
        // tie, contradicting smallest-wins).
        let huff_total = enc.payload.len() + enc.state_bytes.len();
        if huff_total + 1 >= plane.len() {
            return Ok(raw_store(plane));
        }
        let mut payload = Vec::with_capacity(1 + enc.payload.len());
        payload.push(TAG_HUFFLLM);
        payload.extend_from_slice(&enc.payload);
        Ok(Encoded {
            state_bytes: enc.state_bytes,
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
        let Some((&tag, rest)) = payload.split_first() else {
            return Err(PtwmCoreError::CodecDecode {
                codec: "huff_llm_5bit",
                msg: "empty payload".into(),
            });
        };
        match tag {
            TAG_RAW => Ok(rest.to_vec()),
            TAG_HUFFLLM => {
                // Reserve the state-format version for forward compatibility:
                // a v1 decoder must reject any future v2 state layout rather
                // than misparse it. `encode` always writes version 1 here.
                if state_format_version != 1 {
                    return Err(PtwmCoreError::CodecDecode {
                        codec: "huff_llm_5bit",
                        msg: format!("unknown state format version {state_format_version}"),
                    });
                }
                let words = huff_llm::decode_plane(state_bytes, rest)?;
                let cap = words
                    .len()
                    .checked_mul(2)
                    .ok_or_else(|| PtwmCoreError::CodecDecode {
                        codec: "huff_llm_5bit",
                        msg: "decoded length overflows usize".into(),
                    })?;
                let mut out = Vec::with_capacity(cap);
                for w in words {
                    out.extend_from_slice(&w.to_le_bytes());
                }
                Ok(out)
            }
            other => Err(PtwmCoreError::CodecDecode {
                codec: "huff_llm_5bit",
                msg: format!("bad tag {other:#04x}"),
            }),
        }
    }
}

fn raw_store(plane: &[u8]) -> Encoded {
    let mut payload = Vec::with_capacity(1 + plane.len());
    payload.push(TAG_RAW);
    payload.extend_from_slice(plane);
    Encoded {
        state_bytes: Vec::new(),
        state_format_version: 0,
        payload,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::descriptor::{ElementWidth, Layout, PlaneDescriptor};
    use crate::types::role::Role;

    fn desc(width: ElementWidth, len: u64) -> PlaneDescriptor {
        desc_role(Role::Raw, width, len)
    }

    fn desc_role(role: Role, width: ElementWidth, len: u64) -> PlaneDescriptor {
        PlaneDescriptor {
            role,
            element_width: width,
            length_bytes: len,
            layout: Layout::Flat,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        }
    }

    fn rt(bytes: &[u8]) {
        let c = HuffLlm5Bit;
        let enc = c.encode(bytes, None, &PlaneLayout::Flat).unwrap();
        let dec = c
            .decode(
                enc.state_format_version,
                &enc.state_bytes,
                &enc.payload,
                &PlaneLayout::Flat,
                bytes.len(),
            )
            .unwrap();
        assert_eq!(dec, bytes);
    }

    #[test]
    fn accepts_raw_role_any_width() {
        let c = HuffLlm5Bit;
        // The raw source plane reaches the codec with an unreliable
        // element_width (bf16 -> Byte, fp16 -> Word8), so acceptance keys on
        // Role::Raw and must hold for every width.
        for w in [
            ElementWidth::Nibble,
            ElementWidth::Byte,
            ElementWidth::Word2,
            ElementWidth::Word4,
            ElementWidth::Word8,
        ] {
            assert!(
                c.accepts(&desc(w, 4096)),
                "Raw role must be accepted at {w:?}"
            );
        }
        // Non-Raw roles (byte-split terminals, scales, …) are rejected.
        for role in [
            Role::ExponentByte,
            Role::MantissaByte { index: 0, of: 2 },
            Role::Index,
        ] {
            assert!(
                !c.accepts(&desc_role(role.clone(), ElementWidth::Word2, 4096)),
                "non-Raw role {role:?} must be rejected"
            );
        }
    }

    #[test]
    fn priority_is_one() {
        assert_eq!(
            HuffLlm5Bit.priority_for(&desc(ElementWidth::Word2, 4096)),
            1
        );
    }

    #[test]
    fn should_attempt_gates_small_and_odd() {
        let c = HuffLlm5Bit;
        let d = desc(ElementWidth::Word2, 0);
        // Large even plane → attempt.
        assert!(c.should_attempt(&[0u8; 4096], &d, &PlaneLayout::Flat));
        // Tiny plane → skip.
        assert!(!c.should_attempt(&[0u8; 16], &d, &PlaneLayout::Flat));
        // Odd-length plane → skip.
        assert!(!c.should_attempt(&[0u8; 4097], &d, &PlaneLayout::Flat));
    }

    #[test]
    fn roundtrip_compressible() {
        let words: Vec<u16> = (0..4096u32).map(|i| 0x3F00 | (i % 8) as u16).collect();
        let mut bytes = Vec::new();
        for w in words {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        rt(&bytes);
    }

    #[test]
    fn roundtrip_odd_length_uses_raw() {
        let bytes: Vec<u8> = (0..1001u32).map(|i| (i % 251) as u8).collect();
        let c = HuffLlm5Bit;
        let enc = c.encode(&bytes, None, &PlaneLayout::Flat).unwrap();
        assert_eq!(enc.payload[0], TAG_RAW);
        rt(&bytes);
    }

    #[test]
    fn roundtrip_incompressible_uses_raw() {
        // High-entropy 16-bit words → Huffman should not beat raw.
        let bytes: Vec<u8> = (0..8192u32)
            .map(|i| ((i.wrapping_mul(2654435761)) >> 16) as u8)
            .collect();
        let c = HuffLlm5Bit;
        let enc = c.encode(&bytes, None, &PlaneLayout::Flat).unwrap();
        assert_eq!(enc.payload[0], TAG_RAW);
        rt(&bytes);
    }

    #[test]
    fn roundtrip_empty() {
        rt(&[]);
    }

    #[test]
    fn shared_state_rejected() {
        let c = HuffLlm5Bit;
        assert!(
            c.encode(&[0u8; 64], Some(&[1, 2]), &PlaneLayout::Flat)
                .is_err()
        );
    }
}
