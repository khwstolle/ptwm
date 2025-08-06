use crate::codec::{CodecId, Encoded, PlaneCodec};
use crate::entropy::huffman;
use crate::entropy::outcome::CompressOutcome;
use crate::error::PtwmCoreError;
use crate::layout::PlaneLayout;
use crate::types::descriptor::PlaneDescriptor;

pub struct Huffman;

impl Huffman {
    pub fn canonical_id(&self) -> crate::extension::CanonicalId {
        crate::extension::builtin_canonical_id("huffman")
    }
}

const TAG_RAW: u8 = 0;
const TAG_HUF: u8 = 1;

/// Upper bound for Huffman output given the input length.
fn huffman_output_bound(input_len: usize) -> usize {
    input_len.saturating_mul(2).saturating_add(4096)
}

impl PlaneCodec for Huffman {
    fn id(&self) -> CodecId {
        CodecId::Huffman
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
                codec: "huffman",
                msg: "shared state not yet supported".into(),
            });
        }

        if plane.is_empty() {
            // Empty plane: emit a raw tag with no data.
            return Ok(Encoded {
                state_bytes: Vec::new(),
                state_format_version: 0,
                payload: vec![TAG_RAW],
            });
        }

        let mut buf = vec![0u8; huffman_output_bound(plane.len())];
        match huffman::compress_outcome(&mut buf, plane)? {
            CompressOutcome::Encoded(n) => {
                // TAG_HUF | orig_len (4 bytes LE) | huffman-encoded bytes
                let orig_len = plane.len() as u32;
                let mut payload = Vec::with_capacity(1 + 4 + n);
                payload.push(TAG_HUF);
                payload.extend_from_slice(&orig_len.to_le_bytes());
                payload.extend_from_slice(&buf[..n]);
                Ok(Encoded {
                    state_bytes: Vec::new(),
                    state_format_version: 0,
                    payload,
                })
            }
            _outcome => {
                // NotBeneficial / Incompressible / DstTooSmall /
                // LenOverflow → store raw
                let mut payload = Vec::with_capacity(1 + plane.len());
                payload.push(TAG_RAW);
                payload.extend_from_slice(plane);
                Ok(Encoded {
                    state_bytes: Vec::new(),
                    state_format_version: 0,
                    payload,
                })
            }
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
        let Some((&tag, rest)) = payload.split_first() else {
            return Err(PtwmCoreError::CodecDecode {
                codec: "huffman",
                msg: "empty payload".into(),
            });
        };

        match tag {
            TAG_RAW => Ok(rest.to_vec()),
            TAG_HUF => {
                if rest.len() < 4 {
                    return Err(PtwmCoreError::CodecDecode {
                        codec: "huffman",
                        msg: "payload too short for length prefix".into(),
                    });
                }
                let orig_len = u32::from_le_bytes([rest[0], rest[1], rest[2], rest[3]]) as usize;
                let encoded = &rest[4..];
                // Sanity-cap orig_len against encoded.len() so a
                // corrupted header cannot trigger a multi-GiB
                // allocation. Real-world Huffman compression on weight
                // bytes never approaches a 32x ratio (best case ~1
                // bit/byte = 8x), so anything beyond 32 * encoded.len()
                // + 4096 is rejected before alloc.
                let max_allowed = encoded.len().saturating_mul(32).saturating_add(4096);
                if orig_len > max_allowed {
                    return Err(PtwmCoreError::CodecDecode {
                        codec: "huffman",
                        msg: format!(
                            "implausible orig_len {orig_len} for {} encoded bytes",
                            encoded.len()
                        ),
                    });
                }
                let mut out = vec![0u8; orig_len];
                huffman::decompress(&mut out, encoded)?;
                Ok(out)
            }
            other => Err(PtwmCoreError::CodecDecode {
                codec: "huffman",
                msg: format!("bad huffman tag: {other:#04x}"),
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
    fn huffman_accepts_any_descriptor() {
        let c = Huffman;
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
            assert!(c.accepts(d), "Huffman must accept {:?}", d.role);
        }
    }

    #[test]
    fn huffman_priority_is_one() {
        let c = Huffman;
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
    fn huffman_roundtrip() {
        let c = Huffman;
        let data: Vec<u8> = (0..1024u16).map(|i| (i % 17) as u8).collect();
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
    fn huffman_shared_state_rejected() {
        let c = Huffman;
        let data = vec![0u8; 64];
        let result = c.encode(&data, Some(&[1, 2, 3]), &PlaneLayout::Flat);
        assert!(result.is_err());
    }
}
