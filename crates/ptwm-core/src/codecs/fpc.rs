//! FPC-style leading-zero plane codec.
//!
//! Targets the residual streams produced by predictor-based transforms
//! (notably [`crate::transforms::PredictorXor`]) — those streams have a
//! heavy distribution of small-magnitude integers with many leading
//! zero bytes after the high-order byte. The codec encodes each value
//! with a 4-bit leading-zero-byte count (enough to count 0..=8 zero
//! bytes for the widest supported element) followed by the non-zero
//! suffix bytes, packing two values per header byte to keep the
//! framing tight.
//!
//! The codec is intentionally generic: it works on any byte stream
//! whose elements are width-aligned (1/2/4/8) and have a leading-zero-
//! heavy distribution. It does not require a paired predictor.
//!
//! ## Wire layout
//!
//! ```text
//! [u8 width]              element width in bytes (1, 2, 4, or 8)
//! [u32 LE n_elements]     element count
//! [packed headers]        ceil(n / 2) bytes, each carrying two
//!                         4-bit leading-zero-byte counts (high
//!                         nibble = first elem, low nibble = second
//!                         elem).
//! [suffix bytes]          n_elements groups, each (width - lzcount)
//!                         bytes; per FPC convention the leading-zero
//!                         bytes are *implied zero* and never stored.
//! ```
//!
//! Decoder reverses the layout: for each element read the 4-bit
//! header, then the (width - lzcount) suffix bytes, and zero-extend on
//! the high side.
//!
//! ## Single-predictor variant
//!
//! Burtscher's original FPC pairs two predictors and stores a 1-bit
//! per-element selector. PTWM ships only the single-predictor variant
//! (see [`crate::transforms::PredictorXor`]) and therefore omits the
//! selector bit from the header nibble.

use crate::codec::{CodecId, Encoded, PlaneCodec};
use crate::error::PtwmCoreError;
use crate::layout::PlaneLayout;
use crate::types::descriptor::{ElementWidth, PlaneDescriptor};

pub struct Fpc;

impl Fpc {
    pub fn canonical_id(&self) -> crate::extension::CanonicalId {
        crate::extension::builtin_canonical_id("fpc")
    }
}

impl PlaneCodec for Fpc {
    fn id(&self) -> CodecId {
        CodecId::Fpc
    }

    fn accepts(&self, descriptor: &PlaneDescriptor) -> bool {
        // Restricted to byte planes for now: the dispatcher's `encode`
        // path hard-codes width=1 because the descriptor's
        // element_width is not yet plumbed through to codec entry. The
        // wider widths (Word2/4/8) round-trip correctly when driven
        // through the internal `encode_with_width` helper (covered by
        // tests), but admitting them here would silently encode them
        // as independent bytes via the dispatcher, which contradicts
        // the descriptor's stated semantics. Lift this restriction
        // once the descriptor flows into `encode`.
        matches!(descriptor.element_width, ElementWidth::Byte)
    }

    fn priority_for(&self, _descriptor: &PlaneDescriptor) -> i8 {
        // Below the entropy coders' default 1 — the dispatcher should
        // trial this codec only when the predictor pair upstream tells
        // it the stream is leading-zero-heavy. A scorer extension
        // could promote it for those plane shapes.
        0
    }

    fn encode(
        &self,
        plane: &[u8],
        _shared_state: Option<&[u8]>,
        _layout: &PlaneLayout,
    ) -> Result<Encoded, PtwmCoreError> {
        // The codec runs on Byte planes when the caller has already
        // packed an element-wise residual stream into a flat byte
        // buffer. Element width is inferred from the buffer length
        // when descriptor info is absent: any aligned (1/2/4/8) length
        // works. The dispatcher passes the descriptor implicitly via
        // the chain; for the byte-stream encoded form we default to
        // width=1 (the most common case after byte_split + predictor).
        encode_with_width(plane, 1)
    }

    fn decode(
        &self,
        _state_format_version: u8,
        _state_bytes: &[u8],
        payload: &[u8],
        _layout: &PlaneLayout,
        _decoded_len: usize,
    ) -> Result<Vec<u8>, PtwmCoreError> {
        decode_any_width(payload)
    }
}

fn encode_with_width(plane: &[u8], width: usize) -> Result<Encoded, PtwmCoreError> {
    if width == 0 || width > 8 || width.count_ones() != 1 {
        return Err(PtwmCoreError::CodecEncode {
            codec: "fpc",
            msg: format!("unsupported element width {width}"),
        });
    }
    if !plane.len().is_multiple_of(width) {
        return Err(PtwmCoreError::CodecEncode {
            codec: "fpc",
            msg: format!(
                "plane length {} is not a multiple of element width {width}",
                plane.len()
            ),
        });
    }
    let n_elements = plane.len() / width;
    // The element count is serialized as a u32, so the codec is bounded
    // to 2^32 - 1 elements. A larger plane would otherwise be truncated
    // silently by an `as u32` cast and produce a payload that decodes
    // to the wrong length.
    if n_elements > u32::MAX as usize {
        return Err(PtwmCoreError::CodecEncode {
            codec: "fpc",
            msg: format!("element count {n_elements} exceeds u32 wire-format limit"),
        });
    }
    // Capacity: 1 (width) + 4 (n_elements) + ceil(n / 2) (headers) +
    // plane.len() (worst-case suffixes, when every element has zero
    // leading-zero bytes). Use checked arithmetic so a pathological
    // input on a 32-bit target cannot silently roll over.
    let capacity = 1usize
        .checked_add(4)
        .and_then(|c| c.checked_add(n_elements.div_ceil(2)))
        .and_then(|c| c.checked_add(plane.len()))
        .ok_or_else(|| PtwmCoreError::CodecEncode {
            codec: "fpc",
            msg: "output capacity overflows usize".into(),
        })?;
    let mut out = Vec::with_capacity(capacity);
    out.push(width as u8);
    out.extend_from_slice(&(n_elements as u32).to_le_bytes());

    // First pass: compute per-element leading-zero counts (number of
    // *high-order* bytes that are zero in the little-endian
    // representation). For width=W the count is in 0..=W. We store
    // it as a 4-bit field in the header.
    let mut lz_counts: Vec<u8> = Vec::with_capacity(n_elements);
    for chunk in plane.chunks_exact(width) {
        // Bytes are little-endian; the high-order bytes are at the end.
        let mut lz = 0u8;
        for &b in chunk.iter().rev() {
            if b == 0 {
                lz += 1;
            } else {
                break;
            }
        }
        lz_counts.push(lz);
    }

    // Pack two 4-bit headers per byte.
    let header_byte_count = n_elements.div_ceil(2);
    let header_start = out.len();
    out.resize(header_start + header_byte_count, 0);
    for (i, &lz) in lz_counts.iter().enumerate() {
        let nibble = lz & 0x0F;
        let byte_idx = header_start + i / 2;
        if i % 2 == 0 {
            out[byte_idx] |= nibble << 4;
        } else {
            out[byte_idx] |= nibble;
        }
    }

    // Suffix bytes — the (width - lz) low-order bytes of each element.
    for (chunk, &lz) in plane.chunks_exact(width).zip(lz_counts.iter()) {
        let suffix_len = width - lz as usize;
        out.extend_from_slice(&chunk[..suffix_len]);
    }

    Ok(Encoded {
        state_bytes: Vec::new(),
        state_format_version: 0,
        payload: out,
    })
}

fn decode_any_width(payload: &[u8]) -> Result<Vec<u8>, PtwmCoreError> {
    if payload.len() < 5 {
        return Err(PtwmCoreError::CodecDecode {
            codec: "fpc",
            msg: format!("payload too short for header ({} bytes)", payload.len()),
        });
    }
    let width = payload[0] as usize;
    if width == 0 || width > 8 || width.count_ones() != 1 {
        return Err(PtwmCoreError::CodecDecode {
            codec: "fpc",
            msg: format!("unsupported element width {width}"),
        });
    }
    let n_elements = u32::from_le_bytes(payload[1..5].try_into().unwrap()) as usize;
    let header_byte_count = n_elements.div_ceil(2);
    let headers_start = 5usize;
    let suffix_start = headers_start
        .checked_add(header_byte_count)
        .ok_or_else(|| PtwmCoreError::CodecDecode {
            codec: "fpc",
            msg: "suffix offset overflows usize".into(),
        })?;
    if payload.len() < suffix_start {
        return Err(PtwmCoreError::CodecDecode {
            codec: "fpc",
            msg: "payload truncated before suffix bytes".into(),
        });
    }

    // Output capacity is exactly n_elements * width; use checked_mul so
    // a malformed payload claiming an enormous element count on a
    // 32-bit target cannot trigger a silent allocation rollover. The
    // allocation itself will still fail via the allocator if the
    // attacker-controlled count is genuinely huge — but at that point
    // the failure is loud rather than silent corruption.
    let decoded_len = n_elements
        .checked_mul(width)
        .ok_or_else(|| PtwmCoreError::CodecDecode {
            codec: "fpc",
            msg: "decoded length overflows usize".into(),
        })?;
    let mut out = Vec::with_capacity(decoded_len);
    let mut suffix_cursor = suffix_start;
    for i in 0..n_elements {
        let header_byte = payload[headers_start + i / 2];
        let lz = if i % 2 == 0 {
            (header_byte >> 4) & 0x0F
        } else {
            header_byte & 0x0F
        } as usize;
        if lz > width {
            return Err(PtwmCoreError::CodecDecode {
                codec: "fpc",
                msg: format!("leading-zero count {lz} exceeds width {width}"),
            });
        }
        let suffix_len = width - lz;
        // suffix_cursor + suffix_len could overflow usize on a 32-bit
        // target if the payload claims absurd lengths; use checked_add
        // before comparing against payload.len().
        let suffix_end =
            suffix_cursor
                .checked_add(suffix_len)
                .ok_or_else(|| PtwmCoreError::CodecDecode {
                    codec: "fpc",
                    msg: "suffix end offset overflows usize".into(),
                })?;
        if payload.len() < suffix_end {
            return Err(PtwmCoreError::CodecDecode {
                codec: "fpc",
                msg: "payload truncated mid-suffix".into(),
            });
        }
        out.extend_from_slice(&payload[suffix_cursor..suffix_end]);
        // resize is faster than extend(repeat_n) here because the
        // capacity is already sized for the full decoded plane: each
        // resize is a memset on already-allocated memory, with no
        // length-check branch per byte.
        let new_len = out.len() + lz;
        out.resize(new_len, 0);
        suffix_cursor = suffix_end;
    }

    if suffix_cursor != payload.len() {
        return Err(PtwmCoreError::CodecDecode {
            codec: "fpc",
            msg: format!(
                "payload has {} trailing bytes after final suffix",
                payload.len() - suffix_cursor
            ),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::PlaneLayout;
    use crate::types::descriptor::{ElementWidth, Layout, PlaneDescriptor};
    use crate::types::role::Role;

    fn descriptor(width: ElementWidth) -> PlaneDescriptor {
        PlaneDescriptor {
            role: Role::Raw,
            element_width: width,
            length_bytes: 0,
            layout: Layout::Flat,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        }
    }

    #[test]
    fn accepts_byte_width_only() {
        // accepts() is intentionally narrow until the descriptor's
        // element_width is plumbed through to `encode`. The wider
        // widths still round-trip through the internal helper, see
        // `encode_with_explicit_widths_round_trips`.
        let c = Fpc;
        assert!(c.accepts(&descriptor(ElementWidth::Byte)));
        for w in [
            ElementWidth::Nibble,
            ElementWidth::Word2,
            ElementWidth::Word4,
            ElementWidth::Word8,
        ] {
            assert!(!c.accepts(&descriptor(w)));
        }
    }

    #[test]
    fn byte_width_round_trip_through_default_encode() {
        // The default `encode` path uses width=1, which is the byte
        // plane case the dispatcher hits after byte_split + predictor.
        let c = Fpc;
        let data: Vec<u8> = vec![0, 0, 0, 5, 0, 10, 0, 0, 0, 1, 0, 0];
        let enc = c.encode(&data, None, &PlaneLayout::Flat).unwrap();
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
    fn empty_input_round_trips() {
        let c = Fpc;
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
        assert!(dec.is_empty());
    }

    #[test]
    fn dense_zero_input_compresses_hard() {
        // All-zero input: every element has full leading-zero count, so
        // the suffix bytes are empty. Result should be much smaller
        // than input.
        let c = Fpc;
        let data = vec![0u8; 4096];
        let enc = c.encode(&data, None, &PlaneLayout::Flat).unwrap();
        // 5 bytes of header + 2048 bytes of packed headers (each
        // element gets a 4-bit field, two per byte). Suffixes are
        // empty. So output is ~2053 bytes — half of input.
        assert!(
            enc.payload.len() < data.len() / 2 + 16,
            "expected dense-zero to compress to ~half input, got {}",
            enc.payload.len()
        );
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
    fn dense_nonzero_input_expands() {
        // No leading zeros anywhere: header byte per pair plus full
        // suffix bytes; output is bigger than input by the header cost.
        let c = Fpc;
        let data: Vec<u8> = (1..=128).collect(); // all non-zero
        let enc = c.encode(&data, None, &PlaneLayout::Flat).unwrap();
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
    fn encode_with_explicit_widths_round_trips() {
        // Drive the wider encode paths directly. The dispatcher path
        // currently picks width=1; widths 2/4/8 are reachable from a
        // future direct call when the descriptor's element_width is
        // exposed at codec entry. Test them anyway so we have the
        // round-trip invariant locked in.
        for width in [1usize, 2, 4, 8] {
            let n = 16;
            let mut data = Vec::with_capacity(n * width);
            for i in 0..n {
                let v = (i as u64).wrapping_mul(7);
                data.extend_from_slice(&v.to_le_bytes()[..width]);
            }
            let enc = encode_with_width(&data, width).unwrap();
            let dec = decode_any_width(&enc.payload).unwrap();
            assert_eq!(dec, data, "round-trip failed at width={width}");
        }
    }

    #[test]
    fn decode_rejects_truncated_payload() {
        // Header says 64 elements but suffix is empty — must error.
        let mut payload = vec![1u8]; // width=1
        payload.extend_from_slice(&64u32.to_le_bytes());
        // No header / suffix bytes follow.
        let res = decode_any_width(&payload);
        assert!(res.is_err(), "expected truncation error, got {res:?}");
    }

    #[test]
    fn decode_rejects_lz_count_exceeding_width() {
        // Width=1 but the header nibble encodes a leading-zero count of
        // 2 — impossible.
        let mut payload = vec![1u8]; // width=1
        payload.extend_from_slice(&1u32.to_le_bytes()); // 1 element
        payload.push(0x20); // high nibble = 2 (illegal for width=1)
        let res = decode_any_width(&payload);
        assert!(res.is_err(), "expected lz>width rejection, got {res:?}");
    }

    #[test]
    fn decode_rejects_unsupported_width() {
        let mut payload = vec![3u8]; // width=3 — not a power of two
        payload.extend_from_slice(&1u32.to_le_bytes());
        payload.push(0x00);
        payload.push(0x00);
        payload.push(0x00);
        let res = decode_any_width(&payload);
        assert!(res.is_err());
    }
}
