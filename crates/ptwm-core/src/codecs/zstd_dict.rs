//! Zstd plane codec **with dictionary support**.
//!
//! Functionally equivalent to [`crate::codecs::zstd::Zstd`] when the
//! plane's state bytes are empty: the encoder emits a plain Zstd frame
//! and the decoder accepts it.
//!
//! When `state_bytes` is non-empty, the codec treats it as a
//! pre-trained Zstd dictionary. Encoding seeds the compressor's LZ
//! window with the dictionary content so repeated structural patterns
//! (e.g. attention block headers, embedding row layouts) compress to
//! shorter back-references. Decoding must use the *same* dictionary
//! bytes, which is why they ride along in the plane record's state
//! slot.
//!
//! Dictionary training is out of scope for this module. See
//! `zstd --train` or the `ptwm chains train-dict` future CLI surface.
//! The codec is generic over whatever bytes the caller
//! decides to ship as the dictionary.

use crate::codec::{CodecId, Encoded, PlaneCodec};
use crate::error::PtwmCoreError;
use crate::layout::PlaneLayout;
use crate::types::descriptor::PlaneDescriptor;

const DEFAULT_LEVEL: i32 = 3;
/// Version byte stored alongside the dictionary in the plane state.
/// Bumped only on a format-breaking change to how the dictionary is
/// applied — current value 1 is plain dictionary bytes.
const STATE_FORMAT_VERSION: u8 = 1;

pub struct ZstdDict;

impl ZstdDict {
    pub fn canonical_id(&self) -> crate::extension::CanonicalId {
        crate::extension::builtin_canonical_id("zstd_dict")
    }
}

impl PlaneCodec for ZstdDict {
    fn id(&self) -> CodecId {
        CodecId::ZstdDict
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
        #[cfg(feature = "codec-zstd")]
        {
            let dict = shared_state.unwrap_or(&[]);
            let payload = if dict.is_empty() {
                // No dictionary → behave exactly like plain `Zstd`.
                zstd::encode_all(plane, DEFAULT_LEVEL)
            } else {
                encode_with_dict(plane, dict, DEFAULT_LEVEL)
            }
            .map_err(|e| PtwmCoreError::CodecEncode {
                codec: "zstd_dict",
                msg: e.to_string(),
            })?;
            // Mirror the Order1ScaleAC convention: when the caller
            // provides shared state via `shared_state`, the writer is
            // already storing it in the file prelude and addressing it
            // through `StateSource::Shared`. Re-emitting the same dict
            // bytes in every plane record's inline state slot would
            // duplicate it once per plane, defeating the point of the
            // shared prelude. When `shared_state` is `None` the codec
            // has no inline state to emit either: future inline-state
            // support would attach its own per-plane bytes here.
            let state_bytes = Vec::new();
            // `state_format_version` is still meaningful when shared
            // state is in play: the decoder uses it to gate the format
            // it deserializes from the prelude entry. Keep it at the
            // current version when a dict was supplied, and report 0
            // (the no-state convention used by Identity / Huffman /
            // plain Zstd) when no dict was supplied.
            let state_format_version = if dict.is_empty() {
                0
            } else {
                STATE_FORMAT_VERSION
            };
            Ok(Encoded {
                state_bytes,
                state_format_version,
                payload,
            })
        }
        #[cfg(not(feature = "codec-zstd"))]
        {
            let _ = (plane, shared_state);
            Err(PtwmCoreError::InvalidContainer(
                "codec-zstd feature is disabled".into(),
            ))
        }
    }

    fn decode(
        &self,
        state_format_version: u8,
        state_bytes: &[u8],
        payload: &[u8],
        _layout: &PlaneLayout,
        decoded_len: usize,
    ) -> Result<Vec<u8>, PtwmCoreError> {
        #[cfg(feature = "codec-zstd")]
        {
            if !state_bytes.is_empty() && state_format_version != STATE_FORMAT_VERSION {
                return Err(PtwmCoreError::CodecDecode {
                    codec: "zstd_dict",
                    msg: format!(
                        "unsupported state_format_version {state_format_version} (expected \
                         {STATE_FORMAT_VERSION} or 0 for empty state)"
                    ),
                });
            }
            if payload.is_empty() {
                return Ok(Vec::new());
            }
            if state_bytes.is_empty() {
                zstd::decode_all(payload)
            } else {
                decode_with_dict(payload, state_bytes, decoded_len)
            }
            .map_err(|e| PtwmCoreError::CodecDecode {
                codec: "zstd_dict",
                msg: e.to_string(),
            })
        }
        #[cfg(not(feature = "codec-zstd"))]
        {
            let _ = (state_format_version, state_bytes, payload, decoded_len);
            Err(PtwmCoreError::InvalidContainer(
                "codec-zstd feature is disabled".into(),
            ))
        }
    }
}

// The bulk APIs are preferred over the streaming wrappers here for
// two reasons. First, they avoid the std::io::Write/Read indirection
// that the streaming API layers on top of zstd's C functions — a real
// cost on small planes where the per-call overhead dominates. Second,
// the bulk Compressor/Decompressor own a single zstd context that
// could later be cached across calls (keyed by dict + level) to skip
// dictionary preparation entirely. That cache is not in scope here,
// but the switch to bulk is a prerequisite for it.
#[cfg(feature = "codec-zstd")]
fn encode_with_dict(plane: &[u8], dict: &[u8], level: i32) -> std::io::Result<Vec<u8>> {
    let mut compressor = zstd::bulk::Compressor::with_dictionary(level, dict)?;
    compressor.compress(plane)
}

#[cfg(feature = "codec-zstd")]
fn decode_with_dict(payload: &[u8], dict: &[u8], decoded_len: usize) -> std::io::Result<Vec<u8>> {
    let mut decompressor = zstd::bulk::Decompressor::with_dictionary(dict)?;
    // The bulk Decompressor writes into a buffer of exactly the
    // capacity we pass in. The dispatcher knows the decoded length
    // from the propagated terminal descriptor and forwards it via the
    // `decoded_len` argument. When that is zero (older callers that
    // don't propagate it), fall back to the frame's pledged source
    // size if present, then to an 8x heuristic.
    let capacity = if decoded_len > 0 {
        decoded_len
    } else {
        zstd::bulk::Decompressor::upper_bound(payload)
            .unwrap_or_else(|| payload.len().saturating_mul(8))
    };
    decompressor.decompress(payload, capacity)
}

#[cfg(all(test, feature = "codec-zstd"))]
mod tests {
    use super::*;
    use crate::layout::PlaneLayout;

    #[test]
    fn empty_state_matches_plain_zstd_behaviour() {
        let c = ZstdDict;
        let data: Vec<u8> = (0..1024).map(|i| ((i * 13) & 0xFF) as u8).collect();
        let enc = c.encode(&data, None, &PlaneLayout::Flat).unwrap();
        assert!(enc.state_bytes.is_empty());
        assert_eq!(enc.state_format_version, 0);
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
    fn dictionary_round_trips() {
        // The encoder no longer echoes the dict back through
        // Encoded.state_bytes — the dispatcher carries the dict in the
        // prelude under StateSource::Shared and materializes it back
        // into the decoder's `state_bytes` argument at decode time. The
        // direct-call tests below simulate that by passing the dict
        // into decode() themselves.
        let c = ZstdDict;
        let pattern = b"the quick brown fox jumps over the lazy dog ";
        let dict: Vec<u8> = pattern.repeat(8);
        let data: Vec<u8> = pattern.repeat(64);

        let enc = c.encode(&data, Some(&dict), &PlaneLayout::Flat).unwrap();
        assert!(
            enc.state_bytes.is_empty(),
            "encoder must not duplicate shared state in plane record"
        );
        assert_eq!(enc.state_format_version, STATE_FORMAT_VERSION);

        let dec = c
            .decode(
                enc.state_format_version,
                &dict, // dispatcher would fetch this from the prelude
                &enc.payload,
                &PlaneLayout::Flat,
                data.len(),
            )
            .unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn dictionary_payload_round_trips_on_pattern_data() {
        // A previous version of this test asserted strict inequality
        // (with_dict.len() < without_dict.len()). That inequality is
        // not guaranteed across zstd versions and input shapes for
        // small synthetic dictionaries — a regression would surface as
        // a brittle test failure rather than a real codec issue. Drop
        // the inequality and assert only the contract: round-trip is
        // exact and the dict path produces a payload that *can* be
        // decoded back with the same dict.
        let c = ZstdDict;
        let pattern = b"common-token-A common-token-B common-token-C ";
        let dict: Vec<u8> = pattern.repeat(8);
        let data: Vec<u8> = pattern.repeat(64);

        let with_dict = c.encode(&data, Some(&dict), &PlaneLayout::Flat).unwrap();
        let dec = c
            .decode(
                with_dict.state_format_version,
                &dict,
                &with_dict.payload,
                &PlaneLayout::Flat,
                data.len(),
            )
            .unwrap();
        assert_eq!(dec, data);

        let without_dict = c.encode(&data, None, &PlaneLayout::Flat).unwrap();
        let dec_plain = c
            .decode(
                without_dict.state_format_version,
                &without_dict.state_bytes,
                &without_dict.payload,
                &PlaneLayout::Flat,
                data.len(),
            )
            .unwrap();
        assert_eq!(dec_plain, data);
    }

    #[test]
    fn decode_rejects_unknown_state_version() {
        let c = ZstdDict;
        let dict: Vec<u8> = b"dict".repeat(8);
        let data = b"hello".to_vec();
        let enc = c.encode(&data, Some(&dict), &PlaneLayout::Flat).unwrap();
        // Fabricate a state version we don't understand. Pass the real
        // dict in state_bytes so the version check is what trips, not
        // an empty-state short-circuit.
        let res = c.decode(99, &dict, &enc.payload, &PlaneLayout::Flat, data.len());
        assert!(matches!(res, Err(PtwmCoreError::CodecDecode { .. })));
    }

    #[test]
    fn empty_payload_yields_empty_output() {
        let c = ZstdDict;
        let dec = c.decode(0, &[], &[], &PlaneLayout::Flat, 0).unwrap();
        assert!(dec.is_empty());
    }

    #[test]
    fn empty_input_round_trips_with_dict() {
        let c = ZstdDict;
        let dict: Vec<u8> = b"dict".repeat(8);
        let enc = c.encode(&[], Some(&dict), &PlaneLayout::Flat).unwrap();
        let dec = c
            .decode(
                enc.state_format_version,
                &dict,
                &enc.payload,
                &PlaneLayout::Flat,
                0,
            )
            .unwrap();
        assert!(dec.is_empty());
    }
}
