//! Bridge adapter that exposes a third-party `DispatchedPlaneCodec`
//! (native or WASM, dispatched by canonical id) as if it were an in-tree
//! `crate::codec::PlaneCodec`. The container decode loop can then treat
//! every plane uniformly through a single `Box<dyn PlaneCodec>`.
//!
//! Scope: decode is always reachable; encode depends on how the adapter
//! was built. The flat ABI used by third-party codecs is input-bytes →
//! output-bytes, which doesn't expose the structured `Encoded` return
//! shape the in-tree trial-encode loop expects. An adapter built through
//! [`ThirdPartyPlaneCodec::new`] represents a codec the trial-encode loop
//! is considering among several candidates, so `encode` returns
//! `Err(InvalidContainer)` there: reaching `encode` on that path would
//! mean the trial loop reached a third-party codec, which is itself a
//! bug. An adapter built through [`ThirdPartyPlaneCodec::new_explicit`]
//! represents a codec the caller named directly; there is no trial loop
//! to satisfy, so `encode` is reachable.

use std::sync::Arc;

use crate::codec::{CodecId, Encoded, PlaneCodec};
use crate::error::PtwmCoreError;
use crate::flavor::abi::CodecError;
use crate::flavor::router::DispatchedPlaneCodec;
use crate::layout::PlaneLayout;

pub struct ThirdPartyPlaneCodec {
    inner: Arc<dyn DispatchedPlaneCodec>,
    explicitly_selected: bool,
}

impl ThirdPartyPlaneCodec {
    /// Trial-encode path: the dispatcher is considering this codec among
    /// several candidates, so `encode` stays refused (see module docs).
    pub fn new(inner: Arc<dyn DispatchedPlaneCodec>) -> Self {
        Self {
            inner,
            explicitly_selected: false,
        }
    }

    /// Explicit-selection path: the caller named this codec directly, so
    /// there is no trial-encode loop to satisfy and `encode` is reachable.
    pub fn new_explicit(inner: Arc<dyn DispatchedPlaneCodec>) -> Self {
        Self {
            inner,
            explicitly_selected: true,
        }
    }

    pub(crate) fn allows_encode(explicitly_selected: bool) -> bool {
        explicitly_selected
    }
}

impl PlaneCodec for ThirdPartyPlaneCodec {
    fn id(&self) -> CodecId {
        // Third-party codecs aren't covered by the closed `CodecId`
        // enum — they're identified by canonical id at the router
        // boundary. Decode-time callers in container.rs never inspect
        // this value; returning Identity is the safe placeholder.
        CodecId::Identity
    }

    fn encode(
        &self,
        plane: &[u8],
        _shared_state: Option<&[u8]>,
        _layout: &PlaneLayout,
    ) -> Result<Encoded, PtwmCoreError> {
        if !Self::allows_encode(self.explicitly_selected) {
            return Err(PtwmCoreError::InvalidContainer(
                "third-party codecs do not participate in the trial-encode loop; \
                 select this codec explicitly to encode with it"
                    .into(),
            ));
        }
        let mut out = vec![0u8; plane.len() * 2 + 1024];
        let written = self.inner.encode(plane, &mut out).map_err(|e| {
            map_codec_error(e, CodecDirection::Encode, "third-party encode", out.len())
        })?;
        // `written` crosses the flat ABI from the third-party side, so it
        // is untrusted: a codec that reports more bytes than the buffer
        // holds must fail loudly here rather than let `truncate` silently
        // no-op and hand the caller a zero-padded buffer as if it were
        // real encoded output.
        if written > out.len() {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "third-party encode reported {written} bytes written into a {}-byte buffer",
                out.len()
            )));
        }
        out.truncate(written);
        Ok(Encoded {
            state_bytes: Vec::new(),
            state_format_version: 0,
            payload: out,
        })
    }

    fn decode(
        &self,
        state_format_version: u8,
        state_bytes: &[u8],
        payload: &[u8],
        _layout: &PlaneLayout,
        decoded_len: usize,
    ) -> Result<Vec<u8>, PtwmCoreError> {
        let mut out = vec![0u8; decoded_len];
        let written = self
            .inner
            .decode_with_state(state_format_version, state_bytes, payload, &mut out)
            .map_err(|e| {
                map_codec_error(e, CodecDirection::Decode, "third-party decode", out.len())
            })?;
        // The container decode loop for non-chunked planes doesn't
        // separately verify the byte count, so enforce it here:
        // a third-party codec that writes the wrong number of bytes
        // must fail loudly rather than feed truncated/padded data
        // downstream.
        if written != decoded_len {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "third-party decode wrote {written} bytes but plane terminal expected {decoded_len}"
            )));
        }
        Ok(out)
    }
}

/// Which call site is mapping a [`CodecError`], so the non-`BufferTooSmall`
/// fallback below can pick the matching [`PtwmCoreError`] variant instead of
/// always reporting a decode failure.
enum CodecDirection {
    Encode,
    Decode,
}

fn map_codec_error(
    e: CodecError,
    direction: CodecDirection,
    ctx: &'static str,
    got_cap: usize,
) -> PtwmCoreError {
    match e {
        CodecError::BufferTooSmall { needed } => PtwmCoreError::BufferTooSmall {
            expected: needed as usize,
            got: got_cap,
        },
        other => {
            let msg = format!("{other:?}");
            match direction {
                CodecDirection::Encode => PtwmCoreError::CodecEncode { codec: ctx, msg },
                CodecDirection::Decode => PtwmCoreError::CodecDecode { codec: ctx, msg },
            }
        }
    }
}

#[cfg(test)]
mod explicit_selection_tests {
    use super::*;

    /// Minimal `DispatchedPlaneCodec` that echoes its input, so the tests
    /// exercise the gate rather than any real codec's behavior.
    struct EchoCodec;

    impl DispatchedPlaneCodec for EchoCodec {
        fn encode(&self, input: &[u8], output: &mut [u8]) -> Result<usize, CodecError> {
            output[..input.len()].copy_from_slice(input);
            Ok(input.len())
        }
        fn decode(&self, input: &[u8], output: &mut [u8]) -> Result<usize, CodecError> {
            output[..input.len()].copy_from_slice(input);
            Ok(input.len())
        }
    }

    #[test]
    fn encode_is_refused_for_trial_selected_codecs() {
        let codec = ThirdPartyPlaneCodec::new(Arc::new(EchoCodec));
        let err = codec
            .encode(&[1, 2, 3], None, &PlaneLayout::default())
            .expect_err("the trial-encode loop must not reach a third-party encoder");
        assert!(
            format!("{err}").contains("select this codec explicitly"),
            "the error must tell the caller how to proceed, got: {err}"
        );
    }

    #[test]
    fn encode_is_permitted_for_explicitly_selected_codecs() {
        let codec = ThirdPartyPlaneCodec::new_explicit(Arc::new(EchoCodec));
        let encoded = codec
            .encode(&[1, 2, 3], None, &PlaneLayout::default())
            .expect("explicit selection must reach the encoder");
        assert_eq!(encoded.payload.len(), 3, "echo codec returns its input");
    }

    /// `DispatchedPlaneCodec` whose `encode` reports writing more bytes
    /// than the output buffer holds, so the tests can exercise the
    /// out-of-bounds guard without a codec that actually overruns the
    /// buffer (which would panic in the stub instead of in the guard).
    struct OverclaimingCodec;

    impl DispatchedPlaneCodec for OverclaimingCodec {
        fn encode(&self, _input: &[u8], output: &mut [u8]) -> Result<usize, CodecError> {
            Ok(output.len() + 1)
        }
        fn decode(&self, input: &[u8], output: &mut [u8]) -> Result<usize, CodecError> {
            output[..input.len()].copy_from_slice(input);
            Ok(input.len())
        }
    }

    #[test]
    fn encode_fails_loudly_when_third_party_overreports_bytes_written() {
        let codec = ThirdPartyPlaneCodec::new_explicit(Arc::new(OverclaimingCodec));
        let err = codec
            .encode(&[1, 2, 3], None, &PlaneLayout::default())
            .expect_err("an over-large byte count must not become a zero-padded payload");
        assert!(
            format!("{err}").contains("bytes written"),
            "the error should describe the mismatched byte count, got: {err}"
        );
    }
}
