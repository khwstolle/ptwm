//! Bridge adapter that exposes a third-party `DispatchedPlaneCodec`
//! (native or WASM, dispatched by canonical id) as if it were an in-tree
//! `crate::codec::PlaneCodec`. The container decode loop can then treat
//! every plane uniformly through a single `Box<dyn PlaneCodec>`.
//!
//! Scope: decode-only. The flat ABI used by third-party codecs is
//! input-bytes → output-bytes (with an optional `state_bytes` blob on
//! decode), which doesn't expose the structured `Encoded` return shape
//! the in-tree trial-encode loop expects. `encode` therefore returns
//! `Err(InvalidContainer)` — v1 third-party codecs participate only at
//! decode time. Reaching `encode` here would mean the trial-encode loop
//! reached a third-party codec, which is itself a bug.

use std::sync::Arc;

use crate::codec::{CodecId, Encoded, PlaneCodec};
use crate::error::PtwmCoreError;
use crate::flavor::abi::CodecError;
use crate::flavor::router::DispatchedPlaneCodec;
use crate::layout::PlaneLayout;

pub struct ThirdPartyPlaneCodec {
    inner: Arc<dyn DispatchedPlaneCodec>,
}

impl ThirdPartyPlaneCodec {
    pub fn new(inner: Arc<dyn DispatchedPlaneCodec>) -> Self {
        Self { inner }
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
        _plane: &[u8],
        _shared_state: Option<&[u8]>,
        _layout: &PlaneLayout,
    ) -> Result<Encoded, PtwmCoreError> {
        Err(PtwmCoreError::InvalidContainer(
            "third-party codecs cannot participate in the trial-encode loop \
             (v1 dispatches third-party codecs only at decode time)"
                .into(),
        ))
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
            .map_err(|e| map_codec_error(e, "third-party decode", out.len()))?;
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

fn map_codec_error(e: CodecError, ctx: &'static str, got_cap: usize) -> PtwmCoreError {
    match e {
        CodecError::BufferTooSmall { needed } => PtwmCoreError::BufferTooSmall {
            expected: needed as usize,
            got: got_cap,
        },
        other => PtwmCoreError::CodecDecode {
            codec: ctx,
            msg: format!("{other:?}"),
        },
    }
}
