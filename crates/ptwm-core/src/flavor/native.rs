//! Native cdylib loader. Loads a `.so` / `.dylib` / `.dll` via libloading,
//! probes for the per-kind ABI symbols, and exposes a thin handle the
//! dispatcher uses to invoke each method.
//!
//! Safety invariant: the caller MUST have verified the signature against
//! the active trust state before calling `NativeExtension::load`. The
//! `VerifiedToken` opaque type makes this an enforced precondition.

use std::path::Path;

use libloading::{Library, Symbol};

use crate::extension::{CanonicalId, ExtensionTableEntry, Kind, Lifecycle};
use crate::flavor::abi::CodecError;

/// Capability token proving the caller verified the contribution's
/// signature. Construct via `VerifiedToken::new_unchecked` only when
/// the verifier returned Trusted.
pub struct VerifiedToken {
    _private: (),
}

impl VerifiedToken {
    /// Create a token. ONLY call this after the Verifier returned
    /// TrustVerdict::Trusted for the entry.
    pub fn new_unchecked() -> Self {
        Self { _private: () }
    }
}

/// Symbol set discovered at load time. Each Option carries the raw fn
/// pointer for the corresponding ABI export, or None if absent.
///
/// `libloading::Library` must outlive every fn pointer derived from
/// it; we hold the Library inside NativeExtension and the fn pointers
/// alongside.
pub struct NativeSymbols {
    // PlaneCodec
    pub plane_codec_v1_encode: Option<PlaneCodecFn>,
    pub plane_codec_v1_decode: Option<PlaneCodecFn>,
    /// Optional state-aware decode symbol. When present, the dispatcher
    /// uses it for any plane carrying inline / shared state bytes; when
    /// absent, the dispatcher falls back to the stateless
    /// `plane_codec_v1_decode` symbol.
    pub plane_codec_v1_decode_stateful: Option<PlaneCodecStatefulFn>,
    /// Optional CUDA device-pointer path. A codec that only implements
    /// the host path leaves these absent.
    pub plane_codec_v1_encode_cuda: Option<PlaneCodecCudaEncodeFn>,
    pub plane_codec_v1_decode_cuda: Option<PlaneCodecCudaDecodeFn>,
    pub plane_codec_v1_cuda_stream_handle: Option<PlaneCodecCudaStreamHandleFn>,
    // Transform
    pub transform_v1_forward: Option<TransformFn>,
    pub transform_v1_inverse: Option<TransformFn>,
    // DeltaScheme
    pub delta_scheme_v1_encode: Option<DeltaSchemeFn>,
    pub delta_scheme_v1_decode: Option<DeltaSchemeFn>,
    // HardwareBackend
    pub hardware_backend_v1_cuda_stream_handle: Option<HardwareBackendCudaStreamHandleFn>,
    pub hardware_backend_v1_dispatch_decode_cuda: Option<HardwareBackendCudaDispatchDecodeFn>,
}

pub type PlaneCodecFn = unsafe extern "C" fn(
    *const u8, // input ptr
    usize,     // input len
    *mut u8,   // output ptr
    usize,     // output len
) -> i64;

/// State-aware decode symbol.
///
/// Signature: `(state_version, state_ptr, state_len, in_ptr, in_len,
/// out_ptr, out_cap) -> i64`. Return convention matches `PlaneCodecFn`
/// (non-negative output length, or one of the negative error codes
/// decoded by `decode_rc`).
pub type PlaneCodecStatefulFn = unsafe extern "C" fn(
    u8,        // state_format_version
    *const u8, // state ptr
    usize,     // state len
    *const u8, // input ptr
    usize,     // input len
    *mut u8,   // output ptr
    usize,     // output cap
) -> i64;

pub type TransformFn = PlaneCodecFn;

/// `delta_scheme_v1_{encode,decode}` symbol shape.
///
/// Unlike the WASM ABI's `ptwm_delta_scheme_v1_{init,cleanup,encode,decode}`
/// (which threads an opaque state handle because WASM linear memory forces
/// the host to manage allocation explicitly — see
/// `extensions/ref_delta_scheme/rust/src/lib.rs`), the native (dlopen)
/// surface carries no state handle, matching [`PlaneCodecFn`]'s existing
/// native convention: a native contribution manages any internal state as
/// ordinary process-local Rust/C state, not through an FFI-visible handle.
pub type DeltaSchemeFn = unsafe extern "C" fn(
    *const u8, // base ptr
    usize,     // base len
    *const u8, // target (encode) / delta (decode) ptr
    usize,     // target/delta len
    *mut u8,   // output ptr
    usize,     // output len
) -> i64;

/// `hardware_backend_v1_cuda_stream_handle`: returns this backend's own,
/// process-persistent CUDA stream (as a raw pointer value) for the given
/// device ordinal. Callers pass this same handle to `tensor.__dlpack__
/// (stream=...)` for every tensor involved in a subsequent decode call, so
/// PyTorch's DLPack producer inserts the correct cross-stream wait before
/// handing back the device pointer. Returns 0 on failure (no such device,
/// CUDA init failed).
pub type HardwareBackendCudaStreamHandleFn = unsafe extern "C" fn(device_ordinal: u32) -> u64;

/// `hardware_backend_v1_dispatch_decode_cuda`: decode `in_dev_ptr` into
/// `out_dev_ptr`, two distinct device buffers (this is NOT an in-place
/// transform over one buffer; "zero-copy" means no host round-trip, not a
/// shared address). `state_bytes` carries the codec's small per-tensor
/// state (e.g. a 16-entry codebook), in the same `(state_format_version,
/// state_bytes)` shape `plane_codec`'s existing `decode_stateful` path
/// already carries. `codec_id` self-describes the wire format for the
/// extension to validate against (distinct from the backend's own
/// canonical id, which the router already resolved to get here).
///
/// Completion contract: by the time this function returns, the kernel has
/// FULLY COMPLETED (the extension synchronizes its own stream before
/// returning). `out_dev_ptr`'s contents are valid and visible to any
/// subsequent CUDA operation on any stream, with no further caller-side
/// synchronization needed. Both launch-time and execution-time errors are
/// visible via the return code, since the synchronize() call observes both.
pub type HardwareBackendCudaDispatchDecodeFn = unsafe extern "C" fn(
    state_format_version: u8,
    state_ptr: *const u8,
    state_len: usize,
    codec_id_ptr: *const u8,
    codec_id_len: usize,
    in_dev_ptr: u64,
    in_len: usize,
    out_dev_ptr: u64,
    out_len: usize,
    device_ordinal: u32,
) -> i64;

pub const PLANE_CODEC_CUDA_ENCODE_SYMBOL: &[u8] = b"ptwm_plane_codec_v1_encode_cuda\0";
pub const PLANE_CODEC_CUDA_DECODE_SYMBOL: &[u8] = b"ptwm_plane_codec_v1_decode_cuda\0";
pub const PLANE_CODEC_CUDA_STREAM_HANDLE_SYMBOL: &[u8] =
    b"ptwm_plane_codec_v1_cuda_stream_handle\0";

/// Returns the extension's CUDA stream handle for `device_ordinal`, or 0
/// if unavailable. Same contract as the hardware-backend equivalent.
pub type PlaneCodecCudaStreamHandleFn = unsafe extern "C" fn(device_ordinal: u32) -> u64;

/// `plane_codec_v1_decode_cuda`: decode `in_dev_ptr` into `out_dev_ptr`,
/// two distinct device buffers. `state_bytes` carries the codec's small
/// per-tensor state in the same `(state_format_version, state_bytes)`
/// shape the existing `decode_stateful` path uses. `codec_id`
/// self-describes the wire format for the extension to validate.
///
/// Completion contract: by the time this returns, the kernel has FULLY
/// COMPLETED (the extension synchronizes its own stream). `out_dev_ptr`
/// is valid and visible to any subsequent CUDA operation on any stream,
/// with no further caller-side synchronization. Launch-time and
/// execution-time errors are both visible in the return code.
pub type PlaneCodecCudaDecodeFn = unsafe extern "C" fn(
    state_format_version: u8,
    state_ptr: *const u8,
    state_len: usize,
    codec_id_ptr: *const u8,
    codec_id_len: usize,
    in_dev_ptr: u64,
    in_len: usize,
    out_dev_ptr: u64,
    out_cap: usize,
    device_ordinal: u32,
) -> i64;

/// `plane_codec_v1_encode_cuda`: encode `in_dev_ptr` into `out_dev_ptr`.
/// Same completion contract as the decode symbol.
///
/// `needed_out` exists because an encoder cannot size its output before
/// running: when `out_cap` is insufficient the extension writes the
/// required byte count to `*needed_out` and returns `-2`. The generic
/// return-code decoder cannot carry that number, so it is passed
/// explicitly. On success `*needed_out` is left untouched.
#[allow(clippy::too_many_arguments)]
pub type PlaneCodecCudaEncodeFn = unsafe extern "C" fn(
    state_format_version: u8,
    state_ptr: *const u8,
    state_len: usize,
    codec_id_ptr: *const u8,
    codec_id_len: usize,
    in_dev_ptr: u64,
    in_len: usize,
    out_dev_ptr: u64,
    out_cap: usize,
    needed_out: *mut u64,
    device_ordinal: u32,
) -> i64;

pub struct NativeExtension {
    // Held for its drop-time side effect: keeping the dlopen handle alive
    // so all fn pointers in `symbols` remain valid.
    #[allow(dead_code)]
    library: Library,
    pub canonical_id: CanonicalId,
    pub kind: Kind,
    pub lifecycle: Lifecycle,
    pub symbols: NativeSymbols,
}

impl NativeExtension {
    /// Load and resolve symbols. Caller MUST present a VerifiedToken.
    ///
    /// # Safety
    /// The library at `path` will be `dlopen`-ed. The caller has
    /// verified its signature.
    pub fn load(
        path: &Path,
        entry: &ExtensionTableEntry,
        _verified: VerifiedToken,
    ) -> Result<Self, CodecError> {
        let library = unsafe { Library::new(path) }.map_err(|_| CodecError::InvalidInput)?;

        let mut symbols = NativeSymbols {
            plane_codec_v1_encode: None,
            plane_codec_v1_decode: None,
            plane_codec_v1_decode_stateful: None,
            plane_codec_v1_encode_cuda: None,
            plane_codec_v1_decode_cuda: None,
            plane_codec_v1_cuda_stream_handle: None,
            transform_v1_forward: None,
            transform_v1_inverse: None,
            delta_scheme_v1_encode: None,
            delta_scheme_v1_decode: None,
            hardware_backend_v1_cuda_stream_handle: None,
            hardware_backend_v1_dispatch_decode_cuda: None,
        };

        // Probe symbols depending on the kind.
        match entry.kind {
            Kind::PlaneCodec => {
                symbols.plane_codec_v1_encode =
                    resolve::<PlaneCodecFn>(&library, b"ptwm_plane_codec_v1_encode\0");
                symbols.plane_codec_v1_decode =
                    resolve::<PlaneCodecFn>(&library, b"ptwm_plane_codec_v1_decode\0");
                // Optional: a state-aware decoder. Stateless codecs leave
                // this absent and the router falls back to the unadorned
                // decode symbol.
                symbols.plane_codec_v1_decode_stateful = resolve::<PlaneCodecStatefulFn>(
                    &library,
                    b"ptwm_plane_codec_v1_decode_stateful\0",
                );
                // Optional CUDA device-pointer path. A codec that only
                // implements the host path leaves these absent; a codec
                // that implements CUDA must export all three, so a
                // partial set is a packaging error worth failing on.
                symbols.plane_codec_v1_encode_cuda =
                    resolve::<PlaneCodecCudaEncodeFn>(&library, PLANE_CODEC_CUDA_ENCODE_SYMBOL);
                symbols.plane_codec_v1_decode_cuda =
                    resolve::<PlaneCodecCudaDecodeFn>(&library, PLANE_CODEC_CUDA_DECODE_SYMBOL);
                symbols.plane_codec_v1_cuda_stream_handle = resolve::<PlaneCodecCudaStreamHandleFn>(
                    &library,
                    PLANE_CODEC_CUDA_STREAM_HANDLE_SYMBOL,
                );
                validate_cuda_symbol_set([
                    symbols.plane_codec_v1_encode_cuda.is_some(),
                    symbols.plane_codec_v1_decode_cuda.is_some(),
                    symbols.plane_codec_v1_cuda_stream_handle.is_some(),
                ])?;
                if symbols.plane_codec_v1_encode.is_none()
                    || symbols.plane_codec_v1_decode.is_none()
                {
                    return Err(CodecError::Unsupported {
                        feature: "missing required plane_codec_v1 symbol".into(),
                    });
                }
            }
            Kind::Transform => {
                symbols.transform_v1_forward =
                    resolve::<TransformFn>(&library, b"ptwm_transform_v1_forward\0");
                symbols.transform_v1_inverse =
                    resolve::<TransformFn>(&library, b"ptwm_transform_v1_inverse\0");
                if symbols.transform_v1_forward.is_none() || symbols.transform_v1_inverse.is_none()
                {
                    return Err(CodecError::Unsupported {
                        feature: "missing required transform_v1 symbol".into(),
                    });
                }
            }
            Kind::DeltaScheme => {
                symbols.delta_scheme_v1_encode =
                    resolve::<DeltaSchemeFn>(&library, b"ptwm_delta_scheme_v1_encode\0");
                symbols.delta_scheme_v1_decode =
                    resolve::<DeltaSchemeFn>(&library, b"ptwm_delta_scheme_v1_decode\0");
                if symbols.delta_scheme_v1_encode.is_none()
                    || symbols.delta_scheme_v1_decode.is_none()
                {
                    return Err(CodecError::Unsupported {
                        feature: "missing required delta_scheme_v1 symbol".into(),
                    });
                }
            }
            Kind::HardwareBackend => {
                symbols.hardware_backend_v1_cuda_stream_handle =
                    resolve::<HardwareBackendCudaStreamHandleFn>(
                        &library,
                        b"ptwm_hardware_backend_v1_cuda_stream_handle\0",
                    );
                symbols.hardware_backend_v1_dispatch_decode_cuda =
                    resolve::<HardwareBackendCudaDispatchDecodeFn>(
                        &library,
                        b"ptwm_hardware_backend_v1_dispatch_decode_cuda\0",
                    );
                if symbols.hardware_backend_v1_cuda_stream_handle.is_none()
                    || symbols.hardware_backend_v1_dispatch_decode_cuda.is_none()
                {
                    return Err(CodecError::Unsupported {
                        feature: "missing required hardware_backend_v1 symbol".into(),
                    });
                }
            }
            _ => {
                // Other kinds: probe is lenient for v1. The dispatcher
                // will yield Unsupported on invocation if the symbol's
                // missing; per-kind probes can be added incrementally.
            }
        }

        Ok(Self {
            library,
            canonical_id: entry.canonical_id,
            kind: entry.kind,
            lifecycle: entry.lifecycle,
            symbols,
        })
    }

    /// Invoke `ptwm_plane_codec_v1_encode`. Caller-allocates buffers.
    pub fn invoke_plane_codec_encode(
        &self,
        input: &[u8],
        output: &mut [u8],
    ) -> Result<usize, CodecError> {
        let f = self
            .symbols
            .plane_codec_v1_encode
            .ok_or(CodecError::Unsupported {
                feature: "plane_codec_v1_encode not present".into(),
            })?;
        let rc = unsafe {
            f(
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
                output.len(),
            )
        };
        decode_rc(rc, output.len())
    }

    /// Invoke `ptwm_plane_codec_v1_decode`. Caller-allocates buffers.
    pub fn invoke_plane_codec_decode(
        &self,
        input: &[u8],
        output: &mut [u8],
    ) -> Result<usize, CodecError> {
        let f = self
            .symbols
            .plane_codec_v1_decode
            .ok_or(CodecError::Unsupported {
                feature: "plane_codec_v1_decode not present".into(),
            })?;
        let rc = unsafe {
            f(
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
                output.len(),
            )
        };
        decode_rc(rc, output.len())
    }

    /// Invoke `ptwm_plane_codec_v1_decode_stateful`. Caller-allocates
    /// buffers. Falls back to the stateless [`Self::invoke_plane_codec_decode`]
    /// when the extension didn't export the stateful variant — appropriate
    /// for stateless codecs whose state slice is empty.
    pub fn invoke_plane_codec_decode_stateful(
        &self,
        state_version: u8,
        state_bytes: &[u8],
        input: &[u8],
        output: &mut [u8],
    ) -> Result<usize, CodecError> {
        if let Some(f) = self.symbols.plane_codec_v1_decode_stateful {
            // For empty slices, `as_ptr()` returns a non-null dangling
            // pointer (e.g. 0x1). C-side codecs commonly null-check the
            // state pointer to decide whether state is present, so pass
            // a real null when the slice is empty.
            let state_ptr = if state_bytes.is_empty() {
                std::ptr::null()
            } else {
                state_bytes.as_ptr()
            };
            let rc = unsafe {
                f(
                    state_version,
                    state_ptr,
                    state_bytes.len(),
                    input.as_ptr(),
                    input.len(),
                    output.as_mut_ptr(),
                    output.len(),
                )
            };
            return decode_rc(rc, output.len());
        }
        // No stateful symbol — only legal when the caller is passing an
        // empty state. A stateful plane reaching a stateless codec would
        // be a decode-time misconfiguration.
        if !state_bytes.is_empty() {
            return Err(CodecError::Unsupported {
                feature: "plane_codec_v1_decode_stateful not exported but plane carries state \
                          bytes; codec cannot honour the per-plane state"
                    .into(),
            });
        }
        self.invoke_plane_codec_decode(input, output)
    }

    /// Invoke `ptwm_delta_scheme_v1_encode`. Caller-allocates buffers.
    pub fn invoke_delta_scheme_encode(
        &self,
        base: &[u8],
        target: &[u8],
        output: &mut [u8],
    ) -> Result<usize, CodecError> {
        let f = self
            .symbols
            .delta_scheme_v1_encode
            .ok_or(CodecError::Unsupported {
                feature: "delta_scheme_v1_encode not present".into(),
            })?;
        let rc = unsafe {
            f(
                base.as_ptr(),
                base.len(),
                target.as_ptr(),
                target.len(),
                output.as_mut_ptr(),
                output.len(),
            )
        };
        decode_rc(rc, output.len())
    }

    /// Invoke `ptwm_delta_scheme_v1_decode`. Caller-allocates buffers.
    pub fn invoke_delta_scheme_decode(
        &self,
        base: &[u8],
        delta: &[u8],
        output: &mut [u8],
    ) -> Result<usize, CodecError> {
        let f = self
            .symbols
            .delta_scheme_v1_decode
            .ok_or(CodecError::Unsupported {
                feature: "delta_scheme_v1_decode not present".into(),
            })?;
        let rc = unsafe {
            f(
                base.as_ptr(),
                base.len(),
                delta.as_ptr(),
                delta.len(),
                output.as_mut_ptr(),
                output.len(),
            )
        };
        decode_rc(rc, output.len())
    }

    /// Invoke `ptwm_hardware_backend_v1_cuda_stream_handle`. Returns this
    /// backend's process-persistent CUDA stream handle for `device_ordinal`.
    pub fn invoke_hardware_backend_cuda_stream_handle(
        &self,
        device_ordinal: u32,
    ) -> Result<u64, CodecError> {
        let f =
            self.symbols
                .hardware_backend_v1_cuda_stream_handle
                .ok_or(CodecError::Unsupported {
                    feature: "hardware_backend_v1_cuda_stream_handle not present".into(),
                })?;
        let handle = unsafe { f(device_ordinal) };
        if handle == 0 {
            return Err(CodecError::Unsupported {
                feature: "hardware backend failed to produce a CUDA stream handle".into(),
            });
        }
        Ok(handle)
    }

    /// Invoke `ptwm_hardware_backend_v1_dispatch_decode_cuda`. Decodes
    /// `in_dev_ptr` into `out_dev_ptr`, two distinct device buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn invoke_hardware_backend_dispatch_decode_cuda(
        &self,
        state_bytes: &[u8],
        codec_id: &CanonicalId,
        in_dev_ptr: u64,
        in_len: usize,
        out_dev_ptr: u64,
        out_len: usize,
        device_ordinal: u32,
    ) -> Result<usize, CodecError> {
        let f = self
            .symbols
            .hardware_backend_v1_dispatch_decode_cuda
            .ok_or(CodecError::Unsupported {
                feature: "hardware_backend_v1_dispatch_decode_cuda not present".into(),
            })?;
        let codec_id_bytes = codec_id.as_bytes();
        let rc = unsafe {
            f(
                1, // state_format_version
                state_bytes.as_ptr(),
                state_bytes.len(),
                codec_id_bytes.as_ptr(),
                codec_id_bytes.len(),
                in_dev_ptr,
                in_len,
                out_dev_ptr,
                out_len,
                device_ordinal,
            )
        };
        decode_rc(rc, out_len)
    }

    /// Returns the extension's CUDA stream handle for `device_ordinal`,
    /// or 0 when the extension has none.
    pub fn plane_codec_cuda_stream_handle(&self, device_ordinal: u32) -> u64 {
        match self.symbols.plane_codec_v1_cuda_stream_handle {
            Some(f) => unsafe { f(device_ordinal) },
            None => 0,
        }
    }

    /// Invoke `ptwm_plane_codec_v1_decode_cuda`. `in_dev_ptr` and
    /// `out_dev_ptr` are distinct device buffers; see the type's doc
    /// comment for the completion contract.
    #[allow(clippy::too_many_arguments)]
    pub fn invoke_plane_codec_decode_cuda(
        &self,
        state_bytes: &[u8],
        codec_id: &CanonicalId,
        in_dev_ptr: u64,
        in_len: usize,
        out_dev_ptr: u64,
        out_cap: usize,
        device_ordinal: u32,
    ) -> Result<usize, CodecError> {
        let f = self
            .symbols
            .plane_codec_v1_decode_cuda
            .ok_or(CodecError::Unsupported {
                feature: "plane_codec_v1_decode_cuda not present".into(),
            })?;
        let codec_id_bytes = codec_id.as_bytes();
        let rc = unsafe {
            f(
                1,
                state_bytes.as_ptr(),
                state_bytes.len(),
                codec_id_bytes.as_ptr(),
                codec_id_bytes.len(),
                in_dev_ptr,
                in_len,
                out_dev_ptr,
                out_cap,
                device_ordinal,
            )
        };
        decode_rc(rc, out_cap)
    }

    /// Invoke `ptwm_plane_codec_v1_encode_cuda`.
    ///
    /// On `-2` (buffer too small) the extension has written the required
    /// byte count through `needed_out`; that value is returned in
    /// `CodecError::BufferTooSmall { needed }` rather than the zero the
    /// generic return-code decoder would produce.
    #[allow(clippy::too_many_arguments)]
    pub fn invoke_plane_codec_encode_cuda(
        &self,
        state_bytes: &[u8],
        codec_id: &CanonicalId,
        in_dev_ptr: u64,
        in_len: usize,
        out_dev_ptr: u64,
        out_cap: usize,
        device_ordinal: u32,
    ) -> Result<usize, CodecError> {
        let f = self
            .symbols
            .plane_codec_v1_encode_cuda
            .ok_or(CodecError::Unsupported {
                feature: "plane_codec_v1_encode_cuda not present".into(),
            })?;
        let codec_id_bytes = codec_id.as_bytes();
        let mut needed: u64 = 0;
        let rc = unsafe {
            f(
                1,
                state_bytes.as_ptr(),
                state_bytes.len(),
                codec_id_bytes.as_ptr(),
                codec_id_bytes.len(),
                in_dev_ptr,
                in_len,
                out_dev_ptr,
                out_cap,
                &mut needed as *mut u64,
                device_ordinal,
            )
        };
        match decode_rc(rc, out_cap) {
            Err(CodecError::BufferTooSmall { .. }) => Err(CodecError::BufferTooSmall { needed }),
            other => other,
        }
    }
}

fn decode_rc(rc: i64, out_capacity: usize) -> Result<usize, CodecError> {
    if rc < 0 {
        return Err(match rc {
            -1 => CodecError::InvalidInput,
            -2 => CodecError::BufferTooSmall { needed: 0 },
            -3 => CodecError::MissingCapability {
                name: String::new(),
            },
            -4 => CodecError::MissingNativeDep {
                name: String::new(),
                version_constraint: String::new(),
            },
            -5 => CodecError::Unsupported {
                feature: String::new(),
            },
            _ => CodecError::InternalError,
        });
    }
    let n = rc as usize;
    if n > out_capacity {
        return Err(CodecError::InternalError);
    }
    Ok(n)
}

/// A `plane_codec` may implement the CUDA path or not, but not halfway:
/// exporting `decode_cuda` without `encode_cuda` would resolve, then fail
/// only when something tried to encode. Reject the partial set at load.
fn validate_cuda_symbol_set(present: [bool; 3]) -> Result<(), CodecError> {
    let any = present.iter().any(|p| *p);
    let all = present.iter().all(|p| *p);
    if any && !all {
        return Err(CodecError::Unsupported {
            feature: "partial plane_codec_v1 CUDA symbol set: a codec exporting any \
                      of encode_cuda/decode_cuda/cuda_stream_handle must export all three"
                .into(),
        });
    }
    Ok(())
}

fn resolve<F: Copy>(library: &Library, name: &[u8]) -> Option<F> {
    unsafe { library.get::<F>(name).ok().map(|sym: Symbol<F>| *sym) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extension::{
        Attestation, CapabilityMap, ExtensionTableEntry, Kind, Lifecycle, table::FLAVOR_NATIVE,
    };

    fn entry(kind: Kind) -> ExtensionTableEntry {
        ExtensionTableEntry {
            canonical_id: CanonicalId::from_bytes([0xCD; 32]),
            human_label: "io.test.native".into(),
            kind,
            abi_version: 1,
            lifecycle: Lifecycle::Thread,
            flavor_hints: FLAVOR_NATIVE,
            capabilities: CapabilityMap::new(),
            attestation: Attestation::PgpSignature(Vec::new()),
            install_hint: None,
            embedded_wasm_offset: None,
            embedded_wasm_length: None,
        }
    }

    /// Build a bare `NativeExtension` with every `NativeSymbols` field set
    /// to `None`, for exercising "symbol absent" invoke paths without
    /// dlopen-ing a real contribution artifact from disk. Shared across
    /// the hardware-backend and plane-codec CUDA absent-symbol tests below.
    ///
    /// `Library::this()` wraps a handle to the already-loaded host process
    /// instead of opening a new shared object, which gives a real, valid
    /// `Library` (fn pointers can be safely dropped/never resolved against
    /// it) without depending on any particular file existing on disk.
    fn native_extension_with_no_optional_symbols() -> NativeExtension {
        let library: Library = libloading::os::unix::Library::this().into();
        NativeExtension {
            library,
            canonical_id: CanonicalId::from_bytes([0xCD; 32]),
            kind: Kind::HardwareBackend,
            lifecycle: Lifecycle::Thread,
            symbols: NativeSymbols {
                plane_codec_v1_encode: None,
                plane_codec_v1_decode: None,
                plane_codec_v1_decode_stateful: None,
                plane_codec_v1_encode_cuda: None,
                plane_codec_v1_decode_cuda: None,
                plane_codec_v1_cuda_stream_handle: None,
                transform_v1_forward: None,
                transform_v1_inverse: None,
                delta_scheme_v1_encode: None,
                delta_scheme_v1_decode: None,
                hardware_backend_v1_cuda_stream_handle: None,
                hardware_backend_v1_dispatch_decode_cuda: None,
            },
        }
    }

    #[test]
    fn missing_library_path_is_invalid_input() {
        let res = NativeExtension::load(
            Path::new("/this/path/does/not/exist.so"),
            &entry(Kind::PlaneCodec),
            VerifiedToken::new_unchecked(),
        );
        assert!(matches!(res, Err(CodecError::InvalidInput)));
    }

    #[test]
    fn hardware_backend_invoke_stream_handle_errors_when_symbol_absent() {
        // A NativeExtension whose hardware_backend_v1_cuda_stream_handle symbol
        // was never resolved (None) must return Unsupported, not panic/UB.
        let ext = native_extension_with_no_optional_symbols();
        let res = ext.invoke_hardware_backend_cuda_stream_handle(0);
        assert!(matches!(res, Err(CodecError::Unsupported { .. })));
    }

    #[test]
    fn plane_codec_cuda_invocations_report_absent_symbols() {
        // With no library loaded the CUDA symbols are None; the invoke
        // methods must surface Unsupported (or 0 for the stream-handle
        // probe) rather than panic on unwrap.
        let ext = native_extension_with_no_optional_symbols();
        let codec_id = CanonicalId::from_bytes([0xAB; 32]);

        assert_eq!(ext.plane_codec_cuda_stream_handle(0), 0);
        assert!(matches!(
            ext.invoke_plane_codec_encode_cuda(&[], &codec_id, 0, 0, 0, 0, 0),
            Err(CodecError::Unsupported { .. })
        ));
        assert!(matches!(
            ext.invoke_plane_codec_decode_cuda(&[], &codec_id, 0, 0, 0, 0, 0),
            Err(CodecError::Unsupported { .. })
        ));
    }

    #[test]
    fn cuda_plane_codec_symbol_names_are_the_documented_ones() {
        // Guards against a rename drifting from the plugin-facing contract:
        // these exact strings are what a third-party .so must export.
        assert_eq!(
            PLANE_CODEC_CUDA_ENCODE_SYMBOL,
            b"ptwm_plane_codec_v1_encode_cuda\0"
        );
        assert_eq!(
            PLANE_CODEC_CUDA_DECODE_SYMBOL,
            b"ptwm_plane_codec_v1_decode_cuda\0"
        );
        assert_eq!(
            PLANE_CODEC_CUDA_STREAM_HANDLE_SYMBOL,
            b"ptwm_plane_codec_v1_cuda_stream_handle\0"
        );
    }

    #[test]
    fn a_complete_cuda_symbol_set_is_accepted() {
        assert!(validate_cuda_symbol_set([true, true, true]).is_ok());
    }

    #[test]
    fn no_cuda_symbols_is_accepted_as_a_host_only_codec() {
        assert!(validate_cuda_symbol_set([false, false, false]).is_ok());
    }

    #[test]
    fn a_partial_cuda_symbol_set_is_rejected() {
        for present in [
            [true, false, false],
            [false, true, false],
            [true, true, false],
        ] {
            assert!(
                matches!(
                    validate_cuda_symbol_set(present),
                    Err(CodecError::Unsupported { .. })
                ),
                "partial set {present:?} must be rejected"
            );
        }
    }

    #[test]
    fn hardware_backend_missing_required_symbols_is_unsupported() {
        // A library with neither ptwm_hardware_backend_v1_cuda_stream_handle
        // nor ptwm_hardware_backend_v1_dispatch_decode_cuda must fail to load
        // for Kind::HardwareBackend, same strictness as PlaneCodec/DeltaScheme.
        // This test is a placeholder for Task 11's real artifact-backed test;
        // the path-doesn't-exist error fires before symbol probing, so the test
        // passes trivially for now.
        let res = NativeExtension::load(
            Path::new("/this/path/does/not/exist.so"),
            &entry(Kind::HardwareBackend),
            VerifiedToken::new_unchecked(),
        );
        assert!(matches!(res, Err(CodecError::InvalidInput)));
    }
}
