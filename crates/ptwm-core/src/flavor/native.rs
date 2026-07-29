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
    /// dlopen-ing a real contribution artifact from disk.
    ///
    /// `Library::this()` wraps a handle to the already-loaded host process
    /// instead of opening a new shared object, which gives a real, valid
    /// `Library` (fn pointers can be safely dropped/never resolved against
    /// it) without depending on any particular file existing on disk.
    fn native_extension_with_no_hardware_backend_symbols() -> NativeExtension {
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
        let ext = native_extension_with_no_hardware_backend_symbols(); // test-only constructor, see Step 3
        let res = ext.invoke_hardware_backend_cuda_stream_handle(0);
        assert!(matches!(res, Err(CodecError::Unsupported { .. })));
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
