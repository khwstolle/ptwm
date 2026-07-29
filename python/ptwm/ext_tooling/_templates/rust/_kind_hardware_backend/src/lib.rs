//! {{ name }}: a hardware_backend contribution scaffold.
//!
//! The native branch below exports the real device-pointer-shaped ABI
//! (ptwm_hardware_backend_v1_cuda_stream_handle,
//! ptwm_hardware_backend_v1_dispatch_decode_cuda) with a CPU-passthrough
//! body: it reinterprets the u64 pointers as host pointers and memcpy's.
//! This works as a real dispatch-path exercise, not an approximation,
//! because ptwm-core never dereferences these pointers itself; it only
//! forwards them. Replace the body with real CUDA kernel launches for a
//! production backend.
//!
//! The wasm branch below is a CPU-only reference kept for structural
//! parity; it is NOT reachable through HardwareBackendRouter. Native-only
//! by design: a u64 device pointer cannot be expressed in WASM's 32-bit
//! linear address space.

#[cfg(target_arch = "wasm32")]
mod wasm_ref {
    #[unsafe(no_mangle)]
    pub extern "C" fn ptwm_hardware_backend_v1_init() -> u32 {
        1
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn ptwm_hardware_backend_v1_cleanup(_state: u32) {}

    #[unsafe(no_mangle)]
    pub extern "C" fn ptwm_hardware_backend_v1_dispatch_decode(
        _state: u32,
        _codec_id_ptr: u32,
        _codec_id_len: u32,
        in_ptr: u32,
        in_len: u32,
        out_ptr: u32,
        out_len: u32,
    ) -> i64 {
        // Passthrough copy within WASM linear memory, matching
        // extensions/ref_hardware_backend/rust/src/lib.rs's existing shape.
        let n = in_len.min(out_len) as usize;
        unsafe {
            std::ptr::copy_nonoverlapping(in_ptr as *const u8, out_ptr as *mut u8, n);
        }
        n as i64
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    #[unsafe(no_mangle)]
    pub extern "C" fn ptwm_hardware_backend_v1_cuda_stream_handle(_device_ordinal: u32) -> u64 {
        // CPU-passthrough scaffold: no real CUDA stream. Return a nonzero
        // sentinel so callers treat this as "succeeded" for dispatch-path
        // smoke-testing purposes; a real CUDA backend returns a genuine
        // CUstream/cudaStream_t pointer value here.
        1
    }

    #[unsafe(no_mangle)]
    #[allow(clippy::too_many_arguments)]
    pub extern "C" fn ptwm_hardware_backend_v1_dispatch_decode_cuda(
        _state_format_version: u8,
        _state_ptr: *const u8,
        _state_len: usize,
        _codec_id_ptr: *const u8,
        _codec_id_len: usize,
        in_dev_ptr: u64,
        in_len: usize,
        out_dev_ptr: u64,
        out_len: usize,
        _device_ordinal: u32,
    ) -> i64 {
        // Reinterprets the u64 "device" pointers as ordinary host
        // pointers and memcpy's. This is legal ONLY because ptwm-core
        // never dereferences these pointers itself; it exists purely to
        // exercise the real dispatch path (router, native ABI, dlopen,
        // FFI call) with zero GPU/CUDA present, matching this repo's
        // testing plan.
        let n = in_len.min(out_len);
        unsafe {
            std::ptr::copy_nonoverlapping(
                in_dev_ptr as *const u8,
                out_dev_ptr as *mut u8,
                n,
            );
        }
        n as i64
    }
}
