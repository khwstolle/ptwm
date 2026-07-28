//! Reference contribution for the `hardware_backend` kind (CPU passthrough).
//!
//! WASM ABI: ptwm_hardware_backend_v1_{init,cleanup,dispatch_decode}
//! Native ABI: ptwm_hardware_backend_v1_{cuda_stream_handle,dispatch_decode_cuda}
//! Passthrough: copies input to output, ignoring the plane_codec_id.
#![cfg_attr(target_arch = "wasm32", no_main)]

use std::alloc::Layout;

// ---------------------------------------------------------------------------
// Allocator exports (WASM only).
// ---------------------------------------------------------------------------

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn ptwm_alloc(len: u32) -> u32 {
    let layout = Layout::from_size_align(len as usize, 1).unwrap();
    unsafe { std::alloc::alloc(layout) as u32 }
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn ptwm_free(ptr: u32, len: u32) {
    let layout = Layout::from_size_align(len as usize, 1).unwrap();
    unsafe { std::alloc::dealloc(ptr as *mut u8, layout) }
}

// ---------------------------------------------------------------------------
// WASM: Lifecycle and dispatch_decode.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn ptwm_hardware_backend_v1_init() -> u32 {
    1
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn ptwm_hardware_backend_v1_cleanup(_state: u32) {}

#[cfg(target_arch = "wasm32")]
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
    if out_len < in_len {
        return -2;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(in_ptr as *const u8, out_ptr as *mut u8, in_len as usize);
    }
    in_len as i64
}

// ---------------------------------------------------------------------------
// Native: CUDA-shaped exports (CPU-passthrough for testing).
// ---------------------------------------------------------------------------

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
