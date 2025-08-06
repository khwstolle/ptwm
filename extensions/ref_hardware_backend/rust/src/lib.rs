//! Reference contribution for the `hardware_backend` kind (CPU passthrough).
//!
//! WASM ABI: ptwm_hardware_backend_v1_{init,cleanup,dispatch_decode}
//! Passthrough: copies input to output, ignoring the plane_codec_id.
#![no_main]

use std::alloc::Layout;

// ---------------------------------------------------------------------------
// Allocator exports.
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn ptwm_alloc(len: u32) -> u32 {
    let layout = Layout::from_size_align(len as usize, 1).unwrap();
    unsafe { std::alloc::alloc(layout) as u32 }
}

#[unsafe(no_mangle)]
pub extern "C" fn ptwm_free(ptr: u32, len: u32) {
    let layout = Layout::from_size_align(len as usize, 1).unwrap();
    unsafe { std::alloc::dealloc(ptr as *mut u8, layout) }
}

// ---------------------------------------------------------------------------
// Lifecycle.
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn ptwm_hardware_backend_v1_init() -> u32 {
    1
}

#[unsafe(no_mangle)]
pub extern "C" fn ptwm_hardware_backend_v1_cleanup(_state: u32) {}

// ---------------------------------------------------------------------------
// dispatch_decode: passthrough — copies input to output.
//
// Signature: (state, plane_codec_id_ptr, plane_codec_id_len,
//             in_ptr, in_len, out_ptr, out_len) -> i64
// ---------------------------------------------------------------------------

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
