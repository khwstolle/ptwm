//! Reference contribution for the `delta_scheme` kind (passthrough).
//!
//! WASM ABI: ptwm_delta_scheme_v1_{init,cleanup,encode,decode}
//! encode: copies `target` to output (ignores `base`).
//! decode: copies `delta` to output (ignores `base`).
//! Behavioral invariant: decode(encode(base, target)) == target for all inputs.
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
pub extern "C" fn ptwm_delta_scheme_v1_init() -> u32 {
    1
}

#[unsafe(no_mangle)]
pub extern "C" fn ptwm_delta_scheme_v1_cleanup(_state: u32) {}

// ---------------------------------------------------------------------------
// encode(base, target) → copies target unchanged.
// Signature: (state, base_ptr, base_len, target_ptr, target_len, out_ptr, out_len) -> i64
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn ptwm_delta_scheme_v1_encode(
    _state: u32,
    _base_ptr: u32,
    _base_len: u32,
    target_ptr: u32,
    target_len: u32,
    out_ptr: u32,
    out_len: u32,
) -> i64 {
    if out_len < target_len {
        return -2;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(
            target_ptr as *const u8,
            out_ptr as *mut u8,
            target_len as usize,
        );
    }
    target_len as i64
}

// ---------------------------------------------------------------------------
// decode(base, delta) → copies delta unchanged.
// Signature: (state, base_ptr, base_len, delta_ptr, delta_len, out_ptr, out_len) -> i64
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn ptwm_delta_scheme_v1_decode(
    _state: u32,
    _base_ptr: u32,
    _base_len: u32,
    delta_ptr: u32,
    delta_len: u32,
    out_ptr: u32,
    out_len: u32,
) -> i64 {
    if out_len < delta_len {
        return -2;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(
            delta_ptr as *const u8,
            out_ptr as *mut u8,
            delta_len as usize,
        );
    }
    delta_len as i64
}
