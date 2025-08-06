//! Reference contribution for the `transform` kind (passthrough / identity).
//!
//! WASM ABI: ptwm_transform_v1_{init,cleanup,forward,inverse}
//! Behavioral invariant: inverse(forward(x)) == x for all x.
#![no_main]

use std::alloc::Layout;

// ---------------------------------------------------------------------------
// Allocator exports required by every PTWM WASM contribution.
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
// Lifecycle (thread): init/cleanup return a dummy opaque handle (1 = ok).
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn ptwm_transform_v1_init() -> u32 {
    1 // non-null opaque handle
}

#[unsafe(no_mangle)]
pub extern "C" fn ptwm_transform_v1_cleanup(_state: u32) {}

// ---------------------------------------------------------------------------
// Passthrough forward: copy input bytes to output unchanged.
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn ptwm_transform_v1_forward(
    _state: u32,
    in_ptr: u32,
    in_len: u32,
    out_ptr: u32,
    out_len: u32,
) -> i64 {
    if out_len < in_len {
        return -2; // BufferTooSmall
    }
    unsafe {
        std::ptr::copy_nonoverlapping(in_ptr as *const u8, out_ptr as *mut u8, in_len as usize);
    }
    in_len as i64
}

// ---------------------------------------------------------------------------
// Passthrough inverse: identity.
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn ptwm_transform_v1_inverse(
    _state: u32,
    in_ptr: u32,
    in_len: u32,
    out_ptr: u32,
    out_len: u32,
) -> i64 {
    if out_len < in_len {
        return -2; // BufferTooSmall
    }
    unsafe {
        std::ptr::copy_nonoverlapping(in_ptr as *const u8, out_ptr as *mut u8, in_len as usize);
    }
    in_len as i64
}
