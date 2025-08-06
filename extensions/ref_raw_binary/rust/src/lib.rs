//! Reference contribution for the `raw_binary` kind (passthrough).
//!
//! WASM ABI: ptwm_raw_binary_v1_{init,cleanup,encode,decode}
//! Same shape as plane_codec. Ships with an empty schema.cbor placeholder.
//! Behavioral invariant: decode(encode(x)) == x for all x.
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
pub extern "C" fn ptwm_raw_binary_v1_init() -> u32 {
    1
}

#[unsafe(no_mangle)]
pub extern "C" fn ptwm_raw_binary_v1_cleanup(_state: u32) {}

// ---------------------------------------------------------------------------
// Passthrough encode.
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn ptwm_raw_binary_v1_encode(
    _state: u32,
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
// Passthrough decode.
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn ptwm_raw_binary_v1_decode(
    _state: u32,
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
