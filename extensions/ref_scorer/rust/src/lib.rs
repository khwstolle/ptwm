//! Reference contribution for the `scorer` kind.
//!
//! WASM ABI: ptwm_scorer_v1_{init,cleanup,score}
//! Stub: always writes 0.0 to the out_f64_ptr and returns 0 (success).
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
pub extern "C" fn ptwm_scorer_v1_init() -> u32 {
    1
}

#[unsafe(no_mangle)]
pub extern "C" fn ptwm_scorer_v1_cleanup(_state: u32) {}

// ---------------------------------------------------------------------------
// score: writes 0.0f64 to *out_f64_ptr and returns 0.
//
// Signature: (state, ptr, len, out_f64_ptr) -> i64
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn ptwm_scorer_v1_score(_state: u32, _ptr: u32, _len: u32, out_f64_ptr: u32) -> i64 {
    unsafe {
        let out = out_f64_ptr as *mut f64;
        out.write_unaligned(0.0_f64);
    }
    0 // success; byte count is not meaningful for scorer
}
