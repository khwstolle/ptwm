//! {{ name }} — {{ description }}
//!
//! PTWM delta_scheme contribution, ABI v1: `encode(base, target) -> delta`,
//! `decode(base, delta) -> target`. This scaffold is a passthrough —
//! encode copies `target` (ignoring `base`), decode copies `delta`
//! (ignoring `base`) — replace the two `TODO` bodies below with real
//! delta-encoding logic.
//!
//! Built for either target, selected by `ptwm ext build --flavor`:
//! - `wasm32-wasip1` (default): threads an opaque state handle through
//!   `ptwm_delta_scheme_v1_{init,cleanup,encode,decode}`, since WASM
//!   linear memory forces the host to manage allocation explicitly.
//! - native (`cargo build --release`, plain cdylib): no state handle —
//!   manage any internal state as ordinary process-local Rust state.

#![no_main]

#[cfg(target_arch = "wasm32")]
mod wasm_abi {
    #[unsafe(no_mangle)]
    pub extern "C" fn ptwm_alloc(len: u32) -> u32 {
        let layout = std::alloc::Layout::from_size_align(len as usize, 1).unwrap();
        unsafe { std::alloc::alloc(layout) as u32 }
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn ptwm_free(ptr: u32, len: u32) {
        let layout = std::alloc::Layout::from_size_align(len as usize, 1).unwrap();
        unsafe { std::alloc::dealloc(ptr as *mut u8, layout) }
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn ptwm_delta_scheme_v1_init() -> u32 {
        1
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn ptwm_delta_scheme_v1_cleanup(_state: u32) {}

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
        // TODO: replace with real delta-encoding logic.
        unsafe {
            std::ptr::copy_nonoverlapping(
                target_ptr as *const u8,
                out_ptr as *mut u8,
                target_len as usize,
            );
        }
        target_len as i64
    }

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
        // TODO: replace with real delta-decoding logic (the inverse of encode).
        unsafe {
            std::ptr::copy_nonoverlapping(
                delta_ptr as *const u8,
                out_ptr as *mut u8,
                delta_len as usize,
            );
        }
        delta_len as i64
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod native_abi {
    #[unsafe(no_mangle)]
    pub extern "C" fn ptwm_delta_scheme_v1_encode(
        _base_ptr: *const u8,
        _base_len: usize,
        target_ptr: *const u8,
        target_len: usize,
        out_ptr: *mut u8,
        out_len: usize,
    ) -> i64 {
        if out_len < target_len {
            return -2;
        }
        // TODO: replace with real delta-encoding logic.
        unsafe {
            std::ptr::copy_nonoverlapping(target_ptr, out_ptr, target_len);
        }
        target_len as i64
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn ptwm_delta_scheme_v1_decode(
        _base_ptr: *const u8,
        _base_len: usize,
        delta_ptr: *const u8,
        delta_len: usize,
        out_ptr: *mut u8,
        out_len: usize,
    ) -> i64 {
        if out_len < delta_len {
            return -2;
        }
        // TODO: replace with real delta-decoding logic (the inverse of encode).
        unsafe {
            std::ptr::copy_nonoverlapping(delta_ptr, out_ptr, delta_len);
        }
        delta_len as i64
    }
}
