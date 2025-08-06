//! {{ name }} — {{ description }}
//!
//! PTWM {{ kind }} contribution, ABI v1.
//! Built for wasm32-wasip1; see PTWM extension docs.

#![no_main]

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
pub extern "C" fn ptwm_{{ kind }}_v1_encode(
    in_ptr: u32, in_len: u32, out_ptr: u32, out_len: u32,
) -> i64 {
    let input = unsafe { std::slice::from_raw_parts(in_ptr as *const u8, in_len as usize) };
    let output = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut u8, out_len as usize) };
    if output.len() < input.len() {
        return -2;
    }
    output[..input.len()].copy_from_slice(input);
    input.len() as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn ptwm_{{ kind }}_v1_decode(
    in_ptr: u32, in_len: u32, out_ptr: u32, out_len: u32,
) -> i64 {
    let input = unsafe { std::slice::from_raw_parts(in_ptr as *const u8, in_len as usize) };
    let output = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut u8, out_len as usize) };
    if output.len() < input.len() {
        return -2;
    }
    output[..input.len()].copy_from_slice(input);
    input.len() as i64
}
