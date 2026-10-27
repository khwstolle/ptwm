#![no_main]

// `Decompressor.decompress` hands any blob that starts with the compact-frame
// magic to `decode_compact`. The parser must never panic or infinite-loop on
// adversarial input.

use libfuzzer_sys::fuzz_target;
use ptwm_core::compact::{MAGIC_COMPACT, decode_compact};

/// Read one LEB128 varint at `*pos`; `None` when truncated or too long.
fn varint(data: &[u8], pos: &mut usize) -> Option<u64> {
    let mut v = 0u64;
    for i in 0..10 {
        let b = *data.get(*pos)?;
        *pos += 1;
        v |= u64::from(b & 0x7F) << (7 * i);
        if b & 0x80 == 0 {
            return Some(v);
        }
    }
    None
}

fuzz_target!(|data: &[u8]| {
    // The frame declares its decoded size, and a decoder must be able to
    // allocate it: a 100-byte frame of a constant tensor can declare 1 GB.
    // Skip declared sizes above 16 MiB so libFuzzer's RSS limit flags real
    // over-allocation only. Layout after the magic: flags u8, dtype varint,
    // input_format u8, orig_size varint.
    let mut pos = 1;
    if varint(data, &mut pos).is_some() {
        pos += 1;
        if varint(data, &mut pos).is_some_and(|n| n > 1 << 24) {
            return;
        }
    }
    // Prefix the magic so the fuzzer spends its time past the magic check.
    let mut buf = Vec::with_capacity(MAGIC_COMPACT.len() + data.len());
    buf.extend_from_slice(&MAGIC_COMPACT);
    buf.extend_from_slice(data);
    let _ = decode_compact(&buf);
});
