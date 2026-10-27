#![no_main]

// `Decompressor.decompress` hands any blob that starts with the compact-frame
// magic to `decode_compact`. The parser must never panic or infinite-loop on
// adversarial input.

use libfuzzer_sys::fuzz_target;
use ptwm_core::compact::{FLAG_SHAPE, FLAG_SIZE, MAGIC_COMPACT, decode_compact, size_from_shape};
use ptwm_core::varint::read_varint;

/// Raw byte count the frame declares, from `orig_size` or the shape. Layout
/// after the magic: flags u8, dtype_code varint, input_format u8,
/// [orig_size varint] (FLAG_SIZE), [ndim varint, dims varint...]
/// (FLAG_SHAPE). `None` when the header does not parse that far.
fn declared_size(data: &[u8]) -> Option<u64> {
    let flags = *data.first()?;
    let mut pos = 1;
    let dtype_code = u16::try_from(read_varint(data, &mut pos).ok()?).ok()?;
    pos += 1;
    if flags & FLAG_SIZE != 0 {
        return read_varint(data, &mut pos).ok();
    }
    if flags & FLAG_SHAPE == 0 {
        return None;
    }
    let ndim = read_varint(data, &mut pos).ok()?.min(64);
    let mut shape = Vec::new();
    for _ in 0..ndim {
        shape.push(read_varint(data, &mut pos).ok()?);
    }
    // An unknown dtype or an overflowing product fails in the decoder.
    Some(size_from_shape(&shape, dtype_code).unwrap_or(0))
}

fuzz_target!(|data: &[u8]| {
    // The frame declares its decoded size, and a decoder must be able to
    // allocate it: a 100-byte frame of a constant tensor can declare 1 GB.
    // Skip declared sizes above 16 MiB so libFuzzer's RSS limit flags real
    // over-allocation only.
    if declared_size(data).is_some_and(|n| n > 1 << 24) {
        return;
    }
    // Prefix the magic so the fuzzer spends its time past the magic check.
    let mut buf = Vec::with_capacity(MAGIC_COMPACT.len() + data.len());
    buf.extend_from_slice(&MAGIC_COMPACT);
    buf.extend_from_slice(data);
    let _ = decode_compact(&buf);
});
