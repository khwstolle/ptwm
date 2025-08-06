#![no_main]

use libfuzzer_sys::fuzz_target;
use ptwm_core::codec::PlaneCodec;
use ptwm_core::codecs::huffman::Huffman;

fuzz_target!(|data: &[u8]| {
    // Decoder must never panic on arbitrary inputs, regardless of the
    // declared `state_format_version`. Empty `state_bytes` matches what
    // the real Huffman codec emits today.
    let codec = Huffman;
    let _ = codec.decode(
        0,
        &[],
        data,
        &ptwm_core::layout::PlaneLayout::Flat,
        data.len(),
    );
});
