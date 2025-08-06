#![no_main]

use libfuzzer_sys::fuzz_target;
use ptwm_core::codec::PlaneCodec;
use ptwm_core::codecs::rans::Rans;

fuzz_target!(|data: &[u8]| {
    let codec = Rans;
    let _ = codec.decode(
        0,
        &[],
        data,
        &ptwm_core::layout::PlaneLayout::Flat,
        data.len(),
    );
});
