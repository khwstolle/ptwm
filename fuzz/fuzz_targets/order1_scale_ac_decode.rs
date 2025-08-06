#![no_main]

use std::num::NonZero;

use libfuzzer_sys::fuzz_target;
use ptwm_core::codec::PlaneCodec;
use ptwm_core::codecs::order1_scale_ac::Order1ScaleAC;
use ptwm_core::layout::PlaneLayout;

fuzz_target!(|data: &[u8]| {
    if data.len() < 33 {
        return;
    }
    // Split input into state bytes + payload. Accept any split ≥ 33 bytes to
    // maximise coverage of reject paths without requiring a structurally valid
    // state. A valid empty state is 32 + 512 = 544 bytes; we stay below that
    // threshold on most inputs to exercise error-return paths.
    let state_len = (32 + 512).min(data.len());
    let (state, payload) = data.split_at(state_len);
    let codec = Order1ScaleAC;
    let layout = PlaneLayout::Rows {
        row_len: NonZero::new(16).unwrap(),
    };
    let _ = codec.decode(0, state, payload, &layout, 64);
});
