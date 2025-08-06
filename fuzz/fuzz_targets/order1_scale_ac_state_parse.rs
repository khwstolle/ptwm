#![no_main]

use libfuzzer_sys::fuzz_target;
use ptwm_core::codecs::order1_scale_ac::deserialize_state;

fuzz_target!(|data: &[u8]| {
    // `deserialize_state` must never panic on any byte input.
    // All invalid inputs must be rejected via `Err(_)`.
    let _ = deserialize_state(data);
});
