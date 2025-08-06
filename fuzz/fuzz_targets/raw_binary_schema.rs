#![no_main]

use libfuzzer_sys::fuzz_target;

// The raw_binary schema parser doesn't have a dedicated Rust function
// yet (the format is specified, but the parser/validator lands with the
// actual raw_binary dispatcher). For v1 this fuzzes the closest available
// shape: CBOR decoding of a CapabilityMap, which reaches into ciborium
// with arbitrary inputs.
use ptwm_core::extension::CapabilityMap;

fuzz_target!(|data: &[u8]| {
    let _ = CapabilityMap::from_cbor(data);
});
