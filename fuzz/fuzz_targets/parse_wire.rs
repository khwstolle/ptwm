#![no_main]

use libfuzzer_sys::fuzz_target;
use ptwm_core::tensor_record::parse_tensor_record;

fuzz_target!(|data: &[u8]| {
    let _ = parse_tensor_record(data);
});
