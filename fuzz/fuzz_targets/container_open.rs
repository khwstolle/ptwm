#![no_main]

use libfuzzer_sys::fuzz_target;
use ptwm_core::container::ContainerReader;

fuzz_target!(|data: &[u8]| {
    let _ = ContainerReader::open(data);
});
