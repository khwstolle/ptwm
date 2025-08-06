#![no_main]

use libfuzzer_sys::fuzz_target;
use ptwm_core::extension::ExtensionTable;

fuzz_target!(|data: &[u8]| {
    let _ = ExtensionTable::from_bytes(data);
});
