#![no_main]

use libfuzzer_sys::fuzz_target;
use ptwm_core::container::{ContainerLayout, ContainerReader};

fuzz_target!(|data: &[u8]| {
    // Both layouts (\x89PTWM/\x89PTWX and \x89PTWC): open, then decode every
    // tensor. Neither may panic or allocate without bound.
    let Ok(reader) = ContainerReader::open(data) else {
        return;
    };
    let _ = reader.decode_model();
    if reader.layout == ContainerLayout::Compact {
        // Names come from the index; every record must parse or fail cleanly.
        if let Ok(names) = reader.tensor_names() {
            for name in names {
                let _ = reader.tensor_record(&name);
            }
        }
    }
});
