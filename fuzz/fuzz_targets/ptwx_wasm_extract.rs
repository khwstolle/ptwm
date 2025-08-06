#![no_main]

use libfuzzer_sys::fuzz_target;
use ptwm_core::extension::ExtensionTable;
use ptwm_core::header::{HEADER_LEN, Header};

fuzz_target!(|data: &[u8]| {
    if data.len() < HEADER_LEN {
        return;
    }
    let Ok(header) = Header::from_bytes(&data[..HEADER_LEN]) else {
        return;
    };
    if !header.is_ptwx() {
        return;
    }

    let off = header.extension_table_offset as usize;
    let len = header.extension_table_length as usize;
    let Some(end) = off.checked_add(len) else {
        return;
    };
    if end > data.len() {
        return;
    }

    let Ok(table) = ExtensionTable::from_bytes(&data[off..end]) else {
        return;
    };
    for entry in &table.entries {
        let Some(boff) = entry.embedded_wasm_offset else {
            continue;
        };
        let Some(blen) = entry.embedded_wasm_length else {
            continue;
        };
        let Some(bend) = (boff as usize).checked_add(blen as usize) else {
            continue;
        };
        if bend <= data.len() {
            // Don't execute — just slice the bytes to make sure the
            // offset arithmetic doesn't panic.
            let _wasm_bytes = &data[boff as usize..bend];
        }
    }
});
