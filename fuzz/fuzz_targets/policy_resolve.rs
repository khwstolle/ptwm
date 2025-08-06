#![no_main]

use std::collections::HashSet;
use std::path::Path;

use libfuzzer_sys::fuzz_target;
use ptwm_core::policy::{PolicyFile, resolve};

fuzz_target!(|data: &[u8]| {
    let Ok(s) = std::str::from_utf8(data) else {
        return;
    };
    let Ok(pf) = PolicyFile::from_toml(s) else {
        return;
    };
    let _ = resolve(&pf, &HashSet::new(), Path::new("."));
});
