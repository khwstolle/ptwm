#![no_main]

// Adversarial chain wire blobs are accepted by `ptwm chains import` after
// `is_legal_chain` validation. The wire parser is the boundary between
// untrusted input and that validator, so it must never panic, infinite-loop,
// or allocate unboundedly.

use libfuzzer_sys::fuzz_target;
use ptwm_core::chain::wire::read_chain_blob;

fuzz_target!(|data: &[u8]| {
    let _ = read_chain_blob(data);
});
