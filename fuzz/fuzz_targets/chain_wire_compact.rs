#![no_main]

// CW2, the compact chain wire, is read from untrusted `\x89PTWS` frames (and
// compact containers). The parser must never panic, infinite-loop or
// allocate unboundedly, and every chain it accepts must re-encode to the
// same bytes (the format is canonical).
//
// Input layout: byte 0 selects the bound Source (bit 0 set: none; else
// params for a [3, 5] fp32 tensor); the rest is the chain.

use libfuzzer_sys::fuzz_target;
use ptwm_core::chain::validate::validate_chain;
use ptwm_core::chain::wire_compact::{
    op_from_wire_ref, op_wire_ref, read_chain_compact_exact, write_chain_compact,
};
use ptwm_core::transforms::source::bound_source_params;

fuzz_target!(|data: &[u8]| {
    let Some((&sel, chain_bytes)) = data.split_first() else {
        return;
    };
    let params = bound_source_params(&[3, 5], 1).expect("fp32 binds");
    let bound = (sel & 1 == 0).then_some(params.as_slice());
    let Ok(chain) = read_chain_compact_exact(chain_bytes, bound, op_from_wire_ref) else {
        return;
    };
    let mut out = Vec::new();
    write_chain_compact(&chain, bound, op_wire_ref, &mut out).expect("accepted chain re-encodes");
    assert_eq!(out, chain_bytes, "CW2 is not canonical");
    let _ = validate_chain(&chain, 0);
});
