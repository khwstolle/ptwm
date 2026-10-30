#![no_main]

use std::num::NonZero;

use libfuzzer_sys::fuzz_target;
use ptwm_core::codec::PlaneCodec;
use ptwm_core::codecs::order1_scale_ac::{
    ALPHABET, Order1ScaleAC, STATE_FORMAT_VERSION, StateV0, serialize_state,
};
use ptwm_core::layout::PlaneLayout;
use ptwm_core::quantize::{ORDER1_QUANT_TOTAL, quantize_pmf};

/// Header: decoded_len (u32 LE), row_len (u16 LE, 0 = flat), the `prev`
/// byte that gets a conditional row, then 256 count bytes.
const HEADER: usize = 4 + 2 + 1 + ALPHABET;

fuzz_target!(|data: &[u8]| {
    if data.len() < HEADER {
        return;
    }
    let (head, payload) = data.split_at(HEADER);
    let decoded_len = u32::from_le_bytes([head[0], head[1], head[2], head[3]]) as usize;
    let row_len = u16::from_le_bytes([head[4], head[5]]);
    let cond_prev = head[6];
    let counts_bytes = &head[7..];

    // Build a valid state from the input so decode gets past state
    // validation and exercises the capacity bound and the coder loop.
    // Squared counts allow PMFs from near-uniform to maximally skewed.
    let counts: Vec<u32> = counts_bytes.iter().map(|&b| (b as u32) * (b as u32)).collect();
    let rotated: Vec<u32> = counts.iter().cycle().skip(1).take(ALPHABET).copied().collect();
    let marginal: [u16; ALPHABET] = quantize_pmf(&counts, ORDER1_QUANT_TOTAL)
        .try_into()
        .unwrap();
    let row: [u16; ALPHABET] = quantize_pmf(&rotated, ORDER1_QUANT_TOTAL)
        .try_into()
        .unwrap();
    let mut bitmap = [0u8; 32];
    bitmap[(cond_prev / 8) as usize] |= 1 << (cond_prev % 8);
    let Ok(state) = StateV0::from_parts(bitmap, marginal, vec![row]) else {
        return;
    };
    let state_bytes = serialize_state(&state);

    let layout = match NonZero::new(row_len as u32) {
        Some(row_len) => PlaneLayout::Rows { row_len },
        None => PlaneLayout::Flat,
    };
    if let Ok(out) = Order1ScaleAC.decode(
        STATE_FORMAT_VERSION,
        &state_bytes,
        payload,
        &layout,
        decoded_len,
    ) {
        assert_eq!(out.len(), decoded_len);
    }
});
