//! Four byte-history context models (orders 0–3). Each maps a context —
//! a hash of recent whole bytes combined with the partial-byte register
//! `c0` — to an adaptive 12-bit probability counter in a fixed hash table.

/// Number of context models.
pub const N_MODELS: usize = 4;

/// log2 of each model's counter table. 2^16 u16 counters = 128 KiB/model.
const TABLE_BITS: u32 = 16;
const TABLE_SIZE: usize = 1 << TABLE_BITS;
const TABLE_MASK: u32 = (TABLE_SIZE as u32) - 1;

/// Counter adaptation rate (shift). Smaller = faster adaptation.
const COUNTER_RATE: u16 = 5;

/// Knuth multiplicative hash constant (`floor(2^32 / phi)`), for spreading
/// an order-N history hash across the counter table.
const KNUTH_HASH: u32 = 0x9E37_79B1;

/// Move a 12-bit probability counter (`0..=4095`, stored in u16) toward the
/// observed `bit` (target 4095 for `bit == 1`, 0 for `bit == 0`).
#[inline]
pub fn update_counter(p: &mut u16, bit: u8) {
    let target: i32 = if bit == 1 { 4095 } else { 0 };
    let cur = *p as i32;
    *p = (cur + ((target - cur) >> COUNTER_RATE)) as u16;
}

/// The four models. The counter tables share one flat `Vec<u16>` of
/// `N_MODELS * TABLE_SIZE` (~512 KiB, one allocation, no double indirection in
/// the per-bit hot path); model `m` owns `[m*TABLE_SIZE, (m+1)*TABLE_SIZE)`.
/// Allocated fresh per encode/decode call — CM is opt-in and
/// per-bit-compute-dominated, so the once-per-plane allocation is negligible.
pub struct Models {
    tables: Vec<u16>,
    /// Rolling per-order history hashes of recent whole bytes (orders 1..=3).
    hist: [u32; N_MODELS],
    /// Recent bytes ring for rebuilding order hashes.
    bytes: [u8; 3],
    /// Per-bit flat slot indices chosen at predict(), reused at update().
    slots: [usize; N_MODELS],
}

impl Models {
    pub fn new() -> Self {
        Self {
            tables: vec![2048u16; N_MODELS * TABLE_SIZE],
            hist: [0; N_MODELS],
            bytes: [0; 3],
            slots: [0; N_MODELS],
        }
    }

    /// Predict `P(bit=1)` (12-bit) for each model under partial byte `c0`.
    /// Records the chosen flat table slots for the matching [`Self::update`].
    pub fn predict(&mut self, c0: u32) -> [i32; N_MODELS] {
        let mut out = [0i32; N_MODELS];
        for m in 0..N_MODELS {
            let idx =
                (self.hist[m].wrapping_mul(KNUTH_HASH).wrapping_add(c0) & TABLE_MASK) as usize;
            let slot = m * TABLE_SIZE + idx;
            self.slots[m] = slot;
            out[m] = self.tables[slot] as i32;
        }
        out
    }

    /// Update every model's last-predicted counter toward `bit`.
    pub fn update(&mut self, _c0: u32, bit: u8) {
        for m in 0..N_MODELS {
            update_counter(&mut self.tables[self.slots[m]], bit);
        }
    }

    /// Advance the per-order history hashes after a whole byte is coded.
    pub fn push_byte(&mut self, byte: u8) {
        self.bytes[2] = self.bytes[1];
        self.bytes[1] = self.bytes[0];
        self.bytes[0] = byte;
        // order-0 sees no whole-byte history.
        self.hist[0] = 0;
        self.hist[1] = self.bytes[0] as u32;
        self.hist[2] = (self.bytes[0] as u32) | ((self.bytes[1] as u32) << 8);
        self.hist[3] =
            (self.bytes[0] as u32) | ((self.bytes[1] as u32) << 8) | ((self.bytes[2] as u32) << 16);
    }
}

impl Default for Models {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_moves_toward_observed_bit() {
        let mut p: u16 = 2048;
        for _ in 0..50 {
            update_counter(&mut p, 1);
        }
        assert!(p > 3500, "counter should approach 1, got {p}");
        for _ in 0..200 {
            update_counter(&mut p, 0);
        }
        assert!(p < 500, "counter should approach 0, got {p}");
    }

    #[test]
    fn models_predict_in_range_and_update_deterministically() {
        let data = b"the quick brown fox the quick brown fox the quick";
        // Two independent passes over the same bytes must yield identical
        // prediction sequences (determinism / lockstep).
        let seq1 = collect_predictions(data);
        let seq2 = collect_predictions(data);
        assert_eq!(seq1, seq2);
        for &p in &seq1 {
            assert!(
                (0..=4095).contains(&p),
                "prediction {p} out of 12-bit range"
            );
        }
    }

    // Drive the models like the coder will, recording the mixed-free raw
    // model[0] prediction per bit, to assert determinism.
    fn collect_predictions(data: &[u8]) -> Vec<i32> {
        let mut models = Models::new();
        let mut out = Vec::new();
        for &byte in data {
            let mut c0: u32 = 1;
            for i in (0..8).rev() {
                let bit = (byte >> i) & 1;
                let preds = models.predict(c0);
                out.push(preds[0]);
                models.update(c0, bit);
                c0 = (c0 << 1) | bit as u32;
            }
            models.push_byte(byte);
        }
        out
    }
}
