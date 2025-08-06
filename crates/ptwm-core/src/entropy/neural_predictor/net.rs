//! Tiny online-learning neural predictor: a trainable per-context embedding
//! feeds a ReLU hidden layer and a linear output. Pure integer/fixed-point and
//! fully deterministic, so encode and decode evolve in lockstep — losslessness
//! does NOT depend on the net compressing well. No weights cross the wire.

use crate::entropy::context_mixing::squash::squash;

const N_CTX_BITS: u32 = 15;
const N_CTX: usize = 1 << N_CTX_BITS;
const CTX_MASK: u32 = (N_CTX as u32) - 1;
const E_DIM: usize = 8;
const H: usize = 32;

/// Weight fixed-point shift (W1/W2 are 16.16, like the cmix mixer).
const W_SHIFT: u32 = 16;
/// Leaky-ReLU negative-branch shift (slope 1/64) — derivative is table-free.
const LEAK_SHIFT: u32 = 6;
/// Learning-rate shifts (larger = slower). Tuned against the compresses-below-raw test.
const LR2_SHIFT: u32 = 13; // output weights W2
const LR1_SHIFT: u32 = 15; // hidden weights W1
const LRE_SHIFT: u32 = 4; // embedding rows (applied after the >>W_SHIFT)
/// Value-domain clamp (the squash logit domain, ±2047).
const VMAX: i32 = 2047;
/// Knuth multiplicative hash (floor(2^32/phi)) + a second odd spreader.
const KNUTH: u32 = 0x9E37_79B1;
const SPREAD: u32 = 0x85EB_CA6B;

#[inline]
fn leaky_relu(x: i32) -> i32 {
    if x > 0 { x } else { x >> LEAK_SHIFT }
}

/// Saturating `i64 → i32`. Every accumulator/gradient is widened to `i64`;
/// narrowing back must saturate, not truncate (a wrapped value could overflow a
/// later add and panic under `overflow-checks`). Losslessness already holds via
/// lockstep, but the codec must never panic on a hostile stream — so all
/// narrowing casts and weight/state adds saturate.
#[inline]
fn sat_i32(x: i64) -> i32 {
    x.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

/// Tiny MLP predictor. `predict` records per-bit activations that `update`
/// consumes, so the two must be called in lockstep (one `update` per `predict`).
pub struct Net {
    /// Flat embedding table `N_CTX × E_DIM` in the ±VMAX value domain. One heap
    /// allocation (~1 MiB at the defaults); freed when the coder drops it.
    embed: Vec<i32>,
    w1: [[i32; E_DIM]; H], // hidden weights, 16.16
    b1: [i32; H],          // hidden biases, value domain
    w2: [i32; H],          // output weights, 16.16
    bytes: [u8; 3],        // rolling whole-byte history for the context hash
    // per-bit state recorded at predict(), reused at update():
    emb_off: usize,
    pre: [i32; H],
    hid: [i32; H],
    p: i32,
}

impl Net {
    pub fn new() -> Self {
        // Deterministic symmetry-breaking init: all-zero would freeze the net
        // (zero pre-activations → zero gradient). Distinct nonzero biases +
        // small sign-varied W1 + ~1/H output weights start it live and
        // unsaturated; the embedding learns from zero.
        let mut w1 = [[0i32; E_DIM]; H];
        for (j, row) in w1.iter_mut().enumerate() {
            for (k, w) in row.iter_mut().enumerate() {
                *w = (((j * E_DIM + k) % 7) as i32 - 3) * ((1 << W_SHIFT) / 64);
            }
        }
        let mut b1 = [0i32; H];
        for (j, b) in b1.iter_mut().enumerate() {
            *b = ((j as i32) - (H as i32) / 2) * (VMAX / H as i32);
        }
        Self {
            embed: vec![0i32; N_CTX * E_DIM],
            w1,
            b1,
            w2: [(1 << W_SHIFT) / H as i32; H],
            bytes: [0; 3],
            emb_off: 0,
            pre: [0; H],
            hid: [0; H],
            p: 2048,
        }
    }

    #[inline]
    fn context(&self, c0: u32) -> usize {
        let h =
            (self.bytes[0] as u32) | ((self.bytes[1] as u32) << 8) | ((self.bytes[2] as u32) << 16);
        ((h.wrapping_mul(KNUTH) ^ c0.wrapping_mul(SPREAD)) & CTX_MASK) as usize
    }

    /// Predict `P(bit=1)` (12-bit) under the partial-byte register `c0`.
    pub fn predict(&mut self, c0: u32) -> i32 {
        let off = self.context(c0) * E_DIM;
        self.emb_off = off;
        for j in 0..H {
            let mut acc: i64 = 0;
            for k in 0..E_DIM {
                acc += self.w1[j][k] as i64 * self.embed[off + k] as i64;
            }
            let pre = sat_i32(acc >> W_SHIFT)
                .saturating_add(self.b1[j])
                .clamp(-VMAX, VMAX);
            self.pre[j] = pre;
            self.hid[j] = leaky_relu(pre);
        }
        let mut acc: i64 = 0;
        for j in 0..H {
            acc += self.w2[j] as i64 * self.hid[j] as i64;
        }
        let logit = sat_i32(acc >> W_SHIFT).clamp(-VMAX, VMAX);
        self.p = squash(logit).clamp(1, 4095);
        self.p
    }

    /// Online backprop toward the observed `bit`. Uses pre-update weights for
    /// every gradient (accumulates `demb` from the old `w1` before mutating it,
    /// and the `w1` gradient from the old embedding), so it's a clean SGD step —
    /// though even an approximate step would stay bit-exact across encode/decode.
    pub fn update(&mut self, bit: u8) {
        let e = ((bit as i32) << 12) - self.p; // ±4096
        let off = self.emb_off;
        let mut dpre = [0i32; H];
        for j in 0..H {
            let mut d = sat_i32((e as i64 * self.w2[j] as i64) >> W_SHIFT);
            if self.pre[j] <= 0 {
                d >>= LEAK_SHIFT; // leaky-ReLU derivative on the negative branch
            }
            dpre[j] = d;
            let g2 = sat_i32((e as i64 * self.hid[j] as i64) >> LR2_SHIFT);
            self.w2[j] = self.w2[j].saturating_add(g2);
        }
        // `demb[k]` accumulates `dpre[j] * w1[j][k]` over H=32 terms.
        let mut demb = [0i64; E_DIM];
        for j in 0..H {
            let dj = dpre[j] as i64;
            for k in 0..E_DIM {
                // saturating: the all-saturated worst case sums to ~9.21e18,
                // just under i64::MAX — saturate so the bound can't be crossed
                // by a future LR/shift retune.
                demb[k] = demb[k].saturating_add(dj * self.w1[j][k] as i64); // old W1
            }
            for k in 0..E_DIM {
                let g1 = sat_i32((dj * self.embed[off + k] as i64) >> LR1_SHIFT);
                self.w1[j][k] = self.w1[j][k].saturating_add(g1); // old emb
            }
        }
        for k in 0..E_DIM {
            let ge = sat_i32(demb[k] >> (W_SHIFT + LRE_SHIFT));
            self.embed[off + k] = self.embed[off + k].saturating_add(ge).clamp(-VMAX, VMAX);
        }
    }

    pub fn push_byte(&mut self, byte: u8) {
        self.bytes[2] = self.bytes[1];
        self.bytes[1] = self.bytes[0];
        self.bytes[0] = byte;
    }
}

impl Default for Net {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(data: &[u8]) -> Vec<i32> {
        let mut net = Net::new();
        let mut ps = Vec::new();
        for &byte in data {
            let mut c0: u32 = 1;
            for i in (0..8).rev() {
                let bit = (byte >> i) & 1;
                ps.push(net.predict(c0));
                net.update(bit);
                c0 = (c0 << 1) | bit as u32;
            }
            net.push_byte(byte);
        }
        ps
    }

    #[test]
    fn predictions_in_range_and_deterministic() {
        let data = b"the quick brown fox the quick brown fox the quick brown";
        let a = run(data);
        let b = run(data);
        assert_eq!(a, b, "net must be deterministic for lockstep coding");
        for &p in &a {
            assert!((1..=4095).contains(&p), "p={p} out of 12-bit range");
        }
    }

    #[test]
    fn update_saturates_on_extreme_weights() {
        // Regression (review): drive the net into a saturated-weight state and
        // ensure `update` saturates rather than panicking under the test
        // profile's `overflow-checks`. Exercises the i64→i32 narrowing casts and
        // the embedding add that previously could overflow.
        let mut net = Net::new();
        for w in net.w2.iter_mut() {
            *w = i32::MAX;
        }
        for row in net.w1.iter_mut() {
            for w in row.iter_mut() {
                *w = i32::MAX;
            }
        }
        for v in net.embed.iter_mut() {
            *v = VMAX;
        }
        // Each update follows its own predict (the "one update per predict"
        // contract). predict() drives p to a saturated 4095; update(0) then
        // yields the maximal-magnitude error e = -4095, the worst case for the
        // gradient chain that feeds the embedding add.
        net.predict(0xAB);
        net.update(0);
        net.predict(0xCD);
        net.update(1);
        // Embedding values stay clamped to the value domain.
        assert!(net.embed.iter().all(|&v| (-VMAX..=VMAX).contains(&v)));
    }

    #[test]
    fn learns_a_repeating_pattern() {
        // On a highly repetitive stream the net should push predictions away
        // from the 0.5 (=2048) prior — i.e. it is not dead.
        let data = vec![0xABu8; 4096];
        let ps = run(&data);
        let tail = &ps[ps.len() - 64..];
        let moved = tail.iter().filter(|&&p| (p - 2048).abs() > 256).count();
        assert!(
            moved > 16,
            "net failed to learn a constant stream: {tail:?}"
        );
    }
}
