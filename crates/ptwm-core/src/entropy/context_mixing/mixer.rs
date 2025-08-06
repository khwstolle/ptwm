//! Context-gated logistic mixer. One weight vector per gating context
//! (the partial-byte register `c0`, bounded to 256). Mixes the models'
//! stretched logits, squashes to a probability, and learns the weights
//! online by logistic gradient. Integer/deterministic.

use crate::entropy::context_mixing::model::N_MODELS;
use crate::entropy::context_mixing::squash::squash;

/// Number of gating contexts (weight sets): `c0` is `1..=255`, so 256.
const N_GATES: usize = 256;
/// Total weights: one `N_MODELS`-vector per gate.
const N_WEIGHTS: usize = N_GATES * N_MODELS;
/// Fixed-point shift for weights (weights are 16.16).
const WEIGHT_SHIFT: u32 = 16;
/// Learning-rate shift for the gradient update.
const LR_SHIFT: u32 = 10;

pub struct Mixer {
    /// `weights[gate * N_MODELS + i]`, 16.16 fixed point. Fixed-size (4 KiB),
    /// stored inline to avoid a per-call heap allocation.
    weights: [i32; N_WEIGHTS],
    /// State carried from `mix()` to `update()`.
    last_inputs: [i32; N_MODELS],
    last_gate: usize,
    last_p: i32,
}

impl Mixer {
    pub fn new() -> Self {
        // Init weights to ~1/N each (16.16): 65536 / N_MODELS.
        let w0 = (1i32 << WEIGHT_SHIFT) / N_MODELS as i32;
        Self {
            weights: [w0; N_WEIGHTS],
            last_inputs: [0; N_MODELS],
            last_gate: 0,
            last_p: 2048,
        }
    }

    /// Mix the stretched inputs under `gate`, returning `P(bit=1)` (12-bit).
    pub fn mix(&mut self, gate: u32, inputs: &[i32; N_MODELS]) -> i32 {
        let g = (gate as usize) & (N_GATES - 1);
        let base = g * N_MODELS;
        let mut dot: i64 = 0;
        for i in 0..N_MODELS {
            dot += self.weights[base + i] as i64 * inputs[i] as i64;
        }
        let logit = (dot >> WEIGHT_SHIFT) as i32;
        let p = squash(logit.clamp(-2047, 2047));
        self.last_inputs = *inputs;
        self.last_gate = g;
        self.last_p = p;
        p.clamp(1, 4095)
    }

    /// Update the weights used by the last `mix()` toward the observed `bit`.
    pub fn update(&mut self, bit: u8) {
        let err = ((bit as i32) << 12) - self.last_p; // target(0 or 4096) - p
        let base = self.last_gate * N_MODELS;
        for i in 0..N_MODELS {
            let grad = (err as i64 * self.last_inputs[i] as i64) >> LR_SHIFT;
            // saturating: a pathological stream must not panic (debug) or wrap
            // (release) the weights. Saturation is deterministic on both sides.
            self.weights[base + i] = self.weights[base + i].saturating_add(grad as i32);
        }
    }
}

impl Default for Mixer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::context_mixing::squash::stretch;

    #[test]
    fn mix_output_is_a_valid_probability() {
        let mut mx = Mixer::new();
        let inputs = [stretch(3000), stretch(2900), stretch(3100), stretch(3000)];
        let p = mx.mix(7, &inputs);
        assert!((1..=4095).contains(&p), "mixed p={p}");
    }

    #[test]
    fn mixer_learns_a_consistent_bit() {
        // Feed models that always predict "1" strongly; after training, the
        // mixed probability for that gate should be high.
        let mut mx = Mixer::new();
        let inputs = [stretch(3500), stretch(3500), stretch(3500), stretch(3500)];
        let mut last = 0;
        for _ in 0..200 {
            last = mx.mix(7, &inputs);
            mx.update(1);
        }
        assert!(last > 3000, "mixer should converge high, got {last}");
    }

    #[test]
    fn mixer_is_deterministic() {
        let run = || {
            let mut mx = Mixer::new();
            let inputs = [stretch(2600), stretch(3000), stretch(1500), stretch(3000)];
            let mut v = Vec::new();
            for g in 0..16u32 {
                v.push(mx.mix(g, &inputs));
                mx.update((g & 1) as u8);
            }
            v
        };
        assert_eq!(run(), run());
    }
}
