//! lpaq `squash` (logistic) and `stretch` (logit) over 12-bit probabilities.
//! `squash(d)` maps a stretched logit `d` to a probability in `0..=4095`;
//! `stretch(p)` is its inverse. Both are integer and deterministic.

/// Inverse-logit: map a logit `d` (clamped to ±2047) to a probability `0..=4095`.
/// `const fn` so `STRETCH_TABLE` can be built at compile time.
pub const fn squash(d: i32) -> i32 {
    const T: [i32; 33] = [
        1, 2, 3, 6, 10, 16, 27, 45, 73, 120, 194, 310, 488, 747, 1101, 1546, 2047, 2549, 2994,
        3348, 3607, 3785, 3901, 3975, 4022, 4050, 4068, 4079, 4085, 4089, 4092, 4093, 4094,
    ];
    if d > 2047 {
        return 4095;
    }
    if d < -2047 {
        return 0;
    }
    let w = d & 127;
    let idx = ((d >> 7) + 16) as usize;
    (T[idx] * (128 - w) + T[idx + 1] * w + 64) >> 7
}

/// Logit table: `STRETCH_TABLE[p]` is the inverse of [`squash`], computed at
/// compile time by walking `squash` over `-2047..=2047` (no runtime init).
const STRETCH_TABLE: [i16; 4096] = {
    let mut t = [0i16; 4096];
    let mut pi = 0usize;
    let mut d = -2047;
    while d <= 2047 {
        let v = squash(d) as usize;
        while pi <= v {
            t[pi] = d as i16;
            pi += 1;
        }
        d += 1;
    }
    while pi < 4096 {
        t[pi] = 2047;
        pi += 1;
    }
    t
};

/// Logit of a 12-bit probability `p` (`0..=4095`), clamped to `±2047`.
pub fn stretch(p: i32) -> i32 {
    STRETCH_TABLE[(p.clamp(0, 4095)) as usize] as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn squash_is_monotonic_and_bounded() {
        let mut prev = -1i32;
        for d in -2047..=2047 {
            let p = squash(d);
            assert!((0..=4095).contains(&p), "squash({d}) = {p} out of range");
            assert!(p >= prev, "squash not monotonic at {d}");
            prev = p;
        }
        assert_eq!(squash(-3000), 0);
        assert_eq!(squash(3000), 4095);
    }

    #[test]
    fn stretch_inverts_squash_approximately() {
        // stretch(squash(d)) should be close to d in the central region.
        // Outside this band squash saturates into wide flat buckets where the
        // (single-valued) inverse cannot land within tolerance — intrinsic to
        // the canonical lpaq quantization and harmless (stretch is internal and
        // applied identically on encode/decode, so it never affects losslessness).
        for d in (-1500..=1500).step_by(7) {
            let p = squash(d);
            let back = stretch(p);
            assert!(
                (back - d).abs() <= 64,
                "stretch(squash({d}))={back} drifted too far"
            );
        }
    }

    #[test]
    fn stretch_is_bounded() {
        for p in 0..4096 {
            let s = stretch(p);
            assert!((-2047..=2047).contains(&s), "stretch({p})={s} out of range");
        }
    }
}
