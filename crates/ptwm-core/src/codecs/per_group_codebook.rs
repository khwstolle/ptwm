//! PerGroupCodebook codec.
//!
//! Models the per-group nibble distribution of a quantized plane (MXFP4
//! nibble VALUE plane). Offers per-group histogram extraction, k-means
//! codebook fitting (multi-seed, deterministic), and a `PlaneCodec` impl
//! that range-codes nibbles under the per-group PMF.

pub const ALPHABET: usize = 16; // MXFP4 nibble alphabet
pub const GROUP_SIZE: usize = 32; // MXFP4 block size (nibbles per block)

pub fn group_count(plane_nibble_count: usize) -> usize {
    plane_nibble_count.div_ceil(GROUP_SIZE)
}

/// Extract per-group histograms. `nibbles` is a slice where each byte
/// carries a 4-bit nibble in its low nibble (high nibble ignored).
/// Returns one `[u32; ALPHABET]` histogram per group of `GROUP_SIZE`
/// nibbles; the last group may be partial.
pub fn histograms(nibbles: &[u8]) -> Vec<[u32; ALPHABET]> {
    let n_groups = group_count(nibbles.len());
    let mut out = vec![[0u32; ALPHABET]; n_groups];
    for (i, &v) in nibbles.iter().enumerate() {
        let g = i / GROUP_SIZE;
        out[g][(v & 0x0F) as usize] += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_counts() {
        let nibbles: Vec<u8> = (0..64u32).map(|i| (i % 16) as u8).collect();
        let h = histograms(&nibbles);
        assert_eq!(h.len(), 2);
        for g in 0..2 {
            for v in 0..16 {
                assert_eq!(h[g][v], 2);
            }
        }
    }

    #[test]
    fn histogram_partial_last_group() {
        let nibbles = vec![0u8; 40]; // 1 full group + 8 nibbles of next
        let h = histograms(&nibbles);
        assert_eq!(h.len(), 2);
        assert_eq!(h[0][0], 32);
        assert_eq!(h[1][0], 8);
    }

    #[test]
    fn histogram_ignores_high_nibble() {
        let nibbles: Vec<u8> = (0..32u32).map(|i| ((i % 16) as u8) | 0xF0).collect();
        let h = histograms(&nibbles);
        assert_eq!(h.len(), 1);
        for v in 0..16 {
            assert_eq!(h[0][v], 2);
        }
    }
}

pub const K: usize = 16;
pub const KMEANS_MAX_ITERS: usize = 50;
pub const KMEANS_PP_SEED: u64 = 0x5A17ED;

pub struct Codebook {
    pub centroids: [[f32; ALPHABET]; K], // PMF rows, each row sums to ~1.0
    pub assignments: Vec<u8>,            // one cluster index (0..K) per group
}

fn l2_dist(a: &[f32; ALPHABET], b: &[f32; ALPHABET]) -> f32 {
    let mut s = 0f32;
    for i in 0..ALPHABET {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

fn normalize(h: &[u32; ALPHABET]) -> [f32; ALPHABET] {
    let total: u32 = h.iter().sum();
    let total = total.max(1) as f32;
    let mut out = [0f32; ALPHABET];
    for i in 0..ALPHABET {
        out[i] = h[i] as f32 / total;
    }
    out
}

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn gen_below(&mut self, bound: usize) -> usize {
        if bound == 0 {
            0
        } else {
            (self.next_u64() as usize) % bound
        }
    }
    fn gen_f32(&mut self, upper: f32) -> f32 {
        let x = (self.next_u64() & 0xFFFFFF) as f32 / (1u64 << 24) as f32;
        x * upper
    }
}

/// In-cluster L2 variance: `sum_i ||pmf_i - centroid_assignment_i||²`.
/// Lower means tighter clusters → better k-means convergence.
fn codebook_variance(pmfs: &[[f32; ALPHABET]], cb: &Codebook) -> f32 {
    let mut total = 0f32;
    for (i, p) in pmfs.iter().enumerate() {
        let k = cb.assignments[i] as usize;
        total += l2_dist(p, &cb.centroids[k]);
    }
    total
}

/// Multi-seed k-means: parallel evaluation of every seed, pick the
/// codebook with lowest in-cluster L2 variance. Deterministic — ties
/// break to the earliest seed in input order.
pub fn fit_codebook_multi_seed(
    hists: &[[u32; ALPHABET]],
    seeds: &[u64],
) -> Result<Codebook, PtwmCoreError> {
    use rayon::prelude::*;

    if hists.is_empty() {
        return Ok(fit_codebook(
            hists,
            seeds.first().copied().unwrap_or(KMEANS_PP_SEED),
        ));
    }
    if seeds.is_empty() {
        return Err(PtwmCoreError::InvalidContainer(
            "fit_codebook_multi_seed: empty seed list".into(),
        ));
    }
    let pmfs: Vec<[f32; ALPHABET]> = hists.iter().map(normalize).collect();
    let mut scored: Vec<(usize, f32, Codebook)> = seeds
        .par_iter()
        .enumerate()
        .map(|(idx, &seed)| {
            let cb = fit_codebook(hists, seed);
            let var = codebook_variance(&pmfs, &cb);
            (idx, var, cb)
        })
        .collect();
    // NaN partial_cmp yields None and was previously coerced to Equal,
    // letting a degenerate codebook displace a valid one.
    scored.retain(|(_, v, _)| v.is_finite());
    if scored.is_empty() {
        return Err(PtwmCoreError::InvalidContainer(
            "fit_codebook_multi_seed: every seed produced non-finite variance".into(),
        ));
    }
    scored.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    Ok(scored.into_iter().next().map(|(_, _, cb)| cb).unwrap())
}

/// k-means++ init + Lloyd iterations on per-group histogram PMFs. Returns
/// cluster centroids (each a length-ALPHABET PMF) and per-group cluster
/// assignments. Deterministic given the seed.
pub fn fit_codebook(hists: &[[u32; ALPHABET]], seed: u64) -> Codebook {
    let pmfs: Vec<[f32; ALPHABET]> = hists.iter().map(normalize).collect();
    let n = pmfs.len();
    let mut centroids = [[0f32; ALPHABET]; K];

    if n == 0 {
        return Codebook {
            centroids,
            assignments: Vec::new(),
        };
    }

    let mut rng = Lcg::new(seed);

    // k-means++ init
    centroids[0] = pmfs[rng.gen_below(n)];
    for k in 1..K {
        let mut dists = vec![f32::INFINITY; n];
        for (i, p) in pmfs.iter().enumerate() {
            for c in &centroids[..k] {
                let d = l2_dist(p, c);
                if d < dists[i] {
                    dists[i] = d;
                }
            }
        }
        let total: f32 = dists.iter().sum();
        if total == 0.0 {
            centroids[k] = pmfs[rng.gen_below(n)];
            continue;
        }
        let target = rng.gen_f32(total);
        let mut cum = 0f32;
        let mut picked = false;
        for (i, d) in dists.iter().enumerate() {
            cum += d;
            if cum >= target {
                centroids[k] = pmfs[i];
                picked = true;
                break;
            }
        }
        if !picked {
            // Fallback for numerical drift
            centroids[k] = pmfs[n - 1];
        }
    }

    // Lloyd iterations
    let mut assignments = vec![0u8; n];
    for _iter in 0..KMEANS_MAX_ITERS {
        let mut changed = false;
        for (i, p) in pmfs.iter().enumerate() {
            let mut best_k = 0usize;
            let mut best_d = f32::INFINITY;
            for (k, c) in centroids.iter().enumerate() {
                let d = l2_dist(p, c);
                if d < best_d {
                    best_d = d;
                    best_k = k;
                }
            }
            if assignments[i] != best_k as u8 {
                assignments[i] = best_k as u8;
                changed = true;
            }
        }
        if !changed {
            break;
        }
        let mut accum = [[0f32; ALPHABET]; K];
        let mut counts = [0u32; K];
        for (i, p) in pmfs.iter().enumerate() {
            let k = assignments[i] as usize;
            counts[k] += 1;
            for j in 0..ALPHABET {
                accum[k][j] += p[j];
            }
        }
        for k in 0..K {
            if counts[k] == 0 {
                continue;
            }
            for j in 0..ALPHABET {
                centroids[k][j] = accum[k][j] / counts[k] as f32;
            }
        }
    }

    Codebook {
        centroids,
        assignments,
    }
}

/// Quantise a PMF (floats summing to ~1.0) into integer counts whose sum
/// equals `total`, rounding each entry to at least 1 (so no zero-
/// probability symbol appears in the final coder PMF). Used to convert
/// float centroids into the integer denominators the range coder needs.
pub fn quantize_centroid(c: &[f32; ALPHABET], total: u32) -> [u32; ALPHABET] {
    let mut counts = [0u32; ALPHABET];
    let mut sum = 0u32;
    for i in 0..ALPHABET {
        counts[i] = (c[i] * total as f32).round().max(1.0) as u32;
        sum += counts[i];
    }
    while sum > total {
        let (argmax, _) = counts.iter().enumerate().max_by_key(|x| x.1).unwrap();
        if counts[argmax] <= 1 {
            // No further decrement available without producing a zero-prob
            // symbol (range coder would loop forever on encode).
            break;
        }
        counts[argmax] -= 1;
        sum -= 1;
    }
    while sum < total {
        let (argmin, _) = counts.iter().enumerate().min_by_key(|x| x.1).unwrap();
        counts[argmin] += 1;
        sum += 1;
    }
    counts
}

use crate::codec::{CodecId, Encoded, PlaneCodec};
use crate::error::PtwmCoreError;
use crate::layout::PlaneLayout;
use crate::quantize::{PGC_QUANT_TOTAL, validate_pmf_sum};
use crate::range_coder::{RangeDecoder, RangeEncoder};
use crate::types::descriptor::{ElementWidth, PlaneDescriptor};
use crate::types::role::Role;

pub const QUANT_TOTAL: u32 = 256; // per-centroid PMF denominator

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateV0 {
    pub codebook: [[u32; ALPHABET]; K], // quantized counts per centroid
}

fn serialize_state(state: &StateV0) -> Vec<u8> {
    let mut out = Vec::with_capacity(K * ALPHABET);
    for k in 0..K {
        for v in 0..ALPHABET {
            out.push(state.codebook[k][v] as u8);
        }
    }
    out
}

/// Public byte-only validator. Used by the shared-state construction
/// path to fail loudly if a fitter (or external caller) emits bytes that
/// would only fail on read. Mirrors the parse path of `deserialize_state`.
pub fn validate_state_bytes(state_format_version: u8, bytes: &[u8]) -> Result<(), PtwmCoreError> {
    if state_format_version != 0 {
        return Err(PtwmCoreError::CodecDecode {
            codec: "PerGroupCodebook",
            msg: format!("unknown state_format_version {state_format_version}"),
        });
    }
    deserialize_state(bytes).map(|_| ())
}

fn deserialize_state(bytes: &[u8]) -> Result<StateV0, PtwmCoreError> {
    if bytes.len() != K * ALPHABET {
        return Err(PtwmCoreError::CodecDecode {
            codec: "PerGroupCodebook",
            msg: format!("expected {} state bytes, got {}", K * ALPHABET, bytes.len()),
        });
    }
    let mut state = StateV0 {
        codebook: [[0u32; ALPHABET]; K],
    };
    for k in 0..K {
        for v in 0..ALPHABET {
            state.codebook[k][v] = bytes[k * ALPHABET + v] as u32;
        }
        // Validate this row sums to PGC_QUANT_TOTAL (= 256). Trips silent
        // state corruption.
        let row_u16: Vec<u16> = state.codebook[k].iter().map(|&c| c as u16).collect();
        validate_pmf_sum(&row_u16, PGC_QUANT_TOTAL, "PerGroupCodebook")?;
    }
    Ok(state)
}

/// Convert per-cluster quantized count tables into per-cluster float PMFs
/// used for assignment selection.
fn centroid_pmfs(state: &StateV0) -> [[f32; ALPHABET]; K] {
    let mut out = [[0f32; ALPHABET]; K];
    for k in 0..K {
        let sum: u32 = state.codebook[k].iter().sum();
        let sum = sum.max(1) as f32;
        for v in 0..ALPHABET {
            out[k][v] = state.codebook[k][v] as f32 / sum;
        }
    }
    out
}

/// Assign each per-group histogram to its nearest centroid under L2.
fn assign_groups(hists: &[[u32; ALPHABET]], centroids: &[[f32; ALPHABET]; K]) -> Vec<u8> {
    let mut out = Vec::with_capacity(hists.len());
    for h in hists {
        let p = normalize(h);
        let mut best_k = 0usize;
        let mut best_d = f32::INFINITY;
        for (k, c) in centroids.iter().enumerate() {
            let d = l2_dist(&p, c);
            if d < best_d {
                best_d = d;
                best_k = k;
            }
        }
        out.push(best_k as u8);
    }
    out
}

pub struct PerGroupCodebook;

impl PerGroupCodebook {
    pub fn canonical_id(&self) -> crate::extension::CanonicalId {
        crate::extension::builtin_canonical_id("per_group_codebook")
    }
}

impl PlaneCodec for PerGroupCodebook {
    fn id(&self) -> CodecId {
        CodecId::PerGroupCodebook
    }

    /// Accept expanded FP4 value planes whose length divides into whole
    /// groups.
    ///
    /// The codec models one nibble per byte, and `GROUP_SIZE` of 32 is one
    /// MXFP4 block: the unit that shares a scale, and therefore the unit
    /// worth giving its own codebook. `MxFp4Deinterleave` produces exactly
    /// that, 32 single-nibble slots per block.
    ///
    /// Two exclusions are deliberate. A *packed* plane, two values per
    /// byte, is declined rather than misread: reading one would take low
    /// nibbles only and return them with the high half zeroed, silently
    /// losing half the data. Such a plane must be deinterleaved first. And
    /// a length that is not a whole number of groups is declined here
    /// rather than failing inside `encode`, because dispatch drops the
    /// whole candidate chain when a codec errors and packed FP4 has only
    /// one chain to drop. Declining keeps an `encode` error meaning what it
    /// should: an invariant was violated, not merely an unsuitable input.
    fn accepts(&self, descriptor: &PlaneDescriptor) -> bool {
        !descriptor.is_nibble_packed
            && matches!(descriptor.role, Role::Value { .. })
            && descriptor.element_width == ElementWidth::Nibble
            && (descriptor.length_bytes as usize).is_multiple_of(GROUP_SIZE)
    }

    fn priority_for(&self, descriptor: &PlaneDescriptor) -> i8 {
        if self.accepts(descriptor) {
            10
        } else {
            i8::MIN
        }
    }

    fn encode(
        &self,
        plane: &[u8],
        shared_state: Option<&[u8]>,
        _layout: &PlaneLayout,
    ) -> Result<Encoded, PtwmCoreError> {
        if !plane.len().is_multiple_of(GROUP_SIZE) {
            return Err(PtwmCoreError::CodecDecode {
                codec: "PerGroupCodebook",
                msg: format!(
                    "encode: plane length {} not a multiple of GROUP_SIZE {GROUP_SIZE}",
                    plane.len(),
                ),
            });
        }

        let hists = histograms(plane);

        // Either consume provided shared state or fit a per-tensor codebook.
        let (state, inline_state_bytes): (StateV0, Vec<u8>) = match shared_state {
            Some(bytes) => (deserialize_state(bytes)?, Vec::new()),
            None => {
                let cb = fit_codebook(&hists, KMEANS_PP_SEED);
                let mut codebook = [[0u32; ALPHABET]; K];
                for k in 0..K {
                    codebook[k] = quantize_centroid(&cb.centroids[k], QUANT_TOTAL);
                }
                let state = StateV0 { codebook };
                let bytes = serialize_state(&state);
                (state, bytes)
            }
        };

        // Compute assignments from the (possibly shared) codebook's PMFs.
        let pmfs = centroid_pmfs(&state);
        let assignments = assign_groups(&hists, &pmfs);

        // Payload layout:
        //   [n_assignments: u32]
        //   [packed assignments: ceil(n/2) bytes, 4-bit indices, low nibble first]
        //   [coded_len: u32]
        //   [range-coded nibble stream]
        let mut payload = Vec::new();
        payload.extend_from_slice(&(assignments.len() as u32).to_le_bytes());

        let packed_len = assignments.len().div_ceil(2);
        let mut packed = vec![0u8; packed_len];
        for (i, &a) in assignments.iter().enumerate() {
            let nib = a & 0x0F;
            if i % 2 == 0 {
                packed[i / 2] = nib;
            } else {
                packed[i / 2] |= nib << 4;
            }
        }
        payload.extend_from_slice(&packed);

        // Range-coded per-nibble payload
        let mut enc = RangeEncoder::new();
        for (i, &nib_byte) in plane.iter().enumerate() {
            let sym = (nib_byte & 0x0F) as usize;
            let g = i / GROUP_SIZE;
            let k = assignments[g] as usize;
            let counts = &state.codebook[k];
            let cumulative: u32 = counts[..sym].iter().sum();
            let probability = counts[sym];
            enc.encode_symbol(cumulative, probability, QUANT_TOTAL);
        }
        let coded = enc.finish();
        payload.extend_from_slice(&(coded.len() as u32).to_le_bytes());
        payload.extend_from_slice(&coded);

        Ok(Encoded {
            state_bytes: inline_state_bytes,
            state_format_version: 0,
            payload,
        })
    }

    fn decode(
        &self,
        state_format_version: u8,
        state_bytes: &[u8],
        payload: &[u8],
        _layout: &PlaneLayout,
        decoded_len: usize,
    ) -> Result<Vec<u8>, PtwmCoreError> {
        if state_format_version != 0 {
            return Err(PtwmCoreError::CodecDecode {
                codec: "PerGroupCodebook",
                msg: format!("unknown state_format_version {}", state_format_version),
            });
        }
        let state = deserialize_state(state_bytes)?;

        if payload.len() < 4 {
            return Err(PtwmCoreError::CodecDecode {
                codec: "PerGroupCodebook",
                msg: "payload too short for assignment count".into(),
            });
        }
        let n_assignments = u32::from_le_bytes(payload[..4].try_into().unwrap()) as usize;
        // `decoded_len` comes from the trusted spec/plane-record path; a
        // mismatch here is corruption, not a soft failure.
        if !decoded_len.is_multiple_of(GROUP_SIZE) {
            return Err(PtwmCoreError::CodecDecode {
                codec: "PerGroupCodebook",
                msg: format!("decoded_len {decoded_len} not divisible by GROUP_SIZE {GROUP_SIZE}"),
            });
        }
        let expected_assignments = decoded_len / GROUP_SIZE;
        if n_assignments != expected_assignments {
            return Err(PtwmCoreError::CodecDecode {
                codec: "PerGroupCodebook",
                msg: format!(
                    "payload claims {n_assignments} groups (={} bytes), \
                     plane expects {expected_assignments} (={decoded_len} bytes)",
                    n_assignments * GROUP_SIZE
                ),
            });
        }
        let packed_len = n_assignments.div_ceil(2);
        let mut pos = 4usize;
        if payload.len() < pos + packed_len + 4 {
            return Err(PtwmCoreError::CodecDecode {
                codec: "PerGroupCodebook",
                msg: "payload truncated mid-assignments".into(),
            });
        }

        let mut assignments = Vec::with_capacity(n_assignments);
        for i in 0..n_assignments {
            let byte = payload[pos + i / 2];
            let nib = if i % 2 == 0 {
                byte & 0x0F
            } else {
                (byte >> 4) & 0x0F
            };
            assignments.push(nib);
        }
        pos += packed_len;

        let coded_len = u32::from_le_bytes(payload[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        if payload.len() < pos + coded_len {
            return Err(PtwmCoreError::CodecDecode {
                codec: "PerGroupCodebook",
                msg: "coded stream truncated".into(),
            });
        }
        let coded = &payload[pos..pos + coded_len];

        let total_nibbles = n_assignments * GROUP_SIZE;
        let mut dec = RangeDecoder::new(coded);
        let mut out = Vec::with_capacity(total_nibbles);
        for i in 0..total_nibbles {
            let g = i / GROUP_SIZE;
            let k = assignments[g] as usize;
            let counts = &state.codebook[k];
            let target = dec.decode_symbol(QUANT_TOTAL);
            let mut cum = 0u32;
            let mut sym: Option<u32> = None;
            for v in 0..ALPHABET {
                let next = cum + counts[v];
                if target < next {
                    sym = Some(v as u32);
                    dec.advance(cum, counts[v], QUANT_TOTAL);
                    break;
                }
                cum = next;
            }
            // Fall-through is impossible if validate_pmf_sum (called inside
            // deserialize_state) holds, because target ∈ [0, QUANT_TOTAL)
            // and the cumulative counts must cover the full range. A miss
            // here means either the stream desynced (corruption) or the
            // PMF row sums to less than QUANT_TOTAL — either way, error.
            let sym = sym.ok_or_else(|| PtwmCoreError::CodecDecode {
                codec: "PerGroupCodebook",
                msg: format!(
                    "decode_symbol target {target} unreachable in cumulative \
                     counts (sum={cum}, expected={QUANT_TOTAL}); \
                     payload corrupt or PMF underspecified"
                ),
            })?;
            out.push(sym as u8);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod codec_tests {
    use super::*;
    use crate::layout::PlaneLayout;

    fn build_plane_nibbles(len: usize, modulus: u8) -> Vec<u8> {
        // Produce plane nibbles with a biased distribution so the
        // codebook has something to learn.
        (0..len).map(|i| ((i as u8) % modulus) & 0x0F).collect()
    }

    #[test]
    fn codec_roundtrip_inline_state() {
        let c = PerGroupCodebook;
        // 32 groups × GROUP_SIZE nibbles = 1024 nibbles total
        let nibbles = build_plane_nibbles(32 * GROUP_SIZE, 16);
        let enc = c.encode(&nibbles, None, &PlaneLayout::Flat).unwrap();
        let dec = c
            .decode(
                enc.state_format_version,
                &enc.state_bytes,
                &enc.payload,
                &PlaneLayout::Flat,
                nibbles.len(),
            )
            .unwrap();
        assert_eq!(dec, nibbles);
    }

    #[test]
    fn codec_roundtrip_shared_state() {
        let c = PerGroupCodebook;
        // Fit codebook on A, reuse for B.
        let a = build_plane_nibbles(64 * GROUP_SIZE, 16);
        let b = build_plane_nibbles(64 * GROUP_SIZE, 11);
        let enc_a = c.encode(&a, None, &PlaneLayout::Flat).unwrap();
        let shared = enc_a.state_bytes.clone();

        let enc_b = c.encode(&b, Some(&shared), &PlaneLayout::Flat).unwrap();
        assert!(
            enc_b.state_bytes.is_empty(),
            "shared-state encode must not emit its own state bytes"
        );
        let dec = c
            .decode(
                enc_b.state_format_version,
                &shared,
                &enc_b.payload,
                &PlaneLayout::Flat,
                b.len(),
            )
            .unwrap();
        assert_eq!(dec, b);
    }

    #[test]
    fn decoder_rejects_bad_state_version() {
        let c = PerGroupCodebook;
        let nibbles = build_plane_nibbles(GROUP_SIZE, 16);
        let enc = c.encode(&nibbles, None, &PlaneLayout::Flat).unwrap();
        let err = c
            .decode(
                99,
                &enc.state_bytes,
                &enc.payload,
                &PlaneLayout::Flat,
                nibbles.len(),
            )
            .unwrap_err();
        match err {
            PtwmCoreError::CodecDecode { codec, .. } => assert_eq!(codec, "PerGroupCodebook"),
            _ => panic!("expected CodecDecode"),
        }
    }

    #[test]
    fn decoder_rejects_corrupted_state_pmf_sum() {
        let c = PerGroupCodebook;
        let nibbles = build_plane_nibbles(GROUP_SIZE, 16);
        let enc = c.encode(&nibbles, None, &PlaneLayout::Flat).unwrap();
        let mut corrupted = enc.state_bytes.clone();
        corrupted[0] = corrupted[0].wrapping_add(1); // breaks row-0 sum
        let err = c
            .decode(
                enc.state_format_version,
                &corrupted,
                &enc.payload,
                &PlaneLayout::Flat,
                nibbles.len(),
            )
            .unwrap_err();
        match err {
            PtwmCoreError::CodecDecode { codec, .. } => assert_eq!(codec, "PerGroupCodebook"),
            _ => panic!("expected CodecDecode"),
        }
    }
}

#[cfg(test)]
mod fit_tests {
    use super::*;

    #[test]
    fn fit_on_two_clusters_separates_them() {
        let mut hists = Vec::new();
        for _ in 0..100 {
            let mut h = [0u32; ALPHABET];
            for v in 0..4 {
                h[v] = 8;
            }
            hists.push(h);
        }
        for _ in 0..100 {
            let mut h = [0u32; ALPHABET];
            for v in 12..16 {
                h[v] = 8;
            }
            hists.push(h);
        }
        let cb = fit_codebook(&hists, KMEANS_PP_SEED);
        let unique_used: std::collections::HashSet<_> = cb.assignments.iter().copied().collect();
        assert!(
            unique_used.len() >= 2,
            "expected at least 2 distinct clusters, got {:?}",
            unique_used
        );
    }

    #[test]
    fn fit_deterministic() {
        let hists: Vec<[u32; ALPHABET]> = (0..50)
            .map(|i| {
                let mut h = [0u32; ALPHABET];
                for v in 0..ALPHABET {
                    h[v] = ((i + v) as u32) % 17;
                }
                h
            })
            .collect();
        let cb1 = fit_codebook(&hists, 42);
        let cb2 = fit_codebook(&hists, 42);
        assert_eq!(cb1.assignments, cb2.assignments);
    }

    #[test]
    fn quantize_centroid_sums_to_total() {
        let mut c = [0f32; ALPHABET];
        for i in 0..ALPHABET {
            c[i] = 1.0 / ALPHABET as f32;
        }
        let q = quantize_centroid(&c, 256);
        assert_eq!(q.iter().sum::<u32>(), 256);
        for &x in &q {
            assert!(x >= 1);
        }
    }

    #[test]
    fn fit_empty_input_is_ok() {
        let cb = fit_codebook(&[], KMEANS_PP_SEED);
        assert!(cb.assignments.is_empty());
    }
}

#[cfg(test)]
mod capability_tests {
    use super::*;
    use crate::types::descriptor::{ElementWidth, Layout, PlaneDescriptor};
    use crate::types::role::{Role, ScaleFormat, ValueFormat};

    fn descriptor(role: Role, width: ElementWidth, is_nibble_packed: bool) -> PlaneDescriptor {
        PlaneDescriptor {
            role,
            element_width: width,
            length_bytes: 512,
            layout: Layout::Flat,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed,
            vendor_bytes: vec![],
        }
    }

    #[test]
    fn pgc_accepts_expanded_value_nibble() {
        let c = PerGroupCodebook;
        let d = descriptor(
            Role::Value {
                format: ValueFormat::Fp4E2m1,
            },
            ElementWidth::Nibble,
            false,
        );
        assert!(c.accepts(&d));
    }

    #[test]
    fn pgc_rejects_packed_plane() {
        let c = PerGroupCodebook;
        let d = descriptor(
            Role::Value {
                format: ValueFormat::Fp4E2m1,
            },
            ElementWidth::Nibble,
            true,
        );
        assert!(!c.accepts(&d));
    }

    #[test]
    fn pgc_rejects_non_value_role() {
        let c = PerGroupCodebook;
        let d = descriptor(
            Role::Scale {
                format: ScaleFormat::E4M3,
            },
            ElementWidth::Nibble,
            false,
        );
        assert!(!c.accepts(&d));
    }

    #[test]
    fn pgc_rejects_byte_width() {
        let c = PerGroupCodebook;
        let d = descriptor(
            Role::Value {
                format: ValueFormat::Fp4E2m1,
            },
            ElementWidth::Byte,
            false,
        );
        assert!(!c.accepts(&d));
    }

    #[test]
    fn pgc_priority_high_when_accepted() {
        let c = PerGroupCodebook;
        let d = descriptor(
            Role::Value {
                format: ValueFormat::Fp4E2m1,
            },
            ElementWidth::Nibble,
            false,
        );
        assert_eq!(c.priority_for(&d), 10);
    }

    #[test]
    fn pgc_priority_min_when_rejected() {
        let c = PerGroupCodebook;
        let d = descriptor(
            Role::Scale {
                format: ScaleFormat::E4M3,
            },
            ElementWidth::Nibble,
            false,
        );
        assert_eq!(c.priority_for(&d), i8::MIN);
    }
}

/// End-to-end coverage for the producer this codec is actually built for.
///
/// The suite previously exercised the codec only on hand-built planes. It
/// never drove the op that supplies it in practice, which is how a
/// descriptor mismatch between the two survived: `MxFp4Deinterleave`
/// labelled its output nibble-packed while emitting one nibble per byte,
/// and the codec's own contract expects the expanded form.
#[cfg(test)]
mod deinterleave_integration_tests {
    use super::*;
    use crate::transforms::mxfp4_deinterleave::MxFp4Deinterleave;
    use crate::transforms::op::{Op, Plane};
    use crate::types::descriptor::{Layout, PlaneDescriptor};
    use crate::types::role::Role;

    const BLOCK_BYTES: usize = 17;

    /// OCP MXFP4 wire bytes: 16 packed-nibble bytes then one E8M0 scale.
    fn mxfp4_blocks(n_blocks: usize, seed: u64) -> Vec<u8> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut out = Vec::with_capacity(n_blocks * BLOCK_BYTES);
        for b in 0..n_blocks {
            for _ in 0..16 {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                // Skewed within a block, varying across blocks, so a
                // per-block codebook has structure to find.
                let lo = (s % 4 + (b as u64 % 3) * 4) as u8 & 0x0F;
                let hi = ((s >> 8) % 4 + (b as u64 % 3) * 4) as u8 & 0x0F;
                out.push(lo | (hi << 4));
            }
            out.push(0x80 | (b as u8 & 0x0F));
        }
        out
    }

    fn raw_descriptor(len: usize) -> PlaneDescriptor {
        PlaneDescriptor {
            role: Role::Raw,
            element_width: ElementWidth::Byte,
            length_bytes: len as u64,
            layout: Layout::Flat,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        }
    }

    #[test]
    fn codec_accepts_and_round_trips_the_deinterleaved_value_plane() {
        let n_blocks = 8;
        let raw = mxfp4_blocks(n_blocks, 11);
        let op = MxFp4Deinterleave::new(32).unwrap();

        let descs = op
            .propagate_descriptors(&[raw_descriptor(raw.len())])
            .unwrap();
        let value_desc = descs[0].clone();

        // One group per MXFP4 block is the whole premise of the codec.
        assert_eq!(value_desc.length_bytes as usize, n_blocks * GROUP_SIZE);

        let c = PerGroupCodebook;
        assert!(
            c.accepts(&value_desc),
            "the codec must accept the plane its own producer emits: {value_desc:?}"
        );

        let planes = op
            .forward(&[Plane::new(raw.clone(), raw_descriptor(raw.len())).unwrap()])
            .unwrap();
        let value_plane = planes[0].bytes.to_vec();
        assert_eq!(value_plane.len(), n_blocks * GROUP_SIZE);
        assert!(
            value_plane.iter().all(|&b| b <= 0x0F),
            "the value plane is one nibble per byte, so no byte exceeds 0x0F"
        );

        let enc = c.encode(&value_plane, None, &PlaneLayout::Flat).unwrap();
        let dec = c
            .decode(
                enc.state_format_version,
                &enc.state_bytes,
                &enc.payload,
                &PlaneLayout::Flat,
                value_plane.len(),
            )
            .unwrap();
        assert_eq!(
            dec, value_plane,
            "codec must be lossless on its own producer"
        );
    }

    #[test]
    fn codec_declines_a_genuinely_packed_plane_instead_of_halving_it() {
        // Two values per byte. Reading this as one-nibble-per-byte would
        // drop every high nibble, which is exactly the silent loss the
        // `accepts` guard exists to prevent.
        let mut d = raw_descriptor(64);
        d.role = Role::Value {
            format: crate::types::role::ValueFormat::Fp4E2m1,
        };
        d.element_width = ElementWidth::Nibble;
        d.is_nibble_packed = true;
        assert!(!PerGroupCodebook.accepts(&d));
    }

    #[test]
    fn codec_declines_a_length_that_is_not_whole_groups() {
        // Declined rather than failed in `encode`: packed FP4 has one
        // candidate chain, and a codec error drops the chain entirely.
        let mut d = raw_descriptor(GROUP_SIZE + 1);
        d.role = Role::Value {
            format: crate::types::role::ValueFormat::Fp4E2m1,
        };
        d.element_width = ElementWidth::Nibble;
        assert!(!PerGroupCodebook.accepts(&d));
    }
}
