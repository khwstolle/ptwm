//! Order1ScaleAC shared-state fitter with auto-K clustering.

use crate::codecs::order1_scale_ac::{
    ALPHABET, JointCounts, MarginalCounts, STATE_FORMAT_VERSION, StateV0, accumulate_counts_single,
    bitmap_has, bitmap_rank, empty_joint, quantize_and_smooth, serialize_state,
};
use crate::quantize::ORDER1_QUANT_TOTAL;

const K_MAX: usize = 8;

const ROW_TOTAL: f64 = ORDER1_QUANT_TOTAL as f64;

#[derive(Debug, Clone)]
pub(crate) struct PerTensor {
    pub joint: JointCounts,
    pub marginal: MarginalCounts,
    #[allow(dead_code)] // diagnostic-only
    pub n_symbols: usize,
}

/// Stage 1 — pre-fit. Accumulate (joint, marginal) counts per SCALE plane.
/// Output is one `PerTensor` per input. Used by stages 2–4 in subsequent
/// tasks.
pub(crate) fn pre_fit(planes_scale: &[(&[u8], u32)]) -> Vec<PerTensor> {
    planes_scale
        .iter()
        .map(|(plane, row_len)| {
            let (joint, marginal) = accumulate_counts_single(plane, *row_len);
            PerTensor {
                joint,
                marginal,
                n_symbols: plane.len(),
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Stage 2 — pre_fit_states + symmetric KL distance
// ---------------------------------------------------------------------------

fn pre_fit_states(per_tensor: &[PerTensor]) -> Vec<StateV0> {
    per_tensor
        .iter()
        .map(|pt| quantize_and_smooth(pt.joint.clone(), pt.marginal))
        .collect()
}

fn pmf_to_prob(pmf: &[u16; ALPHABET]) -> [f64; ALPHABET] {
    let mut out = [0f64; ALPHABET];
    for i in 0..ALPHABET {
        out[i] = pmf[i] as f64 / ROW_TOTAL;
    }
    out
}

fn row_kl(p: &[u16; ALPHABET], q: &[u16; ALPHABET]) -> f64 {
    let mut s = 0f64;
    let prob_p = pmf_to_prob(p);
    let prob_q = pmf_to_prob(q);
    for i in 0..ALPHABET {
        if prob_p[i] > 0.0 && prob_q[i] > 0.0 {
            s += prob_p[i] * (prob_p[i] / prob_q[i]).ln();
        }
    }
    s / std::f64::consts::LN_2
}

fn lookup_row_or_marginal(state: &StateV0, prev: u8) -> &[u16; ALPHABET] {
    if bitmap_has(&state.present_bitmap, prev) {
        &state.conditional_rows[bitmap_rank(&state.present_bitmap, prev)]
    } else {
        &state.marginal
    }
}

pub(crate) fn symmetric_kl(p: &StateV0, q: &StateV0) -> f64 {
    let mut total = 0f64;
    let p_marg = pmf_to_prob(&p.marginal);
    let q_marg = pmf_to_prob(&q.marginal);
    let mix_marg: [f64; ALPHABET] = {
        let mut m = [0f64; ALPHABET];
        for i in 0..ALPHABET {
            m[i] = 0.5 * (p_marg[i] + q_marg[i]);
        }
        m
    };
    for prev in 0..ALPHABET {
        let p_row = lookup_row_or_marginal(p, prev as u8);
        let q_row = lookup_row_or_marginal(q, prev as u8);
        let kl_pq = row_kl(p_row, q_row);
        let kl_qp = row_kl(q_row, p_row);
        total += mix_marg[prev] * (kl_pq + kl_qp);
    }
    total
}

// ---------------------------------------------------------------------------
// Stage 3 — agglomerative clustering
// ---------------------------------------------------------------------------

fn pairwise_distances(states: &[StateV0]) -> Vec<f64> {
    let n = states.len();
    let mut m = vec![0f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = symmetric_kl(&states[i], &states[j]);
            m[i * n + j] = d;
            m[j * n + i] = d;
        }
    }
    m
}

#[derive(Debug, Clone)]
pub(crate) struct MergeStep {
    pub left: usize,
    pub right: usize,
    #[allow(dead_code)] // diagnostic-only
    pub distance: f64,
}

pub(crate) fn agglomerative(distances: Vec<f64>, n: usize) -> Vec<MergeStep> {
    if n < 2 {
        return Vec::new();
    }
    let mut sizes: Vec<usize> = vec![1; n];
    let mut active: Vec<bool> = vec![true; n];
    let mut dist = distances;
    let mut merges = Vec::with_capacity(n - 1);

    for _ in 0..(n - 1) {
        let mut best = (0usize, 1usize);
        let mut best_d = f64::INFINITY;
        for i in 0..n {
            if !active[i] {
                continue;
            }
            for j in (i + 1)..n {
                if !active[j] {
                    continue;
                }
                let d = dist[i * n + j];
                if d < best_d {
                    best_d = d;
                    best = (i, j);
                }
            }
        }
        let (l, r) = best;
        merges.push(MergeStep {
            left: l,
            right: r,
            distance: best_d,
        });
        let s_l = sizes[l] as f64;
        let s_r = sizes[r] as f64;
        for k in 0..n {
            if k == l || k == r || !active[k] {
                continue;
            }
            let d_lk = dist[l * n + k];
            let d_rk = dist[r * n + k];
            let d_new = (s_l * d_lk + s_r * d_rk) / (s_l + s_r);
            dist[l * n + k] = d_new;
            dist[k * n + l] = d_new;
        }
        sizes[l] += sizes[r];
        active[r] = false;
    }
    merges
}

pub(crate) fn slice_dendrogram(merges: &[MergeStep], n: usize, k: usize) -> Vec<u16> {
    debug_assert!(k <= n, "slice_dendrogram: k={k} must be <= n={n}");
    let mut parent: Vec<usize> = (0..n).collect();
    // Iterative path-compressing find — avoids stack growth on deep
    // chains in the (currently latent) bigger-corpus path.
    fn find(parent: &mut Vec<usize>, mut x: usize) -> usize {
        let mut root = x;
        while parent[root] != root {
            root = parent[root];
        }
        while parent[x] != root {
            let next = parent[x];
            parent[x] = root;
            x = next;
        }
        root
    }
    let n_merges = n.saturating_sub(k).min(merges.len());
    for merge in &merges[..n_merges] {
        let rl = find(&mut parent, merge.left);
        let rr = find(&mut parent, merge.right);
        if rl != rr {
            parent[rr.max(rl)] = rl.min(rr);
        }
    }
    let roots: Vec<usize> = (0..n).map(|i| find(&mut parent, i)).collect();
    let mut unique: Vec<usize> = roots.clone();
    unique.sort();
    unique.dedup();
    let mut assignment = vec![0u16; n];
    for i in 0..n {
        let pos = unique.iter().position(|&r| r == roots[i]).unwrap();
        assignment[i] = pos as u16;
    }
    assignment
}

// ---------------------------------------------------------------------------
// Stage 4 — MDL evaluation
// ---------------------------------------------------------------------------

fn pool_cluster_counts(
    per_tensor: &[PerTensor],
    assignment: &[u16],
    n_clusters: usize,
) -> Vec<(JointCounts, MarginalCounts)> {
    let mut clusters: Vec<_> = (0..n_clusters)
        .map(|_| (empty_joint(), [0u32; ALPHABET]))
        .collect();
    for (idx, pt) in per_tensor.iter().enumerate() {
        let c = assignment[idx] as usize;
        for prev in 0..ALPHABET {
            for cur in 0..ALPHABET {
                clusters[c].0[prev][cur] += pt.joint[prev][cur];
            }
        }
        for cur in 0..ALPHABET {
            clusters[c].1[cur] += pt.marginal[cur];
        }
    }
    clusters
}

fn cross_entropy_bits(pt: &PerTensor, cluster_state: &StateV0) -> f64 {
    let mut bits = 0f64;
    for prev in 0..ALPHABET {
        let row = lookup_row_or_marginal(cluster_state, prev as u8);
        for cur in 0..ALPHABET {
            let n = pt.joint[prev][cur];
            if n == 0 {
                continue;
            }
            let prob = row[cur] as f64 / ROW_TOTAL;
            if prob > 0.0 {
                bits += n as f64 * -prob.log2();
            }
        }
    }
    for cur in 0..ALPHABET {
        let n = pt.marginal[cur];
        if n == 0 {
            continue;
        }
        let prob = cluster_state.marginal[cur] as f64 / ROW_TOTAL;
        if prob > 0.0 {
            bits += n as f64 * -prob.log2();
        }
    }
    bits
}

#[derive(Debug, Clone)]
pub(crate) struct McdlResult {
    pub k: usize,
    pub total_bits: f64,
    pub cluster_states: Vec<StateV0>,
    #[allow(dead_code)] // assignment retained for diagnostics; not used by emit.
    pub assignment: Vec<u16>,
}

/// `<` against NaN is always false, so a NaN total_bits would silently
/// anchor as `best` and mask every later valid K. Refuse it loudly.
pub(crate) fn validate_mdl_finite(total_bits: f64, k: usize) -> Result<(), PtwmCoreError> {
    if total_bits.is_finite() {
        Ok(())
    } else {
        Err(PtwmCoreError::InvalidContainer(format!(
            "Order1ScaleAC fit: non-finite MDL cost {total_bits} at k={k}"
        )))
    }
}

pub(crate) fn evaluate_mdl(per_tensor: &[PerTensor], merges: &[MergeStep], k: usize) -> McdlResult {
    let assignment = slice_dendrogram(merges, per_tensor.len(), k);
    let pooled = pool_cluster_counts(per_tensor, &assignment, k);
    let cluster_states: Vec<StateV0> = pooled
        .into_iter()
        .map(|(j, m)| quantize_and_smooth(j, m))
        .collect();
    let mut bits = 0f64;
    for state in &cluster_states {
        bits += 8.0 * serialize_state(state).len() as f64;
    }
    for (idx, pt) in per_tensor.iter().enumerate() {
        bits += cross_entropy_bits(pt, &cluster_states[assignment[idx] as usize]);
    }
    McdlResult {
        k,
        total_bits: bits,
        cluster_states,
        assignment,
    }
}

// ---------------------------------------------------------------------------
// Stage 5 — top-level fit() driver
// ---------------------------------------------------------------------------

use super::SharedStateEntry;
use crate::codec::CodecId;
use crate::error::PtwmCoreError;

/// Top-level Order1ScaleAC fitter. Returns 0..=K* `SharedStateEntry`s
/// where K* minimizes total MDL cost (Occam tie-break: smaller K wins).
pub(super) fn fit(planes_scale: &[(&[u8], u32)]) -> Result<Vec<SharedStateEntry>, PtwmCoreError> {
    if planes_scale.is_empty() {
        return Ok(Vec::new());
    }
    let per_tensor = pre_fit(planes_scale);
    let n = per_tensor.len();

    // N=1 → skip clustering, single state.
    if n == 1 {
        let pt = &per_tensor[0];
        let state = quantize_and_smooth(pt.joint.clone(), pt.marginal);
        let bytes = serialize_state(&state);
        return Ok(vec![SharedStateEntry::new(
            CodecId::Order1ScaleAC,
            STATE_FORMAT_VERSION,
            0b0010,
            "order1_scale_ac/cluster_0_of_1".to_string(),
            bytes,
        )?]);
    }

    let states = pre_fit_states(&per_tensor);
    let dist = pairwise_distances(&states);
    let merges = agglomerative(dist, n);

    // Try K = 1..=K_MAX; pick min total_bits (Occam: smaller K wins ties).
    let k_max = K_MAX.min(n);
    let mut best: Option<McdlResult> = None;
    for k in 1..=k_max {
        let r = evaluate_mdl(&per_tensor, &merges, k);
        validate_mdl_finite(r.total_bits, k)?;
        let take = match &best {
            None => true,
            Some(b) => r.total_bits < b.total_bits,
        };
        if take {
            best = Some(r);
        }
    }
    let result = best.ok_or_else(|| {
        PtwmCoreError::InvalidContainer("Order1ScaleAC fit: no MDL result".into())
    })?;
    let k_star = result.k;
    result
        .cluster_states
        .into_iter()
        .enumerate()
        .map(|(idx, state)| {
            SharedStateEntry::new(
                CodecId::Order1ScaleAC,
                STATE_FORMAT_VERSION,
                0b0010,
                format!("order1_scale_ac/cluster_{idx}_of_{k_star}"),
                serialize_state(&state),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pre_fit_aggregates_counts() {
        let plane: Vec<u8> = (0..32u8).collect();
        let pre = pre_fit(&[(&plane, 16)]);
        assert_eq!(pre.len(), 1);
        assert_eq!(pre[0].n_symbols, 32);
        // Two rows, marginal[0] and marginal[16] each set once.
        assert_eq!(pre[0].marginal[0], 1);
        assert_eq!(pre[0].marginal[16], 1);
    }

    #[test]
    fn pre_fit_empty_input_is_empty() {
        let pre = pre_fit(&[]);
        assert!(pre.is_empty());
    }

    #[test]
    fn kl_identical_states_is_zero() {
        let plane = vec![100u8; 256];
        let pre = pre_fit(&[(&plane, 16)]);
        let states = pre_fit_states(&pre);
        let d = symmetric_kl(&states[0], &states[0]);
        assert!(d.abs() < 1e-9, "expected ~0, got {d}");
    }

    #[test]
    fn kl_different_states_is_positive() {
        // Planes must be long enough to overcome Laplace smoothing (256 pseudo-counts).
        // Use 4096 symbols each to ensure a strong distributional signal.
        let plane_a: Vec<u8> = (0..4096u32).map(|i| (i % 50) as u8).collect();
        let plane_b: Vec<u8> = (0..4096u32).map(|i| (200 + (i % 30)) as u8).collect();
        let pre = pre_fit(&[(&plane_a, 16), (&plane_b, 16)]);
        let states = pre_fit_states(&pre);
        let d = symmetric_kl(&states[0], &states[1]);
        assert!(d > 0.5, "expected > 0.5 bits/symbol, got {d}");
    }

    #[test]
    fn agglomerative_two_clusters() {
        let plane_a: Vec<u8> = (0..256u32).map(|i| (i * 7 % 40) as u8).collect();
        let plane_b: Vec<u8> = (0..256u32).map(|i| (180 + (i * 3 % 40)) as u8).collect();
        let pre = pre_fit(&[
            (&plane_a, 16),
            (&plane_a, 16),
            (&plane_b, 16),
            (&plane_b, 16),
        ]);
        let states = pre_fit_states(&pre);
        let dist = pairwise_distances(&states);
        let merges = agglomerative(dist, 4);
        assert_eq!(merges.len(), 3);
        let assign_k2 = slice_dendrogram(&merges, 4, 2);
        assert_eq!(assign_k2[0], assign_k2[1]);
        assert_eq!(assign_k2[2], assign_k2[3]);
        assert_ne!(assign_k2[0], assign_k2[2]);
    }

    #[test]
    fn fit_empty_input_returns_no_entries() {
        let entries = fit(&[]).unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn fit_single_plane_returns_one_entry() {
        let plane: Vec<u8> = (0..256u32).map(|i| (i % 64) as u8).collect();
        let entries = fit(&[(&plane, 16)]).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].codec_id, crate::codec::CodecId::Order1ScaleAC);
        assert!(entries[0].name.contains("cluster_0_of_1"));
    }

    #[test]
    fn fit_caps_at_k_max() {
        // 20 distinct random chains → auto-K should not exceed K_MAX = 8.
        let mut planes_owned: Vec<Vec<u8>> = Vec::new();
        for seed in 0..20u64 {
            let mut x = seed.wrapping_mul(0x9E3779B97F4A7C15);
            let mut p = Vec::with_capacity(512);
            for _ in 0..512 {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                p.push((x >> 56) as u8);
            }
            planes_owned.push(p);
        }
        let refs: Vec<(&[u8], u32)> = planes_owned.iter().map(|p| (p.as_slice(), 16)).collect();
        let entries = fit(&refs).unwrap();
        assert!(!entries.is_empty(), "expected >= 1 cluster");
        assert!(
            entries.len() <= K_MAX,
            "K* {} > K_MAX {}",
            entries.len(),
            K_MAX
        );
    }

    #[test]
    fn fit_deterministic() {
        let plane_a: Vec<u8> = (0..512u32).map(|i| (i % 32) as u8).collect();
        let plane_b: Vec<u8> = (0..512u32).map(|i| (200 + (i % 32)) as u8).collect();
        let refs = [(plane_a.as_slice(), 16u32), (plane_b.as_slice(), 16u32)];
        let e1 = fit(&refs).unwrap();
        let e2 = fit(&refs).unwrap();
        assert_eq!(e1.len(), e2.len());
        for (a, b) in e1.iter().zip(e2.iter()) {
            assert_eq!(a.state_bytes, b.state_bytes);
            assert_eq!(a.name, b.name);
        }
    }

    #[test]
    fn validate_mdl_finite_rejects_nan_and_inf() {
        assert!(validate_mdl_finite(f64::NAN, 3).is_err());
        assert!(validate_mdl_finite(f64::INFINITY, 1).is_err());
        assert!(validate_mdl_finite(f64::NEG_INFINITY, 1).is_err());
        assert!(validate_mdl_finite(0.0, 1).is_ok());
        assert!(validate_mdl_finite(1.5e9, 8).is_ok());
    }

    #[test]
    fn mdl_unimodal_corpus_picks_k1() {
        let plane: Vec<u8> = (0..512u32).map(|i| ((i * 7) % 64) as u8).collect();
        let refs: Vec<(&[u8], u32)> = vec![(plane.as_slice(), 16u32); 4];
        let pre = pre_fit(&refs);
        let states = pre_fit_states(&pre);
        let dist = pairwise_distances(&states);
        let merges = agglomerative(dist, 4);
        let mut best_k = 0;
        let mut best_bits = f64::INFINITY;
        for k in 1..=4 {
            let r = evaluate_mdl(&pre, &merges, k);
            if r.total_bits < best_bits {
                best_bits = r.total_bits;
                best_k = k;
            }
        }
        assert_eq!(best_k, 1);
    }
}
