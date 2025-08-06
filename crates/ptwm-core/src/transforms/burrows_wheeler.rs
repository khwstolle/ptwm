//! `BurrowsWheeler` op — block-wise Burrows–Wheeler Transform (BWT).
//!
//! The BWT is a reversible permutation of a byte block that clusters
//! bytes sharing the same following context into long runs. It carries no
//! compression on its own; its value is as a *preprocessing* stage: after
//! the bit-reorder + byte-split chain the exponent plane is mostly long
//! runs of similar bytes, and BWT — followed by
//! [`crate::transforms::MoveToFront`] and an entropy coder — extracts the
//! remaining structure that a block-local Huffman / rANS pass cannot see.
//! This is the classic bzip2 pipeline.
//!
//! Encode is `O(n log² n)` (prefix-doubling rotation sort); decode is
//! `O(n)` (LF-mapping walk). The asymmetry is intentional — this op
//! targets a release-time high-ratio mode where slow encode is acceptable
//! but fast decode matters.
//!
//! ## Block structure and wire layout
//!
//! The input is split into fixed-size blocks of [`BWT_BLOCK_SIZE`] bytes
//! (the final block may be shorter). Each block is transformed
//! independently and emitted as:
//!
//! ```text
//! [ primary_index : u32 little-endian ] [ transformed bytes ]
//! ```
//!
//! The `primary_index` is the row of the sorted-rotation matrix that holds
//! the original block — the standard datum the inverse needs. Block
//! boundaries are recovered on decode without extra metadata: every block
//! but the last contributes exactly `BWT_BLOCK_SIZE` payload bytes, so a
//! greedy walk (`4 + min(BWT_BLOCK_SIZE, remaining)` per block) re-derives
//! them unambiguously. The output is therefore `n + 4·⌈n / BWT_BLOCK_SIZE⌉`
//! bytes long.
//!
//! ## Element widths
//!
//! BWT is defined on the raw byte stream. The op requires a `Byte`-width,
//! non-nibble-packed plane — its production target is the byte-wide
//! exponent plane — and rejects everything else.
//!
//! Round-trip is exact for every input: `inverse(forward(x)) == x`.

use std::sync::Arc;

use crate::error::PtwmCoreError;
use crate::transforms::op::{Op, OpId, Plane};
use crate::types::descriptor::{ElementWidth, PlaneDescriptor};

/// BWT block size in bytes. Larger blocks compress slightly better but
/// cost `O(block log² block)` to sort; 256 KiB is a balance comparable to
/// bzip2's tunable range while keeping per-block encode time bounded.
pub const BWT_BLOCK_SIZE: usize = 1 << 18;

/// Per-block header size: the little-endian `u32` primary index.
const HEADER_BYTES: usize = 4;

/// Block-wise Burrows–Wheeler Transform.
#[derive(Debug, Default, Clone, Copy)]
pub struct BurrowsWheeler;

impl BurrowsWheeler {
    pub const fn new() -> Self {
        Self
    }
}

/// Number of `BWT_BLOCK_SIZE` blocks an `n`-byte input splits into.
#[inline]
fn block_count(n: usize) -> usize {
    n.div_ceil(BWT_BLOCK_SIZE)
}

impl Op for BurrowsWheeler {
    fn propagate_descriptors(
        &self,
        inputs: &[PlaneDescriptor],
    ) -> Result<Vec<PlaneDescriptor>, PtwmCoreError> {
        if inputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "BurrowsWheeler.propagate_descriptors: expected 1 input, got {}",
                inputs.len()
            )));
        }
        let d = &inputs[0];
        require_byte_plane(d, "propagate_descriptors")?;
        let n = d.length_bytes as usize;
        let mut out = d.clone();
        out.length_bytes = (n + HEADER_BYTES * block_count(n)) as u64;
        Ok(vec![out])
    }

    fn forward(&self, inputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        if inputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "BurrowsWheeler.forward: expected 1 input, got {}",
                inputs.len()
            )));
        }
        let p = &inputs[0];
        require_byte_plane(&p.descriptor, "forward")?;
        let bytes = bwt_encode(&p.bytes);
        let mut descriptor = p.descriptor.clone();
        descriptor.length_bytes = bytes.len() as u64;
        Ok(vec![Plane {
            bytes: Arc::from(bytes.into_boxed_slice()),
            descriptor,
        }])
    }

    fn inverse(&self, outputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        if outputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "BurrowsWheeler.inverse: expected 1 input, got {}",
                outputs.len()
            )));
        }
        let p = &outputs[0];
        require_byte_plane(&p.descriptor, "inverse")?;
        let bytes = bwt_decode(&p.bytes)?;
        let mut descriptor = p.descriptor.clone();
        descriptor.length_bytes = bytes.len() as u64;
        Ok(vec![Plane {
            bytes: Arc::from(bytes.into_boxed_slice()),
            descriptor,
        }])
    }

    fn id(&self) -> OpId {
        OpId::BurrowsWheeler
    }

    fn write_params(&self, _out: &mut Vec<u8>) {
        // No parameters: block size is a fixed compile-time constant.
    }
}

// ---------------------------------------------------------------------------
// Block stream encode / decode
// ---------------------------------------------------------------------------

fn bwt_encode(input: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len() + HEADER_BYTES * block_count(input.len()));
    for chunk in input.chunks(BWT_BLOCK_SIZE) {
        let (last_column, primary) = bwt_block_forward(chunk);
        out.extend_from_slice(&primary.to_le_bytes());
        out.extend_from_slice(&last_column);
    }
    out
}

fn bwt_decode(data: &[u8]) -> Result<Vec<u8>, PtwmCoreError> {
    let mut out = Vec::with_capacity(data.len());
    let mut pos = 0usize;
    while pos < data.len() {
        if pos + HEADER_BYTES > data.len() {
            return Err(PtwmCoreError::InvalidContainer(
                "BurrowsWheeler.inverse: truncated block header".into(),
            ));
        }
        let primary =
            u32::from_le_bytes(data[pos..pos + HEADER_BYTES].try_into().unwrap()) as usize;
        pos += HEADER_BYTES;
        // Every block but the last carries exactly BWT_BLOCK_SIZE payload
        // bytes, so the greedy minimum re-derives the boundary.
        let block_len = core::cmp::min(BWT_BLOCK_SIZE, data.len() - pos);
        if block_len == 0 {
            return Err(PtwmCoreError::InvalidContainer(
                "BurrowsWheeler.inverse: block header with no payload".into(),
            ));
        }
        let block = &data[pos..pos + block_len];
        pos += block_len;
        out.extend_from_slice(&bwt_block_inverse(block, primary)?);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Single-block transform
// ---------------------------------------------------------------------------

/// Forward BWT of one block: returns the last column `L` of the
/// sorted-rotation matrix and the `primary` row index (the row equal to
/// the original block). `block` must be non-empty and at most
/// `BWT_BLOCK_SIZE` bytes, so `primary` fits in a `u32`.
fn bwt_block_forward(block: &[u8]) -> (Vec<u8>, u32) {
    let n = block.len();
    debug_assert!((1..=BWT_BLOCK_SIZE).contains(&n));
    let sa = sorted_rotations(block);
    let mut last_column = vec![0u8; n];
    let mut primary = 0u32;
    for (row, &start) in sa.iter().enumerate() {
        let start = start as usize;
        // Last character of the rotation starting at `start`. With
        // `start < n`, the cyclic predecessor is `n - 1` when `start == 0`
        // and `start - 1` otherwise — cheaper than a modulo per element.
        let idx = if start == 0 { n - 1 } else { start - 1 };
        last_column[row] = block[idx];
        if start == 0 {
            primary = row as u32;
        }
    }
    (last_column, primary)
}

/// Inverse BWT of one block via the LF-mapping. `last_column` is `L`;
/// `primary` is the row of the original block in the sorted matrix.
fn bwt_block_inverse(last_column: &[u8], primary: usize) -> Result<Vec<u8>, PtwmCoreError> {
    let n = last_column.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if primary >= n {
        return Err(PtwmCoreError::InvalidContainer(format!(
            "BurrowsWheeler.inverse: primary index {primary} out of range for block of {n}"
        )));
    }

    // Cumulative counts: `cum[c]` is the number of bytes in L strictly less
    // than `c` — i.e. the first row of the sorted matrix whose first column
    // equals `c`.
    let mut counts = [0usize; 256];
    for &b in last_column {
        counts[b as usize] += 1;
    }
    let mut cum = [0usize; 256];
    let mut running = 0usize;
    for (c, slot) in cum.iter_mut().enumerate() {
        *slot = running;
        running += counts[c];
    }

    // LF mapping: lf[i] is the row whose first column holds the byte L[i].
    // The j-th occurrence of a byte in L maps to the j-th occurrence in the
    // (sorted) first column, which preserves the rotation order.
    let mut occ = [0usize; 256];
    let mut lf = vec![0u32; n];
    for (i, &b) in last_column.iter().enumerate() {
        let c = b as usize;
        lf[i] = (cum[c] + occ[c]) as u32;
        occ[c] += 1;
    }

    // Walk the mapping backwards from `primary`, which is the row holding
    // the original block: L[primary] is the last byte, L[lf[primary]] the
    // one before it, and so on.
    let mut out = vec![0u8; n];
    let mut row = primary;
    for slot in out.iter_mut().rev() {
        *slot = last_column[row];
        row = lf[row] as usize;
    }
    Ok(out)
}

/// Sort the `n` cyclic rotations of `block` lexicographically and return
/// their start positions in sorted order, via prefix doubling on cyclic
/// ranks (`O(n log² n)`).
fn sorted_rotations(block: &[u8]) -> Vec<u32> {
    let n = block.len();
    let mut sa: Vec<u32> = (0..n as u32).collect();
    if n <= 1 {
        return sa;
    }

    let mut rank: Vec<u32> = block.iter().map(|&b| b as u32).collect();
    let mut next_rank = vec![0u32; n];
    // `k < n` holds throughout the loop body (the doubling break fires
    // before re-entry once `k >= n`), so `i + k < 2n` and the cyclic
    // wrap is a single conditional subtraction rather than a modulo.
    let wrap = |i: usize, k: usize| -> usize {
        let s = i + k;
        if s >= n { s - n } else { s }
    };
    let mut k = 1usize;
    loop {
        // Sort rotations by the pair (rank at start, rank `k` positions on,
        // wrapping) — i.e. by their first 2k cyclic characters. An unstable
        // sort suffices: the comparator is a total order on the full keys,
        // and ties occur only between genuinely-equal rotations, for which
        // any order round-trips.
        sa.sort_unstable_by(|&a, &b| {
            let a = a as usize;
            let b = b as usize;
            (rank[a], rank[wrap(a, k)]).cmp(&(rank[b], rank[wrap(b, k)]))
        });

        // Re-rank: equal keys share a rank, distinct keys increment.
        next_rank[sa[0] as usize] = 0;
        for w in 1..n {
            let prev = sa[w - 1] as usize;
            let cur = sa[w] as usize;
            let prev_key = (rank[prev], rank[wrap(prev, k)]);
            let cur_key = (rank[cur], rank[wrap(cur, k)]);
            next_rank[cur] = next_rank[prev] + u32::from(cur_key != prev_key);
        }
        rank.copy_from_slice(&next_rank);

        // All rotations distinct → fully sorted.
        if rank[sa[n - 1] as usize] as usize == n - 1 {
            break;
        }
        k <<= 1;
        if k >= n {
            // Doubling has covered ≥ n characters: the order is complete
            // (remaining ties are genuinely-equal rotations of a periodic
            // block, for which any consistent order round-trips).
            break;
        }
    }
    sa
}

fn require_byte_plane(d: &PlaneDescriptor, ctx: &str) -> Result<(), PtwmCoreError> {
    if d.element_width != ElementWidth::Byte || d.is_nibble_packed {
        return Err(PtwmCoreError::InvalidContainer(format!(
            "BurrowsWheeler.{ctx}: requires a byte-width, non-nibble-packed plane \
             (got element_width={:?}, is_nibble_packed={})",
            d.element_width, d.is_nibble_packed
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::descriptor::Layout;
    use crate::types::role::Role;

    fn plane(bytes: Vec<u8>) -> Plane {
        Plane {
            descriptor: PlaneDescriptor {
                role: Role::Raw,
                element_width: ElementWidth::Byte,
                length_bytes: bytes.len() as u64,
                layout: Layout::Flat,
                derives_from_tensor: None,
                residual_of: None,
                is_nibble_packed: false,
                vendor_bytes: vec![],
            },
            bytes: Arc::from(bytes.into_boxed_slice()),
        }
    }

    fn round_trip(raw: &[u8]) {
        let op = BurrowsWheeler::new();
        let fwd = op.forward(&[plane(raw.to_vec())]).unwrap();
        // Output length matches the descriptor prediction.
        let descs = op
            .propagate_descriptors(&[plane(raw.to_vec()).descriptor])
            .unwrap();
        assert_eq!(descs[0].length_bytes, fwd[0].bytes.len() as u64);
        let inv = op.inverse(&[fwd[0].clone()]).unwrap();
        assert_eq!(
            inv[0].bytes.as_ref(),
            raw,
            "round-trip mismatch for {raw:?}"
        );
    }

    #[test]
    fn block_forward_known_vector() {
        // BWT of "banana" (rotation form, no sentinel). Sorted rotations:
        //   anaban | nanaba | abanan | bananá... compute L directly.
        let (l, primary) = bwt_block_forward(b"banana");
        // Reconstruct to confirm the (L, primary) pair is self-consistent.
        let back = bwt_block_inverse(&l, primary as usize).unwrap();
        assert_eq!(back, b"banana");
    }

    #[test]
    fn round_trip_small_strings() {
        round_trip(b"banana");
        round_trip(b"mississippi");
        round_trip(b"the quick brown fox jumps over the lazy dog");
    }

    #[test]
    fn round_trip_single_byte() {
        round_trip(b"X");
    }

    #[test]
    fn round_trip_empty() {
        let op = BurrowsWheeler::new();
        let fwd = op.forward(&[plane(vec![])]).unwrap();
        assert!(fwd[0].bytes.is_empty());
        let inv = op.inverse(&[fwd[0].clone()]).unwrap();
        assert!(inv[0].bytes.is_empty());
    }

    #[test]
    fn round_trip_all_same_byte() {
        // Maximally periodic: every rotation is identical.
        round_trip(&vec![0xAAu8; 257]);
    }

    #[test]
    fn round_trip_short_period() {
        // Periodic block with two distinct rotations ("abab…").
        let raw: Vec<u8> = (0..512)
            .map(|i| if i % 2 == 0 { b'a' } else { b'b' })
            .collect();
        round_trip(&raw);
    }

    #[test]
    fn round_trip_pseudo_random_multi_block() {
        // Larger than one block to exercise the block-stream framing and
        // the greedy boundary recovery on decode.
        let mut state = 0x9E37_79B9u32;
        let n = BWT_BLOCK_SIZE * 2 + 12_345;
        let raw: Vec<u8> = (0..n)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 24) as u8
            })
            .collect();
        let op = BurrowsWheeler::new();
        let fwd = op.forward(&[plane(raw.clone())]).unwrap();
        // Three blocks → three u32 headers of overhead.
        assert_eq!(fwd[0].bytes.len(), raw.len() + HEADER_BYTES * 3);
        let inv = op.inverse(&[fwd[0].clone()]).unwrap();
        assert_eq!(inv[0].bytes.as_ref(), raw);
    }

    #[test]
    fn round_trip_low_cardinality_runs() {
        // Resembles a byte-split exponent plane: long runs of a few values.
        let mut raw = Vec::new();
        for block in 0..200u32 {
            let byte = (block % 5) as u8 + 0x7E;
            raw.extend(std::iter::repeat_n(byte, 64));
        }
        round_trip(&raw);
    }

    #[test]
    fn inverse_rejects_truncated_header() {
        // 2 bytes — fewer than a 4-byte block header.
        let op = BurrowsWheeler::new();
        let bad = plane(vec![0u8, 1]);
        assert!(op.inverse(&[bad]).is_err());
    }

    #[test]
    fn inverse_rejects_header_with_no_payload() {
        // Exactly a header and nothing after it.
        let op = BurrowsWheeler::new();
        let bad = plane(vec![0u8, 0, 0, 0]);
        assert!(op.inverse(&[bad]).is_err());
    }

    #[test]
    fn inverse_rejects_out_of_range_primary() {
        // Header claims primary=5 for a 2-byte payload.
        let err = bwt_block_inverse(b"ab", 5);
        assert!(err.is_err());
    }

    #[test]
    fn rejects_non_byte_width() {
        let op = BurrowsWheeler::new();
        let mut p = plane(vec![0u8; 8]);
        p.descriptor.element_width = ElementWidth::Word2;
        assert!(op.forward(&[p.clone()]).is_err());
        assert!(op.inverse(&[p]).is_err());
    }

    #[test]
    fn rejects_wrong_input_count() {
        let op = BurrowsWheeler::new();
        let p = plane(vec![1u8, 2, 3, 4]);
        assert!(op.forward(&[p.clone(), p.clone()]).is_err());
        assert!(op.inverse(&[]).is_err());
    }

    #[test]
    fn write_params_is_empty() {
        let op = BurrowsWheeler::new();
        let mut buf = Vec::new();
        op.write_params(&mut buf);
        assert!(buf.is_empty());
    }

    #[test]
    fn descriptor_length_grows_by_header_per_block() {
        let op = BurrowsWheeler::new();
        // Empty → no blocks, no overhead.
        let d0 = op
            .propagate_descriptors(&[plane(vec![]).descriptor])
            .unwrap();
        assert_eq!(d0[0].length_bytes, 0);
        // One partial block → one header.
        let d1 = op
            .propagate_descriptors(&[plane(vec![0u8; 100]).descriptor])
            .unwrap();
        assert_eq!(d1[0].length_bytes, 100 + HEADER_BYTES as u64);
    }
}
