use crate::entropy::outcome::CompressOutcome;
use crate::entropy::stream_frame::{JUMP_TABLE_BYTES, split_four, write_jump_table};
use crate::error::PtwmCoreError;

use super::bitstream::BitWriter;
use super::tree::HuffTree;
use super::weights::encode_weights;

pub(crate) fn compress(dst: &mut [u8], src: &[u8]) -> Result<CompressOutcome, PtwmCoreError> {
    if src.is_empty() {
        return Ok(CompressOutcome::Incompressible);
    }

    let mut freqs = [0u32; 256];
    for &b in src {
        freqs[b as usize] += 1;
    }

    let Some(tree) = HuffTree::from_frequencies(&freqs) else {
        return Ok(CompressOutcome::Incompressible);
    };

    let weights = tree.weights_prefix();

    // Write header (weight table + jump-table placeholder) directly into dst.
    let mut hdr = Vec::with_capacity(256 + JUMP_TABLE_BYTES);
    if encode_weights(&weights, &mut hdr).is_err() {
        return Ok(CompressOutcome::Incompressible);
    }
    let jump_table_pos = hdr.len();
    hdr.resize(hdr.len() + JUMP_TABLE_BYTES, 0);
    let header_len = hdr.len();

    if header_len >= dst.len() {
        return Ok(CompressOutcome::DstTooSmall);
    }
    dst[..header_len].copy_from_slice(&hdr);

    // Encode each of the 4 streams, reusing a single scratch buffer sized for
    // the largest segment. BitWriter writes from the end of the buffer
    // backward, so the written region is always `scratch[scratch.len()-n..]`
    // and prior contents are overwritten as needed — safe to reuse.
    let segments = split_four(src.len());
    let max_seg = segments.iter().map(|(s, e)| e - s).max().unwrap_or(0);
    let max_out = max_seg.saturating_mul(tree.max_bits as usize).div_ceil(8) + 8;
    let mut scratch = vec![0u8; max_out];

    let mut stream_lens = [0u32; 4];
    let mut off = header_len;

    for (i, &(start, end)) in segments.iter().enumerate() {
        let seg = &src[start..end];
        let written = encode_stream_into(seg, &tree, &mut scratch)?;

        if off + written > dst.len() {
            return Ok(CompressOutcome::DstTooSmall);
        }
        // The jump table stores u32 lengths; a single stream exceeding
        // 4 GB would silently truncate on cast and corrupt the table. Not
        // reachable from the pipeline (chunks ≤ ~tens of MB) but cheap to
        // assert defensively — reject such inputs rather than emit a bad
        // blob.
        if written > u32::MAX as usize {
            return Ok(CompressOutcome::LenOverflow);
        }
        let scratch_start = scratch.len() - written;
        dst[off..off + written].copy_from_slice(&scratch[scratch_start..]);
        off += written;
        stream_lens[i] = written as u32;
    }

    let total = off;
    if total >= src.len() {
        return Ok(CompressOutcome::NotBeneficial);
    }

    write_jump_table(
        &mut dst[jump_table_pos..jump_table_pos + JUMP_TABLE_BYTES],
        &stream_lens,
    );

    Ok(CompressOutcome::Encoded(total))
}

/// Encode a segment into the provided scratch buffer using the backward
/// BitWriter. Returns the number of bytes written. The written region
/// occupies `scratch[scratch.len() - returned .. scratch.len()]`.
fn encode_stream_into(
    src: &[u8],
    tree: &HuffTree,
    scratch: &mut [u8],
) -> Result<usize, PtwmCoreError> {
    let mut writer = BitWriter::new(scratch);
    for &b in src {
        let (bits, nbits) = tree.code_for(b).ok_or_else(|| {
            PtwmCoreError::HuffmanCompress(format!("missing code for symbol {b}"))
        })?;
        writer.write_bits(bits, nbits);
    }
    let start = writer.finish();
    let total_len = scratch.len();
    Ok(total_len - start)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compress_returns_incompressible_for_empty_input() {
        let mut dst = [0u8; 64];
        assert_eq!(
            compress(&mut dst, &[]).unwrap(),
            CompressOutcome::Incompressible
        );
    }

    #[test]
    fn compress_rejects_tiny_dst() {
        // 2-symbol input has a 1-byte weight header, so header_len = 1 + 16 = 17.
        // A 10-byte dst can't hold that.
        let src = [0u8; 100];
        let mut dst = [0u8; 10];
        let out = compress(&mut dst, &src).unwrap();
        assert_eq!(out, CompressOutcome::DstTooSmall);
    }

    #[test]
    fn compress_reports_not_beneficial_for_near_random_data() {
        let src: Vec<u8> = (0..64).map(|i| (i * 31 + 7) as u8).collect();
        let mut dst = vec![0u8; 4096];
        // Either NotBeneficial or Incompressible is acceptable for this tiny
        // near-uniform input. The critical guarantee is that it never returns
        // Encoded (which would mean a bogus compressed size).
        match compress(&mut dst, &src).unwrap() {
            CompressOutcome::NotBeneficial | CompressOutcome::Incompressible => {}
            other => panic!("unexpected outcome {other:?}"),
        }
    }
}
