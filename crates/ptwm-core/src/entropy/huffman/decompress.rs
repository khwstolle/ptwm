use crate::entropy::stream_frame::read_jump_table;
use crate::error::PtwmCoreError;

use super::bitstream::BitReader;
use super::tree::HuffTree;
use super::weights::decode_weights;

pub(crate) fn decompress(dst: &mut [u8], src: &[u8]) -> Result<usize, PtwmCoreError> {
    let (weights, mut src_pos) = decode_weights(src)
        .map_err(|e| PtwmCoreError::HuffmanDecompress(format!("weight decode failed: {e}")))?;

    let tree = HuffTree::from_weights(&weights)
        .map_err(|e| PtwmCoreError::HuffmanDecompress(format!("invalid weights: {e}")))?;

    let (lens_u32, consumed) = read_jump_table(&src[src_pos..])
        .map_err(|e| PtwmCoreError::HuffmanDecompress(e.to_string()))?;
    src_pos += consumed;
    let lens: [usize; 4] = [
        lens_u32[0] as usize,
        lens_u32[1] as usize,
        lens_u32[2] as usize,
        lens_u32[3] as usize,
    ];

    let tail = src.len() - src_pos;
    let total = lens
        .iter()
        .try_fold(0usize, |acc, &l| acc.checked_add(l))
        .ok_or_else(|| PtwmCoreError::HuffmanDecompress("jump table overflow".to_string()))?;

    if total != tail {
        return Err(PtwmCoreError::HuffmanDecompress(format!(
            "jump table sum {total} does not match payload tail {tail}"
        )));
    }

    let s0_end = src_pos + lens[0];
    let s1_end = s0_end + lens[1];
    let s2_end = s1_end + lens[2];
    let s3_end = s2_end + lens[3];
    let s1 = &src[src_pos..s0_end];
    let s2 = &src[s0_end..s1_end];
    let s3 = &src[s1_end..s2_end];
    let s4 = &src[s2_end..s3_end];

    let mut readers = [
        BitReader::new(s1),
        BitReader::new(s2),
        BitReader::new(s3),
        BitReader::new(s4),
    ];

    let segments = crate::entropy::stream_frame::split_four(dst.len());
    let table = tree.decode_table_raw();
    let mask = tree.table_mask() as u64;
    let max_bits = tree.max_bits;

    // Interleaved fast path: decode from all 4 streams round-robin.
    let mut out_pos = [segments[0].0, segments[1].0, segments[2].0, segments[3].0];
    let ends = [segments[0].1, segments[1].1, segments[2].1, segments[3].1];

    // Interleaved fast path: decode 4 symbols per stream per iteration.
    // Each refill adds 32 bits and we decode 2 symbols (max 22 bits at
    // MAX_TABLE_LOG=11) between refills, so the bit budget is always safe.
    // Unrolling to 4 symbols (2 refill+decode rounds) halves loop overhead.
    let fast_end0 = ends[0].saturating_sub(3);
    let fast_end1 = ends[1].saturating_sub(3);
    let fast_end2 = ends[2].saturating_sub(3);
    let fast_end3 = ends[3].saturating_sub(3);

    while out_pos[0] < fast_end0
        && out_pos[1] < fast_end1
        && out_pos[2] < fast_end2
        && out_pos[3] < fast_end3
        && readers[0].can_fast_refill()
        && readers[1].can_fast_refill()
        && readers[2].can_fast_refill()
        && readers[3].can_fast_refill()
    {
        // Round 1: refill + decode 2 symbols per stream.
        readers[0].refill();
        readers[1].refill();
        readers[2].refill();
        readers[3].refill();

        dst[out_pos[0]] = readers[0].decode_symbol(table, mask);
        dst[out_pos[1]] = readers[1].decode_symbol(table, mask);
        dst[out_pos[2]] = readers[2].decode_symbol(table, mask);
        dst[out_pos[3]] = readers[3].decode_symbol(table, mask);
        out_pos[0] += 1;
        out_pos[1] += 1;
        out_pos[2] += 1;
        out_pos[3] += 1;

        dst[out_pos[0]] = readers[0].decode_symbol(table, mask);
        dst[out_pos[1]] = readers[1].decode_symbol(table, mask);
        dst[out_pos[2]] = readers[2].decode_symbol(table, mask);
        dst[out_pos[3]] = readers[3].decode_symbol(table, mask);
        out_pos[0] += 1;
        out_pos[1] += 1;
        out_pos[2] += 1;
        out_pos[3] += 1;

        // Round 2: refill + decode 2 more symbols per stream.
        readers[0].refill();
        readers[1].refill();
        readers[2].refill();
        readers[3].refill();

        dst[out_pos[0]] = readers[0].decode_symbol(table, mask);
        dst[out_pos[1]] = readers[1].decode_symbol(table, mask);
        dst[out_pos[2]] = readers[2].decode_symbol(table, mask);
        dst[out_pos[3]] = readers[3].decode_symbol(table, mask);
        out_pos[0] += 1;
        out_pos[1] += 1;
        out_pos[2] += 1;
        out_pos[3] += 1;

        dst[out_pos[0]] = readers[0].decode_symbol(table, mask);
        dst[out_pos[1]] = readers[1].decode_symbol(table, mask);
        dst[out_pos[2]] = readers[2].decode_symbol(table, mask);
        dst[out_pos[3]] = readers[3].decode_symbol(table, mask);
        out_pos[0] += 1;
        out_pos[1] += 1;
        out_pos[2] += 1;
        out_pos[3] += 1;
    }

    // Tail: finish each stream sequentially with error-checked decoding.
    for s in 0..4 {
        decode_stream_tail(
            &mut readers[s],
            table,
            mask,
            max_bits,
            &mut dst[out_pos[s]..ends[s]],
        )?;
    }

    if readers.iter().any(|r| !r.is_exhausted()) {
        return Err(PtwmCoreError::HuffmanDecompress(
            "bitstream not fully consumed".to_string(),
        ));
    }

    Ok(dst.len())
}

/// Tail decode: handles remaining symbols after the interleaved fast path
/// exits. Uses the fast fused decode when possible, falls back to fully
/// checked decode for the final bytes.
fn decode_stream_tail(
    reader: &mut BitReader<'_>,
    table: &[u16],
    mask: u64,
    max_bits: u8,
    out: &mut [u8],
) -> Result<(), PtwmCoreError> {
    let len = out.len();
    let mut i = 0;

    // Continue fast decode while source bytes remain.
    while i < len && reader.can_fast_refill() {
        if reader.bit_count < max_bits {
            reader.refill();
        }
        out[i] = reader.decode_symbol(table, mask);
        i += 1;
    }

    // Drain remaining bits in the container.
    while i < len {
        let available = reader.available_bits_total();
        if available == 0 {
            return Err(PtwmCoreError::HuffmanDecompress(
                "unexpected end of bitstream".to_string(),
            ));
        }

        let peek = reader.peek_bits(max_bits).ok_or_else(|| {
            PtwmCoreError::HuffmanDecompress("failed to read bitstream".to_string())
        })?;

        let index = peek as usize;
        if index >= table.len() {
            return Err(PtwmCoreError::HuffmanDecompress(
                "invalid huffman code".to_string(),
            ));
        }
        let entry = table[index];
        let nbits = (entry >> 8) as u8;
        let symbol = (entry & 0xFF) as u8;

        if nbits as usize > available {
            return Err(PtwmCoreError::HuffmanDecompress(
                "bitstream ended mid-symbol".to_string(),
            ));
        }

        if !reader.consume(nbits) {
            return Err(PtwmCoreError::HuffmanDecompress(
                "failed to consume symbol bits".to_string(),
            ));
        }

        out[i] = symbol;
        i += 1;
    }

    Ok(())
}
