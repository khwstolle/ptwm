/// Maximum Huffman code length (bits). Controls the decode-table size
/// (`1 << MAX_TABLE_LOG` entries) and the counts-per-length array. Exposed so
/// consumers can probe the limit in tests; raising it requires also widening
/// the decode-table index type in `bitstream.rs`.
pub const MAX_TABLE_LOG: u8 = 11;

/// Size of the per-length counts array: indices `0..=MAX_TABLE_LOG`.
const COUNTS_LEN: usize = MAX_TABLE_LOG as usize + 1;

#[derive(Clone)]
pub(crate) struct HuffTree {
    pub(crate) lengths: [u8; 256],
    pub(crate) max_bits: u8,
    decode_table: Vec<u16>,
    rev_codes: [u16; 256],
}

impl HuffTree {
    pub(crate) fn from_frequencies(freqs: &[u32; 256]) -> Option<Self> {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let symbols: Vec<usize> = freqs
            .iter()
            .enumerate()
            .filter_map(|(i, &f)| if f > 0 { Some(i) } else { None })
            .collect();

        if symbols.is_empty() {
            return None;
        }

        let mut lengths = [0u8; 256];

        if symbols.len() == 1 {
            lengths[symbols[0]] = 1;
            return Self::from_lengths(lengths);
        }

        let n = symbols.len();
        // Pre-allocate for 2n slots (indices 0..2n-1): n leaves + up to n-1 internal.
        let max_nodes = 2 * n;
        let mut node_freq = vec![0u64; max_nodes];
        let mut node_left: Vec<Option<usize>> = vec![None; max_nodes];
        let mut node_right: Vec<Option<usize>> = vec![None; max_nodes];

        for (id, &sym) in symbols.iter().enumerate() {
            node_freq[id] = freqs[sym] as u64;
        }

        let mut heap: BinaryHeap<Reverse<(u64, usize)>> =
            (0..n).map(|id| Reverse((node_freq[id], id))).collect();

        let mut next_id = n;
        while heap.len() > 1 {
            let Reverse((f1, id1)) = heap.pop().unwrap();
            let Reverse((f2, id2)) = heap.pop().unwrap();
            let id = next_id;
            next_id += 1;
            node_freq[id] = f1 + f2;
            node_left[id] = Some(id1);
            node_right[id] = Some(id2);
            heap.push(Reverse((f1 + f2, id)));
        }

        // DFS to assign code lengths from tree depth.
        let root = heap.pop().unwrap().0.1;
        let mut stack = vec![(root, 0u8)];
        while let Some((id, depth)) = stack.pop() {
            if id < n {
                lengths[symbols[id]] = depth;
            } else {
                let next_depth = depth.saturating_add(1);
                if let Some(l) = node_left[id] {
                    stack.push((l, next_depth));
                }
                if let Some(r) = node_right[id] {
                    stack.push((r, next_depth));
                }
            }
        }

        enforce_max_code_length(&mut lengths, MAX_TABLE_LOG);
        Self::from_lengths(lengths)
    }

    pub(crate) fn from_weights(weights: &[u8]) -> Result<Self, String> {
        if weights.is_empty() {
            return Err("empty weight table".to_string());
        }
        let mut lengths = [0u8; 256];
        for (i, &w) in weights.iter().enumerate() {
            if w > MAX_TABLE_LOG {
                return Err(format!("invalid weight {w} at index {i}"));
            }
            lengths[i] = w;
        }

        Self::from_lengths(lengths).ok_or_else(|| "invalid weight table".to_string())
    }

    fn from_lengths(lengths: [u8; 256]) -> Option<Self> {
        let max_bits = lengths.iter().copied().max().unwrap_or(0).max(1);
        if max_bits > MAX_TABLE_LOG {
            return None;
        }

        let mut counts = [0u16; COUNTS_LEN];
        for &len in &lengths {
            if len > 0 {
                counts[len as usize] = counts[len as usize].saturating_add(1);
            }
        }

        if counts.iter().sum::<u16>() == 0 {
            return None;
        }

        let mut next_code = [0u16; COUNTS_LEN];
        let mut code = 0u16;
        for bits in 1..=max_bits as usize {
            code = (code + counts[bits - 1]) << 1;
            if code > (1u16 << bits) {
                return None;
            }
            next_code[bits] = code;
        }

        let mut codes = [0u16; 256];
        for sym in 0..256 {
            let len = lengths[sym] as usize;
            if len == 0 {
                continue;
            }
            let c = next_code[len];
            if c >= (1u16 << len) {
                return None;
            }
            codes[sym] = c;
            next_code[len] = c + 1;
        }

        let table_size = 1usize << max_bits;
        // Initialize with a safe fallback entry (nbits=max_bits, sym=0).
        // Under-complete trees may leave gaps; these entries are unreachable
        // during valid decompression but must not cause UB if hit.
        let safe_fallback = (max_bits as u16) << 8;
        let mut decode_table = vec![safe_fallback; table_size];
        let mut rev_codes = [0u16; 256];

        for sym in 0..256 {
            let len = lengths[sym];
            if len == 0 {
                continue;
            }
            let code = codes[sym];
            let rev = reverse_bits(code, len);
            rev_codes[sym] = rev;

            let spread = 1usize << (max_bits - len);
            for i in 0..spread {
                let idx = rev as usize | (i << len);
                let entry = ((len as u16) << 8) | sym as u16;
                if decode_table[idx] != safe_fallback && decode_table[idx] != entry {
                    return None;
                }
                decode_table[idx] = entry;
            }
        }

        Some(Self {
            lengths,
            max_bits,
            decode_table,
            rev_codes,
        })
    }

    pub(crate) fn code_for(&self, symbol: u8) -> Option<(u16, u8)> {
        let len = self.lengths[symbol as usize];
        if len == 0 {
            None
        } else {
            Some((self.rev_codes[symbol as usize], len))
        }
    }

    /// Raw decode table slice for direct indexing in the fast decode path.
    #[inline(always)]
    pub(crate) fn decode_table_raw(&self) -> &[u16] {
        &self.decode_table
    }

    /// Bitmask for indexing into the decode table: `(1 << max_bits) - 1`.
    #[inline(always)]
    pub(crate) fn table_mask(&self) -> u16 {
        (1u16 << self.max_bits) - 1
    }

    pub(crate) fn max_symbol_index(&self) -> usize {
        self.lengths.iter().rposition(|&w| w > 0).unwrap_or(0) + 1
    }

    pub(crate) fn weights_prefix(&self) -> Vec<u8> {
        let n = self.max_symbol_index();
        self.lengths[..n].to_vec()
    }
}

/// Clamp code lengths to `max_len` and repair the Kraft inequality by
/// lengthening the shortest active codes until the tree fits in the table.
fn enforce_max_code_length(lengths: &mut [u8; 256], max_len: u8) {
    let mut clamped = false;
    for l in lengths.iter_mut() {
        if *l > max_len {
            *l = max_len;
            clamped = true;
        }
    }
    if !clamped {
        return;
    }

    let capacity = 1u32 << max_len;
    let kraft_sum = |ls: &[u8; 256]| -> u32 {
        ls.iter()
            .filter(|&&l| l > 0)
            .map(|&l| 1u32 << (max_len - l))
            .sum()
    };

    while kraft_sum(lengths) > capacity {
        // Lengthen the shortest active code (frees the most slots per operation).
        let victim = (0..256)
            .filter(|&s| lengths[s] > 0 && lengths[s] < max_len)
            .min_by_key(|&s| lengths[s]);
        match victim {
            Some(s) => lengths[s] += 1,
            None => break,
        }
    }
}

fn reverse_bits(mut code: u16, nbits: u8) -> u16 {
    let mut out = 0u16;
    for _ in 0..nbits {
        out = (out << 1) | (code & 1);
        code >>= 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{HuffTree, MAX_TABLE_LOG};

    #[test]
    fn tree_builds_from_freq() {
        let mut freqs = [0u32; 256];
        freqs[0] = 100;
        freqs[1] = 50;
        freqs[2] = 25;
        let tree = HuffTree::from_frequencies(&freqs).unwrap();
        assert!(tree.max_bits <= MAX_TABLE_LOG);
        assert!(tree.code_for(0).is_some());
        assert!(tree.code_for(2).is_some());
    }

    #[test]
    fn tree_rejects_invalid_weights() {
        assert!(HuffTree::from_weights(&[12]).is_err());
    }

    #[test]
    fn tree_handles_skewed_distribution() {
        // One dominant symbol and many rare symbols — produces a tree deeper
        // than MAX_TABLE_LOG.  Before the length-limiting fix this would
        // cause from_frequencies to return None (false incompressible).
        let mut freqs = [0u32; 256];
        freqs[0] = 1 << 20; // dominant
        for f in freqs.iter_mut().skip(1) {
            *f = 1; // 255 rare symbols
        }
        let tree = HuffTree::from_frequencies(&freqs)
            .expect("skewed distribution must produce a valid length-limited tree");
        assert!(tree.max_bits <= MAX_TABLE_LOG);
        // The dominant symbol should get the shortest code.
        assert!(tree.lengths[0] > 0);
        assert!(tree.lengths[0] < tree.lengths[1]);
    }

    #[test]
    fn single_symbol_builds_tree() {
        // A 1-symbol tree produces a 1-bit code. The decode table has 2
        // entries: one for the symbol, one safe fallback. This is valid
        // (compress will return a small output).
        let mut freqs = [0u32; 256];
        freqs[42] = 1000;
        let tree = HuffTree::from_frequencies(&freqs).unwrap();
        assert_eq!(tree.max_bits, 1);
        assert_eq!(tree.lengths[42], 1);
    }

    #[test]
    fn two_symbol_tree_is_valid() {
        let mut freqs = [0u32; 256];
        freqs[0] = 100;
        freqs[255] = 50;
        let tree = HuffTree::from_frequencies(&freqs).unwrap();
        assert_eq!(tree.max_bits, 1);
        // Both symbols should have code length 1.
        assert_eq!(tree.lengths[0], 1);
        assert_eq!(tree.lengths[255], 1);
        // Decode table should have 2 entries, both populated.
        assert_eq!(tree.decode_table_raw().len(), 2);
    }

    #[test]
    fn from_weights_single_symbol() {
        // Single-symbol weight table: produces a valid 1-bit tree.
        let tree = HuffTree::from_weights(&[1]).unwrap();
        assert_eq!(tree.max_bits, 1);
    }

    #[test]
    fn all_zero_frequencies_returns_none() {
        let freqs = [0u32; 256];
        assert!(HuffTree::from_frequencies(&freqs).is_none());
    }

    #[test]
    fn tree_roundtrip_via_weights() {
        // Build a tree from frequencies, extract weights, rebuild from
        // weights, and verify the decode tables match.
        let mut freqs = [0u32; 256];
        freqs[0] = 500;
        freqs[1] = 200;
        freqs[2] = 100;
        freqs[3] = 50;
        freqs[4] = 10;
        let original = HuffTree::from_frequencies(&freqs).unwrap();
        let weights = original.weights_prefix();
        let rebuilt = HuffTree::from_weights(&weights).unwrap();
        assert_eq!(original.max_bits, rebuilt.max_bits);
        assert_eq!(original.lengths, rebuilt.lengths);
        assert_eq!(original.decode_table_raw(), rebuilt.decode_table_raw());
    }
}
