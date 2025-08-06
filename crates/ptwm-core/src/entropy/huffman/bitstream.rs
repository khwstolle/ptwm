/// Backward bitstream writer. Writes bits into a pre-allocated buffer
/// from end to start, avoiding Vec allocations and the final reverse.
pub(crate) struct BitWriter<'a> {
    buf: &'a mut [u8],
    /// Write cursor: next byte to write (moves from end toward 0).
    ptr: usize,
    bit_container: u64,
    bit_count: u32,
}

impl<'a> BitWriter<'a> {
    pub(crate) fn new(buf: &'a mut [u8]) -> Self {
        Self {
            ptr: buf.len(),
            buf,
            bit_container: 0,
            bit_count: 0,
        }
    }

    #[inline(always)]
    pub(crate) fn write_bits(&mut self, bits: u16, nbits: u8) {
        self.bit_container |= (bits as u64) << self.bit_count;
        self.bit_count += nbits as u32;

        if self.bit_count >= 32 {
            // Flush 4 bytes backward. Use BE so the lowest bits (first to
            // be read) end up at the highest index in the buffer.
            self.ptr -= 4;
            let bytes = (self.bit_container as u32).to_be_bytes();
            self.buf[self.ptr..self.ptr + 4].copy_from_slice(&bytes);
            self.bit_container >>= 32;
            self.bit_count -= 32;
        }
    }

    /// Flush remaining bits and return the start offset of the written
    /// region. The written bytes occupy `buf[start..]`, so the byte
    /// count is `buf.len() - start`.
    pub(crate) fn finish(mut self) -> usize {
        // Flush remaining bits as individual bytes (at most 4).
        while self.bit_count > 0 {
            self.ptr -= 1;
            self.buf[self.ptr] = (self.bit_container & 0xFF) as u8;
            self.bit_container >>= 8;
            self.bit_count = self.bit_count.saturating_sub(8);
        }
        // Return start offset of the written region.
        self.ptr
    }
}

/// High-performance backward bitstream reader.
pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    /// Next byte to read (moves from end toward 0).
    byte_index: isize,
    /// Bits loaded so far, LSB-aligned.
    bit_container: u64,
    /// Number of valid bits in bit_container.
    pub(crate) bit_count: u8,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        let mut reader = Self {
            data,
            byte_index: data.len() as isize - 1,
            bit_container: 0,
            bit_count: 0,
        };
        reader.initial_fill();
        reader
    }

    fn initial_fill(&mut self) {
        let avail = (self.byte_index + 1).min(8) as usize;
        for _ in 0..avail {
            self.bit_container |= (self.data[self.byte_index as usize] as u64) << self.bit_count;
            self.bit_count += 8;
            self.byte_index -= 1;
        }
    }

    /// Reload bytes from the backward stream. Uses an 8-byte load when
    /// possible (bit_count == 0), otherwise a 4-byte load. The big-endian
    /// read places the highest-address byte at the LSB, matching the
    /// backward reading order.
    #[inline(always)]
    fn reload(&mut self) {
        if self.bit_count == 0 && self.byte_index >= 7 {
            // Full 8-byte load into empty container.
            let end = self.byte_index as usize;
            let base = end - 7;
            let word = u64::from_be_bytes(self.data[base..=end].try_into().unwrap());
            self.bit_container = word;
            self.bit_count = 64;
            self.byte_index -= 8;
        } else if self.bit_count <= 32 && self.byte_index >= 3 {
            let end = self.byte_index as usize;
            let base = end - 3;
            let word = u32::from_be_bytes(self.data[base..=end].try_into().unwrap());
            self.bit_container |= (word as u64) << self.bit_count;
            self.bit_count += 32;
            self.byte_index -= 4;
        }
    }

    #[inline(always)]
    pub(crate) fn refill(&mut self) {
        if self.bit_count <= 32 && self.byte_index >= 3 {
            self.reload();
        } else {
            while self.bit_count < 56 && self.byte_index >= 0 {
                self.bit_container |=
                    (self.data[self.byte_index as usize] as u64) << self.bit_count;
                self.bit_count += 8;
                self.byte_index -= 1;
            }
        }
    }

    /// Decode one symbol from the bitstream using the decode table.
    ///
    /// # Safety contract (upheld by callers)
    /// `mask` is `(1 << max_bits) - 1` and `table.len() == 1 << max_bits`,
    /// so `index = bit_container & mask` is always in bounds. The unsafe
    /// `get_unchecked` eliminates a bounds check in the hot loop.
    #[inline(always)]
    pub(crate) fn decode_symbol(&mut self, table: &[u16], mask: u64) -> u8 {
        let index = (self.bit_container & mask) as usize;
        // SAFETY: index < table.len() because index <= mask < 1 << max_bits == table.len()
        let entry = unsafe { *table.get_unchecked(index) };
        let nbits = entry >> 8;
        self.bit_container >>= nbits;
        self.bit_count -= nbits as u8;
        (entry & 0xFF) as u8
    }

    #[inline(always)]
    pub(crate) fn can_fast_refill(&self) -> bool {
        self.byte_index >= 3
    }

    pub(crate) fn available_bits_total(&self) -> usize {
        self.bit_count as usize + ((self.byte_index + 1).max(0) as usize) * 8
    }

    pub(crate) fn peek_bits(&mut self, nbits: u8) -> Option<u16> {
        if nbits == 0 {
            return Some(0);
        }
        self.refill_to(nbits);
        if self.bit_count == 0 {
            return None;
        }
        let mask = (1u64 << nbits) - 1;
        Some((self.bit_container & mask) as u16)
    }

    pub(crate) fn consume(&mut self, nbits: u8) -> bool {
        if nbits == 0 {
            return true;
        }
        if self.bit_count < nbits {
            self.refill_to(nbits);
        }
        if self.bit_count < nbits {
            return false;
        }
        self.bit_container >>= nbits;
        self.bit_count -= nbits;
        true
    }

    fn refill_to(&mut self, nbits: u8) {
        while self.bit_count < nbits && self.byte_index >= 0 {
            self.bit_container |= (self.data[self.byte_index as usize] as u64) << self.bit_count;
            self.bit_count += 8;
            self.byte_index -= 1;
        }
    }

    pub(crate) fn is_exhausted(&self) -> bool {
        self.byte_index < 0 && self.bit_container == 0
    }
}

#[cfg(test)]
mod tests {
    use super::{BitReader, BitWriter};

    #[test]
    fn bitstream_roundtrip() {
        let mut buf = [0u8; 64];
        let mut w = BitWriter::new(&mut buf);
        w.write_bits(0b01, 2);
        w.write_bits(0b101, 3);
        w.write_bits(0b1111, 4);
        let start = w.finish();
        let bytes = &buf[start..];

        let mut r = BitReader::new(bytes);
        assert_eq!(r.peek_bits(2), Some(0b01));
        assert!(r.consume(2));
        assert_eq!(r.peek_bits(3), Some(0b101));
        assert!(r.consume(3));
        assert_eq!(r.peek_bits(4), Some(0b1111));
        assert!(r.consume(4));
        assert!(r.is_exhausted());
    }

    #[test]
    fn bitstream_large_roundtrip() {
        // Write enough bits to trigger the 4-byte flush path.
        let mut buf = [0u8; 256];
        let mut w = BitWriter::new(&mut buf);
        let codes: Vec<(u16, u8)> = (0..50).map(|i| ((i * 3 + 1) & 0x7FF, 11)).collect();
        for &(bits, nbits) in &codes {
            w.write_bits(bits, nbits);
        }
        let start = w.finish();
        let bytes = &buf[start..];

        let mut r = BitReader::new(bytes);
        for &(expected_bits, nbits) in &codes {
            let mask = (1u64 << nbits) - 1;
            let got = r.peek_bits(nbits).unwrap();
            assert_eq!(got, expected_bits & mask as u16, "mismatch");
            assert!(r.consume(nbits));
        }
        assert!(r.is_exhausted());
    }
}
