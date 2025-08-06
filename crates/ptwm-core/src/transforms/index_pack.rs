//! `IndexBitwidthPack` op — packs u8 indices into a tight MSB-first bit stream.

use std::sync::Arc;

use crate::error::PtwmCoreError;
use crate::transforms::op::{Op, OpId, Plane};
use crate::types::descriptor::{ElementWidth, Layout, PlaneDescriptor};
use crate::types::role::Role;

/// Pack `n` indices (each `bits` wide) from `src` into `dst` using MSB-first
/// bit packing within each output byte.
fn pack_bits(src: &[u8], dst: &mut [u8], bits: u8) {
    let bits = bits as usize;
    let mut bit_pos = 0usize; // bit position in the output stream
    for &idx in src {
        // Write `bits` bits of `idx`, MSB first, into the output.
        // The MSB of idx is bit `bits-1`.
        for b in (0..bits).rev() {
            let bit = (idx >> b) & 1;
            let byte_idx = bit_pos / 8;
            let bit_in_byte = 7 - (bit_pos % 8); // MSB of each output byte first
            dst[byte_idx] |= bit << bit_in_byte;
            bit_pos += 1;
        }
    }
}

/// Unpack `n` indices (each `bits` wide) from `src` into `dst`.
fn unpack_bits(src: &[u8], dst: &mut [u8], bits: u8) {
    let bits = bits as usize;
    let mut bit_pos = 0usize;
    for idx in dst.iter_mut() {
        let mut val = 0u8;
        for b in (0..bits).rev() {
            let byte_idx = bit_pos / 8;
            let bit_in_byte = 7 - (bit_pos % 8);
            let bit = (src[byte_idx] >> bit_in_byte) & 1;
            val |= bit << b;
            bit_pos += 1;
        }
        *idx = val;
    }
}

/// Compute the packed output length for `n_indices` each `bits` wide.
#[inline]
fn packed_len(n_indices: u64, bits: u8) -> u64 {
    (n_indices * bits as u64).div_ceil(8)
}

/// `IndexBitwidthPack` packs a stream of u8 indices (each ∈ `0..2^bits`) into
/// a tight bit stream, MSB-first within each output byte. Used downstream of
/// codebook ops to recover the bit budget the codebook earned.
///
/// Stores `bits: u8` (must satisfy `1 ≤ bits ≤ 8`).
pub struct IndexBitwidthPack {
    pub bits: u8,
}

impl IndexBitwidthPack {
    fn validate(&self) -> Result<(), PtwmCoreError> {
        if self.bits < 1 || self.bits > 8 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "IndexBitwidthPack: bits {} out of range [1, 8]",
                self.bits
            )));
        }
        Ok(())
    }
}

impl Op for IndexBitwidthPack {
    fn propagate_descriptors(
        &self,
        inputs: &[PlaneDescriptor],
    ) -> Result<Vec<PlaneDescriptor>, PtwmCoreError> {
        self.validate()?;
        if inputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "IndexBitwidthPack.propagate_descriptors: expected 1 input, got {}",
                inputs.len()
            )));
        }
        let input = &inputs[0];
        if input.element_width != ElementWidth::Byte {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "IndexBitwidthPack: input element_width must be Byte, got {:?}",
                input.element_width
            )));
        }
        if input.is_nibble_packed {
            return Err(PtwmCoreError::InvalidContainer(
                "IndexBitwidthPack: input must not be nibble-packed".into(),
            ));
        }
        let out_len = packed_len(input.length_bytes, self.bits);
        Ok(vec![PlaneDescriptor {
            role: Role::Index,
            element_width: ElementWidth::Byte,
            length_bytes: out_len,
            layout: Layout::Flat,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        }])
    }

    fn forward(&self, inputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        self.validate()?;
        if inputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "IndexBitwidthPack.forward: expected 1 input, got {}",
                inputs.len()
            )));
        }
        let input = &inputs[0];
        let out_len = packed_len(input.bytes.len() as u64, self.bits) as usize;
        let mut out_bytes = vec![0u8; out_len];
        pack_bits(&input.bytes, &mut out_bytes, self.bits);

        Ok(vec![Plane {
            bytes: Arc::from(&out_bytes[..]),
            descriptor: PlaneDescriptor {
                role: Role::Index,
                element_width: ElementWidth::Byte,
                length_bytes: out_bytes.len() as u64,
                layout: Layout::Flat,
                derives_from_tensor: None,
                residual_of: None,
                is_nibble_packed: false,
                vendor_bytes: vec![],
            },
        }])
    }

    fn inverse(&self, outputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        self.validate()?;
        if outputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "IndexBitwidthPack.inverse: expected 1 output, got {}",
                outputs.len()
            )));
        }
        let packed = &outputs[0];
        // Derive the number of original indices: total bits / bits_per_index.
        // Since packing may have padded to a byte boundary, the caller must
        // ensure the packed plane's length is consistent. We trust the
        // descriptor's length_bytes for the packed data and recompute n_indices
        // as floor(packed_bits / bits).  Any padding bits in the last byte are
        // zero and ignored.
        let packed_bits = packed.bytes.len() as u64 * 8;
        let n_indices = packed_bits / self.bits as u64;
        let mut out_bytes = vec![0u8; n_indices as usize];
        unpack_bits(&packed.bytes, &mut out_bytes, self.bits);

        Ok(vec![Plane {
            bytes: Arc::from(&out_bytes[..]),
            descriptor: PlaneDescriptor {
                role: Role::Raw,
                element_width: ElementWidth::Byte,
                length_bytes: out_bytes.len() as u64,
                layout: Layout::Flat,
                derives_from_tensor: None,
                residual_of: None,
                is_nibble_packed: false,
                vendor_bytes: vec![],
            },
        }])
    }

    fn id(&self) -> OpId {
        OpId::IndexBitwidthPack
    }

    fn write_params(&self, out: &mut Vec<u8>) {
        out.push(self.bits);
    }
}

/// Parse `IndexBitwidthPack` params from a raw byte buffer.
pub fn read_index_pack_params(buf: &[u8]) -> Result<(IndexBitwidthPack, usize), PtwmCoreError> {
    if buf.is_empty() {
        return Err(PtwmCoreError::InvalidContainer(
            "IndexBitwidthPack params: empty buffer".into(),
        ));
    }
    let bits = buf[0];
    Ok((IndexBitwidthPack { bits }, 1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::descriptor::{ElementWidth, Layout};
    use crate::types::role::Role;

    fn make_plane(bytes: Vec<u8>) -> Plane {
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

    #[test]
    fn pack_unpack_bits_1_through_8() {
        for bits in 1u8..=8 {
            let max_val = (1u16 << bits) - 1;
            // Build an input where each byte cycles through 0..max_val+1
            let n = 64usize;
            let input: Vec<u8> = (0..n).map(|i| (i as u16 % (max_val + 1)) as u8).collect();
            let plane = make_plane(input.clone());

            let op = IndexBitwidthPack { bits };
            let packed = op.forward(&[plane]).unwrap();
            let unpacked = op.inverse(&packed).unwrap();

            assert_eq!(
                unpacked[0].bytes.len(),
                input.len(),
                "bits={bits}: unpacked length mismatch"
            );
            assert_eq!(
                unpacked[0].bytes.as_ref(),
                input,
                "bits={bits}: round-trip mismatch"
            );
        }
    }

    #[test]
    fn propagate_descriptors_rejects_nibble_input() {
        let op = IndexBitwidthPack { bits: 4 };
        let desc = PlaneDescriptor {
            role: Role::Raw,
            element_width: ElementWidth::Byte,
            length_bytes: 32,
            layout: Layout::Flat,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed: true, // ← should be rejected
            vendor_bytes: vec![],
        };
        assert!(op.propagate_descriptors(&[desc]).is_err());
    }

    #[test]
    fn write_params_roundtrip() {
        for bits in 1u8..=8 {
            let op = IndexBitwidthPack { bits };
            let mut buf = Vec::new();
            op.write_params(&mut buf);
            let (decoded, n) = read_index_pack_params(&buf).unwrap();
            assert_eq!(n, 1);
            assert_eq!(decoded.bits, bits);
        }
    }
}
