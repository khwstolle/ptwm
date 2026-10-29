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

/// [`packed_len`] for a length read from a chain: `None` instead of an
/// overflow when `n_indices * bits` does not fit in `u64`.
fn checked_packed_len(n_indices: u64, bits: u8) -> Option<u64> {
    n_indices.checked_mul(bits as u64).map(|b| b.div_ceil(8))
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
        let out_len = checked_packed_len(input.length_bytes, self.bits).ok_or_else(|| {
            PtwmCoreError::InvalidContainer(format!(
                "IndexBitwidthPack.propagate_descriptors: {} indices of {} bits overflow u64",
                input.length_bytes, self.bits
            ))
        })?;
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

    /// Without the input descriptor, the index count is
    /// `floor(packed_bits / bits)`. That is too large when the last byte
    /// holds padding; [`Op::inverse_with_input_descs`] gives the exact count.
    fn inverse(&self, outputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        self.inverse_with_input_descs(outputs, None)
    }

    fn inverse_with_input_descs(
        &self,
        outputs: &[Plane],
        input_descs: Option<&[PlaneDescriptor]>,
    ) -> Result<Vec<Plane>, PtwmCoreError> {
        self.validate()?;
        if outputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "IndexBitwidthPack.inverse: expected 1 output, got {}",
                outputs.len()
            )));
        }
        let packed = &outputs[0];
        let packed_bytes = packed.bytes.len() as u64;
        let n_indices = match input_descs {
            Some([input]) => {
                let n = input.length_bytes;
                if checked_packed_len(n, self.bits) != Some(packed_bytes) {
                    return Err(PtwmCoreError::InvalidContainer(format!(
                        "IndexBitwidthPack.inverse: {packed_bytes} packed bytes do not hold \
                         {n} indices of {} bits",
                        self.bits
                    )));
                }
                n
            }
            Some(descs) => {
                return Err(PtwmCoreError::InvalidContainer(format!(
                    "IndexBitwidthPack.inverse: expected 1 input descriptor, got {}",
                    descs.len()
                )));
            }
            None => packed_bytes * 8 / self.bits as u64,
        };
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
    fn inverse_with_input_descs_returns_exact_count() {
        // 3 indices × 3 bits = 9 bits → 2 bytes. Without the input
        // descriptor the count is floor(16 / 3) = 5.
        let op = IndexBitwidthPack { bits: 3 };
        let input = make_plane(vec![5, 2, 7]);
        let input_desc = input.descriptor.clone();
        let packed = op.forward(&[input]).unwrap();
        assert_eq!(packed[0].bytes.len(), 2);
        assert_eq!(op.inverse(&packed).unwrap()[0].bytes.len(), 5);
        let exact = op
            .inverse_with_input_descs(&packed, Some(std::slice::from_ref(&input_desc)))
            .unwrap();
        assert_eq!(exact[0].bytes.as_ref(), [5, 2, 7]);
    }

    #[test]
    fn inverse_with_input_descs_rejects_inconsistent_length() {
        // 2 packed bytes cannot hold 6 indices of 3 bits (18 bits).
        let op = IndexBitwidthPack { bits: 3 };
        let packed = make_plane(vec![0, 0]);
        let mut desc = packed.descriptor.clone();
        desc.length_bytes = 6;
        assert!(
            op.inverse_with_input_descs(&[packed], Some(std::slice::from_ref(&desc)))
                .is_err()
        );
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

    #[test]
    fn propagate_descriptors_overflow_is_an_error() {
        // `length_bytes * bits` overflows u64 for a corrupt chain.
        let op = IndexBitwidthPack { bits: 8 };
        let d = PlaneDescriptor {
            role: Role::Raw,
            element_width: ElementWidth::Byte,
            length_bytes: u64::MAX,
            layout: Layout::Flat,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        };
        assert!(op.propagate_descriptors(&[d]).is_err());
    }
}
