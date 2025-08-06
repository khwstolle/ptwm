//! `NibbleSplit` op — split each input byte into high-nibble (exponent) and
//! low-nibble (sign+mantissa) planes.
//!
//! This matches the FP8 nibble-split algorithm in `split/dtype8.rs` (modes
//! 20/22): each byte `[HHHH LLLL]` produces two one-nibble-per-byte slots:
//! plane 0 = high nibble `(b >> 4) & 0x0F`, plane 1 = low nibble `b & 0x0F`.
//!
//! Both output planes have `ElementWidth::Nibble` and `length_bytes == input.length_bytes`
//! (one nibble stored per byte slot; the `element_width` flag signals the codec).

use std::sync::Arc;

use crate::error::PtwmCoreError;
use crate::transforms::op::{Op, OpId, Plane};
use crate::types::descriptor::{ElementWidth, Layout, PlaneDescriptor};
use crate::types::role::{NibbleKind, Role};

/// Split each byte into its high nibble (exponent) and low nibble
/// (sign+mantissa), producing two planes of `length_bytes == input.length_bytes`.
///
/// No parameters; the split is fixed at 4-bit granularity.
pub struct NibbleSplit;

impl Op for NibbleSplit {
    fn propagate_descriptors(
        &self,
        inputs: &[PlaneDescriptor],
    ) -> Result<Vec<PlaneDescriptor>, PtwmCoreError> {
        if inputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "NibbleSplit.propagate_descriptors: expected 1 input, got {}",
                inputs.len()
            )));
        }
        let inp = &inputs[0];
        // NibbleSplit operates byte-by-byte. Accept any non-nibble width;
        // post-BitReorder inputs may carry Word2/Word4 element widths but
        // are byte-addressable for the split itself.
        if inp.element_width == ElementWidth::Nibble {
            return Err(PtwmCoreError::InvalidContainer(
                "NibbleSplit.propagate_descriptors: cannot split nibble-width input".into(),
            ));
        }
        Ok(vec![
            PlaneDescriptor {
                role: Role::Nibble {
                    kind: NibbleKind::Exponent,
                },
                element_width: ElementWidth::Nibble,
                length_bytes: inp.length_bytes,
                layout: Layout::Flat,
                derives_from_tensor: inp.derives_from_tensor,
                residual_of: None,
                is_nibble_packed: false,
                vendor_bytes: vec![],
            },
            PlaneDescriptor {
                role: Role::Nibble {
                    kind: NibbleKind::SignMantissa,
                },
                element_width: ElementWidth::Nibble,
                length_bytes: inp.length_bytes,
                layout: Layout::Flat,
                derives_from_tensor: inp.derives_from_tensor,
                residual_of: None,
                is_nibble_packed: false,
                vendor_bytes: vec![],
            },
        ])
    }

    fn forward(&self, inputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        if inputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "NibbleSplit.forward: expected 1 input, got {}",
                inputs.len()
            )));
        }
        let inp = &inputs[0];
        let descriptors = self.propagate_descriptors(std::slice::from_ref(&inp.descriptor))?;
        let mut high = Vec::with_capacity(inp.bytes.len());
        let mut low = Vec::with_capacity(inp.bytes.len());
        for &b in inp.bytes.iter() {
            high.push((b >> 4) & 0x0F);
            low.push(b & 0x0F);
        }
        Ok(vec![
            Plane {
                bytes: Arc::from(high.into_boxed_slice()),
                descriptor: descriptors[0].clone(),
            },
            Plane {
                bytes: Arc::from(low.into_boxed_slice()),
                descriptor: descriptors[1].clone(),
            },
        ])
    }

    fn inverse(&self, outputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        if outputs.len() != 2 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "NibbleSplit.inverse: expected 2 outputs, got {}",
                outputs.len()
            )));
        }
        let high = &outputs[0];
        let low = &outputs[1];
        if high.bytes.len() != low.bytes.len() {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "NibbleSplit.inverse: high plane length {} != low plane length {}",
                high.bytes.len(),
                low.bytes.len()
            )));
        }
        let bytes: Vec<u8> = high
            .bytes
            .iter()
            .zip(low.bytes.iter())
            .map(|(&h, &l)| ((h & 0x0F) << 4) | (l & 0x0F))
            .collect();
        let len = bytes.len() as u64;
        let in_descriptor = PlaneDescriptor {
            role: Role::Raw,
            element_width: ElementWidth::Byte,
            length_bytes: len,
            layout: Layout::Flat,
            derives_from_tensor: high.descriptor.derives_from_tensor,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        };
        Ok(vec![Plane {
            bytes: Arc::from(bytes.into_boxed_slice()),
            descriptor: in_descriptor,
        }])
    }

    fn id(&self) -> OpId {
        OpId::NibbleSplit
    }

    fn write_params(&self, _out: &mut Vec<u8>) {
        // No parameters.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::descriptor::{ElementWidth, Layout};
    use crate::types::role::{NibbleKind, Role};

    fn byte_plane(data: Vec<u8>) -> Plane {
        Plane {
            bytes: Arc::from(&data[..]),
            descriptor: PlaneDescriptor {
                role: Role::Raw,
                element_width: ElementWidth::Byte,
                length_bytes: data.len() as u64,
                layout: Layout::Flat,
                derives_from_tensor: None,
                residual_of: None,
                is_nibble_packed: false,
                vendor_bytes: vec![],
            },
        }
    }

    #[test]
    fn nibble_split_roundtrip() {
        // 32 bytes with varied nibble patterns.
        let data: Vec<u8> = (0u8..32).map(|i| (i << 4) | (i & 0x0F)).collect();
        let op = NibbleSplit;
        let inp = byte_plane(data.clone());
        let fwd = op.forward(&[inp]).unwrap();
        assert_eq!(fwd.len(), 2);
        let inv = op.inverse(&fwd).unwrap();
        assert_eq!(inv.len(), 1);
        assert_eq!(inv[0].bytes.as_ref(), data);
    }

    #[test]
    fn propagate_descriptors_rejects_nibble_input() {
        let op = NibbleSplit;
        let desc = PlaneDescriptor {
            role: Role::Nibble {
                kind: NibbleKind::Exponent,
            },
            element_width: ElementWidth::Nibble,
            length_bytes: 16,
            layout: Layout::Flat,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        };
        assert!(op.propagate_descriptors(&[desc]).is_err());
    }

    #[test]
    fn propagate_descriptors_accepts_word_widths() {
        let op = NibbleSplit;
        for ew in [ElementWidth::Byte, ElementWidth::Word2, ElementWidth::Word4] {
            let desc = PlaneDescriptor {
                role: Role::Raw,
                element_width: ew,
                length_bytes: 16,
                layout: Layout::Flat,
                derives_from_tensor: None,
                residual_of: None,
                is_nibble_packed: false,
                vendor_bytes: vec![],
            };
            assert!(
                op.propagate_descriptors(&[desc]).is_ok(),
                "expected ok for {ew:?}"
            );
        }
    }

    #[test]
    fn forward_high_nibble_correctness() {
        let op = NibbleSplit;
        let inp = byte_plane(vec![0xAB]);
        let fwd = op.forward(&[inp]).unwrap();
        // High nibble of 0xAB = 0x0A
        assert_eq!(fwd[0].bytes[0], 0x0A);
    }

    #[test]
    fn forward_low_nibble_correctness() {
        let op = NibbleSplit;
        let inp = byte_plane(vec![0xAB]);
        let fwd = op.forward(&[inp]).unwrap();
        // Low nibble of 0xAB = 0x0B
        assert_eq!(fwd[1].bytes[0], 0x0B);
    }

    #[test]
    fn inverse_masks_dirty_high_nibble() {
        // Codecs may emit full-byte values in the high-nibble plane (only the
        // low nibble carries data); reconstruction must mask off the upper bits.
        let op = NibbleSplit;
        let high = Plane {
            bytes: Arc::from(&[0xFAu8, 0xFB][..]), // 0xF0 in upper bits is noise; only 0x0A/0x0B is data.
            descriptor: PlaneDescriptor {
                role: Role::Nibble {
                    kind: NibbleKind::Exponent,
                },
                element_width: ElementWidth::Nibble,
                length_bytes: 2,
                layout: Layout::Flat,
                derives_from_tensor: None,
                residual_of: None,
                is_nibble_packed: false,
                vendor_bytes: vec![],
            },
        };
        let low = Plane {
            bytes: Arc::from(&[0x0Cu8, 0x0D][..]),
            descriptor: PlaneDescriptor {
                role: Role::Nibble {
                    kind: NibbleKind::SignMantissa,
                },
                element_width: ElementWidth::Nibble,
                length_bytes: 2,
                layout: Layout::Flat,
                derives_from_tensor: None,
                residual_of: None,
                is_nibble_packed: false,
                vendor_bytes: vec![],
            },
        };
        let inv = op.inverse(&[high, low]).unwrap();
        assert_eq!(inv[0].bytes.as_ref(), vec![0xAC, 0xBD]);
    }

    #[test]
    fn propagate_descriptors_output_roles() {
        let op = NibbleSplit;
        let desc = PlaneDescriptor {
            role: Role::Raw,
            element_width: ElementWidth::Byte,
            length_bytes: 32,
            layout: Layout::Flat,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        };
        let outs = op.propagate_descriptors(&[desc]).unwrap();
        assert_eq!(outs.len(), 2);
        assert_eq!(
            outs[0].role,
            Role::Nibble {
                kind: NibbleKind::Exponent
            }
        );
        assert_eq!(
            outs[1].role,
            Role::Nibble {
                kind: NibbleKind::SignMantissa
            }
        );
        assert_eq!(outs[0].element_width, ElementWidth::Nibble);
        assert_eq!(outs[1].element_width, ElementWidth::Nibble);
        assert_eq!(outs[0].length_bytes, 32);
        assert_eq!(outs[1].length_bytes, 32);
    }
}
