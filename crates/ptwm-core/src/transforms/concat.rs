//! `Concat` op — concatenates N input planes into a single byte plane.

use std::sync::Arc;

use crate::error::PtwmCoreError;
use crate::transforms::op::{Op, OpId, Plane};
use crate::types::descriptor::{ElementWidth, Layout, PlaneDescriptor};
use crate::types::role::Role;

/// `Concat` concatenates N input planes (all with the same `element_width`)
/// into a single byte plane whose `length_bytes` is the sum of the inputs'.
///
/// ### inverse semantics
///
/// To split the concatenated plane back into N sub-planes, the per-input
/// lengths must be stored in `input_lens`. These are validated at both
/// `propagate_descriptors` and `forward` time. `inverse` uses them to slice
/// the single output plane.
pub struct Concat {
    /// Per-input length in bytes. Must match the actual input plane lengths.
    pub input_lens: Vec<u32>,
}

impl Op for Concat {
    fn propagate_descriptors(
        &self,
        inputs: &[PlaneDescriptor],
    ) -> Result<Vec<PlaneDescriptor>, PtwmCoreError> {
        if inputs.len() != self.input_lens.len() {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "Concat.propagate_descriptors: expected {} inputs (matching input_lens), got {}",
                self.input_lens.len(),
                inputs.len()
            )));
        }
        if inputs.is_empty() {
            return Err(PtwmCoreError::InvalidContainer(
                "Concat.propagate_descriptors: requires at least 1 input".into(),
            ));
        }

        let ref_width = inputs[0].element_width;
        let mut total_bytes: u64 = 0;
        for (i, (input, &expected_len)) in inputs.iter().zip(&self.input_lens).enumerate() {
            if input.element_width != ref_width {
                return Err(PtwmCoreError::InvalidContainer(format!(
                    "Concat: input {} has element_width {:?}, expected {:?}",
                    i, input.element_width, ref_width
                )));
            }
            if input.length_bytes != expected_len as u64 {
                return Err(PtwmCoreError::InvalidContainer(format!(
                    "Concat: input {} length_bytes {} does not match input_lens[{}] {}",
                    i, input.length_bytes, i, expected_len
                )));
            }
            total_bytes += input.length_bytes;
        }

        Ok(vec![PlaneDescriptor {
            role: Role::Raw,
            element_width: ElementWidth::Byte,
            length_bytes: total_bytes,
            layout: Layout::Flat,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        }])
    }

    fn forward(&self, inputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        if inputs.len() != self.input_lens.len() {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "Concat.forward: expected {} inputs, got {}",
                self.input_lens.len(),
                inputs.len()
            )));
        }
        if inputs.is_empty() {
            return Err(PtwmCoreError::InvalidContainer(
                "Concat.forward: requires at least 1 input".into(),
            ));
        }

        let ref_width = inputs[0].descriptor.element_width;
        let mut total_bytes: u64 = 0;
        for (i, (input, &expected_len)) in inputs.iter().zip(&self.input_lens).enumerate() {
            if input.descriptor.element_width != ref_width {
                return Err(PtwmCoreError::InvalidContainer(format!(
                    "Concat: input {} has element_width {:?}, expected {:?}",
                    i, input.descriptor.element_width, ref_width
                )));
            }
            if input.bytes.len() as u64 != expected_len as u64 {
                return Err(PtwmCoreError::InvalidContainer(format!(
                    "Concat: input {} byte length {} does not match input_lens[{}] {}",
                    i,
                    input.bytes.len(),
                    i,
                    expected_len
                )));
            }
            total_bytes += input.bytes.len() as u64;
        }

        let mut out_bytes = Vec::with_capacity(total_bytes as usize);
        for input in inputs {
            out_bytes.extend_from_slice(&input.bytes);
        }

        Ok(vec![Plane {
            bytes: Arc::from(out_bytes.into_boxed_slice()),
            descriptor: PlaneDescriptor {
                role: Role::Raw,
                element_width: ElementWidth::Byte,
                length_bytes: total_bytes,
                layout: Layout::Flat,
                derives_from_tensor: None,
                residual_of: None,
                is_nibble_packed: false,
                vendor_bytes: vec![],
            },
        }])
    }

    fn inverse(&self, outputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        if outputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "Concat.inverse: expected 1 output plane, got {}",
                outputs.len()
            )));
        }
        let out = &outputs[0];
        let total: u64 = self.input_lens.iter().map(|&l| l as u64).sum();
        if out.bytes.len() as u64 != total {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "Concat.inverse: output byte length {} does not match sum of input_lens {}",
                out.bytes.len(),
                total
            )));
        }

        let mut result = Vec::with_capacity(self.input_lens.len());
        let mut offset = 0usize;
        for &len in &self.input_lens {
            let end = offset + len as usize;
            result.push(Plane {
                bytes: Arc::from(&out.bytes[offset..end]),
                descriptor: PlaneDescriptor {
                    role: Role::Raw,
                    element_width: ElementWidth::Byte,
                    length_bytes: len as u64,
                    layout: Layout::Flat,
                    derives_from_tensor: None,
                    residual_of: None,
                    is_nibble_packed: false,
                    vendor_bytes: vec![],
                },
            });
            offset = end;
        }
        Ok(result)
    }

    fn id(&self) -> OpId {
        OpId::Concat
    }

    fn write_params(&self, out: &mut Vec<u8>) {
        // n_inputs: u8
        let n = u8::try_from(self.input_lens.len())
            .expect("Concat: number of inputs must fit in u8 (≤ 255)");
        out.push(n);
        // [u32; n] lengths
        for &len in &self.input_lens {
            out.extend_from_slice(&len.to_le_bytes());
        }
    }
}

/// Parse `Concat` params from a raw byte buffer.
pub fn read_concat_params(buf: &[u8]) -> Result<(Concat, usize), PtwmCoreError> {
    if buf.is_empty() {
        return Err(PtwmCoreError::InvalidContainer(
            "Concat params: empty buffer".into(),
        ));
    }
    let n = buf[0] as usize;
    let mut pos = 1;
    if buf.len() < pos + n * 4 {
        return Err(PtwmCoreError::InvalidContainer(
            "Concat params: truncated input_lens".into(),
        ));
    }
    let mut input_lens = Vec::with_capacity(n);
    for _ in 0..n {
        let len = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap());
        input_lens.push(len);
        pos += 4;
    }
    Ok((Concat { input_lens }, pos))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::descriptor::ElementWidth;
    use crate::types::role::Role;

    fn make_plane(bytes: Vec<u8>, element_width: ElementWidth) -> Plane {
        Plane {
            descriptor: PlaneDescriptor {
                role: Role::Raw,
                element_width,
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
    fn concat_two_planes_roundtrip() {
        let plane_a = make_plane(vec![0x01, 0x02, 0x03], ElementWidth::Byte);
        let plane_b = make_plane(vec![0xAA, 0xBB], ElementWidth::Byte);

        let op = Concat {
            input_lens: vec![3, 2],
        };

        // Forward: concatenate
        let fwd = op.forward(&[plane_a.clone(), plane_b.clone()]).unwrap();
        assert_eq!(fwd.len(), 1);
        assert_eq!(fwd[0].bytes.as_ref(), vec![0x01, 0x02, 0x03, 0xAA, 0xBB]);

        // Inverse: split back
        let inv = op.inverse(&fwd).unwrap();
        assert_eq!(inv.len(), 2);
        assert_eq!(inv[0].bytes.as_ref(), plane_a.bytes.as_ref());
        assert_eq!(inv[1].bytes.as_ref(), plane_b.bytes.as_ref());
    }

    #[test]
    fn propagate_descriptors_rejects_mixed_element_widths() {
        let op = Concat {
            input_lens: vec![4, 4],
        };
        let desc_a = PlaneDescriptor {
            role: Role::Raw,
            element_width: ElementWidth::Byte,
            length_bytes: 4,
            layout: Layout::Flat,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        };
        let desc_b = PlaneDescriptor {
            element_width: ElementWidth::Word2,
            ..desc_a.clone()
        };
        assert!(op.propagate_descriptors(&[desc_a, desc_b]).is_err());
    }

    #[test]
    fn write_params_roundtrip() {
        let op = Concat {
            input_lens: vec![128, 64, 32],
        };
        let mut buf = Vec::new();
        op.write_params(&mut buf);
        let (decoded, n) = read_concat_params(&buf).unwrap();
        assert_eq!(n, buf.len());
        assert_eq!(decoded.input_lens, op.input_lens);
    }
}
