//! `Source` op — graph entry node that describes the raw input tensor.

use crate::error::PtwmCoreError;
use crate::transforms::op::{Op, OpId, Plane};
use crate::types::descriptor::{ElementWidth, Layout, PlaneDescriptor};
use crate::types::role::Role;

/// Element width inferred from a chain-internal dtype code. The chain's
/// Source params use a different code space than `Dtype::from_code` (the
/// public registry); see `_CHAIN_DTYPE` in
/// `python/weights/preprocessing/_chains.py` and the parallel match in
/// `compressor::source_descriptor_for`.
/// Chain-internal code for `Float4E2M1FNx2`: two fp4 values share a byte.
const DTYPE_FP4_E2M1FN_X2: u16 = 0x001F;

fn element_width_for(dtype_code: u16) -> ElementWidth {
    match dtype_code {
        // FP16 / BF16 / int16 / uint16
        0x0002 | 0x000F | 0x0007 | 0x0008 => ElementWidth::Word2,
        // FP32 / int32 / uint32
        0x0003 | 0x0009 | 0x000A => ElementWidth::Word4,
        // FP64 / int64 / uint64
        0x0004 | 0x000B | 0x000C => ElementWidth::Word8,
        // Packed FP4: sub-byte, and the only sub-byte dtype so far.
        DTYPE_FP4_E2M1FN_X2 => ElementWidth::Nibble,
        // int8 / uint8 / FP8 — and the catch-all
        _ => ElementWidth::Byte,
    }
}

/// Whether a chain-internal dtype stores two values per byte.
///
/// Kept separate from element width on purpose: nibble width says how wide
/// an element is, this says that elements share a byte. Codecs that read
/// packed nibbles (`PerGroupCodebook`) require both, and transforms that
/// cannot handle sharing (`BurrowsWheeler`, `IndexPack`) reject on this one.
fn is_nibble_packed_dtype(dtype_code: u16) -> bool {
    dtype_code == DTYPE_FP4_E2M1FN_X2
}

/// Storage bytes for `n_elements` of a chain-internal dtype.
///
/// Accumulates in bits so sub-byte widths survive. Dividing
/// bits-per-element by 8 first floors a nibble to zero, which silently
/// yields a zero-length descriptor rather than a wrong-but-visible one.
fn storage_bytes(dtype_code: u16, n_elements: u64) -> u64 {
    let bits = n_elements * element_width_for(dtype_code).bits_per_element() as u64;
    bits.div_ceil(8)
}

/// Mandatory entry node of every PPG chain. Takes no inputs (the runtime
/// injects raw tensor bytes externally) and emits one `Role::Raw` plane
/// whose descriptor follows from the tensor `shape` and `dtype_code`.
///
/// ### forward / inverse semantics
///
/// The runtime wires tensor bytes into the graph, so `forward` expects a
/// **single synthetic input** carrying the raw tensor bytes (the runtime
/// wraps them in a `Plane` before calling). It validates the descriptor
/// and forwards the plane unchanged. `inverse` is symmetric: takes one
/// plane, returns it unchanged.
pub struct Source {
    pub shape: Vec<u32>,
    pub dtype_code: u16,
}

/// Maximum number of dimensions the wire format supports. Encoded as a
/// single `u8` length prefix in `write_params`.
pub const MAX_SOURCE_DIMS: usize = 255;

impl Source {
    /// Construct a `Source` after validating that `shape` fits in the wire
    /// format (≤ 255 dimensions).
    pub fn new(shape: Vec<u32>, dtype_code: u16) -> Result<Self, PtwmCoreError> {
        if shape.len() > MAX_SOURCE_DIMS {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "Source: shape has {} dimensions, max is {MAX_SOURCE_DIMS}",
                shape.len()
            )));
        }
        Ok(Source { shape, dtype_code })
    }

    /// Build the `PlaneDescriptor` for the raw tensor plane this source emits.
    fn output_descriptor(&self) -> PlaneDescriptor {
        // product() on an empty iterator yields 1 (multiplicative
        // identity), so an empty shape represents a scalar (1 element).
        let n_elements: u64 = self.shape.iter().map(|&d| d as u64).product();
        let length_bytes = storage_bytes(self.dtype_code, n_elements);
        let layout = if let Some(&last) = self.shape.last() {
            Layout::Rows { row_len: last }
        } else {
            Layout::Flat
        };
        PlaneDescriptor {
            role: Role::Raw,
            element_width: element_width_for(self.dtype_code),
            length_bytes,
            layout,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed: is_nibble_packed_dtype(self.dtype_code),
            vendor_bytes: vec![],
        }
    }
}

impl Op for Source {
    fn propagate_descriptors(
        &self,
        inputs: &[PlaneDescriptor],
    ) -> Result<Vec<PlaneDescriptor>, PtwmCoreError> {
        if !inputs.is_empty() {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "Source.propagate_descriptors: expected 0 inputs, got {}",
                inputs.len()
            )));
        }
        if self.shape.len() > MAX_SOURCE_DIMS {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "Source: shape has {} dimensions, max is {MAX_SOURCE_DIMS}",
                self.shape.len()
            )));
        }
        Ok(vec![self.output_descriptor()])
    }

    fn forward(&self, inputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        // The runtime wraps the raw tensor bytes as a single synthetic Plane
        // and passes it here. We validate and forward it unchanged.
        if inputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "Source.forward: expected 1 synthetic input from runtime, got {}",
                inputs.len()
            )));
        }
        Ok(vec![inputs[0].clone()])
    }

    fn inverse(&self, outputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        if outputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "Source.inverse: expected 1 plane, got {}",
                outputs.len()
            )));
        }
        Ok(vec![outputs[0].clone()])
    }

    fn id(&self) -> OpId {
        OpId::Source
    }

    fn write_params(&self, out: &mut Vec<u8>) {
        // shape_dim_count: u8. Validation lives in `Source::new` and
        // `propagate_descriptors`; if either was bypassed (e.g. struct literal
        // construction in tests), saturate to u8::MAX rather than panicking
        // since `write_params` cannot return a Result.
        let dim_count = u8::try_from(self.shape.len()).unwrap_or(u8::MAX);
        out.push(dim_count);
        for &dim in self.shape.iter().take(dim_count as usize) {
            out.extend_from_slice(&dim.to_le_bytes());
        }
        // dtype_code: u16
        out.extend_from_slice(&self.dtype_code.to_le_bytes());
    }
}

/// Parse `Source` params from a raw byte buffer. Returns `(Source, bytes_consumed)`.
pub fn read_source_params(buf: &[u8]) -> Result<(Source, usize), PtwmCoreError> {
    if buf.is_empty() {
        return Err(PtwmCoreError::InvalidContainer(
            "Source params: empty buffer".into(),
        ));
    }
    let dim_count = buf[0] as usize;
    let mut pos = 1;
    if buf.len() < pos + dim_count * 4 + 2 {
        return Err(PtwmCoreError::InvalidContainer(
            "Source params: truncated".into(),
        ));
    }
    let mut shape = Vec::with_capacity(dim_count);
    for _ in 0..dim_count {
        let d = u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]);
        shape.push(d);
        pos += 4;
    }
    let dtype_code = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
    pos += 2;
    Ok((Source { shape, dtype_code }, pos))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_source() -> Source {
        Source {
            shape: vec![4, 8, 16],
            // Float16 → chain-internal code 0x0002 (matches Python _CHAIN_DTYPE).
            dtype_code: 0x0002,
        }
    }

    #[test]
    fn propagate_descriptors_with_empty_inputs_returns_one_plane() {
        let src = make_source();
        let descs = src.propagate_descriptors(&[]).unwrap();
        assert_eq!(descs.len(), 1);
        let d = &descs[0];
        assert_eq!(d.role, Role::Raw);
        // fp16 → 2 bytes per element → ElementWidth::Word2
        assert_eq!(d.element_width, ElementWidth::Word2);
        // 4 * 8 * 16 * 2 bytes/elem = 1024
        assert_eq!(d.length_bytes, 1024);
        assert_eq!(d.layout, Layout::Rows { row_len: 16 });
        assert!(!d.is_nibble_packed);
        assert!(d.vendor_bytes.is_empty());
    }

    #[test]
    fn propagate_descriptors_element_width_matches_dtype() {
        // Chain-internal codes (matching Python _CHAIN_DTYPE):
        //   0x0005 = Int8 → Byte
        //   0x0003 = Float32 → Word4
        //   0x000B = Int64 → Word8
        let src = Source {
            shape: vec![16],
            dtype_code: 0x0005,
        };
        assert_eq!(
            src.propagate_descriptors(&[]).unwrap()[0].element_width,
            ElementWidth::Byte
        );
        let src = Source {
            shape: vec![16],
            dtype_code: 0x0003,
        };
        assert_eq!(
            src.propagate_descriptors(&[]).unwrap()[0].element_width,
            ElementWidth::Word4
        );
        let src = Source {
            shape: vec![4],
            dtype_code: 0x000B,
        };
        assert_eq!(
            src.propagate_descriptors(&[]).unwrap()[0].element_width,
            ElementWidth::Word8
        );
    }

    #[test]
    fn propagate_descriptors_rejects_nonempty_inputs() {
        let src = make_source();
        let dummy_desc = PlaneDescriptor {
            role: Role::Raw,
            element_width: ElementWidth::Byte,
            length_bytes: 0,
            layout: Layout::Flat,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        };
        assert!(src.propagate_descriptors(&[dummy_desc]).is_err());
    }

    #[test]
    fn write_params_roundtrip() {
        // Chain-internal code 0x0003 = Float32.
        let src = Source {
            shape: vec![3, 7, 512],
            dtype_code: 0x0003,
        };
        let mut buf = Vec::new();
        src.write_params(&mut buf);
        let (decoded, n) = read_source_params(&buf).unwrap();
        assert_eq!(n, buf.len());
        assert_eq!(decoded.shape, src.shape);
        assert_eq!(decoded.dtype_code, src.dtype_code);
    }

    #[test]
    fn new_rejects_too_many_dims() {
        let shape = vec![1u32; MAX_SOURCE_DIMS + 1];
        assert!(Source::new(shape, 1).is_err());
    }

    #[test]
    fn propagate_descriptors_rejects_too_many_dims() {
        // Bypass `new` via struct literal to verify the trait method also rejects.
        let src = Source {
            shape: vec![1u32; MAX_SOURCE_DIMS + 1],
            dtype_code: 1,
        };
        assert!(src.propagate_descriptors(&[]).is_err());
    }

    #[test]
    fn empty_shape_uses_flat_layout() {
        // Chain-internal 0x0003 = Float32 → 4 bytes/element.
        let src = Source {
            shape: vec![],
            dtype_code: 0x0003,
        };
        let descs = src.propagate_descriptors(&[]).unwrap();
        assert_eq!(descs[0].layout, Layout::Flat);
        // empty shape → scalar (1 element) → 4 bytes for fp32
        assert_eq!(descs[0].length_bytes, 4);
    }

    #[test]
    fn packed_fp4_source_is_nibble_width_and_nibble_packed() {
        // Chain-internal 0x001F = Float4E2M1FNx2: two fp4 values per byte.
        // PerGroupCodebook accepts a plane only when it is nibble-packed at
        // nibble width, so a Byte-width descriptor here leaves that codec
        // unreachable no matter how a tensor is classified or routed.
        let src = Source {
            shape: vec![64, 128],
            dtype_code: 0x001F,
        };
        let d = &src.propagate_descriptors(&[]).unwrap()[0];
        assert_eq!(d.element_width, ElementWidth::Nibble);
        assert!(d.is_nibble_packed);
        // 8192 fp4 values occupy 4096 bytes, not 8192.
        assert_eq!(d.length_bytes, 64 * 128 / 2);
    }

    #[test]
    fn odd_element_count_at_nibble_width_rounds_up_to_whole_bytes() {
        let src = Source {
            shape: vec![7],
            dtype_code: 0x001F,
        };
        let d = &src.propagate_descriptors(&[]).unwrap()[0];
        // 7 nibbles need 4 bytes; the trailing nibble still costs one.
        assert_eq!(d.length_bytes, 4);
    }

    #[test]
    fn byte_and_word_widths_keep_their_previous_lengths() {
        // Guards the bits-based length arithmetic against regressing the
        // non-nibble dtypes it also now covers.
        for (dtype_code, per_elem) in [(0x0006u16, 1u64), (0x0002, 2), (0x0003, 4)] {
            let src = Source {
                shape: vec![10, 10],
                dtype_code,
            };
            let d = &src.propagate_descriptors(&[]).unwrap()[0];
            assert_eq!(d.length_bytes, 100 * per_elem, "dtype 0x{dtype_code:04X}");
            assert!(!d.is_nibble_packed, "dtype 0x{dtype_code:04X}");
        }
    }
}
