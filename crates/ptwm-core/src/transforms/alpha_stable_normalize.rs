//! `AlphaStableNormalize` — per-row robust affine normalization `(v − δ)/γ`
//! with an exact XOR residual. δ = median(row), γ = median(|row − δ|) (MAD).
//! Lossless by construction: the fit is encode-only, `γ, δ` are stored per
//! row, and the residual repairs all round-off — so reconstruction is
//! bit-exact regardless of the (heavy-tailed) distribution.

use std::sync::Arc;

use crate::error::PtwmCoreError;
use crate::transforms::op::{Op, OpId, Plane};
use crate::types::descriptor::{ElementWidth, Layout, PlaneDescriptor};
use crate::types::role::Role;

/// f32 → bf16 bits (round to nearest even). bf16 is the high 16 bits of f32.
fn f32_to_bf16_bits(v: f32) -> u16 {
    let x = v.to_bits();
    if (x & 0x7FFF_FFFF) > 0x7F80_0000 {
        // NaN: set a quiet-NaN mantissa bit so the value stays NaN after
        // truncating to the high 16 bits.
        return ((x >> 16) as u16) | 0x0040;
    }
    let rounding_bias = 0x7FFF + ((x >> 16) & 1);
    ((x + rounding_bias) >> 16) as u16
}

fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// f32 → IEEE half (f16) bits, round to nearest even.
fn f32_to_f16_bits(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xFF) as i32;
    let mant = x & 0x007F_FFFF;
    if exp == 0xFF {
        // Inf / NaN.
        let m = if mant != 0 { 0x0200 } else { 0 }; // quiet NaN payload
        return sign | 0x7C00 | m;
    }
    let unbiased = exp - 127 + 15;
    if unbiased >= 0x1F {
        return sign | 0x7C00; // overflow → inf
    }
    if unbiased <= 0 {
        if unbiased < -10 {
            return sign; // underflow → signed zero
        }
        // Subnormal half.
        let m = (mant | 0x0080_0000) >> (1 - unbiased);
        let round = ((m & 0x0000_1FFF) > 0x0000_1000
            || ((m & 0x0000_1FFF) == 0x0000_1000 && (m & 0x0000_2000) != 0))
            as u32;
        return sign | ((m >> 13) as u16 + round as u16);
    }
    let half = (unbiased as u16) << 10 | (mant >> 13) as u16;
    let round = ((mant & 0x0000_1FFF) > 0x0000_1000
        || ((mant & 0x0000_1FFF) == 0x0000_1000 && (mant & 0x0000_2000) != 0))
        as u16;
    sign | (half + round)
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x03FF) as u32;
    let out = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // Subnormal: normalize.
            let mut e = -1i32;
            let mut m = mant;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03FF;
            sign | (((e + 127 - 15 + 1) as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1F {
        sign | 0x7F80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(out)
}

/// Byte width of an element for a given dtype_code (0=fp32, 1=bf16, 2=fp16).
fn elem_width(dtype: u8) -> usize {
    match dtype {
        0 => 4,
        _ => 2,
    }
}

/// Decode one element's bytes to f32 for the given dtype_code.
fn read_elem_f32(bytes: &[u8], dtype: u8) -> f32 {
    match dtype {
        0 => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        1 => bf16_bits_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])),
        _ => f16_bits_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])),
    }
}

/// Encode an f32 prediction to the dtype's wire bytes (LE) in a stack
/// buffer. Only the first `elem_width(dtype)` bytes are meaningful; the
/// remainder are zero. Returning a fixed array rather than a `Vec` avoids a
/// heap allocation per element in the forward/inverse hot loops (which run
/// once per tensor element during trial-encoding).
fn pred_bits(p: f32, dtype: u8) -> [u8; 4] {
    match dtype {
        0 => p.to_le_bytes(),
        1 => {
            let b = f32_to_bf16_bits(p).to_le_bytes();
            [b[0], b[1], 0, 0]
        }
        _ => {
            let b = f32_to_f16_bits(p).to_le_bytes();
            [b[0], b[1], 0, 0]
        }
    }
}

#[inline]
fn emit_residual(pb: &[u8], src: &[u8], base: usize, i: usize, w: usize, out: &mut Vec<u8>) {
    let orig = &src[base + i * w..base + (i + 1) * w];
    for k in 0..w {
        out.push(orig[k] ^ pb[k]);
    }
}

/// Bytes per row in the scale plane: a `(γ, δ)` f32 pair.
const SCALE_BYTES_PER_ROW: usize = 8;

/// Deterministic "median" of `scratch`: the upper-middle element in
/// `total_cmp` order (NaN-safe). Uses `select_nth_unstable_by` (O(N) average,
/// vs O(N log N) for a full sort) — it places the sorted-position-`mid` value
/// at `mid`, which is exactly the value a full sort would yield there.
/// `scratch` is partially reordered in place. Empty → 0.0.
fn median_in_place(scratch: &mut [f32]) -> f32 {
    if scratch.is_empty() {
        return 0.0;
    }
    let mid = scratch.len() / 2;
    let (_, val, _) = scratch.select_nth_unstable_by(mid, f32::total_cmp);
    *val
}

/// Robust per-row offset/scale: δ = median(v), γ = median(|v − δ|) (MAD).
/// Falls back to `(γ=1, δ=0)` whenever either is non-finite or γ == 0, so the
/// normalization is always well-defined (the residual still makes it exact).
/// `scratch` is a caller-owned reusable buffer (hoisted out of the row loop)
/// to avoid a per-row heap allocation.
fn fit_affine(vals: &[f32], scratch: &mut Vec<f32>) -> (f32, f32) {
    scratch.clear();
    scratch.extend_from_slice(vals);
    let delta = median_in_place(scratch);
    scratch.clear();
    scratch.extend(vals.iter().map(|&x| (x - delta).abs()));
    let gamma = median_in_place(scratch);
    if !delta.is_finite() || !gamma.is_finite() || gamma == 0.0 {
        (1.0, 0.0)
    } else {
        (gamma, delta)
    }
}

/// Per-row robust affine normalization with exact XOR residual.
#[derive(Debug, Clone, Copy)]
pub struct AlphaStableNormalize {
    precision: u8,
    dtype: u8,
}

impl AlphaStableNormalize {
    pub fn new(precision: u8, dtype: u8) -> Result<Self, PtwmCoreError> {
        if precision != 0 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "AlphaStableNormalize: precision {precision} unsupported (v1 only 0=bf16)"
            )));
        }
        if dtype > 2 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "AlphaStableNormalize: unknown dtype_code {dtype}"
            )));
        }
        Ok(Self { precision, dtype })
    }

    fn row_len(d: &PlaneDescriptor) -> Result<usize, PtwmCoreError> {
        match d.layout {
            Layout::Rows { row_len } => Ok(row_len as usize),
            Layout::Flat => Err(PtwmCoreError::InvalidContainer(
                "AlphaStableNormalize requires a Rows layout (2D tensor)".into(),
            )),
        }
    }

    fn raw_descriptor(
        role: Role,
        width: ElementWidth,
        len: u64,
        layout: Layout,
    ) -> PlaneDescriptor {
        PlaneDescriptor {
            role,
            element_width: width,
            length_bytes: len,
            layout,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        }
    }
}

/// Parse the 2 param bytes `[precision, dtype]`.
pub fn read_alpha_stable_normalize_params(
    params: &[u8],
) -> Result<(AlphaStableNormalize, usize), PtwmCoreError> {
    if params.len() < 2 {
        return Err(PtwmCoreError::InvalidContainer(
            "AlphaStableNormalize params: need 2 bytes [precision, dtype]".into(),
        ));
    }
    Ok((AlphaStableNormalize::new(params[0], params[1])?, 2))
}

impl Op for AlphaStableNormalize {
    fn propagate_descriptors(
        &self,
        inputs: &[PlaneDescriptor],
    ) -> Result<Vec<PlaneDescriptor>, PtwmCoreError> {
        if inputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "AlphaStableNormalize: expected 1 input, got {}",
                inputs.len()
            )));
        }
        let d = &inputs[0];
        let row_len = Self::row_len(d)? as u64;
        let w = elem_width(self.dtype) as u64;
        if row_len == 0 || !(d.length_bytes).is_multiple_of(row_len * w) {
            return Err(PtwmCoreError::InvalidContainer(
                "AlphaStableNormalize: length not a multiple of row_len * elem_width".into(),
            ));
        }
        let rows = d.length_bytes / (row_len * w);
        Ok(vec![
            // scale: (γ, δ) f32 pair per row = SCALE_BYTES_PER_ROW bytes/row
            Self::raw_descriptor(
                Role::Raw,
                ElementWidth::Word4,
                rows * SCALE_BYTES_PER_ROW as u64,
                Layout::Flat,
            ),
            // normalized: bf16 per element
            Self::raw_descriptor(
                Role::Raw,
                ElementWidth::Word2,
                rows * row_len * 2,
                Layout::Rows {
                    row_len: row_len as u32,
                },
            ),
            // residual: same width and layout as input
            Self::raw_descriptor(Role::Raw, d.element_width, d.length_bytes, d.layout),
        ])
    }

    fn forward(&self, inputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        if inputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "AlphaStableNormalize.forward: expected 1 input, got {}",
                inputs.len()
            )));
        }
        let p = &inputs[0];
        let d = Self::row_len(&p.descriptor)?;
        let w = elem_width(self.dtype);
        if d == 0 || !p.bytes.len().is_multiple_of(d * w) {
            return Err(PtwmCoreError::InvalidContainer(
                "AlphaStableNormalize.forward: byte length not a multiple of row_len*width".into(),
            ));
        }
        let rows = p.bytes.len() / (d * w);
        let descs = self.propagate_descriptors(std::slice::from_ref(&p.descriptor))?;

        let mut scale_bytes = Vec::with_capacity(rows * SCALE_BYTES_PER_ROW);
        let mut norm_bytes = Vec::with_capacity(rows * d * 2);
        let mut resid_bytes = Vec::with_capacity(p.bytes.len());

        let mut vals = vec![0f32; d];
        // Reused across rows for the median sorts (avoids per-row allocation).
        let mut scratch: Vec<f32> = Vec::with_capacity(d);
        for row in 0..rows {
            let base = row * d * w;
            for (i, slot) in vals.iter_mut().enumerate() {
                *slot = read_elem_f32(&p.bytes[base + i * w..base + (i + 1) * w], self.dtype);
            }
            let (gamma, delta) = fit_affine(&vals, &mut scratch);
            scale_bytes.extend_from_slice(&gamma.to_le_bytes());
            scale_bytes.extend_from_slice(&delta.to_le_bytes());

            for i in 0..d {
                let n = (vals[i] - delta) / gamma;
                let nbits = f32_to_bf16_bits(n);
                norm_bytes.extend_from_slice(&nbits.to_le_bytes());
                let pred = bf16_bits_to_f32(nbits) * gamma + delta;
                emit_residual(
                    &pred_bits(pred, self.dtype),
                    &p.bytes,
                    base,
                    i,
                    w,
                    &mut resid_bytes,
                );
            }
        }

        Ok(vec![
            Plane {
                bytes: Arc::from(scale_bytes.into_boxed_slice()),
                descriptor: descs[0].clone(),
            },
            Plane {
                bytes: Arc::from(norm_bytes.into_boxed_slice()),
                descriptor: descs[1].clone(),
            },
            Plane {
                bytes: Arc::from(resid_bytes.into_boxed_slice()),
                descriptor: descs[2].clone(),
            },
        ])
    }

    fn inverse(&self, outputs: &[Plane]) -> Result<Vec<Plane>, PtwmCoreError> {
        if outputs.len() != 3 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "AlphaStableNormalize.inverse: expected 3 inputs, got {}",
                outputs.len()
            )));
        }
        let scale = &outputs[0].bytes;
        let norm = &outputs[1].bytes;
        let resid = &outputs[2];
        let d = Self::row_len(&resid.descriptor)?;
        let w = elem_width(self.dtype);
        if !scale.len().is_multiple_of(SCALE_BYTES_PER_ROW) {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "AlphaStableNormalize.inverse: scale plane length {} is not a multiple of {SCALE_BYTES_PER_ROW}",
                scale.len()
            )));
        }
        let rows = scale.len() / SCALE_BYTES_PER_ROW;
        if norm.len() != rows * d * 2 || resid.bytes.len() != rows * d * w {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "AlphaStableNormalize.inverse: plane size mismatch (rows={rows}, d={d}, w={w}): \
                 normalized={}, residual={}",
                norm.len(),
                resid.bytes.len()
            )));
        }
        let mut out = vec![0u8; resid.bytes.len()];

        for row in 0..rows {
            let s = row * SCALE_BYTES_PER_ROW;
            let gamma = f32::from_le_bytes([scale[s], scale[s + 1], scale[s + 2], scale[s + 3]]);
            let delta =
                f32::from_le_bytes([scale[s + 4], scale[s + 5], scale[s + 6], scale[s + 7]]);
            for i in 0..d {
                let nidx = (row * d + i) * 2;
                let nbits = u16::from_le_bytes([norm[nidx], norm[nidx + 1]]);
                let pred = bf16_bits_to_f32(nbits) * gamma + delta;
                let pb = pred_bits(pred, self.dtype);
                let bbase = (row * d + i) * w;
                for k in 0..w {
                    out[bbase + k] = resid.bytes[bbase + k] ^ pb[k];
                }
            }
        }
        Ok(vec![Plane {
            bytes: Arc::from(out.into_boxed_slice()),
            descriptor: resid.descriptor.clone(),
        }])
    }

    fn id(&self) -> OpId {
        OpId::AlphaStableNormalize
    }

    fn write_params(&self, out: &mut Vec<u8>) {
        out.push(self.precision);
        out.push(self.dtype);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input_plane(rows: u32, d: u32, dtype: u8, values: &[f32]) -> Plane {
        let mut bytes = Vec::new();
        for &v in values {
            bytes.extend_from_slice(&pred_bits(v, dtype)[..elem_width(dtype)]);
        }
        let width = elem_width(dtype) as u64;
        Plane {
            bytes: Arc::from(bytes.into_boxed_slice()),
            descriptor: PlaneDescriptor {
                role: Role::Raw,
                element_width: if dtype == 0 {
                    ElementWidth::Word4
                } else {
                    ElementWidth::Word2
                },
                length_bytes: rows as u64 * d as u64 * width,
                layout: Layout::Rows { row_len: d },
                derives_from_tensor: None,
                residual_of: None,
                is_nibble_packed: false,
                vendor_bytes: vec![],
            },
        }
    }

    fn roundtrip(dtype: u8, rows: u32, d: u32, values: Vec<f32>) {
        let op = AlphaStableNormalize::new(0, dtype).unwrap();
        let inp = input_plane(rows, d, dtype, &values);
        let orig = inp.bytes.clone();
        let outs = op.forward(std::slice::from_ref(&inp)).unwrap();
        assert_eq!(outs.len(), 3, "scale + normalized + residual");
        let back = op.inverse(&outs).unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(
            back[0].bytes, orig,
            "dtype={dtype} must round-trip bit-exact"
        );
    }

    #[test]
    fn fp32_roundtrip_normal_rows() {
        let vals: Vec<f32> = (0..32).map(|i| (i as f32) * 0.125 - 2.0).collect();
        roundtrip(0, 4, 8, vals);
    }

    #[test]
    fn bf16_roundtrip() {
        let vals: Vec<f32> = (0..16).map(|i| (i as f32) * 0.5 - 3.0).collect();
        roundtrip(1, 4, 4, vals);
    }

    #[test]
    fn fp16_roundtrip() {
        let vals: Vec<f32> = (0..16).map(|i| (i as f32) * 0.25 - 1.0).collect();
        roundtrip(2, 4, 4, vals);
    }

    fn adversarial_vals() -> Vec<f32> {
        let den = f32::from_bits(1);
        vec![
            0.0,
            0.0,
            0.0,
            0.0, // all zero (γ→0 fallback)
            den,
            -den,
            den,
            -den, // denormals
            f32::INFINITY,
            -f32::INFINITY,
            1.0,
            2.0, // inf
            f32::NAN,
            1.0,
            2.0,
            3.0, // nan
            5.0,
            5.0,
            5.0,
            5.0, // all equal (MAD→0 fallback)
            7.0,
            0.0,
            0.0,
            0.0, // single nonzero
            1.0e30,
            -1.0e30,
            1.0e-30,
            -1.0e-30, // extreme magnitudes
        ]
    }

    #[test]
    fn adversarial_all_dtypes() {
        for dtype in [0u8, 1, 2] {
            roundtrip(dtype, 7, 4, adversarial_vals());
        }
    }

    #[test]
    fn inverse_rejects_mismatched_plane_sizes() {
        let op = AlphaStableNormalize::new(0, 0).unwrap();
        let inp = input_plane(4, 8, 0, &[1.0; 32]);
        let mut outs = op.forward(std::slice::from_ref(&inp)).unwrap();
        let short = outs[1].bytes.len() - 2;
        // corrupt normalized plane (shorten it)
        let truncated: Arc<[u8]> = Arc::from(&outs[1].bytes[..short]);
        outs[1] = Plane {
            bytes: truncated,
            descriptor: outs[1].descriptor.clone(),
        };
        assert!(op.inverse(&outs).is_err());
    }

    #[test]
    fn propagate_descriptors_shapes() {
        let op = AlphaStableNormalize::new(0, 0).unwrap();
        let inp = input_plane(4, 8, 0, &[1.0; 32]);
        let outs = op
            .propagate_descriptors(std::slice::from_ref(&inp.descriptor))
            .unwrap();
        assert_eq!(outs.len(), 3);
        assert_eq!(outs[0].length_bytes, 4 * 8); // scale: rows * (γ,δ) f32 pair
        assert_eq!(outs[1].length_bytes, 4 * 8 * 2); // normalized: rows*d*bf16
        assert_eq!(outs[2].length_bytes, inp.descriptor.length_bytes); // residual
    }

    #[test]
    fn write_params_roundtrip() {
        let op = AlphaStableNormalize::new(0, 2).unwrap();
        let mut buf = Vec::new();
        op.write_params(&mut buf);
        assert_eq!(buf, vec![0, 2]);
        let (op2, n) = read_alpha_stable_normalize_params(&buf).unwrap();
        assert_eq!(n, 2);
        assert_eq!(op2.id(), OpId::AlphaStableNormalize);
    }

    #[test]
    fn rejects_nonzero_precision_and_bad_dtype() {
        assert!(AlphaStableNormalize::new(1, 0).is_err());
        assert!(AlphaStableNormalize::new(0, 3).is_err());
    }

    #[test]
    fn rejects_flat_layout() {
        let op = AlphaStableNormalize::new(0, 0).unwrap();
        let mut inp = input_plane(1, 8, 0, &[1.0; 8]);
        inp.descriptor.layout = Layout::Flat;
        assert!(op.forward(&[inp]).is_err());
    }
}
