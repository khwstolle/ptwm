//! `SphericalNormalize` — per-row (radius, direction) reparameterization
//! with an exact XOR residual. Lossless by construction (the inverse
//! reconstructs the prediction from the stored radius/direction and XORs
//! the residual back, so float-op inaccuracy only affects residual size,
//! never correctness).

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

/// Rebuild the unit vector from quantized-angle bf16 bits (shared by forward
/// and inverse so the two sides are provably bit-identical).
///
/// Reconstruction: standard n-sphere product of sines × cosine.
fn unit_from_angle_bits(angle_bits: &[u16], d: usize) -> Vec<f32> {
    let mut unit = vec![0f32; d];
    let mut sin_prod = 1f32;
    for k in 0..d.saturating_sub(1) {
        let theta = bf16_bits_to_f32(angle_bits[k]);
        unit[k] = sin_prod * theta.cos();
        sin_prod *= theta.sin();
    }
    if d > 0 {
        unit[d - 1] = sin_prod;
    }
    unit
}

/// Convert a vector to (d-1) spherical angles, quantize each to a bf16, and
/// return both the quantized angle bits and the reconstructed unit vector
/// (via `unit_from_angle_bits`, so forward/inverse share one code path).
///
/// Angles: theta_k = atan2( ‖v[k+1..]‖ , v[k] ), k = 0..d-1.
fn angles_quantize_and_reconstruct(vals: &[f32]) -> (Vec<u16>, Vec<f32>) {
    let d = vals.len();
    let mut angle_bits = Vec::with_capacity(d.saturating_sub(1));
    for k in 0..d.saturating_sub(1) {
        let mut tail = 0f32;
        for &x in &vals[k + 1..] {
            tail += x * x;
        }
        let theta = tail.sqrt().atan2(vals[k]);
        angle_bits.push(f32_to_bf16_bits(theta));
    }
    let unit = unit_from_angle_bits(&angle_bits, d);
    (angle_bits, unit)
}

const DIR_MODE_UNIT_VECTOR: u8 = 0;
// Wire-supported and tested, but no production chain or explorer currently
// emits mode 1 — `_make_spherical_normalize_chain` hardcodes mode 0. Kept so
// the angle parameterization can be promoted to a candidate chain later
// without a wire-format change.
const DIR_MODE_ANGLES: u8 = 1;

/// Per-row spherical/unit-vector reparameterization with exact XOR residual.
#[derive(Debug, Clone, Copy)]
pub struct SphericalNormalize {
    mode: u8,
    precision: u8,
    dtype: u8,
}

impl SphericalNormalize {
    pub fn new(mode: u8, precision: u8, dtype: u8) -> Result<Self, PtwmCoreError> {
        if precision != 0 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "SphericalNormalize: precision {precision} unsupported (v1 only 0=bf16)"
            )));
        }
        if mode > DIR_MODE_ANGLES {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "SphericalNormalize: unknown direction mode {mode}"
            )));
        }
        if dtype > 2 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "SphericalNormalize: unknown dtype_code {dtype}"
            )));
        }
        Ok(Self {
            mode,
            precision,
            dtype,
        })
    }

    fn row_len(d: &PlaneDescriptor) -> Result<usize, PtwmCoreError> {
        match d.layout {
            Layout::Rows { row_len } => Ok(row_len as usize),
            Layout::Flat => Err(PtwmCoreError::InvalidContainer(
                "SphericalNormalize requires a Rows layout (2D tensor)".into(),
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

/// Parse the 3 param bytes written by `write_params`.
pub fn read_spherical_normalize_params(
    params: &[u8],
) -> Result<(SphericalNormalize, usize), PtwmCoreError> {
    if params.len() < 3 {
        return Err(PtwmCoreError::InvalidContainer(
            "SphericalNormalize params: need 3 bytes [mode, precision, dtype]".into(),
        ));
    }
    Ok((SphericalNormalize::new(params[0], params[1], params[2])?, 3))
}

impl Op for SphericalNormalize {
    fn propagate_descriptors(
        &self,
        inputs: &[PlaneDescriptor],
    ) -> Result<Vec<PlaneDescriptor>, PtwmCoreError> {
        if inputs.len() != 1 {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "SphericalNormalize: expected 1 input, got {}",
                inputs.len()
            )));
        }
        let d = &inputs[0];
        let row_len = Self::row_len(d)? as u64;
        let w = elem_width(self.dtype) as u64;
        if row_len == 0 || !(d.length_bytes).is_multiple_of(row_len * w) {
            return Err(PtwmCoreError::InvalidContainer(
                "SphericalNormalize: length not a multiple of row_len * elem_width".into(),
            ));
        }
        let rows = d.length_bytes / (row_len * w);
        Ok(vec![
            // radius: one f32 per row
            Self::raw_descriptor(Role::Raw, ElementWidth::Word4, rows * 4, Layout::Flat),
            // direction: bf16 per component
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
                "SphericalNormalize.forward: expected 1 input, got {}",
                inputs.len()
            )));
        }
        let p = &inputs[0];
        let d = Self::row_len(&p.descriptor)?;
        let w = elem_width(self.dtype);
        if d == 0 || !p.bytes.len().is_multiple_of(d * w) {
            return Err(PtwmCoreError::InvalidContainer(
                "SphericalNormalize.forward: byte length not a multiple of row_len*width".into(),
            ));
        }
        let rows = p.bytes.len() / (d * w);
        let descs = self.propagate_descriptors(std::slice::from_ref(&p.descriptor))?;

        let mut radius_bytes = Vec::with_capacity(rows * 4);
        let mut dir_bytes = Vec::with_capacity(rows * d * 2);
        let mut resid_bytes = Vec::with_capacity(p.bytes.len());

        // Reused across rows; every slot is overwritten each iteration.
        let mut vals = vec![0f32; d];
        for row in 0..rows {
            let base = row * d * w;
            // Decode the row to f32.
            for (i, slot) in vals.iter_mut().enumerate() {
                *slot = read_elem_f32(&p.bytes[base + i * w..base + (i + 1) * w], self.dtype);
            }
            // r = ‖v‖ with a fixed left-to-right accumulation.
            let mut acc = 0f32;
            for &x in &vals {
                acc += x * x;
            }
            let r = acc.sqrt();
            radius_bytes.extend_from_slice(&r.to_le_bytes());

            match self.mode {
                DIR_MODE_UNIT_VECTOR => {
                    // Direction: bf16(v_i / r). r==0 / non-finite → 0 dir.
                    for i in 0..d {
                        let u = if r != 0.0 && r.is_finite() {
                            vals[i] / r
                        } else {
                            0.0
                        };
                        let dbits = f32_to_bf16_bits(u);
                        dir_bytes.extend_from_slice(&dbits.to_le_bytes());
                        // prediction p_i = bf16_to_f32(dbits) * r.
                        let pred = bf16_bits_to_f32(dbits) * r;
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
                _ => {
                    // mode 1: spherical angles. Direction plane stores (d-1)
                    // angle bf16s + 1 zero pad so it stays rows*d*2 bytes.
                    let (angle_bits, unit) = angles_quantize_and_reconstruct(&vals);
                    for k in 0..d {
                        let ab = if k < angle_bits.len() {
                            angle_bits[k]
                        } else {
                            0u16
                        };
                        dir_bytes.extend_from_slice(&ab.to_le_bytes());
                    }
                    for i in 0..d {
                        let pred = unit[i] * r;
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
            }
        }

        Ok(vec![
            Plane {
                bytes: Arc::from(radius_bytes.into_boxed_slice()),
                descriptor: descs[0].clone(),
            },
            Plane {
                bytes: Arc::from(dir_bytes.into_boxed_slice()),
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
                "SphericalNormalize.inverse: expected 3 inputs, got {}",
                outputs.len()
            )));
        }
        let radius = &outputs[0].bytes;
        let dir = &outputs[1].bytes;
        let resid = &outputs[2];
        let d = Self::row_len(&resid.descriptor)?;
        let w = elem_width(self.dtype);
        // Validate plane sizes before indexing so a corrupt/mismatched
        // container yields an error rather than a panic (mirrors the
        // byte_split inverse convention).
        if !radius.len().is_multiple_of(4) {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "SphericalNormalize.inverse: radius plane length {} is not a multiple of 4",
                radius.len()
            )));
        }
        let rows = radius.len() / 4;
        if dir.len() != rows * d * 2 || resid.bytes.len() != rows * d * w {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "SphericalNormalize.inverse: plane size mismatch (rows={rows}, d={d}, w={w}): \
                 direction={}, residual={}",
                dir.len(),
                resid.bytes.len()
            )));
        }
        let mut out = vec![0u8; resid.bytes.len()];
        // Reused across rows in mode 1; every slot is overwritten per row.
        let mut angle_bits = vec![0u16; d];

        for row in 0..rows {
            let r = f32::from_le_bytes([
                radius[row * 4],
                radius[row * 4 + 1],
                radius[row * 4 + 2],
                radius[row * 4 + 3],
            ]);
            match self.mode {
                DIR_MODE_UNIT_VECTOR => {
                    for i in 0..d {
                        let didx = (row * d + i) * 2;
                        let dbits = u16::from_le_bytes([dir[didx], dir[didx + 1]]);
                        let pred = bf16_bits_to_f32(dbits) * r;
                        let pb = pred_bits(pred, self.dtype);
                        let base = (row * d + i) * w;
                        for k in 0..w {
                            out[base + k] = resid.bytes[base + k] ^ pb[k];
                        }
                    }
                }
                _ => {
                    // mode 1: read d angle slots (first d-1 are angles), rebuild
                    // the unit vector with the identical product loop.
                    for (k, ab) in angle_bits.iter_mut().enumerate() {
                        let didx = (row * d + k) * 2;
                        *ab = u16::from_le_bytes([dir[didx], dir[didx + 1]]);
                    }
                    let unit = unit_from_angle_bits(&angle_bits, d);
                    for i in 0..d {
                        let pred = unit[i] * r;
                        let pb = pred_bits(pred, self.dtype);
                        let base = (row * d + i) * w;
                        for k in 0..w {
                            out[base + k] = resid.bytes[base + k] ^ pb[k];
                        }
                    }
                }
            }
        }
        Ok(vec![Plane {
            bytes: Arc::from(out.into_boxed_slice()),
            descriptor: resid.descriptor.clone(),
        }])
    }

    fn id(&self) -> OpId {
        OpId::SphericalNormalize
    }

    fn write_params(&self, out: &mut Vec<u8>) {
        out.push(self.mode);
        out.push(self.precision);
        out.push(self.dtype);
    }
}

#[cfg(test)]
mod helper_tests {
    use super::*;

    #[test]
    fn bf16_roundtrip_exact_for_truncated_values() {
        // A value whose low 16 mantissa bits are already zero survives bf16.
        let v = f32::from_bits(0x3F80_0000); // 1.0
        let bits = f32_to_bf16_bits(v);
        assert_eq!(bf16_bits_to_f32(bits), 1.0);
    }

    #[test]
    fn f16_roundtrip_representable() {
        let v = 0.5f32;
        let bits = f32_to_f16_bits(v);
        assert_eq!(f16_bits_to_f32(bits), 0.5);
    }

    #[test]
    fn read_write_elem_fp32_roundtrips_bits() {
        let bytes = 1.5f32.to_le_bytes();
        assert_eq!(read_elem_f32(&bytes, 0), 1.5);
        assert_eq!(pred_bits(1.5, 0), bytes); // fp32: all 4 bytes meaningful
    }

    #[test]
    fn read_write_elem_bf16_roundtrips() {
        // bf16 for 1.0 is the high 2 bytes of 1.0f32 (LE): 0x80,0x3F.
        let bytes = [0x80u8, 0x3F];
        assert_eq!(read_elem_f32(&bytes, 1), 1.0);
        // bf16: only the first elem_width(1)=2 bytes are meaningful.
        assert_eq!(&pred_bits(1.0, 1)[..2], &bytes);
    }

    #[test]
    fn elem_width_matches_dtype() {
        assert_eq!(elem_width(0), 4);
        assert_eq!(elem_width(1), 2);
        assert_eq!(elem_width(2), 2);
    }
}

#[cfg(test)]
mod op_tests {
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

    fn roundtrip(dtype: u8, mode: u8, rows: u32, d: u32, values: Vec<f32>) {
        let op = SphericalNormalize::new(mode, 0, dtype).unwrap();
        let inp = input_plane(rows, d, dtype, &values);
        let orig = inp.bytes.clone();
        let outs = op.forward(std::slice::from_ref(&inp)).unwrap();
        assert_eq!(outs.len(), 3, "radius + direction + residual");
        let back = op.inverse(&outs).unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(
            back[0].bytes, orig,
            "dtype={dtype} mode={mode} must round-trip bit-exact"
        );
    }

    #[test]
    fn mode0_fp32_roundtrip_normal_rows() {
        let vals: Vec<f32> = (0..32).map(|i| (i as f32) * 0.125 - 2.0).collect();
        roundtrip(0, 0, 4, 8, vals);
    }

    #[test]
    fn mode0_bf16_roundtrip() {
        let vals: Vec<f32> = (0..16).map(|i| (i as f32) * 0.5).collect();
        roundtrip(1, 0, 4, 4, vals);
    }

    #[test]
    fn mode0_fp16_roundtrip() {
        let vals: Vec<f32> = (0..16).map(|i| (i as f32) * 0.25 - 1.0).collect();
        roundtrip(2, 0, 4, 4, vals);
    }

    #[test]
    fn mode0_adversarial_rows() {
        // zeros, denormal, inf, nan, all-equal, single-nonzero.
        let nan = f32::NAN;
        let inf = f32::INFINITY;
        let den = f32::from_bits(1); // smallest denormal
        let vals = vec![
            0.0, 0.0, 0.0, 0.0, // all zero row
            den, -den, den, -den, // denormals
            inf, -inf, 1.0, 2.0, // inf
            nan, 1.0, 2.0, 3.0, // nan
            5.0, 5.0, 5.0, 5.0, // all equal
            7.0, 0.0, 0.0, 0.0, // single nonzero
        ];
        roundtrip(0, 0, 6, 4, vals);
    }

    #[test]
    fn inverse_rejects_mismatched_plane_sizes() {
        // A direction plane that is too short must error, not panic.
        let op = SphericalNormalize::new(0, 0, 0).unwrap();
        let inp = input_plane(4, 8, 0, &[1.0; 32]);
        let mut outs = op.forward(std::slice::from_ref(&inp)).unwrap();
        let short = outs[1].bytes.len() - 2;
        // corrupt direction plane (shorten it)
        let truncated: Arc<[u8]> = Arc::from(&outs[1].bytes[..short]);
        outs[1] = Plane {
            bytes: truncated,
            descriptor: outs[1].descriptor.clone(),
        };
        assert!(op.inverse(&outs).is_err());
    }

    #[test]
    fn propagate_descriptors_shapes() {
        let op = SphericalNormalize::new(0, 0, 0).unwrap();
        let inp = input_plane(4, 8, 0, &[1.0; 32]);
        let outs = op
            .propagate_descriptors(std::slice::from_ref(&inp.descriptor))
            .unwrap();
        assert_eq!(outs.len(), 3);
        assert_eq!(outs[0].length_bytes, 4 * 4); // radius: rows * f32
        assert_eq!(outs[1].length_bytes, 4 * 8 * 2); // direction: rows*d*bf16
        assert_eq!(outs[2].length_bytes, inp.descriptor.length_bytes); // residual
    }

    #[test]
    fn write_params_roundtrip() {
        let op = SphericalNormalize::new(1, 0, 2).unwrap();
        let mut buf = Vec::new();
        op.write_params(&mut buf);
        assert_eq!(buf, vec![1, 0, 2]);
        let (op2, n) = read_spherical_normalize_params(&buf).unwrap();
        assert_eq!(n, 3);
        assert_eq!(op2.id(), OpId::SphericalNormalize);
    }

    #[test]
    fn mode1_fp32_roundtrip() {
        let vals: Vec<f32> = (0..32).map(|i| (i as f32).sin()).collect();
        roundtrip(0, 1, 4, 8, vals);
    }

    #[test]
    fn mode1_bf16_roundtrip() {
        let vals: Vec<f32> = (0..16).map(|i| (i as f32) * 0.3 - 1.0).collect();
        roundtrip(1, 1, 4, 4, vals);
    }

    #[test]
    fn mode1_fp16_roundtrip() {
        let vals: Vec<f32> = (0..16).map(|i| (i as f32) * 0.25 - 1.0).collect();
        roundtrip(2, 1, 4, 4, vals);
    }

    /// 7 rows × 4 cols of pathological values: all-zero, ±denormal, ±inf,
    /// NaN, all-equal, single-nonzero, and very large/small magnitudes.
    fn adversarial_vals() -> Vec<f32> {
        let den = f32::from_bits(1); // smallest positive denormal
        vec![
            0.0,
            0.0,
            0.0,
            0.0, // all zero
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
            5.0, // all equal
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
    fn adversarial_all_modes_all_dtypes() {
        // Bit-exact roundtrip must hold for every (mode, dtype) combination.
        // Whatever the dtype's representation of each value (incl. fp16/bf16
        // overflow to inf), the XOR residual reproduces those exact bytes.
        for dtype in [0u8, 1, 2] {
            for mode in [0u8, 1] {
                roundtrip(dtype, mode, 7, 4, adversarial_vals());
            }
        }
    }

    #[test]
    fn mode1_adversarial_rows() {
        let vals = vec![
            0.0,
            0.0,
            0.0,
            0.0,
            f32::INFINITY,
            1.0,
            2.0,
            3.0,
            f32::NAN,
            1.0,
            2.0,
            3.0,
            9.0,
            9.0,
            9.0,
            9.0,
        ];
        roundtrip(0, 1, 4, 4, vals);
    }

    #[test]
    fn rejects_nonzero_precision() {
        assert!(SphericalNormalize::new(0, 1, 0).is_err());
    }

    #[test]
    fn rejects_flat_layout() {
        let op = SphericalNormalize::new(0, 0, 0).unwrap();
        let mut inp = input_plane(1, 8, 0, &[1.0; 8]);
        inp.descriptor.layout = Layout::Flat;
        assert!(op.forward(&[inp]).is_err());
    }
}
