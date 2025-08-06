//! Delta (reference-frame) compression primitives.
//!
//! Cross-checkpoint compression stores a tensor as the *difference* from a
//! reference tensor (e.g. a base model checkpoint). When two checkpoints
//! are close, the residual concentrates on very few bits and compresses
//! far better than the raw tensor.
//!
//! Two modes are implemented here:
//!
//! - **Byte-level XOR** ([`xor_encode`] / [`xor_decode`]): bit-exact and safe
//!   for lossless compression of any dtype.
//! - **Float-arithmetic subtraction** ([`float_sub_encode`] / [`float_sub_decode`]):
//!   interprets buffers as little-endian `f32` arrays and computes element-wise
//!   `raw - reference` / `reference + residual`. Best suited for fine-tune
//!   checkpoints where weight perturbations are small.
//!
//! Learned-basis and predictive modes remain out of scope for this module.

use crate::error::PtwmCoreError;

/// Length of a BLAKE3 hash in bytes.
pub const BLAKE3_HASH_LEN: usize = 32;

/// Wire identifier for the byte-level XOR delta scheme. Reserved for the
/// future `DeltaSpec` TLV in the container header (step 7).
pub const DELTA_SCHEME_XOR: u8 = 1;

fn length_mismatch_encode(raw: usize, reference: usize) -> PtwmCoreError {
    PtwmCoreError::CodecEncode {
        codec: "delta",
        msg: format!("length mismatch: raw={raw} reference={reference}"),
    }
}

fn length_mismatch_decode(residual: usize, reference: usize) -> PtwmCoreError {
    PtwmCoreError::CodecDecode {
        codec: "delta",
        msg: format!("length mismatch: residual={residual} reference={reference}"),
    }
}

fn alignment_error_encode(len: usize) -> PtwmCoreError {
    PtwmCoreError::CodecEncode {
        codec: "delta",
        msg: format!("float delta requires length divisible by 4, got {len}"),
    }
}

fn alignment_error_decode(len: usize) -> PtwmCoreError {
    PtwmCoreError::CodecDecode {
        codec: "delta",
        msg: format!("float delta requires length divisible by 4, got {len}"),
    }
}

/// Compute the XOR residual of `raw` and `reference`.
///
/// For equal-length inputs, returns `raw ^ reference` byte-by-byte. XOR is
/// self-inverse, so [`xor_decode`] is identical in effect but named for
/// clarity at call sites.
pub fn xor_encode(raw: &[u8], reference: &[u8]) -> Result<Vec<u8>, PtwmCoreError> {
    if raw.len() != reference.len() {
        return Err(length_mismatch_encode(raw.len(), reference.len()));
    }
    Ok(raw.iter().zip(reference).map(|(&a, &b)| a ^ b).collect())
}

/// Reconstruct the original bytes from an XOR residual and its reference.
pub fn xor_decode(residual: &[u8], reference: &[u8]) -> Result<Vec<u8>, PtwmCoreError> {
    if residual.len() != reference.len() {
        return Err(length_mismatch_decode(residual.len(), reference.len()));
    }
    Ok(residual
        .iter()
        .zip(reference)
        .map(|(&a, &b)| a ^ b)
        .collect())
}

/// BLAKE3 digest of `data`. Used to identify / verify a delta reference.
pub fn blake3_hash(data: &[u8]) -> [u8; BLAKE3_HASH_LEN] {
    blake3::hash(data).into()
}

/// Wire identifier for the float-arithmetic delta scheme.
pub const DELTA_SCHEME_FLOAT: u8 = 2;

/// Compute element-wise f32 subtraction: `raw[i] - reference[i]`.
///
/// Both inputs are interpreted as little-endian `f32` slices. Byte lengths
/// must be equal and divisible by 4. The inverse is [`float_sub_decode`].
///
/// # Caveats
///
/// - **NaN bit patterns are not preserved**: IEEE 754 arithmetic may produce a
///   different NaN payload than the input, so inputs containing NaN values
///   will not round-trip bit-exactly.
/// - **Catastrophic cancellation**: if the raw value and reference have extreme
///   magnitude differences (e.g. `raw = 1.0`, `reference = 2e30`), the
///   subtraction loses precision and the codec is **not lossless** for those
///   elements. For trained neural network fine-tune deltas, where weights are
///   small perturbations of the base, this is not expected to occur; callers
///   requiring guaranteed bit-exact roundtrip should verify with their data.
pub fn float_sub_encode(raw: &[u8], reference: &[u8]) -> Result<Vec<u8>, PtwmCoreError> {
    if raw.len() != reference.len() {
        return Err(length_mismatch_encode(raw.len(), reference.len()));
    }
    if !raw.len().is_multiple_of(4) {
        return Err(alignment_error_encode(raw.len()));
    }
    let mut out = Vec::with_capacity(raw.len());
    for (r, b) in raw.chunks_exact(4).zip(reference.chunks_exact(4)) {
        let rv = f32::from_le_bytes(r.try_into().unwrap());
        let bv = f32::from_le_bytes(b.try_into().unwrap());
        out.extend_from_slice(&(rv - bv).to_le_bytes());
    }
    Ok(out)
}

/// Reconstruct original bytes: `reference[i] + residual[i]`.
///
/// # Caveats
///
/// - **NaN bit patterns are not preserved**: IEEE 754 arithmetic may produce a
///   different NaN payload than the input, so inputs containing NaN values
///   will not round-trip bit-exactly.
/// - **Catastrophic cancellation**: if the residual and reference have extreme
///   magnitude differences the addition loses precision and the codec is **not
///   lossless** for those elements. For trained neural network fine-tune
///   checkpoints where fine-tune deltas are small perturbations of the base
///   model, this is not expected to occur; callers requiring guaranteed
///   bit-exact roundtrip should verify with their data.
pub fn float_sub_decode(residual: &[u8], reference: &[u8]) -> Result<Vec<u8>, PtwmCoreError> {
    if residual.len() != reference.len() {
        return Err(length_mismatch_decode(residual.len(), reference.len()));
    }
    if !residual.len().is_multiple_of(4) {
        return Err(alignment_error_decode(residual.len()));
    }
    let mut out = Vec::with_capacity(residual.len());
    for (d, b) in residual.chunks_exact(4).zip(reference.chunks_exact(4)) {
        let dv = f32::from_le_bytes(d.try_into().unwrap());
        let bv = f32::from_le_bytes(b.try_into().unwrap());
        out.extend_from_slice(&(bv + dv).to_le_bytes());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xor_roundtrip() {
        let raw: Vec<u8> = (0..1024).map(|i| (i * 17 + 3) as u8).collect();
        let reference: Vec<u8> = (0..1024).map(|i| (i * 11 + 5) as u8).collect();
        let residual = xor_encode(&raw, &reference).unwrap();
        let recovered = xor_decode(&residual, &reference).unwrap();
        assert_eq!(raw, recovered);
    }

    #[test]
    fn xor_identical_inputs_produce_zeros() {
        let data = vec![42u8; 256];
        let residual = xor_encode(&data, &data).unwrap();
        assert!(residual.iter().all(|&b| b == 0));
    }

    #[test]
    fn xor_length_mismatch() {
        let err = xor_encode(&[1, 2, 3], &[1, 2]);
        assert!(matches!(err, Err(PtwmCoreError::CodecEncode { .. })));
    }

    #[test]
    fn xor_empty_inputs() {
        assert_eq!(xor_encode(&[], &[]).unwrap(), Vec::<u8>::new());
        assert_eq!(xor_decode(&[], &[]).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn blake3_deterministic() {
        let data = b"hello world";
        assert_eq!(blake3_hash(data), blake3_hash(data));
    }

    #[test]
    fn blake3_differs_for_different_inputs() {
        assert_ne!(blake3_hash(b"abc"), blake3_hash(b"xyz"));
    }

    #[test]
    fn float_sub_roundtrip() {
        let raw: Vec<u8> = (0..256)
            .flat_map(|i| f32::from(i as u16).to_le_bytes())
            .collect();
        let reference: Vec<u8> = (0..256)
            .flat_map(|i| (f32::from(i as u16) * 0.5).to_le_bytes())
            .collect();
        let residual = float_sub_encode(&raw, &reference).unwrap();
        let recovered = float_sub_decode(&residual, &reference).unwrap();
        assert_eq!(raw, recovered);
    }

    #[test]
    fn float_sub_identical_inputs_produce_zeros() {
        let data: Vec<u8> = (0..64)
            .flat_map(|i| f32::from(i as u16).to_le_bytes())
            .collect();
        let residual = float_sub_encode(&data, &data).unwrap();
        // Every f32 should be exactly 0.0
        for chunk in residual.chunks_exact(4) {
            let val = f32::from_le_bytes(chunk.try_into().unwrap());
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn float_sub_length_not_multiple_of_4() {
        let err = float_sub_encode(&[1, 2, 3], &[4, 5, 6]);
        assert!(matches!(err, Err(PtwmCoreError::CodecEncode { .. })));
    }

    #[test]
    fn float_sub_decode_length_not_multiple_of_4() {
        let err = float_sub_decode(&[1, 2, 3], &[4, 5, 6]);
        assert!(matches!(err, Err(PtwmCoreError::CodecDecode { .. })));
    }

    #[test]
    fn float_sub_length_mismatch() {
        let err = float_sub_encode(&[0u8; 8], &[0u8; 4]);
        assert!(matches!(err, Err(PtwmCoreError::CodecEncode { .. })));
    }

    #[test]
    fn float_sub_decode_length_mismatch() {
        let err = float_sub_decode(&[0u8; 8], &[0u8; 4]);
        assert!(matches!(err, Err(PtwmCoreError::CodecDecode { .. })));
    }

    #[test]
    fn xor_decode_length_mismatch_produces_decode_error() {
        let err = xor_decode(&[1, 2, 3], &[1, 2]);
        assert!(matches!(err, Err(PtwmCoreError::CodecDecode { .. })));
    }

    #[test]
    fn float_sub_empty_inputs() {
        assert_eq!(float_sub_encode(&[], &[]).unwrap(), Vec::<u8>::new());
        assert_eq!(float_sub_decode(&[], &[]).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn float_sub_roundtrip_preserves_negatives_and_special_values() {
        // Tests roundtrip with negative values, subnormals, and small finite values.
        // NaN is intentionally excluded — NaN bit patterns are not preserved through
        // IEEE 754 arithmetic (see float_sub_encode / float_sub_decode doc caveats).
        let raw_vals: &[f32] = &[
            -1.5,
            0.0,
            1e-40_f32, // subnormal
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
        ];
        // Use similar-magnitude references so cancellation is not extreme.
        let ref_vals: &[f32] = &[
            -1.0,
            0.0,
            0.5e-40_f32, // subnormal reference
            f32::MIN_POSITIVE * 2.0,
            -f32::MIN_POSITIVE * 0.5,
        ];
        let raw: Vec<u8> = raw_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let reference: Vec<u8> = ref_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let residual = float_sub_encode(&raw, &reference).unwrap();
        let recovered = float_sub_decode(&residual, &reference).unwrap();
        assert_eq!(raw, recovered);
    }
}
