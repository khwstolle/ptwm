//! Tensor metadata convention for the `.ptwm` container.
//!
//! `TensorRecord.tensor_metadata` is an opaque CBOR map.
//! This module defines and enforces the **shape convention** used by the
//! Python bindings:
//!
//! ```text
//! {
//!   "shape": [u64, u64, ...],   // tensor dimensions, row-major
//!   "dtype": "<str>"             // canonical dtype name (e.g. "bf16", "mxfp4")
//! }
//! ```
//!
//! A minimal hand-rolled CBOR encoder/decoder is used — we only need
//! unsigned integers, text strings, fixed-length arrays, and fixed-length
//! maps. No floats, tags, or indefinite-length items.
//!
//! Keeping it dependency-free avoids pulling a CBOR crate for ~50 lines of
//! logic and guarantees the byte layout is exactly what the spec says.

use crate::error::PtwmCoreError;

// CBOR major types (high 3 bits of the first byte).
const MAJOR_UINT: u8 = 0 << 5;
const MAJOR_TSTR: u8 = 3 << 5;
const MAJOR_ARRAY: u8 = 4 << 5;
const MAJOR_MAP: u8 = 5 << 5;

fn write_uint(out: &mut Vec<u8>, major: u8, n: u64) {
    if n <= 23 {
        out.push(major | (n as u8));
    } else if n <= u8::MAX as u64 {
        out.push(major | 24);
        out.push(n as u8);
    } else if n <= u16::MAX as u64 {
        out.push(major | 25);
        out.extend_from_slice(&(n as u16).to_be_bytes());
    } else if n <= u32::MAX as u64 {
        out.push(major | 26);
        out.extend_from_slice(&(n as u32).to_be_bytes());
    } else {
        out.push(major | 27);
        out.extend_from_slice(&n.to_be_bytes());
    }
}

fn write_tstr(out: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    write_uint(out, MAJOR_TSTR, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

/// Encode `{"shape": shape, "dtype": dtype_name}` as CBOR.
pub fn encode_shape_metadata(shape: &[u64], dtype_name: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(16 + shape.len() * 9 + dtype_name.len());
    // map with 2 entries
    write_uint(&mut out, MAJOR_MAP, 2);
    // "shape" -> array of u64
    write_tstr(&mut out, "shape");
    write_uint(&mut out, MAJOR_ARRAY, shape.len() as u64);
    for dim in shape {
        write_uint(&mut out, MAJOR_UINT, *dim);
    }
    // "dtype" -> text string
    write_tstr(&mut out, "dtype");
    write_tstr(&mut out, dtype_name);
    out
}

/// Minimal CBOR reader: enough to decode the shape-metadata map.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn read_head(&mut self, expected_major: u8) -> Result<u64, PtwmCoreError> {
        if self.pos >= self.buf.len() {
            return Err(PtwmCoreError::InvalidContainer(
                "cbor metadata truncated at head".into(),
            ));
        }
        let first = self.buf[self.pos];
        self.pos += 1;
        let got_major = first & 0xe0;
        if got_major != expected_major {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "cbor metadata: expected major {:#x}, got {:#x}",
                expected_major >> 5,
                got_major >> 5
            )));
        }
        let arg = first & 0x1f;
        match arg {
            0..=23 => Ok(arg as u64),
            24 => self.read_fixed::<1>().map(|b| b[0] as u64),
            25 => self.read_fixed::<2>().map(|b| u16::from_be_bytes(b) as u64),
            26 => self.read_fixed::<4>().map(|b| u32::from_be_bytes(b) as u64),
            27 => self.read_fixed::<8>().map(u64::from_be_bytes),
            _ => Err(PtwmCoreError::InvalidContainer(format!(
                "cbor metadata: unsupported argument {}",
                arg
            ))),
        }
    }

    fn read_fixed<const N: usize>(&mut self) -> Result<[u8; N], PtwmCoreError> {
        if self.pos + N > self.buf.len() {
            return Err(PtwmCoreError::InvalidContainer(
                "cbor metadata truncated".into(),
            ));
        }
        let mut out = [0u8; N];
        out.copy_from_slice(&self.buf[self.pos..self.pos + N]);
        self.pos += N;
        Ok(out)
    }

    fn read_tstr(&mut self) -> Result<String, PtwmCoreError> {
        let len = self.read_head(MAJOR_TSTR)? as usize;
        if self.pos + len > self.buf.len() {
            return Err(PtwmCoreError::InvalidContainer(
                "cbor metadata tstr truncated".into(),
            ));
        }
        let s = std::str::from_utf8(&self.buf[self.pos..self.pos + len])
            .map_err(|_| PtwmCoreError::InvalidContainer("cbor metadata: tstr not utf8".into()))?
            .to_string();
        self.pos += len;
        Ok(s)
    }
}

/// Decode a `{"shape": [...], "dtype": "..."}` CBOR map previously written
/// by [`encode_shape_metadata`]. Any other map layout (missing keys, extra
/// keys, wrong types) is rejected as `InvalidContainer`.
pub fn decode_shape_metadata(buf: &[u8]) -> Result<(Vec<u64>, String), PtwmCoreError> {
    let mut r = Reader::new(buf);
    let n_pairs = r.read_head(MAJOR_MAP)?;
    if n_pairs != 2 {
        return Err(PtwmCoreError::InvalidContainer(format!(
            "cbor metadata: expected map of 2 pairs, got {}",
            n_pairs
        )));
    }

    let mut shape: Option<Vec<u64>> = None;
    let mut dtype: Option<String> = None;
    for _ in 0..2 {
        let key = r.read_tstr()?;
        match key.as_str() {
            "shape" => {
                let n = r.read_head(MAJOR_ARRAY)? as usize;
                let mut dims = Vec::with_capacity(n);
                for _ in 0..n {
                    dims.push(r.read_head(MAJOR_UINT)?);
                }
                shape = Some(dims);
            }
            "dtype" => dtype = Some(r.read_tstr()?),
            other => {
                return Err(PtwmCoreError::InvalidContainer(format!(
                    "cbor metadata: unknown key {:?}",
                    other
                )));
            }
        }
    }
    match (shape, dtype) {
        (Some(s), Some(d)) => Ok((s, d)),
        _ => Err(PtwmCoreError::InvalidContainer(
            "cbor metadata: missing shape or dtype".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_2d() {
        let encoded = encode_shape_metadata(&[1024, 768], "bf16");
        let (shape, dtype) = decode_shape_metadata(&encoded).unwrap();
        assert_eq!(shape, vec![1024, 768]);
        assert_eq!(dtype, "bf16");
    }

    #[test]
    fn roundtrip_0d() {
        let encoded = encode_shape_metadata(&[], "fp32");
        let (shape, dtype) = decode_shape_metadata(&encoded).unwrap();
        assert!(shape.is_empty());
        assert_eq!(dtype, "fp32");
    }

    #[test]
    fn roundtrip_large_dim() {
        // Values that cross every uint-size boundary.
        let shape: Vec<u64> = vec![23, 24, 255, 256, 65_535, 65_536, 4_294_967_296];
        let encoded = encode_shape_metadata(&shape, "mxfp4");
        let (decoded, dtype) = decode_shape_metadata(&encoded).unwrap();
        assert_eq!(decoded, shape);
        assert_eq!(dtype, "mxfp4");
    }

    #[test]
    fn rejects_unknown_key() {
        // {"shape":[1], "bogus":"x"} — using encoder for the first pair.
        let mut buf = Vec::new();
        write_uint(&mut buf, MAJOR_MAP, 2);
        write_tstr(&mut buf, "shape");
        write_uint(&mut buf, MAJOR_ARRAY, 1);
        write_uint(&mut buf, MAJOR_UINT, 1);
        write_tstr(&mut buf, "bogus");
        write_tstr(&mut buf, "x");
        assert!(decode_shape_metadata(&buf).is_err());
    }

    #[test]
    fn rejects_truncated() {
        let encoded = encode_shape_metadata(&[4, 4], "fp16");
        assert!(decode_shape_metadata(&encoded[..encoded.len() - 1]).is_err());
    }

    #[test]
    fn rejects_wrong_top_level_type() {
        // An array instead of a map.
        let mut buf = Vec::new();
        write_uint(&mut buf, MAJOR_ARRAY, 0);
        assert!(decode_shape_metadata(&buf).is_err());
    }

    #[test]
    fn exact_bytes_small() {
        // Sanity-check exact wire layout for a fixed input — guards against
        // accidental format changes.
        let out = encode_shape_metadata(&[2, 3], "fp16");
        // Expected:
        //   0xa2                          map(2)
        //   0x65 "shape"                  tstr(5)
        //   0x82                          array(2)
        //     0x02 0x03
        //   0x65 "dtype"                  tstr(5)
        //   0x64 "fp16"                   tstr(4)
        let expected: Vec<u8> = vec![
            0xa2, 0x65, b's', b'h', b'a', b'p', b'e', 0x82, 0x02, 0x03, 0x65, b'd', b't', b'y',
            b'p', b'e', 0x64, b'f', b'p', b'1', b'6',
        ];
        assert_eq!(out, expected);
    }
}
