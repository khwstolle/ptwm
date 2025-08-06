//! Pure-Rust core for lossless neural-network weight compression.
//!
//! No Python dependency lives here; the sibling `ptwm-py` crate carries the
//! PyO3 bindings. The modules below implement the `.ptwm` container format
//! (header, prelude, index, tensor / plane records, compressor, container
//! reader) and the codec plus preprocessing utilities the container and the
//! Python CLI share.
//!
//! # API stability
//!
//! The stable, supported entry point to PTWM is the **Python package**
//! (`pip install ptwm`). This crate's API is intended for the
//! sibling [`ptwm-py`] bindings and may change in any minor release.
//! External Rust consumers should pin an exact version.
//!
//! [`ptwm-py`]: https://crates.io/crates/ptwm-py

pub mod chain;
pub mod codec;
pub mod codec_tagged;
pub mod codecs;
pub mod compressor;
pub mod container;
pub mod delta;
pub mod discovery;
pub mod dispatch;
pub mod dtype;
pub mod entropy;
pub mod error;
pub mod extension;
pub mod fit;
pub mod flavor;
pub mod header;
pub mod index;
pub mod install;
pub mod layout;
pub mod metadata;
pub mod plane_record;
pub mod policy;
pub mod prelude;
pub mod quantize;
pub mod range_coder;
pub mod select;
pub mod split;
pub mod tensor_record;
pub mod transcode;
pub mod transforms;
pub mod trust;
pub mod types;

pub use dtype::{Dtype, PreprocessingModes};
pub use error::PtwmCoreError;

/// Validate `num_buf` and return it as a `usize`.
pub fn validate_num_buf(num_buf: u32) -> Result<usize, PtwmCoreError> {
    match num_buf {
        1 | 2 | 4 | 8 => Ok(num_buf as usize),
        _ => Err(PtwmCoreError::InvalidNumBuf(num_buf)),
    }
}

/// Validate `bits_mode` / `bytes_mode` for the given `num_buf`.
pub fn validate_modes(
    num_buf: usize,
    bits_mode: i32,
    bytes_mode: i32,
) -> Result<(), PtwmCoreError> {
    if bits_mode != 0 && bits_mode != 1 {
        return Err(PtwmCoreError::InvalidBitMode(bits_mode));
    }
    match num_buf {
        1 => {
            if !matches!(bytes_mode, 10..=12) {
                return Err(PtwmCoreError::InvalidByteMode(bytes_mode));
            }
        }
        2 => {
            // 0/1/8/10 = dtype16 modes; 20/22 = FP8 nibble-split modes
            if !matches!(bytes_mode, 0 | 1 | 8 | 10 | 20 | 22) {
                return Err(PtwmCoreError::InvalidByteMode(bytes_mode));
            }
        }
        4 => {
            if !matches!(bytes_mode, 0 | 220) {
                return Err(PtwmCoreError::InvalidByteMode(bytes_mode));
            }
        }
        8 => {
            if bytes_mode != 10 {
                return Err(PtwmCoreError::InvalidByteMode(bytes_mode));
            }
        }
        _ => unreachable!(),
    }
    Ok(())
}

/// Split a single chunk's data into `num_buf` buffers based on dtype.
pub fn split_chunk(data: &[u8], num_buf: usize, bits_mode: i32, bytes_mode: i32) -> Vec<Vec<u8>> {
    // bytes_mode 20/22 are FP8 nibble-split modes that use num_buf=2 but route
    // through dtype8, not dtype16.
    if num_buf == 2 && matches!(bytes_mode, 20 | 22) {
        return split::dtype8::split(data, bytes_mode);
    }
    match num_buf {
        1 => split::dtype8::split(data, bytes_mode),
        2 => split::dtype16::split(data, bits_mode, bytes_mode),
        4 => split::dtype32::split(data, bits_mode, bytes_mode),
        8 => split::dtype64::split(data, bits_mode, bytes_mode),
        _ => unreachable!(),
    }
}

/// Inverse of [`split_chunk`]. Reassembles `planes` into the output buffer.
///
/// `out` must be sized to the original uncompressed chunk length (before
/// splitting). `planes.len()` must equal `num_buf`.
pub fn combine_chunk(
    planes: &[Vec<u8>],
    out: &mut [u8],
    num_buf: usize,
    bits_mode: i32,
    bytes_mode: i32,
) {
    debug_assert_eq!(planes.len(), num_buf, "combine_chunk: plane count mismatch");
    if num_buf == 2 && matches!(bytes_mode, 20 | 22) {
        split::dtype8::combine(planes, out, bytes_mode);
        return;
    }
    match num_buf {
        1 => split::dtype8::combine(planes, out, bytes_mode),
        2 => split::dtype16::combine_bufs(&planes[0], &planes[1], out, bits_mode, bytes_mode),
        4 => {
            let refs: Vec<&[u8]> = planes.iter().map(|b| b.as_slice()).collect();
            let lens: Vec<usize> = planes.iter().map(|b| b.len()).collect();
            split::dtype32::combine_bufs(&refs, &lens, out, bits_mode, bytes_mode);
        }
        8 => {
            let refs: Vec<&[u8]> = planes.iter().map(|b| b.as_slice()).collect();
            let lens: Vec<usize> = planes.iter().map(|b| b.len()).collect();
            split::dtype64::combine_bufs(&refs, &lens, out, bits_mode, bytes_mode);
        }
        _ => unreachable!(),
    }
}

/// Compute per-buffer sizes for a chunk of `chunk_len` bytes.
pub fn buffer_sizes_for_chunk(chunk_len: usize, num_buf: usize, bytes_mode: i32) -> Vec<usize> {
    if num_buf == 2 && matches!(bytes_mode, 20 | 22) {
        return split::dtype8::buffer_sizes(chunk_len, bytes_mode);
    }
    match num_buf {
        1 => split::dtype8::buffer_sizes(chunk_len, bytes_mode),
        2 => split::dtype16::buffer_sizes(chunk_len, bytes_mode),
        4 => split::dtype32::buffer_sizes(chunk_len, bytes_mode),
        8 => split::dtype64::buffer_sizes(chunk_len, bytes_mode),
        _ => unreachable!(),
    }
}
