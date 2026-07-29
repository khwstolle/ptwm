//! PyO3 bindings for `ptwm-core`. Installed as the `weights._core` module.
//!
//! This crate owns every PyO3-touching line in the project. `ptwm-core`
//! holds the algorithms; this file converts Python values at the boundary
//! and releases the GIL around CPU-heavy work.

use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use xxhash_rust::xxh64::xxh64 as xxh64_fn;

use ptwm_core::chain::Chain;
use ptwm_core::chain::wire::read_chain_blob;
use ptwm_core::compressor::{
    CompressorOptions, InputTensor, compress_model as core_compress_model,
};
use ptwm_core::container::ContainerReader;
use ptwm_core::{PtwmCoreError, codec_tagged, delta, entropy};

mod delta_scheme;
mod ext;
mod hardware;
mod host_flavor;
mod inspect;
mod policy;
mod trust;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_xxhash64, m)?)?;
    m.add_function(wrap_pyfunction!(py_shannon, m)?)?;
    m.add_function(wrap_pyfunction!(py_identity_encode, m)?)?;
    m.add_function(wrap_pyfunction!(py_identity_decode, m)?)?;
    m.add_function(wrap_pyfunction!(py_huffman_encode, m)?)?;
    m.add_function(wrap_pyfunction!(py_huffman_decode, m)?)?;
    m.add_function(wrap_pyfunction!(py_rans_encode, m)?)?;
    m.add_function(wrap_pyfunction!(py_rans_decode, m)?)?;
    m.add_function(wrap_pyfunction!(py_zstd_encode, m)?)?;
    m.add_function(wrap_pyfunction!(py_zstd_decode, m)?)?;
    m.add_function(wrap_pyfunction!(py_delta_xor_encode, m)?)?;
    m.add_function(wrap_pyfunction!(py_delta_xor_decode, m)?)?;
    m.add_function(wrap_pyfunction!(py_delta_float_sub_encode, m)?)?;
    m.add_function(wrap_pyfunction!(py_delta_float_sub_decode, m)?)?;
    m.add_function(wrap_pyfunction!(py_blake3_hash, m)?)?;
    m.add_function(wrap_pyfunction!(py_dtype_element_size, m)?)?;
    m.add_function(wrap_pyfunction!(compress_model, m)?)?;
    m.add_function(wrap_pyfunction!(decode_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(decode_model, m)?)?;
    m.add_function(wrap_pyfunction!(decode_tensor_shape, m)?)?;
    m.add_function(wrap_pyfunction!(list_tensor_names, m)?)?;
    m.add_function(wrap_pyfunction!(decode_tensor_info, m)?)?;
    m.add_function(wrap_pyfunction!(decode_tensor_delta_hash, m)?)?;
    m.add_function(wrap_pyfunction!(list_plane_summaries, m)?)?;
    m.add_function(wrap_pyfunction!(explode_ptwm, m)?)?;
    m.add_function(wrap_pyfunction!(implode_ptwm, m)?)?;
    m.add_function(wrap_pyfunction!(codec_id_registry, m)?)?;
    m.add_function(wrap_pyfunction!(py_builtin_canonical_id, m)?)?;
    m.add_function(wrap_pyfunction!(py_write_chain_blob, m)?)?;
    m.add_function(wrap_pyfunction!(py_chain_blob_info, m)?)?;
    trust::register(m)?;
    policy::register(m)?;
    host_flavor::register(m)?;
    ext::register(m)?;
    inspect::register(m)?;
    delta_scheme::register(m)?;
    hardware::register(m)?;
    Ok(())
}

/// Authoritative `(name, wire_id)` pairs from the Rust `CodecId` registry.
/// The Python `CodecId` IntEnum mirrors these; a regression test pulls
/// them at runtime to keep both sides in lockstep.
#[pyfunction]
fn codec_id_registry() -> Vec<(String, u16)> {
    ptwm_core::codec::REGISTRY
        .iter()
        .map(|(name, id)| ((*name).to_string(), *id))
        .collect()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Get a contiguous `&[u8]` from a `PyBuffer`, or error.
fn contiguous_slice<'a>(buf: &'a PyBuffer<u8>, name: &str) -> PyResult<&'a [u8]> {
    if !buf.is_c_contiguous() {
        return Err(pyo3::exceptions::PyBufferError::new_err(format!(
            "{name}: buffer must be C-contiguous"
        )));
    }
    let len = buf.len_bytes();
    if len == 0 {
        return Ok(&[]);
    }
    // SAFETY: len > 0, we verified contiguity, and the PyBuffer holds the GIL reference.
    Ok(unsafe { std::slice::from_raw_parts(buf.buf_ptr() as *const u8, len) })
}

// ---------------------------------------------------------------------------
// Header functions
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(name = "xxhash64")]
fn py_xxhash64(data: PyBuffer<u8>) -> PyResult<u64> {
    let bytes = contiguous_slice(&data, "data")?;
    Ok(xxh64_fn(bytes, 0))
}

// ---------------------------------------------------------------------------
// Preprocessing + entropy: codec-agnostic public API
// ---------------------------------------------------------------------------

/// Base-2 Shannon entropy (bits/byte) of `data`.
#[pyfunction]
#[pyo3(name = "shannon")]
fn py_shannon(data: PyBuffer<u8>) -> PyResult<f64> {
    let bytes = contiguous_slice(&data, "data")?;
    Ok(entropy::shannon(bytes))
}

// ---------------------------------------------------------------------------
// Per-plane codec encode / decode
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(name = "identity_encode")]
fn py_identity_encode<'py>(
    py: Python<'py>,
    data: PyBuffer<u8>,
) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
    let bytes = contiguous_slice(&data, "data")?;
    let out = py.allow_threads(|| codec_tagged::identity_encode(bytes));
    Ok(pyo3::types::PyBytes::new(py, &out))
}

#[pyfunction]
#[pyo3(name = "identity_decode")]
fn py_identity_decode<'py>(
    py: Python<'py>,
    blob: PyBuffer<u8>,
    expected_len: usize,
) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
    let bytes = contiguous_slice(&blob, "blob")?;
    let out = py.allow_threads(|| codec_tagged::identity_decode(bytes, expected_len))?;
    Ok(pyo3::types::PyBytes::new(py, &out))
}

#[pyfunction]
#[pyo3(name = "huffman_encode")]
fn py_huffman_encode<'py>(
    py: Python<'py>,
    data: PyBuffer<u8>,
) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
    let bytes = contiguous_slice(&data, "data")?;
    let out = py.allow_threads(|| codec_tagged::huffman_encode(bytes))?;
    Ok(pyo3::types::PyBytes::new(py, &out))
}

#[pyfunction]
#[pyo3(name = "huffman_decode")]
fn py_huffman_decode<'py>(
    py: Python<'py>,
    blob: PyBuffer<u8>,
    expected_len: usize,
) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
    let bytes = contiguous_slice(&blob, "blob")?;
    let out = py.allow_threads(|| codec_tagged::huffman_decode(bytes, expected_len))?;
    Ok(pyo3::types::PyBytes::new(py, &out))
}

#[pyfunction]
#[pyo3(name = "rans_encode")]
fn py_rans_encode<'py>(
    py: Python<'py>,
    data: PyBuffer<u8>,
) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
    let bytes = contiguous_slice(&data, "data")?;
    let out = py.allow_threads(|| codec_tagged::rans_encode(bytes))?;
    Ok(pyo3::types::PyBytes::new(py, &out))
}

#[pyfunction]
#[pyo3(name = "rans_decode")]
fn py_rans_decode<'py>(
    py: Python<'py>,
    blob: PyBuffer<u8>,
    expected_len: usize,
) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
    let bytes = contiguous_slice(&blob, "blob")?;
    let out = py.allow_threads(|| codec_tagged::rans_decode(bytes, expected_len))?;
    Ok(pyo3::types::PyBytes::new(py, &out))
}

#[pyfunction]
#[pyo3(name = "zstd_encode")]
#[pyo3(signature = (data, level=3))]
fn py_zstd_encode<'py>(
    py: Python<'py>,
    data: PyBuffer<u8>,
    level: i32,
) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
    let bytes = contiguous_slice(&data, "data")?;
    let out = py.allow_threads(|| codec_tagged::zstd_encode(bytes, level))?;
    Ok(pyo3::types::PyBytes::new(py, &out))
}

#[pyfunction]
#[pyo3(name = "zstd_decode")]
fn py_zstd_decode<'py>(
    py: Python<'py>,
    blob: PyBuffer<u8>,
) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
    let bytes = contiguous_slice(&blob, "blob")?;
    let out = py.allow_threads(|| codec_tagged::zstd_decode(bytes))?;
    Ok(pyo3::types::PyBytes::new(py, &out))
}

// ---------------------------------------------------------------------------
// Delta (reference-frame) compression
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(name = "delta_xor_encode")]
fn py_delta_xor_encode<'py>(
    py: Python<'py>,
    raw: PyBuffer<u8>,
    reference: PyBuffer<u8>,
) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
    let raw_bytes = contiguous_slice(&raw, "raw")?;
    let ref_bytes = contiguous_slice(&reference, "reference")?;
    let out = py.allow_threads(|| delta::xor_encode(raw_bytes, ref_bytes))?;
    Ok(pyo3::types::PyBytes::new(py, &out))
}

#[pyfunction]
#[pyo3(name = "delta_xor_decode")]
fn py_delta_xor_decode<'py>(
    py: Python<'py>,
    residual: PyBuffer<u8>,
    reference: PyBuffer<u8>,
) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
    let residual_bytes = contiguous_slice(&residual, "residual")?;
    let ref_bytes = contiguous_slice(&reference, "reference")?;
    let out = py.allow_threads(|| delta::xor_decode(residual_bytes, ref_bytes))?;
    Ok(pyo3::types::PyBytes::new(py, &out))
}

#[pyfunction]
#[pyo3(name = "delta_float_sub_encode")]
fn py_delta_float_sub_encode<'py>(
    py: Python<'py>,
    raw: PyBuffer<u8>,
    reference: PyBuffer<u8>,
) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
    let raw_bytes = contiguous_slice(&raw, "raw")?;
    let ref_bytes = contiguous_slice(&reference, "reference")?;
    let out = py.allow_threads(|| delta::float_sub_encode(raw_bytes, ref_bytes))?;
    Ok(pyo3::types::PyBytes::new(py, &out))
}

#[pyfunction]
#[pyo3(name = "delta_float_sub_decode")]
fn py_delta_float_sub_decode<'py>(
    py: Python<'py>,
    residual: PyBuffer<u8>,
    reference: PyBuffer<u8>,
) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
    let residual_bytes = contiguous_slice(&residual, "residual")?;
    let ref_bytes = contiguous_slice(&reference, "reference")?;
    let out = py.allow_threads(|| delta::float_sub_decode(residual_bytes, ref_bytes))?;
    Ok(pyo3::types::PyBytes::new(py, &out))
}

#[pyfunction]
#[pyo3(name = "blake3_hash")]
fn py_blake3_hash<'py>(
    py: Python<'py>,
    data: PyBuffer<u8>,
) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
    let bytes = contiguous_slice(&data, "data")?;
    let digest = py.allow_threads(|| delta::blake3_hash(bytes));
    Ok(pyo3::types::PyBytes::new(py, &digest))
}

/// Expose [`ptwm_core::Dtype::element_size`] to Python. Bytes per scalar.
#[pyfunction]
#[pyo3(name = "dtype_element_size")]
fn py_dtype_element_size(dtype_code: u8) -> PyResult<u32> {
    let dt = ptwm_core::Dtype::from_code(dtype_code)?;
    Ok(dt.element_size())
}

// ---------------------------------------------------------------------------
// v3 container: compress_model + decode_tensor
// ---------------------------------------------------------------------------

/// Compress a list of tensors into a `.ptwm` v3 container blob.
///
/// Each element of `records` is a 7-tuple:
/// ``(name, dtype_code, input_format, raw_bytes, shape,
///    dtype_name, delta_reference_blake3)``
///
/// `chains_per_tensor` — one list of chain wire-byte blobs per tensor.
/// Each blob is the output of `chain::wire::write_chain`. Pass a single
/// blob per tensor when no multi-chain trial-encode is needed.
///
/// `method_hint` selects the dominant codec for the container header
/// (1=HUFFMAN, 2=ZSTD, 3=MICROSCALE, 4=RANS, 5=IDENTITY).
///
/// Returns the entire container as a ``bytes`` object.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (records, chains_per_tensor, emit_payload_hash, emit_plane_crc, method_hint=3, streaming_chunk=None, codec_menu=None))]
fn compress_model<'py>(
    py: Python<'py>,
    records: Vec<(
        String,           // name
        u16,              // dtype_code
        u8,               // input_format
        PyBuffer<u8>,     // raw_bytes (buffer-protocol; zero-copy)
        Option<Vec<u64>>, // shape
        Option<String>,   // dtype_name
        Option<Vec<u8>>,  // delta_reference_blake3 (32 bytes)
    )>,
    chains_per_tensor: Vec<Vec<Vec<u8>>>, // [tensor][candidate] = wire bytes
    emit_payload_hash: bool,
    emit_plane_crc: bool,
    method_hint: u16,
    streaming_chunk: Option<u32>,
    // Optional wire-stable CodecId list. When provided, the dispatcher
    // trial-encode menu is filtered to codecs in this set on every plane;
    // empty intersection on a plane raises `InvalidContainer`.
    codec_menu: Option<Vec<u16>>,
) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
    if chains_per_tensor.len() != records.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "chains_per_tensor length {} != records length {}",
            chains_per_tensor.len(),
            records.len()
        )));
    }

    // Validate delta hash widths and parse chains before releasing GIL.
    for (name, _, _, _, _, _, delta_hash) in &records {
        if let Some(h) = delta_hash
            && h.len() != 32
        {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "tensor '{name}': delta_reference_blake3 must be 32 bytes, got {}",
                h.len()
            )));
        }
    }

    // Materialise borrowed slices over each PyBuffer while the GIL is held.
    // The buffers stay pinned for the lifetime of `records`, which outlives
    // the `allow_threads` closure below.
    let raw_slices: Vec<&[u8]> = records
        .iter()
        .enumerate()
        .map(|(i, r)| contiguous_slice(&r.3, &format!("records[{i}].raw_bytes")))
        .collect::<PyResult<_>>()?;

    // Deserialize all chains before releasing GIL.
    let mut chains_per_tensor_parsed: Vec<Vec<Chain>> = Vec::with_capacity(chains_per_tensor.len());
    for (tensor_idx, candidate_blobs) in chains_per_tensor.iter().enumerate() {
        if candidate_blobs.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "chains_per_tensor[{tensor_idx}]: at least one candidate chain required"
            )));
        }
        let mut candidates = Vec::with_capacity(candidate_blobs.len());
        for (chain_idx, blob) in candidate_blobs.iter().enumerate() {
            let (chain, _) = read_chain_blob(blob).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "chains_per_tensor[{tensor_idx}][{chain_idx}]: {e}"
                ))
            })?;
            candidates.push(chain);
        }
        chains_per_tensor_parsed.push(candidates);
    }

    let forced_codec = match method_hint {
        2 => Some(ptwm_core::codec::CodecId::Zstd),
        4 => Some(ptwm_core::codec::CodecId::Rans),
        5 => Some(ptwm_core::codec::CodecId::Identity),
        _ => None,
    };

    let allow_codec_ids: Option<Vec<ptwm_core::codec::CodecId>> = codec_menu.map(|ids| {
        ids.into_iter()
            .filter_map(ptwm_core::codec::CodecId::from_u16)
            .collect()
    });

    let mut out: Vec<u8> = Vec::new();
    py.allow_threads(|| -> Result<(), PtwmCoreError> {
        let inputs: Vec<InputTensor<'_>> = records
            .iter()
            .zip(chains_per_tensor_parsed.iter())
            .zip(raw_slices.iter())
            .map(
                |(
                    ((name, dtype_code, input_format, _, shape, dtype_name, delta_hash), chains),
                    raw_bytes,
                )| {
                    let delta_reference_blake3 = delta_hash.as_ref().map(|h| {
                        let mut arr = [0u8; 32];
                        arr.copy_from_slice(&h[..32]);
                        arr
                    });
                    InputTensor {
                        name: name.clone(),
                        candidate_chains: chains.clone(),
                        dtype_code: *dtype_code,
                        input_format: *input_format,
                        raw_bytes,
                        shape: shape.clone(),
                        dtype_name: dtype_name.clone(),
                        delta_reference_blake3,
                    }
                },
            )
            .collect();

        let opts = CompressorOptions {
            method_hint,
            emit_payload_hash,
            emit_plane_crc,
            forced_codec,
            allow_codec_ids,
            chunk_size: streaming_chunk,
        };

        let cursor = std::io::Cursor::new(&mut out);
        core_compress_model(cursor, &inputs, opts)?;
        Ok(())
    })
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok(pyo3::types::PyBytes::new(py, &out))
}

/// Decode a single tensor from a container blob.
///
/// `blob` is the full container ``bytes`` returned by :func:`compress_model`.
/// `name` is the tensor name string. Returns the raw decompressed ``bytes``
/// via the chain-based decode path.
///
/// `skip_missing` — when ``True`` open the container even if non-builtin
/// extensions referenced in its Extension Table are not installed (mirrors
/// ``ContainerReader::open_partial``). Tensors that actually need those
/// missing extensions will still fail at decode time.
///
/// Raises ``RuntimeError`` when the container is malformed or the tensor
/// name is missing.  Raises ``ValueError`` when non-builtin extensions are
/// missing and ``skip_missing`` is ``False`` (the default).
#[pyfunction]
#[pyo3(signature = (blob, name, skip_missing = false))]
fn decode_tensor<'py>(
    py: Python<'py>,
    blob: &Bound<'_, pyo3::types::PyBytes>,
    name: &str,
    skip_missing: bool,
) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
    let data: Vec<u8> = blob.as_bytes().to_vec();
    let name_owned = name.to_string();
    let out = py
        .allow_threads(|| -> Result<Vec<u8>, PtwmCoreError> {
            let reader = if skip_missing {
                ContainerReader::open_partial(&data)?
            } else {
                ContainerReader::open(&data)?
            };
            reader.decode_tensor(&name_owned)
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(pyo3::types::PyBytes::new(py, &out))
}

/// Decode every tensor in `blob` at once, in parallel across the rayon pool
/// (GIL released during decode). Returns `(name, bytes)` per tensor in
/// container index order. Prefer this over per-tensor `decode_tensor` for a
/// full-model load — the per-tensor decodes are independent and scale across
/// cores. `skip_missing` mirrors `decode_tensor`.
#[pyfunction]
#[pyo3(signature = (blob, skip_missing = false))]
fn decode_model<'py>(
    py: Python<'py>,
    blob: &Bound<'_, pyo3::types::PyBytes>,
    skip_missing: bool,
) -> PyResult<Vec<(String, Bound<'py, pyo3::types::PyBytes>)>> {
    // Borrow the blob's bytes directly — no full-model copy. `blob` outlives
    // this call, and the bytes are plain immutable memory, so reading them with
    // the GIL released is sound.
    let data = blob.as_bytes();
    let decoded = py
        .allow_threads(|| -> Result<Vec<(String, Vec<u8>)>, PtwmCoreError> {
            let reader = if skip_missing {
                ContainerReader::open_partial(data)?
            } else {
                ContainerReader::open(data)?
            };
            reader.decode_model()
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    decoded
        .into_iter()
        .map(|(name, bytes)| Ok((name, pyo3::types::PyBytes::new(py, &bytes))))
        .collect()
}

/// Return `(shape, dtype_name, dtype_code, input_format)` for a v3 tensor.
///
/// Returns `None` when the tensor has no CBOR shape metadata embedded.
/// `skip_missing` has the same semantics as for :func:`decode_tensor`.
#[pyfunction]
#[pyo3(signature = (blob, name, skip_missing = false))]
fn decode_tensor_info(
    py: Python<'_>,
    blob: &Bound<'_, pyo3::types::PyBytes>,
    name: &str,
    skip_missing: bool,
) -> PyResult<Option<(Vec<u64>, String, u16, u8)>> {
    use ptwm_core::metadata::decode_shape_metadata;
    use ptwm_core::tensor_record::parse_tensor_record;

    let data: Vec<u8> = blob.as_bytes().to_vec();
    let name_owned = name.to_string();
    let info = py
        .allow_threads(
            || -> Result<Option<(Vec<u64>, String, u16, u8)>, PtwmCoreError> {
                let reader = if skip_missing {
                    ContainerReader::open_partial(&data)?
                } else {
                    ContainerReader::open(&data)?
                };
                let record_bytes =
                    reader.get_tensor_record_bytes(&name_owned).ok_or_else(|| {
                        PtwmCoreError::InvalidContainer(format!("tensor '{name_owned}' not found"))
                    })?;
                let (rec, _) = parse_tensor_record(record_bytes)?;
                match rec.tensor_metadata {
                    Some(meta) => {
                        let (shape, dtype_name) = decode_shape_metadata(&meta)?;
                        Ok(Some((shape, dtype_name, rec.dtype_code, rec.input_format)))
                    }
                    None => Ok(None),
                }
            },
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(info)
}

/// Return the BLAKE3 hash of the delta reference recorded for a v3 tensor,
/// or `None` if the tensor carries no delta dependency.
/// `skip_missing` has the same semantics as for :func:`decode_tensor`.
#[pyfunction]
#[pyo3(signature = (blob, name, skip_missing = false))]
fn decode_tensor_delta_hash<'py>(
    py: Python<'py>,
    blob: &Bound<'_, pyo3::types::PyBytes>,
    name: &str,
    skip_missing: bool,
) -> PyResult<Option<Bound<'py, pyo3::types::PyBytes>>> {
    use ptwm_core::tensor_record::{RefKind, parse_tensor_record};

    let data: Vec<u8> = blob.as_bytes().to_vec();
    let name_owned = name.to_string();
    let hash_opt = py
        .allow_threads(|| -> Result<Option<[u8; 32]>, PtwmCoreError> {
            let reader = if skip_missing {
                ContainerReader::open_partial(&data)?
            } else {
                ContainerReader::open(&data)?
            };
            let record_bytes = reader.get_tensor_record_bytes(&name_owned).ok_or_else(|| {
                PtwmCoreError::InvalidContainer(format!("tensor '{name_owned}' not found"))
            })?;
            let (rec, _) = parse_tensor_record(record_bytes)?;
            for dep in &rec.dependencies {
                if matches!(dep.ref_kind, RefKind::ExternalSafetensors)
                    && let Some(h) = dep.expected_hash
                {
                    return Ok(Some(h));
                }
            }
            Ok(None)
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(hash_opt.map(|h| pyo3::types::PyBytes::new(py, &h)))
}

/// List every tensor name stored in a v3 container blob, in index order.
/// `skip_missing` has the same semantics as for :func:`decode_tensor`.
#[pyfunction]
#[pyo3(signature = (blob, skip_missing = false))]
fn list_tensor_names(
    py: Python<'_>,
    blob: &Bound<'_, pyo3::types::PyBytes>,
    skip_missing: bool,
) -> PyResult<Vec<String>> {
    let data: Vec<u8> = blob.as_bytes().to_vec();
    let names = py
        .allow_threads(|| -> Result<Vec<String>, PtwmCoreError> {
            let reader = if skip_missing {
                ContainerReader::open_partial(&data)?
            } else {
                ContainerReader::open(&data)?
            };
            reader.tensor_names()
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(names)
}

/// Decode the shape-metadata map for a v3 tensor, if present.
///
/// Returns ``(shape, dtype_name)`` or ``None`` when no CBOR shape metadata.
#[pyfunction]
fn decode_tensor_shape(
    py: Python<'_>,
    blob: &Bound<'_, pyo3::types::PyBytes>,
    name: &str,
) -> PyResult<Option<(Vec<u64>, String)>> {
    let data: Vec<u8> = blob.as_bytes().to_vec();
    let name_owned = name.to_string();
    let got = py
        .allow_threads(|| -> Result<Option<(Vec<u64>, String)>, PtwmCoreError> {
            let reader = ContainerReader::open(&data)?;
            reader.tensor_shape(&name_owned)
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(got)
}

/// Per-plane summary for every tensor in a v3 container blob.
///
/// Returns a list of (tensor_name, plane_index, role, codec_id,
/// state_source, payload_len) tuples — one row per plane.
#[pyfunction]
fn list_plane_summaries(
    py: Python<'_>,
    blob: &Bound<'_, pyo3::types::PyBytes>,
) -> PyResult<Vec<(String, u32, u8, u16, u8, u64)>> {
    use ptwm_core::tensor_record::parse_tensor_record;

    let data: Vec<u8> = blob.as_bytes().to_vec();
    let summaries = py
        .allow_threads(
            || -> Result<Vec<(String, u32, u8, u16, u8, u64)>, PtwmCoreError> {
                let reader = ContainerReader::open(&data)?;
                let names = reader.tensor_names()?;
                let mut out = Vec::new();
                for name in names {
                    let record_bytes = reader.get_tensor_record_bytes(&name).ok_or_else(|| {
                        PtwmCoreError::InvalidContainer(format!("tensor '{name}' not found"))
                    })?;
                    let (rec, _) = parse_tensor_record(record_bytes)?;
                    for (idx, plane) in rec.terminals.iter().enumerate() {
                        out.push((
                            name.clone(),
                            idx as u32,
                            plane.role as u8,
                            plane.codec_id.as_u16(),
                            plane.state_source as u8,
                            plane.payload_len,
                        ));
                    }
                }
                Ok(out)
            },
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(summaries)
}

/// Explode a `.ptwm` container into its canonical PTWM-JSON projection.
///
/// Returns ``(registry_json, tensors, members)`` where:
/// * ``registry_json`` is the canonical (JCS) archive-scoped registry manifest;
/// * ``tensors`` is a list of ``(name, key, tensor_manifest_json)`` in file
///   order (``key`` is the zero-padded global tensor index / WebDataset
///   ``__key__``);
/// * ``members`` maps member name → bytes (compressed plane payloads,
///   externalized codec state, shared state).
///
/// `implode_ptwm(registry_json, [t[2] for t in tensors], members)` reproduces
/// the input container byte-for-byte.
#[pyfunction]
fn explode_ptwm<'py>(
    py: Python<'py>,
    blob: &Bound<'_, pyo3::types::PyBytes>,
) -> PyResult<(
    String,
    Vec<(String, String, String)>,
    Bound<'py, pyo3::types::PyDict>,
)> {
    // Borrow the buffer directly: the `&[u8]` is `Ungil`, so the heavy parse
    // can run with the GIL released without copying the (possibly multi-GiB)
    // container.
    let data = blob.as_bytes();
    let exploded = py
        .allow_threads(|| ptwm_core::transcode::explode(data))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let members = pyo3::types::PyDict::new(py);
    for (key, bytes) in &exploded.members {
        members.set_item(key, pyo3::types::PyBytes::new(py, bytes))?;
    }
    let tensors: Vec<(String, String, String)> = exploded
        .tensors
        .into_iter()
        .map(|t| (t.name, t.key, t.manifest_json))
        .collect();
    Ok((exploded.registry_json, tensors, members))
}

/// Reconstruct a `.ptwm` container from a registry manifest, per-tensor
/// manifests (in file order), and a member-name → bytes mapping.
#[pyfunction]
fn implode_ptwm<'py>(
    py: Python<'py>,
    registry_json: String,
    tensor_jsons: Vec<String>,
    members: std::collections::HashMap<String, Bound<'py, pyo3::types::PyBytes>>,
) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
    // Borrow each PyBytes buffer as `&[u8]` (no copy here); the transcoder
    // materializes the owned bytes it needs inside `allow_threads`, so the
    // big payload copies happen with the GIL released.
    let members_borrowed: std::collections::HashMap<String, &[u8]> = members
        .iter()
        .map(|(k, v)| (k.clone(), v.as_bytes()))
        .collect();
    let blob = py
        .allow_threads(move || {
            ptwm_core::transcode::implode(&registry_json, &tensor_jsons, members_borrowed)
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(pyo3::types::PyBytes::new(py, &blob))
}

/// Return the 32-byte canonical id for a built-in op or codec by name.
///
/// The Python chain builder uses this to construct extension table entries
/// that are compatible with the Rust-side canonical ids.
///
/// Example names: ``"source"``, ``"byte_split"``, ``"huffman"``, …
#[pyfunction]
#[pyo3(name = "builtin_canonical_id")]
fn py_builtin_canonical_id<'py>(py: Python<'py>, name: &str) -> Bound<'py, pyo3::types::PyBytes> {
    let id = ptwm_core::extension::builtin_canonical_id(name);
    pyo3::types::PyBytes::new(py, id.as_bytes())
}

/// Serialise a [`Chain`] (supplied as its legacy Python wire bytes) into the
/// new self-contained blob format (local extension table prefix + table-indexed
/// chain bytes). Use in place of the old ``Chain.to_bytes()`` when you need a
/// blob that is compatible with the current ``compress_model`` wire format.
///
/// ``legacy_blob`` must have been produced by the older Python
/// ``Chain.to_bytes()`` method (i.e. using direct ``op_id`` u16 values).
///
/// This function exists as a migration helper; once the Python chain builder
/// is fully updated to produce the current format's blobs natively, it can be
/// removed.
#[pyfunction]
#[pyo3(name = "chain_blob_from_legacy")]
fn py_write_chain_blob<'py>(
    py: Python<'py>,
    legacy_blob: &Bound<'_, pyo3::types::PyBytes>,
) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
    use ptwm_core::chain::wire::write_chain_blob;

    // Parse the legacy blob: the old format stores op_id u16 directly.
    let buf = legacy_blob.as_bytes();
    let chain = read_legacy_chain_blob(buf)?;
    let new_blob = write_chain_blob(&chain)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(pyo3::types::PyBytes::new(py, &new_blob))
}

/// Parse a chain blob and return `(et_size_bytes, [(table_idx, op_id)])`.
///
/// `et_size_bytes` is the number of bytes the extension table occupies at the
/// start of `blob`. The Python `_parse_chain` helper can then skip that many
/// bytes to reach the chain wire section, and use the table to map indices to
/// op_ids.
///
/// Returns an error if `blob` is not a valid new-format chain blob.
#[pyfunction]
#[pyo3(name = "chain_blob_info")]
fn py_chain_blob_info(blob: &[u8]) -> PyResult<(usize, Vec<(u16, u16)>)> {
    use ptwm_core::extension::dispatch::{BuiltinKind, dispatch_builtin};
    use ptwm_core::extension::table::ExtensionTable;

    // Parse the extension table.
    let inline_table = ExtensionTable::from_bytes(blob)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("chain blob et: {e}")))?;
    let et_bytes = inline_table
        .to_bytes()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("chain blob et ser: {e}")))?;
    let et_size = et_bytes.len();

    // Build (table_idx, op_id) pairs for every entry that resolves to an op.
    let mut table: Vec<(u16, u16)> = Vec::with_capacity(inline_table.entries.len());
    for (idx, entry) in inline_table.entries.iter().enumerate() {
        if let Some(BuiltinKind::Op(op)) = dispatch_builtin(&entry.canonical_id) {
            table.push((idx as u16, op.as_u16()));
        }
    }

    Ok((et_size, table))
}

/// Parse a chain from the **legacy wire format**, where node op-ids were
/// stored as raw `OpId` discriminant u16 values rather than extension table
/// indices. Used only by the [`py_write_chain_blob`] migration helper.
fn read_legacy_chain_blob(buf: &[u8]) -> PyResult<ptwm_core::chain::Chain> {
    use ptwm_core::chain::graph::{Chain, ChainEdge, ChainNode, TerminalRef};
    use ptwm_core::transforms::op::OpId;
    use ptwm_core::types::role::Role;

    // Minimum header: 3 × u8 + 3 × u16 = 9 bytes.
    if buf.len() < 9 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "legacy chain blob: header truncated (need ≥ 9 bytes)",
        ));
    }

    fn ivc(msg: impl std::fmt::Display) -> pyo3::PyErr {
        pyo3::exceptions::PyValueError::new_err(msg.to_string())
    }

    let num_nodes = buf[0] as usize;
    let num_edges = buf[1] as usize;
    let num_terminals = buf[2] as usize;
    let nodes_len = u16::from_le_bytes([buf[3], buf[4]]) as usize;
    let edges_len = u16::from_le_bytes([buf[5], buf[6]]) as usize;
    let terminals_len = u16::from_le_bytes([buf[7], buf[8]]) as usize;
    let total = 9 + nodes_len + edges_len + terminals_len;
    if buf.len() < total {
        return Err(ivc(format!(
            "legacy chain blob: buffer too short (need {total}, have {})",
            buf.len()
        )));
    }

    let nodes_bytes = &buf[9..9 + nodes_len];
    let edges_bytes = &buf[9 + nodes_len..9 + nodes_len + edges_len];
    let terminals_bytes =
        &buf[9 + nodes_len + edges_len..9 + nodes_len + edges_len + terminals_len];

    let mut nodes = Vec::with_capacity(num_nodes);
    let mut pos = 0;
    for _ in 0..num_nodes {
        if pos + 3 > nodes_bytes.len() {
            return Err(ivc("legacy chain blob: node header truncated"));
        }
        let op_id_raw = u16::from_le_bytes([nodes_bytes[pos], nodes_bytes[pos + 1]]);
        let op = OpId::from_u16(op_id_raw).map_err(|e| ivc(format!("legacy chain blob: {e}")))?;
        let params_len = nodes_bytes[pos + 2] as usize;
        pos += 3;
        if pos + params_len > nodes_bytes.len() {
            return Err(ivc("legacy chain blob: node params truncated"));
        }
        nodes.push(ChainNode {
            op,
            params: nodes_bytes[pos..pos + params_len].to_vec(),
        });
        pos += params_len;
    }

    let mut edges = Vec::with_capacity(num_edges);
    pos = 0;
    for _ in 0..num_edges {
        if pos + 5 > edges_bytes.len() {
            return Err(ivc("legacy chain blob: edge header truncated"));
        }
        let src_node = edges_bytes[pos];
        let src_output_idx = edges_bytes[pos + 1];
        let dst_node = edges_bytes[pos + 2];
        let dst_input_idx = edges_bytes[pos + 3];
        let role_present = edges_bytes[pos + 4];
        pos += 5;
        let role_override = match role_present {
            0 => None,
            1 => {
                let (role, n) = Role::read(&edges_bytes[pos..])
                    .map_err(|e| ivc(format!("legacy chain blob edge role: {e}")))?;
                pos += n;
                Some(role)
            }
            v => {
                return Err(ivc(format!(
                    "legacy chain blob: edge role_present byte unknown value {v}"
                )));
            }
        };
        if pos + 2 > edges_bytes.len() {
            return Err(ivc("legacy chain blob: edge vendor_bytes_len truncated"));
        }
        let vendor_len = u16::from_le_bytes([edges_bytes[pos], edges_bytes[pos + 1]]) as usize;
        pos += 2;
        if pos + vendor_len > edges_bytes.len() {
            return Err(ivc("legacy chain blob: edge vendor_bytes truncated"));
        }
        edges.push(ChainEdge {
            src_node,
            src_output_idx,
            dst_node,
            dst_input_idx,
            role_override,
            vendor_bytes: edges_bytes[pos..pos + vendor_len].to_vec(),
        });
        pos += vendor_len;
    }

    let mut terminals = Vec::with_capacity(num_terminals);
    pos = 0;
    for _ in 0..num_terminals {
        if pos + 2 > terminals_bytes.len() {
            return Err(ivc("legacy chain blob: terminal header truncated"));
        }
        let node_idx = terminals_bytes[pos];
        let output_idx = terminals_bytes[pos + 1];
        pos += 2;
        let (role, n) = Role::read(&terminals_bytes[pos..])
            .map_err(|e| ivc(format!("legacy chain blob terminal role: {e}")))?;
        pos += n;
        terminals.push(TerminalRef {
            node_idx,
            output_idx,
            role,
        });
    }

    Ok(Chain {
        nodes,
        edges,
        terminals,
    })
}
