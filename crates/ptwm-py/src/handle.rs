//! `ContainerHandle`: one opened container reader over borrowed bytes.
//!
//! Opening a container parses its header, Extension Table, prelude, chain
//! registry and index and checks the referenced extensions. A handle does
//! this once and answers any number of queries from the parsed state. The
//! free functions in `lib.rs` (`decode_tensor`, `list_tensor_names`, ...)
//! open a temporary handle per call.
//!
//! A buffer is borrowed without a copy only when its memory cannot change
//! while the handle exists; every other buffer is copied once. The rule is
//! in [`crate::buffer`]. The handle holds the buffer export, which keeps the
//! object alive and its memory pinned.

use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::buffer::ReadBuffer;

use ptwm_core::PtwmCoreError;
use ptwm_core::container::{ContainerLayout, ContainerReader};
use ptwm_core::header::{ContainerMagic, sniff};
use ptwm_core::metadata::decode_shape_metadata;
use ptwm_core::tensor_record::RefKind;

/// `(shape, dtype_name, dtype_code, input_format)` of a tensor.
pub type TensorInfo = (Vec<u64>, String, u16, u8);

/// `(tensor_name, plane_index, role, codec_id, state_source, payload_len)`.
pub type PlaneSummary = (String, u32, u8, u16, u8, u64);

/// An opened `.ptwm` container (either layout).
///
/// ``ContainerHandle(buffer, skip_missing=False)`` opens the container once;
/// the methods then query it without re-parsing. ``skip_missing`` opens the
/// container even when non-builtin extensions it references are not
/// installed; tensors that need them fail at decode time.
#[pyclass(frozen, module = "ptwm._core")]
pub struct ContainerHandle {
    // Field order matters: `reader` borrows from `_backing` and must drop
    // first. Struct fields drop in declaration order.
    reader: ContainerReader<'static>,
    _backing: ReadBuffer,
}

fn runtime_err(e: PtwmCoreError) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}

impl ContainerHandle {
    /// Open the buffer `obj` exports. The buffer is borrowed when its memory
    /// cannot change and it is C-contiguous; otherwise it is copied once
    /// (see [`ReadBuffer::new`]).
    pub fn open(py: Python<'_>, obj: &Bound<'_, PyAny>, skip_missing: bool) -> PyResult<Self> {
        let backing = ReadBuffer::new(obj)?;
        // SAFETY: `bytes` points into the export or the heap copy owned by
        // `backing`; neither moves when `backing` moves into the handle. The
        // handle drops the reader before `backing` (see the field order of
        // `ContainerHandle`).
        let bytes: &'static [u8] =
            unsafe { std::mem::transmute::<&[u8], &'static [u8]>(backing.as_slice()) };
        let reader = py
            .allow_threads(|| {
                if skip_missing {
                    ContainerReader::open_partial(bytes)
                } else {
                    ContainerReader::open(bytes)
                }
            })
            .map_err(runtime_err)?;
        Ok(Self {
            reader,
            _backing: backing,
        })
    }

    pub fn names_vec(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        py.allow_threads(|| self.reader.tensor_names())
            .map_err(runtime_err)
    }

    pub fn decode_bytes<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyBytes>> {
        let out = py
            .allow_threads(|| self.reader.decode_tensor(name))
            .map_err(runtime_err)?;
        Ok(PyBytes::new(py, &out))
    }

    pub fn decode_all<'py>(&self, py: Python<'py>) -> PyResult<Vec<(String, Bound<'py, PyBytes>)>> {
        let decoded = py
            .allow_threads(|| self.reader.decode_model())
            .map_err(runtime_err)?;
        Ok(decoded
            .into_iter()
            .map(|(name, bytes)| (name, PyBytes::new(py, &bytes)))
            .collect())
    }

    pub fn tensor_info(&self, py: Python<'_>, name: &str) -> PyResult<Option<TensorInfo>> {
        py.allow_threads(|| -> Result<Option<TensorInfo>, PtwmCoreError> {
            let rec = self.reader.tensor_record(name)?;
            match rec.tensor_metadata {
                Some(meta) => {
                    let (shape, dtype_name) = decode_shape_metadata(&meta)?;
                    Ok(Some((shape, dtype_name, rec.dtype_code, rec.input_format)))
                }
                None => Ok(None),
            }
        })
        .map_err(runtime_err)
    }

    pub fn tensor_delta_hash<'py>(
        &self,
        py: Python<'py>,
        name: &str,
    ) -> PyResult<Option<Bound<'py, PyBytes>>> {
        let hash = py
            .allow_threads(|| -> Result<Option<[u8; 32]>, PtwmCoreError> {
                let rec = self.reader.tensor_record(name)?;
                Ok(rec
                    .dependencies
                    .iter()
                    .filter(|d| matches!(d.ref_kind, RefKind::ExternalSafetensors))
                    .find_map(|d| d.expected_hash))
            })
            .map_err(runtime_err)?;
        Ok(hash.map(|h| PyBytes::new(py, &h)))
    }

    pub fn tensor_shape(&self, py: Python<'_>, name: &str) -> PyResult<Option<(Vec<u64>, String)>> {
        py.allow_threads(|| self.reader.tensor_shape(name))
            .map_err(runtime_err)
    }

    pub fn plane_summary_rows(&self, py: Python<'_>) -> PyResult<Vec<PlaneSummary>> {
        py.allow_threads(|| -> Result<Vec<PlaneSummary>, PtwmCoreError> {
            let mut out = Vec::new();
            for rec in self.reader.tensor_records()? {
                for (idx, plane) in rec.terminals.iter().enumerate() {
                    out.push((
                        rec.name.clone(),
                        idx as u32,
                        plane.role as u8,
                        plane.codec_id.as_u16(),
                        plane.state_source as u8,
                        plane.payload_len,
                    ));
                }
            }
            Ok(out)
        })
        .map_err(runtime_err)
    }
}

#[pymethods]
impl ContainerHandle {
    #[new]
    #[pyo3(signature = (buffer, skip_missing = false))]
    fn py_new(py: Python<'_>, buffer: &Bound<'_, PyAny>, skip_missing: bool) -> PyResult<Self> {
        Self::open(py, buffer, skip_missing)
    }

    /// ``"legacy"`` (``\x89PTWM`` / ``\x89PTWX``) or ``"compact"``
    /// (``\x89PTWC``).
    #[getter]
    fn layout(&self) -> &'static str {
        self.reader.layout.as_str()
    }

    /// Whether the buffer is borrowed (``True``) or was copied (``False``).
    #[getter]
    fn zero_copy(&self) -> bool {
        self._backing.is_borrowed()
    }

    /// Every tensor name in write (body) order.
    fn names(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        self.names_vec(py)
    }

    /// Decode one tensor and return its raw bytes.
    fn decode<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyBytes>> {
        self.decode_bytes(py, name)
    }

    /// Decode every tensor in parallel; ``(name, bytes)`` in write order.
    fn decode_model<'py>(&self, py: Python<'py>) -> PyResult<Vec<(String, Bound<'py, PyBytes>)>> {
        self.decode_all(py)
    }

    /// ``(shape, dtype_name, dtype_code, input_format)``, or ``None`` when
    /// the tensor has no shape metadata.
    fn info(&self, py: Python<'_>, name: &str) -> PyResult<Option<TensorInfo>> {
        self.tensor_info(py, name)
    }

    /// The BLAKE3 hash of the tensor's delta reference, or ``None``.
    fn delta_hash<'py>(
        &self,
        py: Python<'py>,
        name: &str,
    ) -> PyResult<Option<Bound<'py, PyBytes>>> {
        self.tensor_delta_hash(py, name)
    }

    /// ``(shape, dtype_name)``, or ``None`` when the tensor has no shape
    /// metadata.
    fn shape(&self, py: Python<'_>, name: &str) -> PyResult<Option<(Vec<u64>, String)>> {
        self.tensor_shape(py, name)
    }

    /// One ``(tensor_name, plane_index, role, codec_id, state_source,
    /// payload_len)`` row per plane, tensors in write order.
    fn plane_summaries(&self, py: Python<'_>) -> PyResult<Vec<PlaneSummary>> {
        self.plane_summary_rows(py)
    }

    fn __len__(&self) -> usize {
        self.reader.index.len()
    }

    fn __contains__(&self, name: &str) -> bool {
        self.reader.get_tensor_record_bytes(name).is_some()
    }
}

/// The multi-tensor layout of `blob` from its magic: ``"legacy"`` for
/// ``\x89PTWM`` / ``\x89PTWX``, ``"compact"`` for ``\x89PTWC``, ``None`` for
/// anything else (including a ``\x89PTWS`` single-tensor frame). Only the
/// first nine bytes are read.
#[pyfunction]
pub fn container_layout(py: Python<'_>, blob: PyBuffer<u8>) -> PyResult<Option<&'static str>> {
    let owned;
    let bytes: &[u8] = if blob.is_c_contiguous() {
        crate::contiguous_slice(&blob, "blob")?
    } else {
        owned = blob.to_vec(py)?;
        &owned
    };
    Ok(match sniff(bytes) {
        Some(ContainerMagic::Ptwm | ContainerMagic::Ptwx) => Some(ContainerLayout::Legacy.as_str()),
        Some(ContainerMagic::Ptwc) => Some(ContainerLayout::Compact.as_str()),
        None => None,
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ContainerHandle>()?;
    m.add_function(pyo3::wrap_pyfunction!(container_layout, m)?)?;
    Ok(())
}
