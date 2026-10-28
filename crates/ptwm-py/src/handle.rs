//! `ContainerHandle`: one opened container reader over borrowed bytes.
//!
//! Opening a container parses its header, Extension Table, prelude, chain
//! registry and index and checks the referenced extensions. A handle does
//! this once and answers any number of queries from the parsed state. The
//! free functions in `lib.rs` (`decode_tensor`, `list_tensor_names`, ...)
//! open a temporary handle per call.
//!
//! A read-only, C-contiguous buffer (``bytes``, a read-only ``memoryview`` or
//! ``mmap``) is borrowed without a copy: the handle holds the buffer export,
//! which keeps the object alive and its memory pinned. A writable or
//! non-contiguous buffer (``bytearray``, a writable ``mmap`` or numpy array)
//! is copied once, because the caller could change it while the reader
//! holds slices into it.

use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use ptwm_core::PtwmCoreError;
use ptwm_core::container::{ContainerLayout, ContainerReader};
use ptwm_core::header::{ContainerMagic, sniff};
use ptwm_core::metadata::decode_shape_metadata;
use ptwm_core::tensor_record::RefKind;

/// `(shape, dtype_name, dtype_code, input_format)` of a tensor.
pub type TensorInfo = (Vec<u64>, String, u16, u8);

/// `(tensor_name, plane_index, role, codec_id, state_source, payload_len)`.
pub type PlaneSummary = (String, u32, u8, u16, u8, u64);

/// The memory a handle's reader points into.
enum Backing {
    /// A read-only, C-contiguous buffer export. Holding it keeps the
    /// exporting object alive and its memory unchanged.
    Borrowed(#[allow(dead_code)] PyBuffer<u8>),
    /// A private copy of a buffer that could change under the reader.
    Owned(#[allow(dead_code)] Box<[u8]>),
}

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
    _backing: Backing,
}

fn runtime_err(e: PtwmCoreError) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}

impl ContainerHandle {
    /// Open `buffer`, borrowing it when it is read-only and C-contiguous.
    pub fn open(py: Python<'_>, buffer: PyBuffer<u8>, skip_missing: bool) -> PyResult<Self> {
        let (bytes, backing): (&[u8], Backing) =
            if buffer.readonly() && buffer.is_c_contiguous() {
                let len = buffer.len_bytes();
                let bytes: &[u8] = if len == 0 {
                    &[]
                } else {
                    // SAFETY: the buffer is C-contiguous and `len` bytes long.
                    // The export stays alive in `Backing::Borrowed` for as
                    // long as the reader, and a read-only export cannot be
                    // written through.
                    unsafe { std::slice::from_raw_parts(buffer.buf_ptr() as *const u8, len) }
                };
                (bytes, Backing::Borrowed(buffer))
            } else {
                let owned: Box<[u8]> = buffer.to_vec(py)?.into_boxed_slice();
                // SAFETY: the boxed slice's heap memory does not move when the
                // box moves into `Backing::Owned`, and it lives as long as the
                // reader.
                let bytes = unsafe { std::slice::from_raw_parts(owned.as_ptr(), owned.len()) };
                (bytes, Backing::Owned(owned))
            };
        // SAFETY: `bytes` points into `backing`, which the handle owns and
        // drops after the reader (see the field order of `ContainerHandle`).
        let bytes: &'static [u8] = unsafe { std::mem::transmute::<&[u8], &'static [u8]>(bytes) };
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
    fn py_new(py: Python<'_>, buffer: PyBuffer<u8>, skip_missing: bool) -> PyResult<Self> {
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
        matches!(self._backing, Backing::Borrowed(_))
    }

    /// Every tensor name in write (body) order.
    fn names(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        self.names_vec(py)
    }

    /// Decode one tensor and return its raw bytes.
    fn decode<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyBytes>> {
        self.decode_bytes(py, name)
    }

    /// Decode every tensor in parallel; ``(name, bytes)`` in index order.
    fn decode_model<'py>(&self, py: Python<'py>) -> PyResult<Vec<(String, Bound<'py, PyBytes>)>> {
        self.decode_all(py)
    }

    /// ``(shape, dtype_name, dtype_code, input_format)``, or ``None`` when
    /// the tensor has no shape metadata.
    fn info(&self, py: Python<'_>, name: &str) -> PyResult<Option<TensorInfo>> {
        self.tensor_info(py, name)
    }

    /// The BLAKE3 hash of the tensor's delta reference, or ``None``.
    fn delta_hash<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Option<Bound<'py, PyBytes>>> {
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
