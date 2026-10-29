//! Caller buffers that Rust code may read with the GIL released.
//!
//! A buffer is borrowed without a copy only when its memory cannot change
//! while the export is held: a C-contiguous export of ``bytes``, or of an
//! ``mmap`` opened with ``ACCESS_READ``, directly or through any chain of
//! ``memoryview`` objects. Every other buffer is copied once while the GIL
//! is held, including read-only views of mutable objects (for example
//! ``memoryview(bytearray).toreadonly()`` or a numpy array with
//! ``writeable=False`` over a writable base): a read-only export only says
//! that the export cannot be written through, not that the owner cannot
//! write the memory. Without the copy, another Python thread could write a
//! ``bytearray`` while Rust code reads it as a shared slice, which is a data
//! race.
//!
//! A read-only ``mmap`` maps a file with ``MAP_SHARED``. Another process (or
//! a second, writable mapping of the same file) can still change the file;
//! the caller must not modify a file while it is borrowed.

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyBufferError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyMemoryView};

/// Whether the memory behind `obj`'s buffer cannot change for as long as
/// an export of it is held.
///
/// True for an exact ``bytes`` object and for an exact ``mmap.mmap`` whose
/// own export is read-only (``ACCESS_READ``), reached directly or through
/// ``memoryview.obj`` links. Everything else (``bytearray``, writable
/// ``mmap``, numpy arrays, subclasses that could override the buffer
/// protocol, memoryviews without an owner object) is treated as mutable.
pub(crate) fn immutable_backing(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = obj.py();
    let mut cur = obj.clone();
    // `memoryview.obj` of a memoryview over a memoryview names the base
    // object, so the loop ends after one step in practice; the bound guards
    // against an unexpected cycle.
    for _ in 0..16 {
        if cur.is_exact_instance_of::<PyMemoryView>() {
            let base = cur.getattr("obj")?;
            if base.is_none() {
                return Ok(false);
            }
            cur = base;
            continue;
        }
        if cur.is_exact_instance_of::<PyBytes>() {
            return Ok(true);
        }
        let mmap_type = py.import("mmap")?.getattr("mmap")?;
        if cur.get_type().is(&mmap_type) {
            return Ok(PyBuffer::<u8>::get(&cur)?.readonly());
        }
        return Ok(false);
    }
    Ok(false)
}

/// The bytes of a caller buffer, safe to read with the GIL released.
pub(crate) enum ReadBuffer {
    /// A C-contiguous export of immutable memory (see
    /// [`immutable_backing`]). Holding it keeps the exporting object alive
    /// and its memory pinned.
    Borrowed(PyBuffer<u8>),
    /// A private copy of a buffer that could change while it is read.
    Owned(Box<[u8]>),
}

impl ReadBuffer {
    /// Export `obj`'s buffer. It is borrowed when its memory cannot change
    /// (see [`immutable_backing`]) and it is C-contiguous; otherwise it is
    /// copied once, in C order, while the GIL is held.
    pub(crate) fn new(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let buffer = PyBuffer::<u8>::get(obj)?;
        Self::from_export(obj, buffer)
    }

    /// Like [`ReadBuffer::new`], but rejects a buffer that is not
    /// C-contiguous with ``BufferError`` and prefixes a ``TypeError`` (not a
    /// buffer, or not a byte buffer) with ``argument '<name>': ``.
    pub(crate) fn contiguous(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Self> {
        let py = obj.py();
        let buffer = PyBuffer::<u8>::get(obj).map_err(|e| {
            if e.is_instance_of::<PyTypeError>(py) {
                PyTypeError::new_err(format!("argument '{name}': {}", e.value(py)))
            } else {
                e
            }
        })?;
        if !buffer.is_c_contiguous() {
            return Err(PyBufferError::new_err(format!(
                "{name}: buffer must be C-contiguous"
            )));
        }
        Self::from_export(obj, buffer)
    }

    fn from_export(obj: &Bound<'_, PyAny>, buffer: PyBuffer<u8>) -> PyResult<Self> {
        if buffer.is_c_contiguous() && immutable_backing(obj)? {
            Ok(Self::Borrowed(buffer))
        } else {
            Ok(Self::Owned(buffer.to_vec(obj.py())?.into_boxed_slice()))
        }
    }

    /// Whether the buffer is borrowed (``true``) or was copied (``false``).
    pub(crate) fn is_borrowed(&self) -> bool {
        matches!(self, Self::Borrowed(_))
    }

    /// The buffer's bytes.
    pub(crate) fn as_slice(&self) -> &[u8] {
        match self {
            Self::Borrowed(buffer) => {
                let len = buffer.len_bytes();
                if len == 0 {
                    return &[];
                }
                // SAFETY: the buffer is C-contiguous and `len` bytes long
                // (checked in `from_export`). The export lives as long as
                // `self`, which keeps the owner alive and (for `mmap`)
                // prevents it from being closed or resized. The owner is
                // `bytes` or a read-only `mmap`, so no Python code can write
                // the memory while the slice exists, even with the GIL
                // released.
                unsafe { std::slice::from_raw_parts(buffer.buf_ptr() as *const u8, len) }
            }
            Self::Owned(bytes) => bytes,
        }
    }
}

/// Whether the GIL-free binding functions borrow `obj`'s buffer without a
/// copy (``True``) or copy it first (``False``). Test hook for the rule in
/// this module.
#[pyfunction]
fn _buffer_is_borrowed(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(ReadBuffer::new(obj)?.is_borrowed())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(_buffer_is_borrowed, m)?)?;
    Ok(())
}
