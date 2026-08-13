//! Shared DLPack capsule parsing and buffer-protocol fallback helpers.
//!
//! Extracted from `hardware.rs` so `plane_codec_cuda.rs` can reuse the same
//! device-pointer extraction and input-resolution logic rather than
//! duplicating a hand-rolled DLPack parser.
//!
//! # DLPack capsule parsing
//!
//! `torch.Tensor.__dlpack__()` returns a `PyCapsule` wrapping a
//! `DLManagedTensor` (frozen C ABI, defined by `dlpack.h`). No `dlpack`
//! crate dependency is used here: the pinned `pyo3 = "0.24"` in this
//! workspace conflicts with the version range the `dlpark` crate's `pyo3`
//! feature requires, so the relevant structs are reproduced below directly
//! against the stable DLPack C layout instead.

use std::os::raw::{c_int, c_void};

use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyCapsuleMethods, PyDict};

pub(crate) fn to_pyerr<E: std::fmt::Display>(e: E) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

// ---------------------------------------------------------------------------
// DLPack: frozen C structs (dlpack.h)
// ---------------------------------------------------------------------------

/// The capsule name mandated by the DLPack Python spec for an
/// unconsumed `__dlpack__()` capsule. A producer that has already
/// consumed the capsule renames it to `"used_dltensor"`; this module only
/// ever sees freshly produced capsules, so `"dltensor"` is the only name
/// accepted here.
pub(crate) const DLPACK_CAPSULE_NAME: &str = "dltensor";

/// `DLDeviceType::kDLCUDA`, per `dlpack.h`.
pub(crate) const DL_CUDA: c_int = 2;

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct DLDevice {
    pub(crate) device_type: c_int, // kDLCUDA = 2
    pub(crate) device_id: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct DLDataType {
    pub(crate) code: u8, // kDLBfloat = 4, kDLUInt = 1, etc.
    pub(crate) bits: u8,
    pub(crate) lanes: u16,
}

#[repr(C)]
pub(crate) struct DLTensor {
    pub(crate) data: *mut c_void,
    pub(crate) device: DLDevice,
    pub(crate) ndim: c_int,
    pub(crate) dtype: DLDataType,
    pub(crate) shape: *mut i64,
    pub(crate) strides: *mut i64, // may be null (implies row-major contiguous)
    pub(crate) byte_offset: u64,
}

#[repr(C)]
pub(crate) struct DLManagedTensor {
    pub(crate) dl_tensor: DLTensor,
    pub(crate) manager_ctx: *mut c_void,
    pub(crate) deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

/// True when `strides` (in elements, per the DLPack spec) matches the
/// row-major-contiguous layout implied by `shape`.
///
/// A dimension of size 0 or 1 is skipped: its stride is a don't-care for
/// contiguity purposes (this mirrors the convention PyTorch's own
/// `is_contiguous()` uses for singleton dimensions), which keeps this
/// check from rejecting perfectly usable tensors that merely have an
/// arbitrary stride recorded on a size-1 axis.
pub(crate) fn is_row_major_contiguous(shape: &[i64], strides: &[i64]) -> bool {
    debug_assert_eq!(shape.len(), strides.len());
    let mut expected: i64 = 1;
    for i in (0..shape.len()).rev() {
        let dim = shape[i];
        if dim < 0 {
            return false;
        }
        if dim > 1 && strides[i] != expected {
            return false;
        }
        expected = expected.saturating_mul(dim.max(1));
    }
    true
}

/// Extract (device pointer as u64, device ordinal, byte length) from a
/// PyCapsule returned by `tensor.__dlpack__(stream=...)`. Rejects
/// non-contiguous tensors (a non-null `strides` pointer whose values do
/// not match the row-major-contiguous stride for `shape`), matching the
/// existing `.contiguous()` convention already used on the CPU path in
/// `python/ptwm/core/_compressor.py::_to_raw_bytes`.
///
/// `role` names the argument this capsule came from (e.g. `"compressed"`,
/// `"out"`, `"src"`, `"dst"`), for the contiguity error message only — it
/// mirrors the `role` prefix `resolve_input`'s own buffer-protocol errors
/// already use, so an error from either branch names the actual argument
/// rather than a hardcoded caller.
pub(crate) fn extract_device_ptr(
    capsule: &Bound<'_, PyAny>,
    role: &str,
) -> PyResult<(u64, i32, usize)> {
    let capsule: &Bound<'_, PyCapsule> = capsule
        .downcast::<PyCapsule>()
        .map_err(|e| to_pyerr(e.to_string()))?;

    // `PyCapsule::pointer()` fetches the capsule's own stored name and
    // passes it straight back into `PyCapsule_GetPointer`, so it always
    // "succeeds" for any valid capsule regardless of what that name is.
    // The DLPack Python spec pins the name to a specific string; check it
    // explicitly here rather than trusting an unnamed or wrongly-named
    // capsule to actually contain a `DLManagedTensor`.
    let name = capsule.name().map_err(to_pyerr)?;
    match name {
        Some(n) if n.to_str().map_err(to_pyerr)? == DLPACK_CAPSULE_NAME => {}
        Some(n) => {
            return Err(to_pyerr(format!(
                "expected a DLPack capsule named '{DLPACK_CAPSULE_NAME}', got '{}' \
                 (has this capsule already been consumed?)",
                n.to_string_lossy()
            )));
        }
        None => {
            return Err(to_pyerr(format!(
                "expected a DLPack capsule named '{DLPACK_CAPSULE_NAME}', capsule has no name"
            )));
        }
    }

    let raw_ptr = capsule.pointer();
    if raw_ptr.is_null() {
        return Err(to_pyerr("DLPack capsule pointer is null"));
    }

    // SAFETY: the name check above confirms this is a "dltensor" capsule
    // per the DLPack Python spec, which guarantees the capsule's opaque
    // pointer references a `DLManagedTensor` laid out per the frozen
    // dlpack.h C ABI. The tensor memory is kept alive by the capsule
    // object itself, which the caller holds for the duration of this
    // call (it is not dropped until the enclosing `#[pyfunction]` returns).
    let managed: &DLManagedTensor = unsafe { &*raw_ptr.cast::<DLManagedTensor>() };
    let dl_tensor = &managed.dl_tensor;

    if dl_tensor.device.device_type != DL_CUDA {
        return Err(to_pyerr(format!(
            "expected a CUDA DLPack tensor (device_type={DL_CUDA}), got device_type={}",
            dl_tensor.device.device_type
        )));
    }

    if dl_tensor.ndim < 0 {
        return Err(to_pyerr("DLPack tensor has negative ndim"));
    }
    let ndim = dl_tensor.ndim as usize;

    let shape: &[i64] = if ndim == 0 {
        &[]
    } else {
        if dl_tensor.shape.is_null() {
            return Err(to_pyerr("DLPack tensor shape pointer is null"));
        }
        // SAFETY: shape is non-null (checked above) and the
        // `DLManagedTensor` contract guarantees `ndim` valid `i64`
        // entries at that pointer.
        unsafe { std::slice::from_raw_parts(dl_tensor.shape, ndim) }
    };

    if !dl_tensor.strides.is_null() {
        // SAFETY: same contract as `shape` above; non-null `strides`
        // points to `ndim` valid `i64` entries per the DLPack spec.
        let strides = unsafe { std::slice::from_raw_parts(dl_tensor.strides, ndim) };
        if !is_row_major_contiguous(shape, strides) {
            return Err(to_pyerr(format!(
                "{role}: non-contiguous tensor passed to device-buffer dispatch; \
                 call .contiguous() on the tensor before passing it in"
            )));
        }
    }

    let elem_bits = dl_tensor.dtype.bits as u64 * dl_tensor.dtype.lanes as u64;
    if elem_bits == 0 || !elem_bits.is_multiple_of(8) {
        return Err(to_pyerr(format!(
            "unsupported DLPack dtype: bits={} lanes={} does not divide evenly into bytes",
            dl_tensor.dtype.bits, dl_tensor.dtype.lanes
        )));
    }
    let elem_bytes = elem_bits / 8;

    let num_elements: u64 = shape
        .iter()
        .try_fold(1u64, |acc, &d| {
            if d < 0 {
                None
            } else {
                acc.checked_mul(d as u64)
            }
        })
        .ok_or_else(|| to_pyerr("DLPack tensor shape has a negative dimension or overflows"))?;

    let byte_len = num_elements
        .checked_mul(elem_bytes)
        .ok_or_else(|| to_pyerr("DLPack tensor byte length overflows usize"))?;

    if dl_tensor.data.is_null() {
        return Err(to_pyerr("DLPack tensor data pointer is null"));
    }
    let device_ptr = (dl_tensor.data as u64)
        .checked_add(dl_tensor.byte_offset)
        .ok_or_else(|| to_pyerr("DLPack tensor data pointer + byte_offset overflows u64"))?;

    Ok((device_ptr, dl_tensor.device.device_id, byte_len as usize))
}

// ---------------------------------------------------------------------------
// Buffer-protocol fallback: plain `bytes` / `bytearray` inputs
// ---------------------------------------------------------------------------

/// Either a DLPack capsule (produced by `tensor.__dlpack__()`) or a plain
/// buffer-protocol object (`bytes`, `bytearray`, ...), resolved once and
/// kept alive for the duration of the dispatch call.
///
/// `resolve_input` picks the variant based on whether the Python object
/// exposes `__dlpack__`; real `torch.Tensor` arguments always do, so the
/// DLPack path is unchanged for them. Plain `bytes`/`bytearray` do not, so
/// they fall back to `Buffer`, which reads/writes the host memory the
/// buffer protocol exposes directly. This lets a CPU-only caller (this
/// crate's own interop tests, and any other host-memory caller) exercise a
/// dispatch contribution without constructing a fake CUDA-shaped DLPack
/// tensor; a contribution that treats its "device" pointers as ordinary
/// host pointers (e.g. a CPU-flavor `hardware_backend`) can be driven
/// through this path exactly as it expects. Both `hardware.rs`'s
/// `compressed`/`out` argument pair and `plane_codec_cuda.rs`'s `src`/`dst`
/// pair resolve through this same enum; the argument names live in the
/// `role` string carried alongside each variant, not in the type itself.
pub(crate) enum InputHandle<'py> {
    Dlpack(Bound<'py, PyAny>, &'static str),
    Buffer(PyBuffer<u8>),
}

impl InputHandle<'_> {
    /// Returns `(pointer, device_ordinal, byte_len)`. `device_ordinal` is
    /// `None` for the buffer-protocol fallback, which carries no device
    /// information; callers must skip the device-ordinal-agreement check
    /// in that case rather than treat `None` as a mismatch.
    pub(crate) fn ptr_len_ordinal(&self) -> PyResult<(u64, Option<i32>, usize)> {
        match self {
            InputHandle::Dlpack(capsule, role) => {
                let (ptr, ordinal, len) = extract_device_ptr(capsule, role)?;
                Ok((ptr, Some(ordinal), len))
            }
            InputHandle::Buffer(buf) => Ok((buf.buf_ptr() as u64, None, buf.len_bytes())),
        }
    }
}

/// Resolve one dispatch argument (named by `role`, e.g. `compressed`/`out`
/// or `src`/`dst` depending on the caller) to an [`InputHandle`].
///
/// `dlpack_kwargs` is only used on the DLPack branch (`stream=` has no
/// meaning for a plain host buffer). `require_writable` rejects a
/// read-only buffer on the fallback branch; the DLPack branch has no
/// equivalent read-only concept at this layer; a decode into a read-only
/// tensor's backing memory is between the caller and whatever `torch`
/// enforces, unchanged from before this fallback was added.
pub(crate) fn resolve_input<'py>(
    obj: &Bound<'py, PyAny>,
    dlpack_kwargs: &Bound<'py, PyDict>,
    role: &'static str,
    require_writable: bool,
) -> PyResult<InputHandle<'py>> {
    if obj.hasattr("__dlpack__")? {
        let capsule = obj.call_method("__dlpack__", (), Some(dlpack_kwargs))?;
        return Ok(InputHandle::Dlpack(capsule, role));
    }

    let buf = PyBuffer::<u8>::get(obj)?;
    if !buf.is_c_contiguous() {
        return Err(to_pyerr(format!("{role}: buffer must be C-contiguous")));
    }
    if require_writable && buf.readonly() {
        return Err(to_pyerr(format!(
            "{role}: buffer-protocol fallback requires a writable buffer \
             (e.g. bytearray), got a read-only buffer"
        )));
    }
    Ok(InputHandle::Buffer(buf))
}

// ---------------------------------------------------------------------------
// Device-ordinal agreement check
// ---------------------------------------------------------------------------

/// Pure comparison behind [`check_device_ordinal_agreement`]: `Some(message)`
/// when the two resolved ordinals disagree with `device_ordinal`, `None` when
/// they agree or either side is unknown (the buffer-protocol fallback,
/// which carries no device information to check against).
///
/// Kept separate from `check_device_ordinal_agreement` itself, and free of
/// any `pyo3` type, so it can be unit-tested directly here (matching this
/// module's existing `is_row_major_contiguous` convention): building or
/// dropping a real `PyErr` requires the CPython C API, which this crate's
/// plain `cargo test` binary cannot link (the `extension-module` pyo3
/// feature deliberately omits linking libpython, since a real build of this
/// crate is loaded into an already-running Python process instead).
fn device_ordinal_mismatch_message(
    device_ordinal: u32,
    in_ordinal: Option<i32>,
    out_ordinal: Option<i32>,
    in_role: &str,
    out_role: &str,
) -> Option<String> {
    let (in_ordinal, out_ordinal) = match (in_ordinal, out_ordinal) {
        (Some(a), Some(b)) => (a, b),
        _ => return None,
    };
    if in_ordinal != device_ordinal as i32 || out_ordinal != device_ordinal as i32 {
        Some(format!(
            "device_ordinal mismatch: caller supplied {device_ordinal}, but {in_role} \
             tensor reports device {in_ordinal} and {out_role} tensor reports device {out_ordinal}"
        ))
    } else {
        None
    }
}

/// Verify that a caller-supplied `device_ordinal` (used to acquire a CUDA
/// stream before any tensor was inspected) agrees with both resolved
/// tensors' own DLPack-reported device ordinals.
///
/// A mismatch means the stream and the tensor memory belong to different
/// devices, which would silently corrupt a decode/encode rather than fail
/// loudly, so this returns an error naming all three ordinals. Either
/// ordinal is `None` on the buffer-protocol fallback branch (see
/// [`InputHandle::ptr_len_ordinal`]), which carries no device information;
/// the check is a no-op in that case rather than treated as a mismatch.
///
/// `in_role`/`out_role` name the two arguments in the error message (e.g.
/// `"compressed"`/`"out"` for `hardware.rs`, `"src"`/`"dst"` for
/// `plane_codec_cuda.rs`), so the message names the actual caller
/// arguments rather than a hardcoded pair.
pub(crate) fn check_device_ordinal_agreement(
    device_ordinal: u32,
    in_ordinal: Option<i32>,
    out_ordinal: Option<i32>,
    in_role: &str,
    out_role: &str,
) -> PyResult<()> {
    match device_ordinal_mismatch_message(
        device_ordinal,
        in_ordinal,
        out_ordinal,
        in_role,
        out_role,
    ) {
        Some(msg) => Err(to_pyerr(msg)),
        None => Ok(()),
    }
}

#[cfg(test)]
mod dlpack_tests {
    use super::*;

    #[test]
    fn dl_data_type_bf16_matches_dlpack_spec_code() {
        // DLPack's DLDataTypeCode for bfloat16 is kDLBfloat = 4, per the
        // frozen DLPack C header (dlpack.h, DLDataTypeCode enum). This
        // guards against a transcription error in the hand-rolled struct
        // above; it is not a full DLPack conformance test.
        let dt = DLDataType {
            code: 4,
            bits: 16,
            lanes: 1,
        };
        assert_eq!(dt.code, 4);
        assert_eq!(dt.bits, 16);
    }

    #[test]
    fn row_major_contiguous_accepts_standard_layout() {
        // A [2, 3] row-major tensor has strides [3, 1] (in elements).
        assert!(is_row_major_contiguous(&[2, 3], &[3, 1]));
    }

    #[test]
    fn row_major_contiguous_rejects_transposed_layout() {
        // The same [2, 3] tensor transposed (a view, not a copy) has
        // strides [1, 2], which is not row-major-contiguous.
        assert!(!is_row_major_contiguous(&[2, 3], &[1, 2]));
    }

    #[test]
    fn row_major_contiguous_ignores_singleton_dimension_stride() {
        // A [1, 3] tensor's size-1 leading dimension carries an
        // arbitrary/don't-care stride in many producers; only the size-3
        // trailing dimension's stride must be 1.
        assert!(is_row_major_contiguous(&[1, 3], &[999, 1]));
    }
}

#[cfg(test)]
mod device_ordinal_agreement_tests {
    use super::*;

    // These exercise `device_ordinal_mismatch_message` directly rather than
    // `check_device_ordinal_agreement` itself: the latter's compiled body
    // unconditionally contains a call to `to_pyerr`/`PyErr::new_err`, so
    // even a test run that only takes the `Ok` branch at runtime would make
    // that call statically reachable from this test binary and fail to
    // link (see the doc comment on `device_ordinal_mismatch_message`).

    #[test]
    fn matching_ordinals_pass() {
        assert!(device_ordinal_mismatch_message(0, Some(0), Some(0), "src", "dst").is_none());
    }

    #[test]
    fn mismatched_ordinal_produces_descriptive_message() {
        let msg = device_ordinal_mismatch_message(0, Some(0), Some(1), "src", "dst")
            .expect("mismatched ordinals must be reported");
        assert!(msg.contains("device_ordinal mismatch"), "{msg}");
        assert!(msg.contains("src"), "{msg}");
        assert!(msg.contains("dst"), "{msg}");
    }

    #[test]
    fn buffer_protocol_fallback_both_none_passes() {
        // Neither side carries a device ordinal on the buffer-protocol
        // fallback branch, so there is nothing to check and this must not
        // be treated as a mismatch.
        assert!(device_ordinal_mismatch_message(0, None, None, "src", "dst").is_none());
    }

    // The next two pin the exact wording for each call site's argument
    // names: `hardware.rs` uses "compressed"/"out", `plane_codec_cuda.rs`
    // uses "src"/"dst" for both its encode and decode functions. A silent
    // change to either message would fail one of these.

    #[test]
    fn message_names_the_hardware_backend_argument_pair() {
        assert_eq!(
            device_ordinal_mismatch_message(0, Some(1), Some(1), "compressed", "out").unwrap(),
            "device_ordinal mismatch: caller supplied 0, but compressed tensor reports \
             device 1 and out tensor reports device 1"
        );
    }

    #[test]
    fn message_names_the_plane_codec_cuda_argument_pair() {
        assert_eq!(
            device_ordinal_mismatch_message(0, Some(1), Some(1), "src", "dst").unwrap(),
            "device_ordinal mismatch: caller supplied 0, but src tensor reports \
             device 1 and dst tensor reports device 1"
        );
    }
}
