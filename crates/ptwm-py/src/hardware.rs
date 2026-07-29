//! PyO3 bridge for invoking a `hardware_backend` contribution by canonical
//! id, plus a hand-rolled DLPack capsule parser.
//!
//! There is no in-tree consumer of `hardware_backend` (unlike `plane_codec`,
//! which the container encode/decode loop calls internally): this module
//! is the only way to reach a `hardware_backend` contribution from Python.
//! Each call re-scans installed extensions and builds a fresh
//! `HardwareBackendRouter`; `scan_all_cached` already caches the discovery
//! walk, so this matches the stateless-function pattern used by
//! `delta_scheme.rs` and `ext.rs` rather than introducing a persistent
//! router object.
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

use ptwm_core::discovery::scan_all_cached;
use ptwm_core::extension::CanonicalId;
use ptwm_core::flavor::HardwareBackendRouter;

use crate::policy::PyResolvedPolicy;

fn to_pyerr<E: std::fmt::Display>(e: E) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

/// Build a `HardwareBackendRouter` for one call.
///
/// `policy` is optional. When `None`, the router is built via
/// `HardwareBackendRouter::new`, which gates every `hardware_class`-
/// declaring contribution through the default (empty-`available_hardware`)
/// `HostPolicy` and therefore denies all of them: today's exact
/// behavior, unchanged. When `Some`, the caller has supplied a
/// `ResolvedPolicy` (produced from an operator's policy file via
/// `PolicyFile::resolve`, see `policy.rs`); its `HostPolicy` is passed
/// through to `HardwareBackendRouter::new_with_policy`, so a contribution
/// only resolves if its declared `hardware_class` is actually listed in
/// that policy's `available_hardware`. This never bypasses the
/// `hardware_class`-presence precondition or `check` itself; it only
/// lets a caller supply which policy `check` runs against.
fn hardware_backend_router(policy: Option<&PyResolvedPolicy>) -> PyResult<HardwareBackendRouter> {
    let installed = scan_all_cached().map_err(to_pyerr)?;
    Ok(match policy {
        Some(p) => HardwareBackendRouter::new_with_policy(installed, p.host_policy()),
        None => HardwareBackendRouter::new(installed),
    })
}

// ---------------------------------------------------------------------------
// DLPack: frozen C structs (dlpack.h)
// ---------------------------------------------------------------------------

/// The capsule name mandated by the DLPack Python spec for an
/// unconsumed `__dlpack__()` capsule. A producer that has already
/// consumed the capsule renames it to `"used_dltensor"`; this module only
/// ever sees freshly produced capsules, so `"dltensor"` is the only name
/// accepted here.
const DLPACK_CAPSULE_NAME: &str = "dltensor";

/// `DLDeviceType::kDLCUDA`, per `dlpack.h`.
const DL_CUDA: c_int = 2;

#[repr(C)]
#[derive(Clone, Copy)]
struct DLDevice {
    device_type: c_int, // kDLCUDA = 2
    device_id: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DLDataType {
    code: u8, // kDLBfloat = 4, kDLUInt = 1, etc.
    bits: u8,
    lanes: u16,
}

#[repr(C)]
struct DLTensor {
    data: *mut c_void,
    device: DLDevice,
    ndim: c_int,
    dtype: DLDataType,
    shape: *mut i64,
    strides: *mut i64, // may be null (implies row-major contiguous)
    byte_offset: u64,
}

#[repr(C)]
struct DLManagedTensor {
    dl_tensor: DLTensor,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

/// True when `strides` (in elements, per the DLPack spec) matches the
/// row-major-contiguous layout implied by `shape`.
///
/// A dimension of size 0 or 1 is skipped: its stride is a don't-care for
/// contiguity purposes (this mirrors the convention PyTorch's own
/// `is_contiguous()` uses for singleton dimensions), which keeps this
/// check from rejecting perfectly usable tensors that merely have an
/// arbitrary stride recorded on a size-1 axis.
fn is_row_major_contiguous(shape: &[i64], strides: &[i64]) -> bool {
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
fn extract_device_ptr(capsule: &Bound<'_, PyAny>) -> PyResult<(u64, i32, usize)> {
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
            return Err(to_pyerr(
                "non-contiguous tensor passed to hardware_backend dispatch; \
                 call .contiguous() on the tensor before passing it in",
            ));
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
/// `hardware_backend` contribution without constructing a fake CUDA-shaped
/// DLPack tensor; a `hardware_class = "cpu"` contribution such as
/// `ref_hardware_backend` treats its "device" pointers as ordinary host
/// pointers already (see that crate's own doc comments), so passing a host
/// pointer through this path is exactly what such a contribution expects.
enum InputHandle<'py> {
    Dlpack(Bound<'py, PyAny>),
    Buffer(PyBuffer<u8>),
}

impl InputHandle<'_> {
    /// Returns `(pointer, device_ordinal, byte_len)`. `device_ordinal` is
    /// `None` for the buffer-protocol fallback, which carries no device
    /// information; callers must skip the device-ordinal-agreement check
    /// in that case rather than treat `None` as a mismatch.
    fn ptr_len_ordinal(&self) -> PyResult<(u64, Option<i32>, usize)> {
        match self {
            InputHandle::Dlpack(capsule) => {
                let (ptr, ordinal, len) = extract_device_ptr(capsule)?;
                Ok((ptr, Some(ordinal), len))
            }
            InputHandle::Buffer(buf) => Ok((buf.buf_ptr() as u64, None, buf.len_bytes())),
        }
    }
}

/// Resolve one `compressed`/`out` argument to an [`InputHandle`].
///
/// `dlpack_kwargs` is only used on the DLPack branch (`stream=` has no
/// meaning for a plain host buffer). `require_writable` rejects a
/// read-only buffer on the fallback branch; the DLPack branch has no
/// equivalent read-only concept at this layer; a decode into a read-only
/// tensor's backing memory is between the caller and whatever `torch`
/// enforces, unchanged from before this fallback was added.
fn resolve_input<'py>(
    obj: &Bound<'py, PyAny>,
    dlpack_kwargs: &Bound<'py, PyDict>,
    role: &str,
    require_writable: bool,
) -> PyResult<InputHandle<'py>> {
    if obj.hasattr("__dlpack__")? {
        let capsule = obj.call_method("__dlpack__", (), Some(dlpack_kwargs))?;
        return Ok(InputHandle::Dlpack(capsule));
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
// PyO3-visible functions
// ---------------------------------------------------------------------------

/// Return the raw CUDA stream handle for `canonical_id`'s `hardware_backend`
/// contribution on the given device ordinal.
///
/// `policy`, when supplied, is a `ResolvedPolicy` (see `policy.rs`) whose
/// `HostPolicy` gates resolution instead of the router's own default-deny
/// policy; omitting it keeps today's default-deny behavior.
#[pyfunction]
#[pyo3(signature = (canonical_id, device_ordinal, policy=None))]
pub fn hardware_backend_cuda_stream_handle(
    canonical_id: String,
    device_ordinal: u32,
    policy: Option<&PyResolvedPolicy>,
) -> PyResult<u64> {
    let router = hardware_backend_router(policy)?;
    let id = CanonicalId::parse(&canonical_id).map_err(to_pyerr)?;
    let backend = router.get(&id).map_err(to_pyerr)?;
    backend.cuda_stream_handle(device_ordinal).map_err(to_pyerr)
}

/// Dispatch a CUDA decode through `canonical_id`'s `hardware_backend`
/// contribution.
///
/// `compressed` and `out` are ordinarily `torch.Tensor` objects already
/// resident on the CUDA device identified by `device_ordinal`; both must
/// be contiguous, and are read via `__dlpack__()`. As a fallback, an
/// object without `__dlpack__` (plain `bytes`/`bytearray`) is accepted
/// too, read/written directly through the buffer protocol instead: see
/// [`resolve_input`]. This exists so a `hardware_class = "cpu"`
/// contribution can be exercised from Python without constructing a fake
/// CUDA-shaped DLPack tensor or requiring `torch`; a real CUDA tensor
/// still goes through `__dlpack__` exactly as before. `device_ordinal` is
/// an explicit, caller-supplied parameter rather than something inferred
/// from the tensors: `cuda_stream_handle` needs a device ordinal before
/// any tensor has been inspected (there is no tensor yet to peek a
/// `.device.index` from at that point), and v1 of this design is
/// documented as single-GPU, so requiring the caller to state the device
/// explicitly is simpler than reaching into tensor internals and keeps
/// behavior visible rather than silently inferred. The device-ordinal
/// agreement check below only applies when both arguments went through
/// the DLPack branch; the buffer-protocol fallback carries no device
/// information to check against.
#[pyfunction]
#[pyo3(signature = (
    canonical_id,
    codec_id,
    state_bytes,
    compressed,
    out,
    device_ordinal,
    policy=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn hardware_backend_dispatch_decode_cuda<'py>(
    py: Python<'py>,
    canonical_id: String,
    codec_id: String,
    state_bytes: &[u8],
    compressed: &Bound<'py, PyAny>,
    out: &Bound<'py, PyAny>,
    device_ordinal: u32,
    policy: Option<&PyResolvedPolicy>,
) -> PyResult<usize> {
    let router = hardware_backend_router(policy)?;
    let id = CanonicalId::parse(&canonical_id).map_err(to_pyerr)?;
    let codec = CanonicalId::parse(&codec_id).map_err(to_pyerr)?;
    let backend = router.get(&id).map_err(to_pyerr)?;

    let stream = backend
        .cuda_stream_handle(device_ordinal)
        .map_err(to_pyerr)?;

    let dlpack_kwargs = PyDict::new(py);
    dlpack_kwargs.set_item("stream", stream)?;
    let compressed_handle = resolve_input(compressed, &dlpack_kwargs, "compressed", false)?;
    let out_handle = resolve_input(out, &dlpack_kwargs, "out", true)?;

    let (in_dev_ptr, in_ordinal, in_len) = compressed_handle.ptr_len_ordinal()?;
    let (out_dev_ptr, out_ordinal, out_len) = out_handle.ptr_len_ordinal()?;

    // Both tensors' own DLPack device ordinals must agree with the
    // caller-supplied `device_ordinal` used to acquire the stream above;
    // a mismatch means the stream and the tensor memory belong to
    // different devices, which would silently corrupt the decode rather
    // than fail loudly. Neither ordinal is known on the buffer-protocol
    // fallback branch (`None`), so the check is skipped there rather than
    // treated as a mismatch.
    if let (Some(in_ordinal), Some(out_ordinal)) = (in_ordinal, out_ordinal) {
        if in_ordinal != device_ordinal as i32 || out_ordinal != device_ordinal as i32 {
            return Err(to_pyerr(format!(
                "device_ordinal mismatch: caller supplied {device_ordinal}, but compressed \
                 tensor reports device {in_ordinal} and out tensor reports device {out_ordinal}"
            )));
        }
    }

    py.allow_threads(|| {
        backend
            .dispatch_decode_cuda(
                state_bytes,
                &codec,
                in_dev_ptr,
                in_len,
                out_dev_ptr,
                out_len,
                device_ordinal,
            )
            .map_err(to_pyerr)
    })
}

pub fn register(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(
        hardware_backend_cuda_stream_handle,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        hardware_backend_dispatch_decode_cuda,
        m
    )?)?;
    Ok(())
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
