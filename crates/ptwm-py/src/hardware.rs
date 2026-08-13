//! PyO3 bridge for invoking a `hardware_backend` contribution by canonical
//! id.
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
//! DLPack capsule parsing and the buffer-protocol fallback live in
//! `crate::device_buffer`, shared with `plane_codec_cuda.rs`.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use ptwm_core::discovery::scan_all_cached;
use ptwm_core::extension::CanonicalId;
use ptwm_core::flavor::HardwareBackendRouter;

use crate::device_buffer::{check_device_ordinal_agreement, resolve_input, to_pyerr};
use crate::policy::PyResolvedPolicy;

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
    // treated as a mismatch. See `check_device_ordinal_agreement`, shared
    // with `plane_codec_cuda.rs`.
    check_device_ordinal_agreement(device_ordinal, in_ordinal, out_ordinal, "compressed", "out")?;

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
