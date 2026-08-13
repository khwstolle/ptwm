//! PyO3 bridge for dispatching a `plane_codec` CUDA (device-resident)
//! encode/decode by selector.
//!
//! Mirrors `hardware.rs`'s `hardware_backend_dispatch_decode_cuda` in
//! structure: build a `PlaneCodecCudaRouter` for the call (gated by an
//! optional caller-supplied policy, same as `hardware_backend_router`),
//! resolve device pointers through the shared DLPack/buffer-protocol
//! helpers in `crate::device_buffer`, enforce device-ordinal agreement,
//! then dispatch with the GIL released.
//!
//! Unlike `hardware_backend_dispatch_decode_cuda`, both `selector` and
//! `codec_id` are resolved through `resolve_codec_selector` rather than
//! parsed directly as canonical ids: this lets a caller pass a
//! human-typed name (a builtin like `"identity"`, or an installed
//! contribution's label) as well as a canonical id, matching the
//! selector-resolution convenience `resolve_codec_selector` provides
//! elsewhere. Both selectors are resolved against the same `installed`
//! scan used to build the router, so name resolution and router lookup
//! agree on what is actually installed for this call.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use ptwm_core::discovery::DiscoveredContribution;
use ptwm_core::discovery::scan_all_cached;
use ptwm_core::extension::resolve_codec_selector;
use ptwm_core::flavor::PlaneCodecCudaRouter;

use crate::device_buffer::{check_device_ordinal_agreement, resolve_input, to_pyerr};
use crate::policy::PyResolvedPolicy;

/// Build a `PlaneCodecCudaRouter` for one call, plus the `installed` scan
/// it was built from (also needed by the caller to resolve selectors
/// against the same install snapshot).
///
/// `policy` follows exactly the same optional-gating convention as
/// `hardware.rs`'s `hardware_backend_router`: `None` keeps the router's
/// own default-deny `HostPolicy` (today's behavior, unchanged); `Some`
/// passes the caller-supplied `ResolvedPolicy`'s `HostPolicy` through to
/// `PlaneCodecCudaRouter::new_with_policy` instead.
fn plane_codec_cuda_router(
    policy: Option<&PyResolvedPolicy>,
) -> PyResult<(PlaneCodecCudaRouter, Vec<DiscoveredContribution>)> {
    let installed = scan_all_cached().map_err(to_pyerr)?;
    let router = match policy {
        Some(p) => PlaneCodecCudaRouter::new_with_policy(installed.clone(), p.host_policy()),
        None => PlaneCodecCudaRouter::new(installed.clone()),
    };
    Ok((router, installed))
}

// ---------------------------------------------------------------------------
// PyO3-visible functions
// ---------------------------------------------------------------------------

/// Dispatch a CUDA plane-codec encode through `selector`'s `plane_codec`
/// CUDA contribution.
///
/// `selector` names the `plane_codec` contribution to dispatch through
/// (a builtin name, an installed contribution's label, or a canonical
/// id); `codec_id` names the specific codec variant passed through to the
/// contribution's `encode_cuda`. Both are resolved via
/// `resolve_codec_selector` against the same installed-extension snapshot
/// used to build the router.
///
/// `src`/`dst` are ordinarily `torch.Tensor` objects already resident on
/// the CUDA device identified by `device_ordinal`, read via
/// `__dlpack__()`; a plain buffer-protocol object (`bytes`/`bytearray`)
/// is accepted as a fallback too. See `hardware.rs`'s
/// `hardware_backend_dispatch_decode_cuda` doc comment for the full
/// reasoning behind the buffer-protocol fallback, the explicit
/// `device_ordinal` parameter, and the device-ordinal agreement check —
/// all identical here.
#[pyfunction]
#[pyo3(signature = (
    selector,
    codec_id,
    state_bytes,
    src,
    dst,
    device_ordinal,
    policy=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn plane_codec_encode_cuda<'py>(
    py: Python<'py>,
    selector: String,
    codec_id: String,
    state_bytes: &[u8],
    src: &Bound<'py, PyAny>,
    dst: &Bound<'py, PyAny>,
    device_ordinal: u32,
    policy: Option<&PyResolvedPolicy>,
) -> PyResult<usize> {
    let (router, installed) = plane_codec_cuda_router(policy)?;
    let id = resolve_codec_selector(&selector, &installed).map_err(to_pyerr)?;
    let codec = resolve_codec_selector(&codec_id, &installed).map_err(to_pyerr)?;
    let backend = router.get(&id).map_err(to_pyerr)?;

    // `DispatchedPlaneCodecCuda::cuda_stream_handle` is infallible (`u64`,
    // not `Result<u64, CodecError>`): unlike `hardware.rs`'s
    // `HardwareBackendCuda::cuda_stream_handle`, there is no `?`/`map_err`
    // here.
    let stream = backend.cuda_stream_handle(device_ordinal);

    let dlpack_kwargs = PyDict::new(py);
    dlpack_kwargs.set_item("stream", stream)?;
    let src_handle = resolve_input(src, &dlpack_kwargs, "src", false)?;
    let dst_handle = resolve_input(dst, &dlpack_kwargs, "dst", true)?;

    let (in_dev_ptr, in_ordinal, in_len) = src_handle.ptr_len_ordinal()?;
    let (out_dev_ptr, out_ordinal, out_len) = dst_handle.ptr_len_ordinal()?;

    // Same device-ordinal agreement check as
    // `hardware_backend_dispatch_decode_cuda`: skipped on the
    // buffer-protocol fallback branch, which carries no device ordinal.
    check_device_ordinal_agreement(device_ordinal, in_ordinal, out_ordinal, "src", "dst")?;

    py.allow_threads(|| {
        backend
            .encode_cuda(
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

/// Dispatch a CUDA plane-codec decode through `selector`'s `plane_codec`
/// CUDA contribution. See [`plane_codec_encode_cuda`] for the full
/// argument and behavior documentation; this differs only in calling
/// `decode_cuda` instead of `encode_cuda`.
#[pyfunction]
#[pyo3(signature = (
    selector,
    codec_id,
    state_bytes,
    src,
    dst,
    device_ordinal,
    policy=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn plane_codec_decode_cuda<'py>(
    py: Python<'py>,
    selector: String,
    codec_id: String,
    state_bytes: &[u8],
    src: &Bound<'py, PyAny>,
    dst: &Bound<'py, PyAny>,
    device_ordinal: u32,
    policy: Option<&PyResolvedPolicy>,
) -> PyResult<usize> {
    let (router, installed) = plane_codec_cuda_router(policy)?;
    let id = resolve_codec_selector(&selector, &installed).map_err(to_pyerr)?;
    let codec = resolve_codec_selector(&codec_id, &installed).map_err(to_pyerr)?;
    let backend = router.get(&id).map_err(to_pyerr)?;

    // Infallible: see the identical comment in `plane_codec_encode_cuda`.
    let stream = backend.cuda_stream_handle(device_ordinal);

    let dlpack_kwargs = PyDict::new(py);
    dlpack_kwargs.set_item("stream", stream)?;
    let src_handle = resolve_input(src, &dlpack_kwargs, "src", false)?;
    let dst_handle = resolve_input(dst, &dlpack_kwargs, "dst", true)?;

    let (in_dev_ptr, in_ordinal, in_len) = src_handle.ptr_len_ordinal()?;
    let (out_dev_ptr, out_ordinal, out_len) = dst_handle.ptr_len_ordinal()?;

    check_device_ordinal_agreement(device_ordinal, in_ordinal, out_ordinal, "src", "dst")?;

    py.allow_threads(|| {
        backend
            .decode_cuda(
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
    m.add_function(pyo3::wrap_pyfunction!(plane_codec_encode_cuda, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(plane_codec_decode_cuda, m)?)?;
    Ok(())
}
