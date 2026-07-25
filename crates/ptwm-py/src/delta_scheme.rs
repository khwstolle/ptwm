//! PyO3 bridge for invoking a `delta_scheme` contribution by canonical id.
//!
//! There is no in-tree consumer of `delta_scheme` (unlike `plane_codec`,
//! which the container encode/decode loop calls internally) — this module
//! is the only way to reach a `delta_scheme` contribution from Python.
//! Each call re-scans installed extensions and builds a fresh
//! `DeltaSchemeRouter`; `scan_all_cached` already caches the discovery
//! walk, so this matches the stateless-function pattern used by `ext.rs`
//! rather than introducing a persistent router object.

use pyo3::prelude::*;

use ptwm_core::discovery::scan_all_cached;
use ptwm_core::extension::CanonicalId;
use ptwm_core::flavor::DeltaSchemeRouter;

fn to_pyerr<E: std::fmt::Display>(e: E) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

fn router() -> PyResult<DeltaSchemeRouter> {
    let installed = scan_all_cached().map_err(to_pyerr)?;
    Ok(DeltaSchemeRouter::new(installed))
}

/// Encode `target` against `base` through the `delta_scheme` contribution
/// identified by `canonical_id` (the `"blake3:<hex>"` string from the
/// contribution's manifest). `max_output_len` bounds the output buffer;
/// `len(target)` is a safe default for schemes that don't expand data.
#[pyfunction]
pub fn delta_scheme_encode<'py>(
    py: Python<'py>,
    canonical_id: String,
    base: &[u8],
    target: &[u8],
    max_output_len: usize,
) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
    let id = CanonicalId::parse(&canonical_id).map_err(to_pyerr)?;
    // `base`/`target` are `Ungil` (borrowed from immutable Python `bytes`),
    // so the router resolve + encode can run with the GIL released without
    // copying the (possibly large) tensor payloads.
    let out = py.allow_threads(|| -> PyResult<Vec<u8>> {
        let scheme = router()?.get(&id).map_err(to_pyerr)?;
        let mut output = vec![0u8; max_output_len];
        let n = scheme.encode(base, target, &mut output).map_err(to_pyerr)?;
        output.truncate(n);
        Ok(output)
    })?;
    Ok(pyo3::types::PyBytes::new(py, &out))
}

/// Decode `delta` against `base` through the `delta_scheme` contribution
/// identified by `canonical_id`. `expected_len` is the reconstructed
/// target's exact byte length.
#[pyfunction]
pub fn delta_scheme_decode<'py>(
    py: Python<'py>,
    canonical_id: String,
    base: &[u8],
    delta: &[u8],
    expected_len: usize,
) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
    let id = CanonicalId::parse(&canonical_id).map_err(to_pyerr)?;
    let out = py.allow_threads(|| -> PyResult<Vec<u8>> {
        let scheme = router()?.get(&id).map_err(to_pyerr)?;
        let mut output = vec![0u8; expected_len];
        let n = scheme.decode(base, delta, &mut output).map_err(to_pyerr)?;
        output.truncate(n);
        Ok(output)
    })?;
    Ok(pyo3::types::PyBytes::new(py, &out))
}

pub fn register(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(delta_scheme_encode, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(delta_scheme_decode, m)?)?;
    Ok(())
}
