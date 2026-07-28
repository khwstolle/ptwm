//! PyO3 bridge for the PTWM policy system.

use pyo3::prelude::*;
use std::collections::HashSet;
use std::path::PathBuf;

use ptwm_core::extension::CanonicalId;
use ptwm_core::policy::{HostPolicy, PolicyFile, ResolvedPolicy, resolve};

fn to_pyerr<E: std::fmt::Display>(e: E) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

#[pyclass(module = "ptwm._core", name = "PolicyFile")]
pub struct PyPolicyFile {
    inner: PolicyFile,
    base_dir: PathBuf,
}

#[pymethods]
impl PyPolicyFile {
    #[staticmethod]
    pub fn load(path: String) -> PyResult<Self> {
        let p = PathBuf::from(&path);
        let src = std::fs::read_to_string(&p).map_err(to_pyerr)?;
        let inner = PolicyFile::from_toml(&src).map_err(to_pyerr)?;
        let base_dir = p
            .parent()
            .map(|d| d.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));
        Ok(Self { inner, base_dir })
    }

    #[staticmethod]
    pub fn default_empty() -> Self {
        Self {
            inner: PolicyFile::default(),
            base_dir: PathBuf::from("."),
        }
    }

    pub fn resolve(&self, trusted_ids_hex: Vec<String>) -> PyResult<PyResolvedPolicy> {
        let mut trusted: HashSet<CanonicalId> = HashSet::new();
        for s in trusted_ids_hex {
            let s = s.strip_prefix("blake3:").unwrap_or(&s).to_string();
            let bytes = hex::decode(&s).map_err(to_pyerr)?;
            if bytes.len() != 32 {
                return Err(to_pyerr("canonical id must be 32 bytes"));
            }
            let mut out = [0u8; 32];
            out.copy_from_slice(&bytes);
            trusted.insert(CanonicalId::from_bytes(out));
        }
        let resolved = resolve(&self.inner, &trusted, &self.base_dir).map_err(to_pyerr)?;
        Ok(PyResolvedPolicy { inner: resolved })
    }
}

#[pyclass(module = "ptwm._core", name = "ResolvedPolicy")]
pub struct PyResolvedPolicy {
    inner: ResolvedPolicy,
}

#[pymethods]
impl PyResolvedPolicy {
    pub fn to_toml(&self) -> PyResult<String> {
        self.inner.to_toml().map_err(to_pyerr)
    }

    pub fn allowed_count(&self) -> usize {
        self.inner.allowed.len()
    }

    pub fn effective_global_count(&self) -> usize {
        self.inner.effective_global.len()
    }

    /// Sorted lowercase-hex canonical ids of the effective_global set.
    /// Format matches `builtin_canonical_id(...).hex()`, with no prefix.
    pub fn effective_global_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self
            .inner
            .effective_global
            .iter()
            .map(|id| hex::encode(id.as_bytes()))
            .collect();
        ids.sort();
        ids
    }
}

impl PyResolvedPolicy {
    /// Crate-internal accessor for the resolved `HostPolicy`, used by
    /// other PyO3 bindings (e.g. `hardware.rs`) that need to pass a
    /// caller-supplied policy into a router such as
    /// `HardwareBackendRouter::new_with_policy` instead of that router's
    /// own default-deny policy. Not exposed to Python: callers on the
    /// Python side only ever hand a whole `ResolvedPolicy` object back
    /// into another PyO3 function, never read `HostPolicy` fields
    /// directly.
    pub(crate) fn host_policy(&self) -> HostPolicy {
        self.inner.host.clone()
    }
}

pub fn register(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_class::<PyPolicyFile>()?;
    m.add_class::<PyResolvedPolicy>()?;
    Ok(())
}
