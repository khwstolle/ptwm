//! PyO3 bridge for the install + discovery surface.

use pyo3::prelude::*;
use std::path::PathBuf;

use ptwm_core::discovery::{
    DiscoveredContribution, FLAVOR_HOST, FLAVOR_NATIVE, FLAVOR_WASM, scan_all_cached,
    user_extensions_dir,
};
use ptwm_core::install::{install, parse_source};

fn to_pyerr<E: std::fmt::Display>(e: E) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

fn flavor_str(bits: u8) -> Vec<String> {
    let mut v = Vec::new();
    if bits & FLAVOR_WASM != 0 {
        v.push("wasm".into());
    }
    if bits & FLAVOR_NATIVE != 0 {
        v.push("native".into());
    }
    if bits & FLAVOR_HOST != 0 {
        v.push("host".into());
    }
    v
}

#[pyclass(module = "ptwm._core", name = "InstalledContribution")]
#[derive(Clone)]
pub struct PyInstalledContribution {
    #[pyo3(get)]
    pub bundle_name: String,
    #[pyo3(get)]
    pub bundle_version: String,
    #[pyo3(get)]
    pub author_pubkey: String,
    #[pyo3(get)]
    pub manifest_path: String,
    #[pyo3(get)]
    pub bundle_dir: String,
    #[pyo3(get)]
    pub installed_flavors: Vec<String>,
    #[pyo3(get)]
    pub contribution_count: usize,
}

impl From<&DiscoveredContribution> for PyInstalledContribution {
    fn from(d: &DiscoveredContribution) -> Self {
        Self {
            bundle_name: d.manifest.bundle.name.clone(),
            bundle_version: d.manifest.bundle.version.clone(),
            author_pubkey: d.manifest.bundle.author_pubkey.clone(),
            manifest_path: d.manifest_path.to_string_lossy().into_owned(),
            bundle_dir: d.bundle_dir.to_string_lossy().into_owned(),
            installed_flavors: flavor_str(d.installed_flavors),
            contribution_count: d.manifest.contributions.len(),
        }
    }
}

#[pyfunction]
pub fn ext_list() -> PyResult<Vec<PyInstalledContribution>> {
    let found = scan_all_cached().map_err(to_pyerr)?;
    Ok(found.iter().map(PyInstalledContribution::from).collect())
}

#[pyfunction]
pub fn ext_install(source_str: String) -> PyResult<String> {
    let src = parse_source(&source_str).map_err(to_pyerr)?;
    let res = install(src).map_err(to_pyerr)?;
    Ok(res.bundle_dir.to_string_lossy().into_owned())
}

#[pyfunction]
pub fn ext_user_dir() -> String {
    user_extensions_dir().to_string_lossy().into_owned()
}

#[pyfunction]
pub fn ext_remove(bundle_dir: String) -> PyResult<bool> {
    let p = PathBuf::from(&bundle_dir);
    if !p.exists() {
        return Ok(false);
    }
    std::fs::remove_dir_all(&p).map_err(to_pyerr)?;
    Ok(true)
}

pub fn register(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_class::<PyInstalledContribution>()?;
    m.add_function(pyo3::wrap_pyfunction!(ext_list, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ext_install, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ext_user_dir, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(ext_remove, m)?)?;
    Ok(())
}
