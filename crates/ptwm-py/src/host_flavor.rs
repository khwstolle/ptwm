//! Host-language (Python) contribution registry, exposed to Python.
//!
//! A Python extension package registers its contributions via the
//! `ptwm.extensions` entry-point group. At PTWM startup, the loader
//! (python/ptwm/_loader.py) walks those entry points, calls each one,
//! and the entry point invokes `ContributionRegistry.register(...)`.

use std::collections::HashMap;
use std::sync::Mutex;

use pyo3::prelude::*;

use ptwm_core::extension::{Kind, Lifecycle};

struct Registered {
    kind_str: String,
    lifecycle_str: String,
    /// Stored for future dispatch; not yet called by the Rust side.
    #[allow(dead_code)]
    callable: PyObject,
}

static REGISTRY: Mutex<Option<Inner>> = Mutex::new(None);

#[derive(Default)]
struct Inner {
    by_id: HashMap<String, Registered>,
}

#[pyclass(module = "ptwm._core", name = "ContributionRegistry")]
pub struct PyContributionRegistry;

#[pymethods]
impl PyContributionRegistry {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Register a contribution. `canonical_id_hex` is the 64-char hex
    /// (with or without "blake3:" prefix). `kind` is one of the snake_case
    /// names from the Kind enum. `lifecycle` is "none" | "thread" | "process".
    pub fn register(
        &self,
        canonical_id_hex: String,
        kind: String,
        lifecycle: String,
        callable: PyObject,
    ) -> PyResult<()> {
        let _ = parse_kind(&kind).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("unknown kind: {kind}"))
        })?;
        let _ = parse_lifecycle(&lifecycle).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("unknown lifecycle: {lifecycle}"))
        })?;
        let id_norm = canonical_id_hex
            .strip_prefix("blake3:")
            .unwrap_or(&canonical_id_hex)
            .to_lowercase();
        let mut guard = REGISTRY.lock().unwrap();
        let inner = guard.get_or_insert_with(Inner::default);
        inner.by_id.insert(
            id_norm,
            Registered {
                kind_str: kind,
                lifecycle_str: lifecycle,
                callable,
            },
        );
        Ok(())
    }

    /// Number of registered contributions (test/debug helper).
    pub fn count(&self) -> usize {
        REGISTRY
            .lock()
            .unwrap()
            .as_ref()
            .map(|i| i.by_id.len())
            .unwrap_or(0)
    }

    /// Clear all registrations (test helper).
    pub fn clear(&self) {
        *REGISTRY.lock().unwrap() = None;
    }
}

fn parse_kind(s: &str) -> Option<Kind> {
    match s {
        "transform" => Some(Kind::Transform),
        "plane_codec" => Some(Kind::PlaneCodec),
        "chain_builder" => Some(Kind::ChainBuilder),
        "chain_explorer" => Some(Kind::ChainExplorer),
        "classifier" => Some(Kind::Classifier),
        "scorer" => Some(Kind::Scorer),
        "delta_scheme" => Some(Kind::DeltaScheme),
        "integration_adapter" => Some(Kind::IntegrationAdapter),
        "container_layout" => Some(Kind::ContainerLayout),
        "hardware_backend" => Some(Kind::HardwareBackend),
        "benchmark_metric" => Some(Kind::BenchmarkMetric),
        "training_hook" => Some(Kind::TrainingHook),
        "raw_binary" => Some(Kind::RawBinary),
        _ => None,
    }
}

fn parse_lifecycle(s: &str) -> Option<Lifecycle> {
    match s {
        "none" => Some(Lifecycle::None),
        "thread" => Some(Lifecycle::Thread),
        "process" => Some(Lifecycle::Process),
        _ => None,
    }
}

/// Return list of registered (canonical_id_hex, kind, lifecycle) tuples.
/// Used by the Rust dispatcher to enumerate host-flavor contributions.
#[pyfunction]
pub fn host_registry_entries() -> Vec<(String, String, String)> {
    REGISTRY
        .lock()
        .unwrap()
        .as_ref()
        .map(|i| {
            i.by_id
                .iter()
                .map(|(id, r)| (id.clone(), r.kind_str.clone(), r.lifecycle_str.clone()))
                .collect()
        })
        .unwrap_or_default()
}

pub fn register(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_class::<PyContributionRegistry>()?;
    m.add_function(pyo3::wrap_pyfunction!(host_registry_entries, m)?)?;
    Ok(())
}
