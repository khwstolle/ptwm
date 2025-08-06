//! PyO3 bridge for the PTWM trust system.

use pyo3::prelude::*;
use std::path::PathBuf;

use ptwm_core::trust::{
    BundleStatus, Keyring, PublicKey, SecretKey, TrustEntry, accept_update, apply_fresh_install,
    bundled_hash, evaluate_bundled, lock_path,
};

#[pyclass(module = "ptwm._core", name = "TrustEntry")]
#[derive(Clone)]
pub struct PyTrustEntry {
    #[pyo3(get)]
    pub kind: String,
    #[pyo3(get)]
    pub pubkey: Option<String>,
    #[pyo3(get)]
    pub canonical_id: Option<String>,
    #[pyo3(get)]
    pub label: Option<String>,
    #[pyo3(get)]
    pub allowed_capabilities: Option<Vec<String>>,
}

impl From<&TrustEntry> for PyTrustEntry {
    fn from(e: &TrustEntry) -> Self {
        match e {
            TrustEntry::AuthorKey { pubkey, label } => Self {
                kind: "author_key".into(),
                pubkey: Some(pubkey.clone()),
                canonical_id: None,
                label: label.clone(),
                allowed_capabilities: None,
            },
            TrustEntry::ContributionHash {
                canonical_id,
                label,
            } => Self {
                kind: "contribution_hash".into(),
                pubkey: None,
                canonical_id: Some(canonical_id.clone()),
                label: label.clone(),
                allowed_capabilities: None,
            },
            TrustEntry::KeyWithCapabilityConstraints {
                pubkey,
                allowed_capabilities,
                label,
            } => Self {
                kind: "key_with_capability_constraints".into(),
                pubkey: Some(pubkey.clone()),
                canonical_id: None,
                label: label.clone(),
                allowed_capabilities: Some(allowed_capabilities.clone()),
            },
        }
    }
}

fn entry_matches(e: &TrustEntry, identifier: &str) -> bool {
    match e {
        TrustEntry::AuthorKey { pubkey, label }
        | TrustEntry::KeyWithCapabilityConstraints { pubkey, label, .. } => {
            pubkey == identifier || label.as_deref() == Some(identifier)
        }
        TrustEntry::ContributionHash {
            canonical_id,
            label,
        } => canonical_id == identifier || label.as_deref() == Some(identifier),
    }
}

#[pyclass(module = "ptwm._core", name = "Keyring")]
pub struct PyKeyring {
    inner: Keyring,
    path: Option<PathBuf>,
}

#[pymethods]
impl PyKeyring {
    /// Load (or create an empty) keyring from `path`.
    ///
    /// If the file does not exist, returns an empty keyring associated with
    /// that path so `save()` knows where to write it.
    #[staticmethod]
    #[pyo3(signature = (path))]
    pub fn load(path: String) -> PyResult<Self> {
        let p = PathBuf::from(&path);
        let inner = Keyring::load_from_path(&p).map_err(to_pyerr)?;
        Ok(Self {
            inner,
            path: Some(p),
        })
    }

    /// Write the keyring to its associated path.
    pub fn save(&self) -> PyResult<()> {
        let p = self.path.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("keyring has no associated path")
        })?;
        self.inner.save_to_path(p).map_err(to_pyerr)
    }

    /// Append an `AuthorKey` entry.
    pub fn add_author_key(&mut self, pubkey: String, label: Option<String>) {
        self.inner
            .entries
            .push(TrustEntry::AuthorKey { pubkey, label });
    }

    /// Append a `ContributionHash` entry.
    pub fn add_contribution_hash(&mut self, canonical_id: String, label: Option<String>) {
        self.inner.entries.push(TrustEntry::ContributionHash {
            canonical_id,
            label,
        });
    }

    /// Remove entries matching `identifier` (pubkey, canonical_id, or label).
    ///
    /// Returns `True` if at least one entry was removed.
    pub fn remove(&mut self, identifier: String) -> bool {
        let before = self.inner.entries.len();
        self.inner
            .entries
            .retain(|e| !entry_matches(e, &identifier));
        before != self.inner.entries.len()
    }

    /// Return all entries as a list of `TrustEntry` objects.
    pub fn entries(&self) -> Vec<PyTrustEntry> {
        self.inner.entries.iter().map(PyTrustEntry::from).collect()
    }
}

#[pyclass(module = "ptwm._core", name = "BundleStatus")]
#[derive(Clone)]
pub struct PyBundleStatus {
    /// One of `"FreshInstall"`, `"Match"`, `"Mismatch"`.
    #[pyo3(get)]
    pub kind: String,
    /// Hex-encoded old hash (only set when `kind == "Mismatch"`).
    #[pyo3(get)]
    pub old_hash_hex: Option<String>,
    /// Hex-encoded new bundled hash (set for `"FreshInstall"` and `"Mismatch"`).
    #[pyo3(get)]
    pub new_hash_hex: Option<String>,
    /// Entries present in the new bundled keyring but not the old one.
    #[pyo3(get)]
    pub added: Vec<PyTrustEntry>,
    /// Entries present in the old bundled keyring but not the new one.
    #[pyo3(get)]
    pub removed: Vec<PyTrustEntry>,
}

fn bytes_to_hex(b: &[u8]) -> String {
    b.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[pyfunction]
#[pyo3(name = "trust_evaluate_bundled")]
pub fn py_evaluate_bundled() -> PyResult<PyBundleStatus> {
    let status = evaluate_bundled().map_err(to_pyerr)?;
    Ok(match status {
        BundleStatus::FreshInstall { hash, .. } => PyBundleStatus {
            kind: "FreshInstall".into(),
            old_hash_hex: None,
            new_hash_hex: Some(bytes_to_hex(&hash)),
            added: vec![],
            removed: vec![],
        },
        BundleStatus::Match { .. } => PyBundleStatus {
            kind: "Match".into(),
            old_hash_hex: None,
            new_hash_hex: None,
            added: vec![],
            removed: vec![],
        },
        BundleStatus::Mismatch {
            old_hash,
            new_hash,
            diff,
        } => PyBundleStatus {
            kind: "Mismatch".into(),
            old_hash_hex: Some(bytes_to_hex(&old_hash)),
            new_hash_hex: Some(bytes_to_hex(&new_hash)),
            added: diff.added.iter().map(PyTrustEntry::from).collect(),
            removed: diff.removed.iter().map(PyTrustEntry::from).collect(),
        },
    })
}

#[pyfunction]
#[pyo3(name = "trust_apply_fresh_install")]
pub fn py_apply_fresh_install() -> PyResult<()> {
    let h = bundled_hash();
    apply_fresh_install(&h).map_err(to_pyerr)
}

#[pyfunction]
#[pyo3(name = "trust_accept_update")]
pub fn py_accept_update() -> PyResult<()> {
    let h = bundled_hash();
    accept_update(&h).map_err(to_pyerr)
}

#[pyfunction]
#[pyo3(name = "trust_user_keys_path")]
pub fn py_user_keys_path() -> String {
    lock_path()
        .with_file_name("keys.toml")
        .to_string_lossy()
        .into_owned()
}

fn to_pyerr<E: std::fmt::Display>(e: E) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

// ---------------------------------------------------------------------------
// Signing key pair (for ext sign / author tooling)
// ---------------------------------------------------------------------------

/// An Ed25519 public key.
#[pyclass(module = "ptwm._core", name = "PublicKey")]
#[derive(Clone)]
pub struct PyPublicKey(PublicKey);

#[pymethods]
impl PyPublicKey {
    /// Return the 64-character lowercase hex encoding of the 32-byte key.
    pub fn hex(&self) -> String {
        hex::encode(self.0.0)
    }

    /// Return the first 16 bytes (32 hex chars) of the BLAKE3 hash of the key.
    pub fn fingerprint(&self) -> String {
        self.0.fingerprint()
    }
}

/// An Ed25519 signing key.  Load from a 32-byte raw seed file produced by
/// ``ptwm trust keygen``.
#[pyclass(module = "ptwm._core", name = "SecretKey")]
pub struct PySecretKey(SecretKey);

#[pymethods]
impl PySecretKey {
    /// Generate a fresh random key (useful for tests).
    #[staticmethod]
    pub fn generate() -> Self {
        Self(SecretKey::generate())
    }

    /// Load a key from a 32-byte raw seed file.
    ///
    /// Raises `ValueError` when the file cannot be read or is not exactly
    /// 32 bytes.
    #[staticmethod]
    pub fn load(path: String) -> PyResult<Self> {
        let bytes = std::fs::read(&path)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        if bytes.len() != 32 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "secret key file must be exactly 32 bytes, got {}",
                bytes.len()
            )));
        }
        let arr: [u8; 32] = bytes.try_into().unwrap();
        Ok(Self(SecretKey::from_bytes(&arr)))
    }

    /// Return the corresponding public key.
    pub fn public(&self) -> PyPublicKey {
        PyPublicKey(self.0.public())
    }

    /// Sign `message` (a bytes-like object) and return the 64-byte signature.
    pub fn sign(&self, message: Vec<u8>) -> Vec<u8> {
        self.0.sign(&message)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKeyring>()?;
    m.add_class::<PyTrustEntry>()?;
    m.add_class::<PyBundleStatus>()?;
    m.add_class::<PySecretKey>()?;
    m.add_class::<PyPublicKey>()?;
    m.add_function(wrap_pyfunction!(py_evaluate_bundled, m)?)?;
    m.add_function(wrap_pyfunction!(py_apply_fresh_install, m)?)?;
    m.add_function(wrap_pyfunction!(py_accept_update, m)?)?;
    m.add_function(wrap_pyfunction!(py_user_keys_path, m)?)?;
    Ok(())
}
