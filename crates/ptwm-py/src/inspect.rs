//! Read-only inspection helpers exposed to Python.

use std::fs::File;
use std::io::Read;

use pyo3::prelude::*;

use ptwm_core::container::read_extension_table;

fn to_pyerr<E: std::fmt::Display>(e: E) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

/// List the extension-table entries recorded in a `.ptwm` file.
///
/// Returns a list of ``(canonical_id_hex, human_label, kind_u16, flavor_hints)``
/// tuples -- one per entry in the file's extension table.  Returns an empty
/// list when the file carries no extension table.
///
/// Both layouts are read: a legacy (``\x89PTWM``) file has its table hash
/// verified, and a compact (``\x89PTWC``) file has its compact entries
/// expanded to full entries. Built-in entries written by another release are
/// reported with the ids and labels of the current release (see
/// ``normalize_builtin_ids``), so the ids of files written by different
/// releases compare equal.
///
/// Raises ``ValueError`` for any parse error (bad magic, truncated header,
/// out-of-bounds extension table region, hash mismatch, ...).
#[pyfunction]
pub fn list_extension_table_entries(path: String) -> PyResult<Vec<(String, String, u16, u8)>> {
    let mut f =
        File::open(&path).map_err(|e| to_pyerr(format!("cannot open '{}': {}", path, e)))?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)
        .map_err(|e| to_pyerr(format!("cannot read '{}': {}", path, e)))?;
    let table = read_extension_table(&buf).map_err(|e| to_pyerr(format!("'{path}': {e}")))?;
    Ok(table
        .entries
        .iter()
        .map(|e| {
            (
                e.canonical_id.to_string(),
                e.human_label.clone(),
                e.kind.as_u16(),
                e.flavor_hints,
            )
        })
        .collect())
}

pub fn register(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(list_extension_table_entries, m)?)?;
    Ok(())
}
