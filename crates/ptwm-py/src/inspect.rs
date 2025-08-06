//! Read-only inspection helpers exposed to Python.

use std::fs::File;
use std::io::Read;

use pyo3::prelude::*;

use ptwm_core::extension::ExtensionTable;
use ptwm_core::header::{HEADER_LEN, Header};

fn to_pyerr<E: std::fmt::Display>(e: E) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

/// List the extension-table entries recorded in a `.ptwm` file.
///
/// Returns a list of ``(canonical_id_hex, human_label, kind_u16, flavor_hints)``
/// tuples -- one per entry in the file's extension table.  Returns an empty
/// list when the file carries no extension table (``extension_table_length == 0``).
///
/// Raises ``ValueError`` for any parse error (bad magic, truncated header,
/// out-of-bounds extension table region, ...).
#[pyfunction]
pub fn list_extension_table_entries(path: String) -> PyResult<Vec<(String, String, u16, u8)>> {
    let mut f =
        File::open(&path).map_err(|e| to_pyerr(format!("cannot open '{}': {}", path, e)))?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)
        .map_err(|e| to_pyerr(format!("cannot read '{}': {}", path, e)))?;
    if buf.len() < HEADER_LEN {
        return Err(to_pyerr(format!(
            "'{}': file shorter than header ({} < {})",
            path,
            buf.len(),
            HEADER_LEN
        )));
    }
    let header = Header::from_bytes(&buf[..HEADER_LEN]).map_err(to_pyerr)?;
    if header.extension_table_length == 0 {
        return Ok(Vec::new());
    }
    let off = header.extension_table_offset as usize;
    let len = header.extension_table_length as usize;
    if off + len > buf.len() {
        return Err(to_pyerr(format!(
            "'{}': extension table region [{}..{}] exceeds file size {}",
            path,
            off,
            off + len,
            buf.len()
        )));
    }
    let table_bytes = &buf[off..off + len];
    let table = ExtensionTable::from_bytes(table_bytes).map_err(to_pyerr)?;
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
