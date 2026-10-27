//! Read-only inspection helpers exposed to Python.

use std::fs::File;
use std::io::Read;

use pyo3::prelude::*;

use ptwm_core::extension::ExtensionTable;
use ptwm_core::header::{HEADER_LEN, Header};

fn to_pyerr<E: std::fmt::Display>(e: E) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

/// Return `buf[off..off + len]`, or `None` when the range overflows or runs
/// past the end of `buf`.
fn extension_table_region(buf: &[u8], off: u64, len: u64) -> Option<&[u8]> {
    let start = usize::try_from(off).ok()?;
    let end = start.checked_add(usize::try_from(len).ok()?)?;
    buf.get(start..end)
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
    let off = header.extension_table_offset;
    let len = header.extension_table_length;
    let table_bytes = extension_table_region(&buf, off, len).ok_or_else(|| {
        to_pyerr(format!(
            "'{}': extension table region at offset {} with length {} exceeds file size {}",
            path,
            off,
            len,
            buf.len()
        ))
    })?;
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

#[cfg(test)]
mod tests {
    use super::extension_table_region;

    #[test]
    fn region_rejects_overflow_and_out_of_bounds() {
        let buf = [0u8; 16];
        assert_eq!(extension_table_region(&buf, 4, 8), Some(&buf[4..12]));
        assert_eq!(extension_table_region(&buf, 16, 0), Some(&buf[16..16]));
        assert!(extension_table_region(&buf, 8, 9).is_none());
        assert!(extension_table_region(&buf, u64::MAX - 4, 8).is_none());
        assert!(extension_table_region(&buf, 4, u64::MAX).is_none());
    }
}
