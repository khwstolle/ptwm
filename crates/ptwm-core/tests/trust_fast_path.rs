//! Opening a container whose Extension Table holds only built-ins must not
//! touch the filesystem: no extension discovery, no keyring load and no
//! first-run trust lock (synthesis E9).
//!
//! The test points `HOME` and the XDG directories at a temporary directory,
//! so it runs in its own test binary: environment changes are process-wide.

use std::io::Cursor;
use std::path::{Path, PathBuf};

use ptwm_core::container::ContainerReader;
use ptwm_core::container::ContainerWriter;
use ptwm_core::extension::{CanonicalId, ExtensionTable, Kind, builtin_entry};

fn fixture(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Every file below `dir`, recursively.
fn files_under(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else {
                out.push(path);
            }
        }
    }
    out
}

/// A container whose table holds one third-party codec entry.
fn third_party_container() -> Vec<u8> {
    let mut entry = builtin_entry("identity", Kind::PlaneCodec, true);
    entry.canonical_id = CanonicalId::derive(&[0x42; 32], "demo", "0.1.0");
    entry.human_label = "com.example.demo@0.1.0".into();
    let mut cursor = Cursor::new(Vec::new());
    {
        let mut w = ContainerWriter::new(&mut cursor).unwrap();
        w.set_extension_table(ExtensionTable {
            entries: vec![entry],
        });
        w.finalize().unwrap();
    }
    cursor.into_inner()
}

#[test]
fn builtin_only_files_open_without_filesystem_access() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    // SAFETY: this binary has a single test, so no other thread reads the
    // environment while it is changed.
    unsafe {
        std::env::set_var("HOME", root.join("home"));
        std::env::set_var("XDG_CONFIG_HOME", root.join("config"));
        std::env::set_var("XDG_DATA_HOME", root.join("data"));
        std::env::set_var("XDG_CACHE_HOME", root.join("cache"));
        std::env::remove_var("PTWM_EXTENSION_PATH");
    }

    for name in [
        "legacy_v1_1_0.ptwm",
        "legacy_v1_0_0.ptwm",
        "legacy_v1_0_0_native.ptwm",
        "legacy_trimmed.ptwm",
    ] {
        let blob = fixture(name);
        let reader = ContainerReader::open(&blob).unwrap_or_else(|e| panic!("{name}: {e}"));
        reader.decode_model().unwrap();
        assert!(reader.missing_extensions.is_empty());
    }
    assert_eq!(
        files_under(root),
        Vec::<PathBuf>::new(),
        "opening built-in-only files created files"
    );

    // Control: a table with a third-party entry runs the discovery and
    // trust pass, which records the first-run trust lock. This shows the
    // assertion above would catch the pass running.
    let blob = third_party_container();
    let reader = ContainerReader::open_partial(&blob).unwrap();
    assert_eq!(reader.missing_extensions.len(), 1);
    assert!(
        !files_under(root).is_empty(),
        "the trust pass was expected to write its lock file"
    );
}
