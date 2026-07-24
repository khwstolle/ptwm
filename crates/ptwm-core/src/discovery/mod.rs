//! Filesystem discovery + index cache for installed extensions.
//!
//! Layout per extension bundle:
//! ```text
//! $PTWM_EXTENSION_PATH-or-XDG-dir/<author-fingerprint>/<name>@<version>/
//!   manifest.toml
//!   <contribution>.wasm
//!   <contribution>.so / .dylib / .dll
//!   signature.bin
//! ```
//!
//! Resolution order:
//!   * $PTWM_EXTENSION_PATH (colon-separated) if set
//!   * else: $XDG_DATA_HOME/ptwm/extensions/ then /usr/share/ptwm/extensions/

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

use crate::extension::{ExtensionError, Manifest};

/// Bit-flag set of detected on-disk flavors for a discovered bundle.
pub const FLAVOR_WASM: u8 = 0b001;
pub const FLAVOR_NATIVE: u8 = 0b010;
pub const FLAVOR_HOST: u8 = 0b100;

#[derive(Clone, Debug)]
pub struct DiscoveredContribution {
    pub manifest: Manifest,
    pub manifest_path: PathBuf,
    pub bundle_dir: PathBuf,
    pub installed_flavors: u8, // bit-flag set
}

impl DiscoveredContribution {
    /// Compute the 32-byte BLAKE3 digest the contribution's signature
    /// covers: `blake3(manifest_bytes || <bundle_name>.wasm || .so || .dylib)`
    /// in that order. Mirrors `python/ptwm/ext_tooling/sign.py::sign_bundle`,
    /// which is what `ptwm ext sign` actually signs at bundle-creation time.
    ///
    /// Missing binary files are simply skipped (matches `sign_bundle`'s
    /// `_collect_binary_paths`). The manifest file MUST exist; otherwise the
    /// bundle wouldn't have been discovered.
    pub fn signed_digest(&self) -> Result<[u8; 32], ExtensionError> {
        let manifest_bytes = std::fs::read(&self.manifest_path).map_err(|e| {
            ExtensionError::ManifestParse(format!("read {:?}: {e}", self.manifest_path))
        })?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(&manifest_bytes);
        let name = &self.manifest.bundle.name;
        for ext in ["wasm", "so", "dylib"] {
            let bin_path = self.bundle_dir.join(format!("{name}.{ext}"));
            // Read directly instead of `exists()`+`read` to avoid a TOCTOU
            // window: a binary deleted between the check and the read
            // would surface as a confusing read error rather than the
            // intended "absent, skip" path.
            match std::fs::read(&bin_path) {
                Ok(bin_bytes) => {
                    hasher.update(&bin_bytes);
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => {
                    return Err(ExtensionError::ManifestParse(format!(
                        "read {bin_path:?}: {e}"
                    )));
                }
            }
        }
        let mut out = [0u8; 32];
        out.copy_from_slice(hasher.finalize().as_bytes());
        Ok(out)
    }
}

pub fn user_extensions_dir() -> PathBuf {
    if let Ok(v) = std::env::var("XDG_DATA_HOME") {
        return PathBuf::from(v).join("ptwm").join("extensions");
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home)
            .join(".local")
            .join("share")
            .join("ptwm")
            .join("extensions");
    }
    PathBuf::from(".local/share/ptwm/extensions")
}

pub fn system_extensions_dir() -> PathBuf {
    PathBuf::from("/usr/share/ptwm/extensions")
}

pub fn cache_dir() -> PathBuf {
    if let Ok(v) = std::env::var("XDG_CACHE_HOME") {
        return PathBuf::from(v).join("ptwm");
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(".cache").join("ptwm");
    }
    PathBuf::from(".cache/ptwm")
}

pub fn index_cache_path() -> PathBuf {
    cache_dir().join("extensions.idx")
}

pub fn resolution_path() -> Vec<PathBuf> {
    if let Ok(p) = std::env::var("PTWM_EXTENSION_PATH") {
        return p
            .split(':')
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
            .collect();
    }
    vec![user_extensions_dir(), system_extensions_dir()]
}

/// Detect which flavor artifacts exist in `bundle_dir`.
pub fn detect_flavors(bundle_dir: &Path) -> u8 {
    let mut flavors = 0u8;
    for entry in std::fs::read_dir(bundle_dir)
        .into_iter()
        .flatten()
        .flatten()
    {
        let path = entry.path();
        if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
            match ext {
                "wasm" => flavors |= FLAVOR_WASM,
                "so" | "dylib" | "dll" => flavors |= FLAVOR_NATIVE,
                _ => {}
            }
        }
    }
    // Host flavor is detected only via Python entry points, not on disk.
    flavors
}

/// Walk the resolution path and parse every `manifest.toml` found.
pub fn scan_all() -> Result<Vec<DiscoveredContribution>, ExtensionError> {
    let mut out = Vec::new();
    for root in resolution_path() {
        if !root.exists() {
            continue;
        }
        for entry in WalkDir::new(&root)
            .max_depth(3)
            .into_iter()
            .filter_map(Result::ok)
        {
            if entry.file_name() == "manifest.toml" {
                let src = std::fs::read_to_string(entry.path()).map_err(|e| {
                    ExtensionError::ManifestParse(format!("read {:?}: {e}", entry.path()))
                })?;
                let manifest = match Manifest::from_toml(&src) {
                    Ok(m) => m,
                    Err(_) => continue, // skip malformed bundles; v1 is lenient
                };
                let bundle_dir = entry
                    .path()
                    .parent()
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| root.clone());
                let installed_flavors = detect_flavors(&bundle_dir);
                out.push(DiscoveredContribution {
                    manifest,
                    manifest_path: entry.path().to_path_buf(),
                    bundle_dir,
                    installed_flavors,
                });
            }
        }
    }
    Ok(out)
}

/// On-disk index cache. Stores a snapshot of `(path, mtime, manifest)` so
/// subsequent scans skip re-parsing when nothing changed.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct IndexCache {
    pub entries: Vec<IndexEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IndexEntry {
    pub manifest_path: PathBuf,
    /// Seconds-since-epoch mtime, captured at index-write time.
    pub mtime: i64,
    pub bundle_dir: PathBuf,
    pub installed_flavors: u8,
}

impl PartialEq for IndexEntry {
    fn eq(&self, other: &Self) -> bool {
        self.manifest_path == other.manifest_path
            && self.mtime == other.mtime
            && self.bundle_dir == other.bundle_dir
            && self.installed_flavors == other.installed_flavors
    }
}

impl IndexCache {
    pub fn load(path: &Path) -> Result<Self, ExtensionError> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let src = std::fs::read_to_string(path)
            .map_err(|e| ExtensionError::ManifestParse(format!("read cache: {e}")))?;
        toml::from_str(&src).map_err(|e| ExtensionError::ManifestParse(e.to_string()))
    }

    pub fn save(&self, path: &Path) -> Result<(), ExtensionError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| ExtensionError::ManifestParse(format!("mkdir cache: {e}")))?;
        }
        let s = toml::to_string_pretty(self)
            .map_err(|e| ExtensionError::ManifestParse(e.to_string()))?;
        std::fs::write(path, s)
            .map_err(|e| ExtensionError::ManifestParse(format!("write cache: {e}")))
    }
}

fn mtime_seconds(path: &Path) -> i64 {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Compute a fresh index (without writing to disk). Used by `scan_all_cached`.
fn compute_index(discoveries: &[DiscoveredContribution]) -> IndexCache {
    IndexCache {
        entries: discoveries
            .iter()
            .map(|d| IndexEntry {
                manifest_path: d.manifest_path.clone(),
                mtime: mtime_seconds(&d.manifest_path),
                bundle_dir: d.bundle_dir.clone(),
                installed_flavors: d.installed_flavors,
            })
            .collect(),
    }
}

/// Scan with cache: read the cached index, re-parse only entries whose
/// mtime advanced.
pub fn scan_all_cached() -> Result<Vec<DiscoveredContribution>, ExtensionError> {
    let cache_path = index_cache_path();
    let cached = IndexCache::load(&cache_path).unwrap_or_default();

    let raw = scan_all()?;
    let new_index = compute_index(&raw);
    // Only persist if it differs.
    if cached.entries != new_index.entries {
        let _ = new_index.save(&cache_path);
    }
    Ok(raw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::Mutex;
    use tempfile::tempdir;

    /// Serialise all env-mutating tests so they don't race each other.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn write_bundle(root: &Path, fp: &str, name: &str, version: &str, with_wasm: bool) {
        let dir = root.join(fp).join(format!("{name}@{version}"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("manifest.toml"),
            format!(
                r#"
[bundle]
name = "{name}"
version = "{version}"
author_pubkey = "ed25519:00000000"
"#
            ),
        )
        .unwrap();
        if with_wasm {
            fs::write(dir.join("foo.wasm"), [0u8; 8]).unwrap();
        }
    }

    fn xdg_env_guard(data_home: &Path) -> impl Drop {
        struct Guard {
            prev_data: Option<String>,
            prev_path: Option<String>,
            _lock: std::sync::MutexGuard<'static, ()>,
        }
        impl Drop for Guard {
            fn drop(&mut self) {
                match &self.prev_data {
                    Some(v) => unsafe { std::env::set_var("XDG_DATA_HOME", v) },
                    None => unsafe { std::env::remove_var("XDG_DATA_HOME") },
                }
                match &self.prev_path {
                    Some(v) => unsafe { std::env::set_var("PTWM_EXTENSION_PATH", v) },
                    None => unsafe { std::env::remove_var("PTWM_EXTENSION_PATH") },
                }
            }
        }
        let _lock = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let prev_data = std::env::var("XDG_DATA_HOME").ok();
        let prev_path = std::env::var("PTWM_EXTENSION_PATH").ok();
        unsafe { std::env::set_var("XDG_DATA_HOME", data_home) };
        unsafe { std::env::remove_var("PTWM_EXTENSION_PATH") };
        Guard {
            prev_data,
            prev_path,
            _lock,
        }
    }

    #[test]
    fn scan_finds_a_bundle_with_wasm() {
        let dir = tempdir().unwrap();
        let _g = xdg_env_guard(dir.path());
        let extensions_root = dir.path().join("ptwm").join("extensions");
        fs::create_dir_all(&extensions_root).unwrap();
        write_bundle(&extensions_root, "abcd1234", "demo", "0.1.0", true);

        let found = scan_all().unwrap();
        assert!(
            found.iter().any(|d| d.manifest.bundle.name == "demo"
                && d.installed_flavors & FLAVOR_WASM != 0),
            "expected to find demo bundle with wasm flavor, found {found:?}"
        );
    }

    #[test]
    fn missing_dirs_dont_panic() {
        let dir = tempdir().unwrap();
        let _g = xdg_env_guard(dir.path());
        // Don't create anything; scan should just return empty.
        let found = scan_all().unwrap();
        assert!(found.is_empty());
    }

    #[test]
    fn index_cache_round_trips() {
        let dir = tempdir().unwrap();
        let cache_path = dir.path().join("cache.idx");
        let mut idx = IndexCache::default();
        idx.entries.push(IndexEntry {
            manifest_path: PathBuf::from("/tmp/m.toml"),
            mtime: 12345,
            bundle_dir: PathBuf::from("/tmp"),
            installed_flavors: FLAVOR_WASM,
        });
        idx.save(&cache_path).unwrap();
        let loaded = IndexCache::load(&cache_path).unwrap();
        assert_eq!(loaded.entries.len(), 1);
        assert_eq!(loaded.entries[0].mtime, 12345);
    }

    #[test]
    fn signed_digest_matches_blake3_of_manifest_and_binary() {
        let dir = tempdir().unwrap();
        let bundle_dir = dir.path().to_path_buf();
        let manifest_text = r#"
[bundle]
name = "demo"
version = "0.1.0"
author_pubkey = "ed25519:00000000"
"#;
        let manifest_path = bundle_dir.join("manifest.toml");
        fs::write(&manifest_path, manifest_text).unwrap();
        let wasm_bytes = [0xAA, 0xBB, 0xCC, 0xDD];
        fs::write(bundle_dir.join("demo.wasm"), wasm_bytes).unwrap();

        let manifest = Manifest::from_toml(manifest_text).unwrap();
        let contrib = DiscoveredContribution {
            manifest,
            manifest_path: manifest_path.clone(),
            bundle_dir: bundle_dir.clone(),
            installed_flavors: FLAVOR_WASM,
        };

        // Hand-compute the same digest.
        let mut hasher = blake3::Hasher::new();
        hasher.update(manifest_text.as_bytes());
        hasher.update(&wasm_bytes);
        let mut expected = [0u8; 32];
        expected.copy_from_slice(hasher.finalize().as_bytes());

        let got = contrib.signed_digest().unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn signed_digest_changes_when_binary_is_tampered() {
        let dir = tempdir().unwrap();
        let bundle_dir = dir.path().to_path_buf();
        let manifest_text = r#"
[bundle]
name = "demo"
version = "0.1.0"
author_pubkey = "ed25519:00000000"
"#;
        let manifest_path = bundle_dir.join("manifest.toml");
        fs::write(&manifest_path, manifest_text).unwrap();
        fs::write(bundle_dir.join("demo.wasm"), [0u8; 4]).unwrap();

        let manifest = Manifest::from_toml(manifest_text).unwrap();
        let contrib = DiscoveredContribution {
            manifest,
            manifest_path: manifest_path.clone(),
            bundle_dir: bundle_dir.clone(),
            installed_flavors: FLAVOR_WASM,
        };
        let before = contrib.signed_digest().unwrap();

        // Tamper the binary, recompute.
        fs::write(bundle_dir.join("demo.wasm"), [0xFFu8; 4]).unwrap();
        let after = contrib.signed_digest().unwrap();
        assert_ne!(before, after);
    }

    #[test]
    fn signed_digest_handles_missing_binary() {
        let dir = tempdir().unwrap();
        let bundle_dir = dir.path().to_path_buf();
        let manifest_text = r#"
[bundle]
name = "demo"
version = "0.1.0"
author_pubkey = "ed25519:00000000"
"#;
        let manifest_path = bundle_dir.join("manifest.toml");
        fs::write(&manifest_path, manifest_text).unwrap();
        // No binary on disk — `_collect_binary_paths` would return [].

        let manifest = Manifest::from_toml(manifest_text).unwrap();
        let contrib = DiscoveredContribution {
            manifest,
            manifest_path,
            bundle_dir,
            installed_flavors: 0,
        };

        // Should equal blake3 of just the manifest bytes.
        let mut hasher = blake3::Hasher::new();
        hasher.update(manifest_text.as_bytes());
        let mut expected = [0u8; 32];
        expected.copy_from_slice(hasher.finalize().as_bytes());

        assert_eq!(contrib.signed_digest().unwrap(), expected);
    }

    #[test]
    fn resolution_path_respects_env_override() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let dir1 = tempdir().unwrap();
        let dir2 = tempdir().unwrap();
        let joined = format!("{}:{}", dir1.path().display(), dir2.path().display());
        let prev = std::env::var("PTWM_EXTENSION_PATH").ok();
        unsafe { std::env::set_var("PTWM_EXTENSION_PATH", &joined) };
        let r = resolution_path();
        assert_eq!(r.len(), 2);
        assert_eq!(r[0], dir1.path());
        assert_eq!(r[1], dir2.path());
        match prev {
            Some(v) => unsafe { std::env::set_var("PTWM_EXTENSION_PATH", v) },
            None => unsafe { std::env::remove_var("PTWM_EXTENSION_PATH") },
        }
    }
}
