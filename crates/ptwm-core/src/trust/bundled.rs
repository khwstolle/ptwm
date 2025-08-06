//! Bundled-then-pinned trust state evaluator.

use std::path::{Path, PathBuf};

use super::keyring::{Keyring, TrustEntry};
use crate::extension::ExtensionError;

/// Bundled keyring TOML, embedded at build time. Currently ships
/// as an empty `entries = []`; later releases may add curated keys.
pub const BUNDLED_TOML: &str = include_str!("../../data/bundled-keys.toml");

pub fn bundled_keyring() -> Result<Keyring, ExtensionError> {
    Keyring::load_from_toml(BUNDLED_TOML)
}

pub fn bundled_hash() -> [u8; 32] {
    let h = blake3::hash(BUNDLED_TOML.as_bytes());
    let mut out = [0u8; 32];
    out.copy_from_slice(h.as_bytes());
    out
}

/// $XDG_CONFIG_HOME/ptwm/trust/bundled.lock — or ~/.config/... fallback.
pub fn xdg_config_dir() -> PathBuf {
    if let Ok(p) = std::env::var("XDG_CONFIG_HOME") {
        return PathBuf::from(p);
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(".config");
    }
    PathBuf::from(".config")
}

pub fn lock_path() -> PathBuf {
    xdg_config_dir()
        .join("ptwm")
        .join("trust")
        .join("bundled.lock")
}

pub fn prev_keyring_path() -> PathBuf {
    lock_path().with_file_name("bundled.prev.toml")
}

pub fn read_lock(path: &Path) -> Option<[u8; 32]> {
    let s = std::fs::read_to_string(path).ok()?;
    let bytes = hex::decode(s.trim()).ok()?;
    if bytes.len() != 32 {
        return None;
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes);
    Some(out)
}

pub fn write_lock(path: &Path, hash: &[u8; 32], bundled_toml: &str) -> Result<(), ExtensionError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| ExtensionError::ManifestParse(e.to_string()))?;
    }
    std::fs::write(path, hex::encode(hash))
        .map_err(|e| ExtensionError::ManifestParse(e.to_string()))?;
    // Snapshot the bundled TOML next to the lock so we can diff against
    // the next bundled version.
    let prev = prev_keyring_path();
    std::fs::write(&prev, bundled_toml)
        .map_err(|e| ExtensionError::ManifestParse(e.to_string()))?;
    Ok(())
}

#[derive(Clone, Debug, Default)]
pub struct KeyringDiff {
    pub added: Vec<TrustEntry>,
    pub removed: Vec<TrustEntry>,
}

pub fn diff_keyrings(old: &Keyring, new: &Keyring) -> KeyringDiff {
    // Use the serialized form of each TrustEntry as a comparison key,
    // since TrustEntry itself does not derive Eq/Hash cleanly across
    // variants. Stable + good enough for human-facing diff display.
    fn key_of(e: &TrustEntry) -> String {
        toml::to_string(e).unwrap_or_default()
    }

    let mut old_keys: std::collections::BTreeMap<String, &TrustEntry> = Default::default();
    let mut new_keys: std::collections::BTreeMap<String, &TrustEntry> = Default::default();
    for e in &old.entries {
        old_keys.insert(key_of(e), e);
    }
    for e in &new.entries {
        new_keys.insert(key_of(e), e);
    }

    let added: Vec<TrustEntry> = new_keys
        .iter()
        .filter(|(k, _)| !old_keys.contains_key(*k))
        .map(|(_, v)| (*v).clone())
        .collect();
    let removed: Vec<TrustEntry> = old_keys
        .iter()
        .filter(|(k, _)| !new_keys.contains_key(*k))
        .map(|(_, v)| (*v).clone())
        .collect();
    KeyringDiff { added, removed }
}

pub enum BundleStatus {
    /// No lock file → fresh install. Caller should auto-trust the
    /// bundled keyring and persist the lock.
    FreshInstall { keyring: Keyring, hash: [u8; 32] },
    /// Lock matches bundled hash → trust active, no action needed.
    Match { keyring: Keyring },
    /// Lock present but bundled file has changed → bundled keyring
    /// NOT active until the user runs `ptwm trust update --bundled`.
    Mismatch {
        old_hash: [u8; 32],
        new_hash: [u8; 32],
        diff: KeyringDiff,
    },
}

pub fn evaluate_bundled() -> Result<BundleStatus, ExtensionError> {
    let bundled = bundled_keyring()?;
    let hash = bundled_hash();
    let path = lock_path();
    match read_lock(&path) {
        None => Ok(BundleStatus::FreshInstall {
            keyring: bundled,
            hash,
        }),
        Some(prev) if prev == hash => Ok(BundleStatus::Match { keyring: bundled }),
        Some(prev) => {
            // Try to reconstruct the previous keyring from the snapshot
            // file. Fall back to an empty diff if the snapshot is missing.
            let prev_toml = std::fs::read_to_string(prev_keyring_path()).unwrap_or_default();
            let prev_keyring = Keyring::load_from_toml(&prev_toml).unwrap_or_default();
            let diff = diff_keyrings(&prev_keyring, &bundled);
            Ok(BundleStatus::Mismatch {
                old_hash: prev,
                new_hash: hash,
                diff,
            })
        }
    }
}

/// Apply a fresh-install state: persists the lock + snapshot.
pub fn apply_fresh_install(hash: &[u8; 32]) -> Result<(), ExtensionError> {
    write_lock(&lock_path(), hash, BUNDLED_TOML)
}

/// Accept an update — overwrites the lock and snapshot with the new state.
pub fn accept_update(new_hash: &[u8; 32]) -> Result<(), ExtensionError> {
    write_lock(&lock_path(), new_hash, BUNDLED_TOML)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn xdg(dir: &Path) -> impl Drop {
        // Serialize all XDG-mutating tests within this process via a mutex.
        // `set_var` is unsafe on edition 2024 when other threads could be
        // reading the environment concurrently.
        static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

        struct Guard {
            _prev: Option<String>,
            _lock: std::sync::MutexGuard<'static, ()>,
        }
        impl Drop for Guard {
            fn drop(&mut self) {
                match &self._prev {
                    Some(v) => unsafe { std::env::set_var("XDG_CONFIG_HOME", v) },
                    None => unsafe { std::env::remove_var("XDG_CONFIG_HOME") },
                }
            }
        }

        let lock = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let prev = std::env::var("XDG_CONFIG_HOME").ok();
        unsafe { std::env::set_var("XDG_CONFIG_HOME", dir) };
        Guard {
            _prev: prev,
            _lock: lock,
        }
    }

    #[test]
    fn fresh_install_when_no_lock() {
        let d = tempdir().unwrap();
        let _g = xdg(d.path());
        match evaluate_bundled().unwrap() {
            BundleStatus::FreshInstall { .. } => {}
            _ => panic!("expected FreshInstall"),
        }
    }

    #[test]
    fn match_when_lock_equals_bundled() {
        let d = tempdir().unwrap();
        let _g = xdg(d.path());
        // Write lock with the current bundled hash.
        let h = bundled_hash();
        apply_fresh_install(&h).unwrap();
        match evaluate_bundled().unwrap() {
            BundleStatus::Match { .. } => {}
            _ => panic!("expected Match"),
        }
    }

    #[test]
    fn mismatch_when_lock_differs() {
        let d = tempdir().unwrap();
        let _g = xdg(d.path());
        // Plant a lock with a hash that won't match the current bundled.
        let path = lock_path();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, hex::encode([0x99u8; 32])).unwrap();
        // Also plant a "previous" keyring snapshot for diff reconstruction.
        std::fs::write(prev_keyring_path(), "entries = []\n").unwrap();
        match evaluate_bundled().unwrap() {
            BundleStatus::Mismatch {
                old_hash, new_hash, ..
            } => {
                assert_eq!(old_hash, [0x99u8; 32]);
                assert_ne!(old_hash, new_hash);
            }
            _ => panic!("expected Mismatch"),
        }
    }

    #[test]
    fn diff_detects_added_and_removed() {
        let mut old = Keyring::default();
        old.entries.push(TrustEntry::AuthorKey {
            pubkey: format!("ed25519:{}", "11".repeat(32)),
            label: Some("alice".into()),
        });
        let mut new = Keyring::default();
        new.entries.push(TrustEntry::AuthorKey {
            pubkey: format!("ed25519:{}", "22".repeat(32)),
            label: Some("bob".into()),
        });
        let d = diff_keyrings(&old, &new);
        assert_eq!(d.added.len(), 1);
        assert_eq!(d.removed.len(), 1);
    }
}
