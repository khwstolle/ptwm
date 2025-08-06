//! User-facing trust keyring.

use std::path::Path;

use serde::{Deserialize, Serialize};

use super::signature::PublicKey;
use crate::extension::{CanonicalId, ExtensionError};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Keyring {
    #[serde(default)]
    pub entries: Vec<TrustEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TrustEntry {
    /// Trust everything this author signs.
    AuthorKey {
        pubkey: String, // "ed25519:<hex>"
        label: Option<String>,
    },
    /// Trust only the specific contribution with this canonical id.
    ContributionHash {
        canonical_id: String, // "blake3:<hex>"
        label: Option<String>,
    },
    /// Trust this author's signatures, but only for contributions
    /// declaring capabilities that are a subset of `allowed_capabilities`.
    KeyWithCapabilityConstraints {
        pubkey: String,
        allowed_capabilities: Vec<String>,
        label: Option<String>,
    },
}

impl Keyring {
    pub fn load_from_toml(src: &str) -> Result<Self, ExtensionError> {
        toml::from_str(src).map_err(|e| ExtensionError::ManifestParse(e.to_string()))
    }

    pub fn save_to_toml(&self) -> Result<String, ExtensionError> {
        toml::to_string_pretty(self).map_err(|e| ExtensionError::ManifestParse(e.to_string()))
    }

    pub fn load_from_path(path: &Path) -> Result<Self, ExtensionError> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let src = std::fs::read_to_string(path)
            .map_err(|e| ExtensionError::ManifestParse(e.to_string()))?;
        Self::load_from_toml(&src)
    }

    pub fn save_to_path(&self, path: &Path) -> Result<(), ExtensionError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| ExtensionError::ManifestParse(e.to_string()))?;
        }
        std::fs::write(path, self.save_to_toml()?)
            .map_err(|e| ExtensionError::ManifestParse(e.to_string()))
    }

    pub fn trusts_pubkey(&self, pk: &PublicKey) -> Option<&TrustEntry> {
        self.entries.iter().find(|e| match e {
            TrustEntry::AuthorKey { pubkey, .. }
            | TrustEntry::KeyWithCapabilityConstraints { pubkey, .. } => {
                PublicKey::from_hex(pubkey)
                    .map(|p| p == *pk)
                    .unwrap_or(false)
            }
            TrustEntry::ContributionHash { .. } => false,
        })
    }

    pub fn trusts_contribution_id(&self, id: &CanonicalId) -> bool {
        let want = id.to_string();
        self.entries.iter().any(|e| {
            matches!(
                e,
                TrustEntry::ContributionHash { canonical_id, .. } if canonical_id == &want
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn empty_keyring_round_trips_through_toml() {
        let k = Keyring::default();
        let s = k.save_to_toml().unwrap();
        let back = Keyring::load_from_toml(&s).unwrap();
        assert!(back.entries.is_empty());
    }

    #[test]
    fn author_key_round_trips() {
        let mut k = Keyring::default();
        k.entries.push(TrustEntry::AuthorKey {
            pubkey: "ed25519:00"
                .repeat(32)
                .chars()
                .take("ed25519:".len() + 64)
                .collect(),
            label: Some("core team".into()),
        });
        let s = k.save_to_toml().unwrap();
        let back = Keyring::load_from_toml(&s).unwrap();
        assert_eq!(back.entries.len(), 1);
    }

    #[test]
    fn trusts_pubkey_returns_the_right_entry() {
        let mut k = Keyring::default();
        let hex32 = "11".repeat(32);
        k.entries.push(TrustEntry::AuthorKey {
            pubkey: format!("ed25519:{hex32}"),
            label: Some("alice".into()),
        });
        let pk = PublicKey([0x11; 32]);
        let entry = k.trusts_pubkey(&pk).expect("alice should be trusted");
        match entry {
            TrustEntry::AuthorKey { label, .. } => assert_eq!(label.as_deref(), Some("alice")),
            _ => panic!("wrong variant"),
        }
        let other_pk = PublicKey([0x22; 32]);
        assert!(k.trusts_pubkey(&other_pk).is_none());
    }

    #[test]
    fn trusts_contribution_id_matches_by_hash() {
        let id = CanonicalId::derive(&[0x33; 32], "foo", "1.0.0");
        let mut k = Keyring::default();
        k.entries.push(TrustEntry::ContributionHash {
            canonical_id: id.to_string(),
            label: None,
        });
        assert!(k.trusts_contribution_id(&id));
        let other = CanonicalId::derive(&[0x44; 32], "bar", "1.0.0");
        assert!(!k.trusts_contribution_id(&other));
    }

    #[test]
    fn load_save_path_round_trip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("trust").join("keys.toml");

        let mut k = Keyring::default();
        k.entries.push(TrustEntry::AuthorKey {
            pubkey: format!("ed25519:{}", "aa".repeat(32)),
            label: Some("x".into()),
        });
        k.save_to_path(&path).unwrap();

        let loaded = Keyring::load_from_path(&path).unwrap();
        assert_eq!(loaded.entries.len(), 1);
    }

    #[test]
    fn load_from_missing_path_returns_empty() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("does-not-exist.toml");
        let k = Keyring::load_from_path(&path).unwrap();
        assert!(k.entries.is_empty());
    }
}
