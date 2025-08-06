//! Canonical contribution identity = blake3(pubkey || name || version).

use blake3::Hasher;
use serde::{Deserialize, Serialize};

/// 32-byte canonical contribution identifier.
#[derive(Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct CanonicalId([u8; 32]);

impl CanonicalId {
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Derive a canonical ID from author key, contribution name, and version
    /// string. The author pubkey is the 32-byte Ed25519 raw public key.
    pub fn derive(author_pubkey: &[u8; 32], name: &str, version: &str) -> Self {
        let mut h = Hasher::new();
        h.update(author_pubkey);
        h.update(b"\x00"); // separator
        h.update(name.as_bytes());
        h.update(b"\x00");
        h.update(version.as_bytes());
        let mut out = [0u8; 32];
        out.copy_from_slice(h.finalize().as_bytes());
        Self(out)
    }
}

impl core::fmt::Debug for CanonicalId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "CanonicalId(blake3:{})", hex(&self.0))
    }
}

impl core::fmt::Display for CanonicalId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "blake3:{}", hex(&self.0))
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Human-readable contribution reference, e.g. "io.example.foo@1.2.3".
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ContributionRef {
    pub namespace: String,
    pub name: String,
    pub version: String,
}

impl ContributionRef {
    pub fn parse(s: &str) -> Result<Self, super::ExtensionError> {
        let (path, version) = s
            .rsplit_once('@')
            .ok_or_else(|| super::ExtensionError::InvalidRef(s.to_string()))?;
        let (namespace, name) = path
            .rsplit_once('.')
            .ok_or_else(|| super::ExtensionError::InvalidRef(s.to_string()))?;
        Ok(Self {
            namespace: namespace.to_string(),
            name: name.to_string(),
            version: version.to_string(),
        })
    }
}

impl core::fmt::Display for ContributionRef {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}.{}@{}", self.namespace, self.name, self.version)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derive_is_deterministic_and_distinct() {
        let pk = [0x42u8; 32];
        let a = CanonicalId::derive(&pk, "foo", "1.0.0");
        let b = CanonicalId::derive(&pk, "foo", "1.0.0");
        let c = CanonicalId::derive(&pk, "foo", "1.0.1");
        let d = CanonicalId::derive(&pk, "bar", "1.0.0");
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
    }

    #[test]
    fn ref_parse_roundtrip() {
        let r = ContributionRef::parse("io.example.foo@1.2.3").unwrap();
        assert_eq!(r.namespace, "io.example");
        assert_eq!(r.name, "foo");
        assert_eq!(r.version, "1.2.3");
        assert_eq!(r.to_string(), "io.example.foo@1.2.3");
    }

    #[test]
    fn ref_parse_rejects_malformed() {
        assert!(ContributionRef::parse("no-version").is_err());
        assert!(ContributionRef::parse("@1.0.0").is_err());
    }
}
