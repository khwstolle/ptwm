//! TOML manifest schema for an extension bundle (manifest.toml).

use serde::{Deserialize, Serialize};

use super::{CapabilityMap, ExtensionError, Kind, Lifecycle};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub bundle: BundleHeader,
    #[serde(default)]
    pub contributions: Vec<ContributionDecl>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BundleHeader {
    pub name: String,
    pub version: String,
    pub author_pubkey: String, // "ed25519:<hex>"
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ContributionDecl {
    pub id: String,    // "blake3:<hex>"
    pub label: String, // "io.example.foo"
    pub kind: Kind,
    pub abi_version: u16,
    pub lifecycle: Lifecycle,
    #[serde(default)]
    pub flavors: Vec<String>, // ["wasm", "native", "host"]
    #[serde(default)]
    pub capabilities: CapabilityMap,
    #[serde(default)]
    pub install_hint: Option<String>,
}

impl Manifest {
    pub fn from_toml(src: &str) -> Result<Self, ExtensionError> {
        toml::from_str(src).map_err(|e| ExtensionError::ManifestParse(e.to_string()))
    }

    pub fn to_toml(&self) -> Result<String, ExtensionError> {
        toml::to_string_pretty(self).map_err(|e| ExtensionError::ManifestParse(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extension::capability::CapabilityValue;

    #[test]
    fn parse_minimal() {
        let src = r#"
[bundle]
name = "demo"
version = "0.1.0"
author_pubkey = "ed25519:00000000"
"#;
        let m = Manifest::from_toml(src).unwrap();
        assert_eq!(m.bundle.name, "demo");
        assert!(m.contributions.is_empty());
    }

    #[test]
    fn parse_with_contribution() {
        let src = r#"
[bundle]
name = "demo"
version = "0.1.0"
author_pubkey = "ed25519:00000000"

[[contributions]]
id = "blake3:0000"
label = "io.example.foo"
kind = "transform"
abi_version = 1
lifecycle = "thread"
flavors = ["wasm"]
capabilities = { determinism = true }
"#;
        let m = Manifest::from_toml(src).unwrap();
        assert_eq!(m.contributions.len(), 1);
        let c = &m.contributions[0];
        assert_eq!(c.kind, Kind::Transform);
        assert_eq!(c.lifecycle, Lifecycle::Thread);
        assert_eq!(c.flavors, vec!["wasm"]);
        assert_eq!(
            c.capabilities.get("determinism"),
            Some(&CapabilityValue::Bool(true))
        );
    }

    #[test]
    fn rejects_unknown_kind() {
        let src = r#"
[bundle]
name = "demo"
version = "0.1.0"
author_pubkey = "ed25519:00000000"

[[contributions]]
id = "blake3:0000"
label = "io.example.foo"
kind = "not_a_kind"
abi_version = 1
lifecycle = "thread"
"#;
        let err = Manifest::from_toml(src).unwrap_err();
        assert!(matches!(err, ExtensionError::ManifestParse(_)));
    }
}
