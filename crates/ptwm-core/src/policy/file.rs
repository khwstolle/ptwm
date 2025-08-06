//! Policy TOML schema.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::capability_check::HostPolicy;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PolicyFile {
    #[serde(default)]
    pub include: Vec<String>, // paths merged first
    #[serde(default)]
    pub allow: AllowSection,
    #[serde(default)]
    pub ignore: IgnoreSection,
    #[serde(default)]
    pub per_role: BTreeMap<String, RoleOverride>,
    #[serde(default)]
    pub capabilities: HostPolicy,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AllowSection {
    #[serde(default)]
    pub extra: Vec<String>, // additional canonical IDs to allow on top of trusted
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct IgnoreSection {
    #[serde(default)]
    pub extensions: Vec<String>, // canonical IDs to exclude
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RoleOverride {
    #[serde(default)]
    pub allow_extra: Vec<String>,
    #[serde(default)]
    pub ignore: Vec<String>,
}

impl PolicyFile {
    pub fn from_toml(src: &str) -> Result<Self, crate::extension::ExtensionError> {
        toml::from_str(src)
            .map_err(|e| crate::extension::ExtensionError::ManifestParse(e.to_string()))
    }

    pub fn to_toml(&self) -> Result<String, crate::extension::ExtensionError> {
        toml::to_string_pretty(self)
            .map_err(|e| crate::extension::ExtensionError::ManifestParse(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_empty() {
        let p: PolicyFile = PolicyFile::from_toml("").unwrap();
        assert!(p.include.is_empty());
        assert!(p.allow.extra.is_empty());
        assert!(p.ignore.extensions.is_empty());
        assert!(p.per_role.is_empty());
    }

    #[test]
    fn parse_full_example() {
        let src = r#"
include = ["~/.ptwm/policies/cpu-only.toml"]

[allow]
extra = ["blake3:abcd"]

[ignore]
extensions = ["blake3:ef01"]

[per_role."attention.*"]
allow_extra = ["blake3:1111"]

[per_role."mlp.gate"]
ignore = ["blake3:hadamard"]

[capabilities]
allow_host_imports = []
deny_capabilities  = ["network"]
"#;
        let p = PolicyFile::from_toml(src).unwrap();
        assert_eq!(p.include, vec!["~/.ptwm/policies/cpu-only.toml"]);
        assert_eq!(p.allow.extra, vec!["blake3:abcd"]);
        assert_eq!(p.ignore.extensions, vec!["blake3:ef01"]);
        assert!(p.per_role.contains_key("attention.*"));
        assert!(p.per_role.contains_key("mlp.gate"));
        assert_eq!(
            p.capabilities.deny_capabilities,
            vec!["network".to_string()]
        );
    }

    #[test]
    fn round_trip_toml() {
        let mut p = PolicyFile::default();
        p.allow.extra.push("blake3:abcd".into());
        let s = p.to_toml().unwrap();
        let back = PolicyFile::from_toml(&s).unwrap();
        assert_eq!(back.allow.extra, p.allow.extra);
    }
}
