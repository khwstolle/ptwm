//! Org trust manifests — signed TOML files vouching for member author keys.
//!
//! Lifecycle:
//!   1. Org publishes `<manifest>.toml` at a URL + a sibling `<manifest>.toml.sig`.
//!   2. User runs `ptwm trust add --org <URL>`.
//!   3. PTWM fetches both, verifies the manifest signature against
//!      `org.key`, parses members, adds the *org key* to the user's
//!      keyring; member keys are trusted transitively by the verifier
//!      when the contribution's pubkey matches a member.
//!
//! `fetch_and_verify` does the network work for tests we mock with a
//! local file path (so this file's tests don't make HTTP calls).

use std::io::Read;
use std::path::Path;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::signature::PublicKey;
use crate::extension::ExtensionError;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct OrgManifest {
    pub org: OrgHeader,
    #[serde(default)]
    pub members: Vec<OrgMember>,
    #[serde(default)]
    pub scope: OrgScope,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct OrgHeader {
    pub name: String,
    pub key: String, // "ed25519:<hex>"
    pub url: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct OrgMember {
    pub label: String,
    pub pubkey: String,
    pub since: String, // ISO date
    #[serde(default)]
    pub until: Option<String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct OrgScope {
    #[serde(default)]
    pub allowed_capabilities: Vec<String>,
    #[serde(default)]
    pub allowed_kinds: Vec<String>,
}

impl OrgManifest {
    /// Parse the TOML body and verify the detached signature against
    /// the embedded org key.
    pub fn parse_and_verify(body: &[u8], signature: &[u8]) -> Result<Self, ExtensionError> {
        let manifest: OrgManifest = toml::from_str(
            std::str::from_utf8(body)
                .map_err(|e| ExtensionError::ManifestParse(format!("non-utf8 manifest: {e}")))?,
        )
        .map_err(|e| ExtensionError::ManifestParse(e.to_string()))?;
        let pubkey = PublicKey::from_hex(&manifest.org.key)?;
        pubkey.verify(body, signature)?;
        Ok(manifest)
    }

    /// Return the set of currently-active member pubkey strings. A
    /// member is active when:
    ///   * `since` <= today (we accept any non-empty `since` string as
    ///     a placeholder; date arithmetic would add chrono as a dep;
    ///     for v1, the field is informational and not gating).
    ///   * `until` is None *or* > today (same caveat — we accept any
    ///     `until` value as a marker; the verifier skips members whose
    ///     `until` is present and lexicographically <= today).
    /// The caller is responsible for parsing dates; for now this treats
    /// `until.is_some()` as "potentially expired" and lets the policy
    /// layer decide.
    pub fn active_members(&self, today_yyyymmdd: &str) -> Vec<&OrgMember> {
        self.members
            .iter()
            .filter(|m| match &m.until {
                None => true,
                Some(u) => u.as_str() > today_yyyymmdd,
            })
            .collect()
    }
}

/// Fetch the manifest and signature from a URL pair `(<url>, <url>.sig)`.
/// Uses `ureq` (blocking) with a 30 s timeout. Returns the verified manifest.
pub fn fetch_and_verify(url: &str) -> Result<OrgManifest, ExtensionError> {
    let body = http_get(url)?;
    let sig = http_get(&format!("{url}.sig"))?;
    OrgManifest::parse_and_verify(&body, &sig)
}

fn http_get(url: &str) -> Result<Vec<u8>, ExtensionError> {
    let resp = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(30))
        .build()
        .get(url)
        .call()
        .map_err(|e| ExtensionError::ManifestParse(format!("HTTP GET {url}: {e}")))?;
    let mut buf = Vec::new();
    resp.into_reader()
        .read_to_end(&mut buf)
        .map_err(|e| ExtensionError::ManifestParse(format!("read body {url}: {e}")))?;
    Ok(buf)
}

/// Fixture-friendly variant: load body+sig from a local pair of files.
pub fn load_and_verify_files(
    body_path: &Path,
    sig_path: &Path,
) -> Result<OrgManifest, ExtensionError> {
    let body = std::fs::read(body_path)
        .map_err(|e| ExtensionError::ManifestParse(format!("read {body_path:?}: {e}")))?;
    let sig = std::fs::read(sig_path)
        .map_err(|e| ExtensionError::ManifestParse(format!("read {sig_path:?}: {e}")))?;
    OrgManifest::parse_and_verify(&body, &sig)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trust::signature::SecretKey;

    fn sample_manifest_body(pubkey_hex: &str) -> String {
        format!(
            r#"[org]
name = "Sample Lab"
key = "ed25519:{pubkey_hex}"
url = "https://lab.example/ptwm-trust.toml"

[[members]]
label = "alice"
pubkey = "ed25519:{member}"
since = "2020-01-01"

[[members]]
label = "bob"
pubkey = "ed25519:{member}"
since = "2020-03-15"
until = "2020-09-30"

[scope]
allowed_capabilities = ["determinism", "hardware_class:cpu"]
allowed_kinds = ["transform", "plane_codec"]
"#,
            member = "22".repeat(32),
        )
    }

    #[test]
    fn parse_and_verify_accepts_valid_signature() {
        let sk = SecretKey::generate();
        let pk_hex = hex::encode(sk.public().0);
        let body = sample_manifest_body(&pk_hex);
        let sig = sk.sign(body.as_bytes());
        let m = OrgManifest::parse_and_verify(body.as_bytes(), &sig).unwrap();
        assert_eq!(m.org.name, "Sample Lab");
        assert_eq!(m.members.len(), 2);
        assert_eq!(m.scope.allowed_capabilities.len(), 2);
    }

    #[test]
    fn parse_and_verify_rejects_tampered_body() {
        let sk = SecretKey::generate();
        let pk_hex = hex::encode(sk.public().0);
        let body = sample_manifest_body(&pk_hex);
        let sig = sk.sign(body.as_bytes());
        let mut tampered = body.into_bytes();
        tampered[0] ^= 0xFF;
        let res = OrgManifest::parse_and_verify(&tampered, &sig);
        assert!(res.is_err());
    }

    #[test]
    fn parse_and_verify_rejects_wrong_key() {
        let sk = SecretKey::generate();
        let other = SecretKey::generate();
        let body = sample_manifest_body(&hex::encode(sk.public().0));
        let sig = other.sign(body.as_bytes());
        let res = OrgManifest::parse_and_verify(body.as_bytes(), &sig);
        assert!(res.is_err());
    }

    #[test]
    fn active_members_filters_expired_until() {
        let sk = SecretKey::generate();
        let body = sample_manifest_body(&hex::encode(sk.public().0));
        let sig = sk.sign(body.as_bytes());
        let m = OrgManifest::parse_and_verify(body.as_bytes(), &sig).unwrap();
        // alice has no until → always active. bob has until=2020-09-30,
        // so today=2020-12-01 → bob filtered out.
        let active = m.active_members("2020-12-01");
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].label, "alice");
        // today=2020-05-01 → both active.
        let active2 = m.active_members("2020-05-01");
        assert_eq!(active2.len(), 2);
    }

    #[test]
    fn load_and_verify_files_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let sk = SecretKey::generate();
        let body = sample_manifest_body(&hex::encode(sk.public().0));
        let sig = sk.sign(body.as_bytes());
        let body_path = dir.path().join("org.toml");
        let sig_path = dir.path().join("org.toml.sig");
        std::fs::write(&body_path, body).unwrap();
        std::fs::write(&sig_path, sig).unwrap();
        let m = load_and_verify_files(&body_path, &sig_path).unwrap();
        assert_eq!(m.org.name, "Sample Lab");
    }
}
