//! Source-format dispatch for `ptwm ext install`.
//!
//! v1 supports: local directory, local `.tar.zst`, https:// archive,
//! oci:// artifact, git+https#sha, and pip package name. This `mod.rs`
//! exposes the parse + dispatch surface.

pub mod git;
pub mod https;
pub mod local;
pub mod oci;
pub mod pip;

use std::path::{Path, PathBuf};

use crate::extension::ExtensionError;

#[derive(Clone, Debug)]
pub enum Source {
    LocalDir(PathBuf),
    LocalArchive(PathBuf),
    Https(String),
    Oci(String),
    Git { url: String, sha: String },
    Pip(String),
    Bundled,
}

pub fn parse_source(s: &str) -> Result<Source, ExtensionError> {
    if s == "--bundled" {
        return Ok(Source::Bundled);
    }
    if let Some(rest) = s.strip_prefix("oci://") {
        return Ok(Source::Oci(rest.to_string()));
    }
    if s.starts_with("https://") {
        return Ok(Source::Https(s.to_string()));
    }
    if let Some(rest) = s.strip_prefix("git+") {
        return git::parse_git_source(rest);
    }
    let path = Path::new(s);
    if path.is_dir() {
        return Ok(Source::LocalDir(path.to_path_buf()));
    }
    if path.is_file() && s.ends_with(".tar.zst") {
        return Ok(Source::LocalArchive(path.to_path_buf()));
    }
    // Last resort: assume pip package name.
    Ok(Source::Pip(s.to_string()))
}

pub struct InstallResult {
    pub bundle_dir: PathBuf,
}

/// Stage and install a source into the user extensions directory.
///
/// Signature verification is not yet plugged in here.
/// For now: parse manifest in the staged area to validate; copy into
/// `user_extensions_dir() / <author-fp> / <name>@<version>/`.
pub fn install(source: Source) -> Result<InstallResult, ExtensionError> {
    let staging = match &source {
        Source::LocalDir(p) => local::stage_local_dir(p)?,
        Source::LocalArchive(p) => local::stage_local_archive(p)?,
        Source::Https(url) => https::stage_https(url)?,
        Source::Oci(reference) => oci::stage_oci(reference)?,
        Source::Git { url, sha } => git::stage_git(url, sha)?,
        Source::Pip(name) => {
            // pip installs into site-packages, not into the user extensions
            // directory. Return a synthetic InstallResult pointing at the
            // Python interpreter's site-packages — discovery picks it up via
            // Python entry points on next PTWM start.
            pip::install_pip_package(name)?;
            return Ok(InstallResult {
                bundle_dir: PathBuf::new(), // no on-disk bundle dir for pip sources
            });
        }
        Source::Bundled => {
            return Err(ExtensionError::ManifestParse(
                "bundled install not in this task".into(),
            ));
        }
    };

    let manifest_path = staging.path().join("manifest.toml");
    let src = std::fs::read_to_string(&manifest_path)
        .map_err(|e| ExtensionError::ManifestParse(format!("read manifest: {e}")))?;
    let manifest = crate::extension::Manifest::from_toml(&src)?;

    // Compute target directory.
    let fp = author_fingerprint(&manifest.bundle.author_pubkey)?;
    let target_root = crate::discovery::user_extensions_dir();
    let target = target_root.join(&fp).join(format!(
        "{}@{}",
        manifest.bundle.name, manifest.bundle.version
    ));

    if target.exists() {
        std::fs::remove_dir_all(&target)
            .map_err(|e| ExtensionError::ManifestParse(format!("clear target: {e}")))?;
    }
    std::fs::create_dir_all(&target)
        .map_err(|e| ExtensionError::ManifestParse(format!("mkdir target: {e}")))?;

    // Copy staged contents (not the staging dir itself) into target.
    let opts = fs_extra::dir::CopyOptions {
        content_only: true,
        overwrite: true,
        ..Default::default()
    };
    fs_extra::dir::copy(staging.path(), &target, &opts)
        .map_err(|e| ExtensionError::ManifestParse(format!("install copy: {e}")))?;

    Ok(InstallResult { bundle_dir: target })
}

fn author_fingerprint(pubkey_with_prefix: &str) -> Result<String, ExtensionError> {
    let s = pubkey_with_prefix
        .strip_prefix("ed25519:")
        .unwrap_or(pubkey_with_prefix);
    let bytes = hex::decode(s)
        .map_err(|e| ExtensionError::ManifestParse(format!("bad author_pubkey: {e}")))?;
    if bytes.len() != 32 {
        // For built-in / placeholder author_pubkey strings like "00000000"
        // (8 chars), just hash the raw string.
        return Ok(blake3::hash(pubkey_with_prefix.as_bytes())
            .to_hex()
            .to_string()[..16]
            .to_string());
    }
    Ok(blake3::hash(&bytes).to_hex().to_string()[..16].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_https_source() {
        let s = parse_source("https://example.com/foo.tar.zst").unwrap();
        assert!(matches!(s, Source::Https(_)));
    }

    #[test]
    fn parses_oci_source() {
        let s = parse_source("oci://reg.example/foo:1.0").unwrap();
        assert!(matches!(s, Source::Oci(_)));
    }

    #[test]
    fn parses_bundled_marker() {
        assert!(matches!(
            parse_source("--bundled").unwrap(),
            Source::Bundled
        ));
    }

    #[test]
    fn unknown_source_falls_through_to_pip() {
        let s = parse_source("some-pip-pkg").unwrap();
        assert!(matches!(s, Source::Pip(_)));
    }
}
