//! Git source: `git+https://...#<commit-sha>`. Only commit-SHA pins
//! are accepted so that re-installs are reproducible.

use std::path::PathBuf;

use git2::{Oid, Repository, build::CheckoutBuilder};
use tempfile::TempDir;

use crate::extension::ExtensionError;
use crate::install::Source;

pub fn parse_git_source(rest: &str) -> Result<Source, ExtensionError> {
    let (url, sha) = rest
        .rsplit_once('#')
        .ok_or_else(|| ExtensionError::InvalidRef(format!("git source needs '#<sha>': {rest}")))?;
    if sha.len() < 7 || !sha.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(ExtensionError::InvalidRef(format!(
            "git source needs hex commit SHA, got '{sha}'"
        )));
    }
    Ok(Source::Git {
        url: url.to_string(),
        sha: sha.to_string(),
    })
}

pub fn stage_git(url: &str, sha: &str) -> Result<TempDir, ExtensionError> {
    let staging =
        tempfile::tempdir().map_err(|e| ExtensionError::ManifestParse(format!("tempdir: {e}")))?;
    let work = PathBuf::from(staging.path());
    let repo = Repository::clone(url, &work)
        .map_err(|e| ExtensionError::ManifestParse(format!("git clone {url}: {e}")))?;
    let oid = Oid::from_str(sha)
        .map_err(|e| ExtensionError::ManifestParse(format!("bad sha {sha}: {e}")))?;
    repo.set_head_detached(oid)
        .map_err(|e| ExtensionError::ManifestParse(format!("checkout {sha}: {e}")))?;
    repo.checkout_head(Some(CheckoutBuilder::new().force()))
        .map_err(|e| ExtensionError::ManifestParse(format!("checkout HEAD: {e}")))?;
    Ok(staging)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_git_accepts_full_sha() {
        let s = parse_git_source(
            "https://example.com/foo.git#deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        )
        .unwrap();
        match s {
            Source::Git { url, sha } => {
                assert_eq!(url, "https://example.com/foo.git");
                assert_eq!(sha.len(), 40);
            }
            _ => panic!("expected Git"),
        }
    }

    #[test]
    fn parse_git_accepts_short_sha() {
        let s = parse_git_source("https://example.com/foo.git#deadbee").unwrap();
        assert!(matches!(s, Source::Git { .. }));
    }

    #[test]
    fn parse_git_rejects_too_short() {
        let res = parse_git_source("https://example.com/foo.git#abc");
        assert!(res.is_err());
    }

    #[test]
    fn parse_git_rejects_non_hex() {
        let res = parse_git_source("https://example.com/foo.git#main");
        assert!(res.is_err());
    }

    #[test]
    fn parse_git_rejects_missing_sha() {
        let res = parse_git_source("https://example.com/foo.git");
        assert!(res.is_err());
    }
}
