//! Resolve a user-supplied codec selector to a `CanonicalId`.
//!
//! Selectors are a convenience for humans at the call site. The container
//! records the resolved `CanonicalId`, so renaming a contribution can
//! never invalidate an existing archive.

use thiserror::Error;

use crate::discovery::DiscoveredContribution;
use crate::extension::{CanonicalId, builtin_canonical_id};

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ResolveError {
    #[error("no codec named or identified by {selector:?}")]
    NotFound { selector: String },

    #[error(
        "codec name {selector:?} is ambiguous across {} contributions: {}; \
         select by canonical id instead",
        candidates.len(),
        candidates.join(", ")
    )]
    Ambiguous {
        selector: String,
        candidates: Vec<String>,
    },
}

fn looks_like_canonical_id(s: &str) -> bool {
    let body = s.strip_prefix("blake3:").unwrap_or(s);
    body.len() == 64 && body.chars().all(|c| c.is_ascii_hexdigit())
}

pub fn resolve_codec_selector(
    selector: &str,
    installed: &[DiscoveredContribution],
) -> Result<CanonicalId, ResolveError> {
    if looks_like_canonical_id(selector) {
        let to_parse = if selector.starts_with("blake3:") {
            selector.to_string()
        } else {
            format!("blake3:{}", selector)
        };
        return CanonicalId::parse(&to_parse).map_err(|_| ResolveError::NotFound {
            selector: selector.to_string(),
        });
    }

    let mut hits: Vec<CanonicalId> = Vec::new();
    if crate::codec::REGISTRY
        .iter()
        .any(|(name, _)| *name == selector)
    {
        hits.push(builtin_canonical_id(selector));
    }
    for d in installed {
        for c in &d.manifest.contributions {
            if c.label == selector {
                if let Ok(id) = CanonicalId::parse(&c.id) {
                    hits.push(id);
                }
            }
        }
    }

    match hits.len() {
        0 => Err(ResolveError::NotFound {
            selector: selector.to_string(),
        }),
        1 => Ok(hits[0]),
        _ => Err(ResolveError::Ambiguous {
            selector: selector.to_string(),
            candidates: hits.iter().map(|id| id.to_string()).collect(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_contributions_labelled(label: &str) -> Vec<DiscoveredContribution> {
        use crate::extension::manifest::{BundleHeader, ContributionDecl, Manifest};
        use crate::extension::{Kind, Lifecycle};
        use std::path::PathBuf;

        let manifest1 = Manifest {
            bundle: BundleHeader {
                name: "bundle1".to_string(),
                version: "1.0.0".to_string(),
                author_pubkey:
                    "ed25519:0000000000000000000000000000000000000000000000000000000000000000"
                        .to_string(),
                description: None,
            },
            contributions: vec![ContributionDecl {
                id: crate::extension::builtin_canonical_id("huffman").to_string(),
                label: label.to_string(),
                kind: Kind::PlaneCodec,
                abi_version: 1,
                lifecycle: Lifecycle::Thread,
                flavors: vec![],
                capabilities: Default::default(),
                install_hint: None,
            }],
        };

        let manifest2 = Manifest {
            bundle: BundleHeader {
                name: "bundle2".to_string(),
                version: "1.0.0".to_string(),
                author_pubkey:
                    "ed25519:0000000000000000000000000000000000000000000000000000000000000000"
                        .to_string(),
                description: None,
            },
            contributions: vec![ContributionDecl {
                id: crate::extension::builtin_canonical_id("zstd").to_string(),
                label: label.to_string(),
                kind: Kind::PlaneCodec,
                abi_version: 1,
                lifecycle: Lifecycle::Thread,
                flavors: vec![],
                capabilities: Default::default(),
                install_hint: None,
            }],
        };

        vec![
            DiscoveredContribution {
                manifest: manifest1,
                manifest_path: PathBuf::from("/tmp/manifest1.toml"),
                bundle_dir: PathBuf::from("/tmp/bundle1"),
                installed_flavors: 0,
            },
            DiscoveredContribution {
                manifest: manifest2,
                manifest_path: PathBuf::from("/tmp/manifest2.toml"),
                bundle_dir: PathBuf::from("/tmp/bundle2"),
                installed_flavors: 0,
            },
        ]
    }

    #[test]
    fn resolves_a_builtin_name() {
        let got = resolve_codec_selector("huffman", &[] as &[DiscoveredContribution])
            .expect("builtin must resolve");
        assert_eq!(got, crate::extension::builtin_canonical_id("huffman"));
    }

    #[test]
    fn resolves_a_canonical_id_verbatim() {
        let id = crate::extension::builtin_canonical_id("zstd");
        let got = resolve_codec_selector(&id.to_string(), &[] as &[DiscoveredContribution])
            .expect("id must resolve");
        assert_eq!(got, id);
    }

    #[test]
    fn resolves_bare_hex_canonical_id() {
        let id = crate::extension::builtin_canonical_id("huffman");
        let id_str = id.to_string();
        let bare_hex = &id_str[7..]; // Skip "blake3:"
        let got = resolve_codec_selector(bare_hex, &[] as &[DiscoveredContribution])
            .expect("bare hex must resolve");
        assert_eq!(got, id);
    }

    #[test]
    fn unknown_name_reports_not_found() {
        match resolve_codec_selector("no_such_codec", &[] as &[DiscoveredContribution]) {
            Err(ResolveError::NotFound { selector }) => assert_eq!(selector, "no_such_codec"),
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[test]
    fn ambiguous_name_names_every_candidate() {
        // Two installed contributions sharing a label must not silently
        // resolve to whichever was discovered first.
        let installed = two_contributions_labelled("gpu_codec");
        match resolve_codec_selector("gpu_codec", &installed) {
            Err(ResolveError::Ambiguous {
                selector,
                candidates,
            }) => {
                assert_eq!(selector, "gpu_codec");
                assert_eq!(candidates.len(), 2, "both candidates must be named");
            }
            other => panic!("expected Ambiguous, got {other:?}"),
        }
    }
}
