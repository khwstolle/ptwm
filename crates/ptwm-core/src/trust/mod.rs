//! PTWM trust system: Ed25519 signatures, bundled-then-pinned keyring,
//! org-trust-manifest delegation, and the Verifier that decides whether
//! an ExtensionTableEntry is trusted.
//!
//! This file re-exports the submodules that make up the trust system.

pub mod bundled;
pub mod keyring;
pub mod org;
pub mod signature;
pub mod verifier;

// Re-exports added as each sub-module is populated by subsequent tasks.
pub use bundled::{
    BundleStatus, KeyringDiff, accept_update, apply_fresh_install, bundled_hash, bundled_keyring,
    evaluate_bundled, lock_path,
};
pub use keyring::{Keyring, TrustEntry};
pub use org::{
    OrgHeader, OrgManifest, OrgMember, OrgScope, fetch_and_verify, load_and_verify_files,
};
pub use signature::{PublicKey, SecretKey};
pub use verifier::{TrustVerdict, Verifier};
