//! PTWM v1 extension system: contributions, manifests, capabilities,
//! lifecycle, and the per-file Extension Table that ties them to the
//! on-disk wire format.

pub mod builder;
pub mod builtins;
pub mod capability;
pub mod dispatch;
pub mod error;
pub mod id;
pub mod kind;
pub mod lifecycle;
pub mod manifest;
pub mod resolve;
pub mod table;

pub use builder::ExtensionTableBuilder;
pub use builtins::{builtin_entries, builtin_entry, is_builtin};
pub use capability::{Capability, CapabilityMap};
pub use dispatch::{BuiltinKind, dispatch_builtin};
pub use error::ExtensionError;
pub use id::{CanonicalId, ContributionRef};
pub use kind::Kind;
pub use lifecycle::Lifecycle;
pub use manifest::{ContributionDecl, Manifest};
pub use resolve::{ResolveError, resolve_codec_selector};
pub use table::{Attestation, ExtensionTable, ExtensionTableEntry};

/// Sentinel public key for in-tree built-ins. Built-ins are trusted by
/// construction; the verifier short-circuits on this key.
pub const BUILTIN_PUBKEY: [u8; 32] = [0xBB; 32];

/// Compute the canonical id of an in-tree built-in by name.
/// The version is the PTWM crate version (`CARGO_PKG_VERSION`), so
/// built-in IDs are stable per release.
pub fn builtin_canonical_id(name: &str) -> CanonicalId {
    CanonicalId::derive(&BUILTIN_PUBKEY, name, env!("CARGO_PKG_VERSION"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_ids_are_distinct() {
        assert_ne!(
            builtin_canonical_id("huffman"),
            builtin_canonical_id("rans")
        );
    }

    #[test]
    fn builtin_id_is_deterministic() {
        assert_eq!(
            builtin_canonical_id("huffman"),
            builtin_canonical_id("huffman")
        );
    }
}
