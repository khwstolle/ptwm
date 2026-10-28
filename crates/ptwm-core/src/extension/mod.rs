//! PTWM v1 extension system: contributions, manifests, capabilities,
//! lifecycle, and the per-file Extension Table that ties them to the
//! on-disk wire format.

pub mod builder;
pub mod builtins;
pub mod capability;
pub mod compact_table;
pub mod dispatch;
pub mod error;
pub mod id;
pub mod kind;
pub mod lifecycle;
pub mod manifest;
pub mod resolve;
pub mod table;

pub use builder::ExtensionTableBuilder;
pub use builtins::{
    BUILTIN_LABEL_PREFIX, BuiltinRef, BuiltinRegistry, CURRENT_BUILTIN_VERSION, builtin_entries,
    builtin_entry, builtin_entry_at_version, builtin_registry, is_builtin,
    is_valid_builtin_version, normalize_builtin_ids,
};
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
/// built-in IDs change with each release. Files from other releases are
/// mapped to these ids by [`normalize_builtin_ids`].
pub fn builtin_canonical_id(name: &str) -> CanonicalId {
    builtins::current_builtin_id(name)
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
