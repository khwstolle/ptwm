//! PTWM policy system: capability checks, native-dep verification,
//! policy-file TOML schema, and the resolver that computes the
//! encode-time candidate pool.
//!
//! This file re-exports the submodules that make up the policy system.

pub mod capability_check;
pub mod file;
pub mod native_deps;
pub mod resolve;

// Re-exports added as each sub-module is populated by subsequent tasks.
pub use capability_check::{CapabilityVerdict, HostPolicy, check, check_with};
pub use file::{AllowSection, IgnoreSection, PolicyFile, RoleOverride};
pub use native_deps::{
    CommandRunner, NativeDep, NativeDepVerdict, RealCommandRunner, RunResult, VendorEntry,
    VendorSignature, VendorTable, bundled_vendor_table, verify_native_dep, verify_native_dep_with,
};
pub use resolve::{ResolvedPolicy, resolve};
