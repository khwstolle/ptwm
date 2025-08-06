//! Host-language (Python) extension descriptors.
//!
//! For Python: the actual contribution lives in user-installed Python
//! code, discovered via `importlib.metadata` entry points under the
//! `ptwm.extensions` group. The Rust side just holds the metadata; the
//! PyO3 bridge in `crates/ptwm-py/src/host_flavor.rs` owns the callable.

use crate::extension::{CanonicalId, Kind, Lifecycle};

#[derive(Clone, Debug)]
pub struct HostExtension {
    pub canonical_id: CanonicalId,
    pub kind: Kind,
    pub lifecycle: Lifecycle,
    pub py_label: String, // e.g. "myorg.foo.HadamardTransform"
}

impl HostExtension {
    pub fn new(
        canonical_id: CanonicalId,
        kind: Kind,
        lifecycle: Lifecycle,
        py_label: String,
    ) -> Self {
        Self {
            canonical_id,
            kind,
            lifecycle,
            py_label,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_extension_holds_metadata() {
        let h = HostExtension::new(
            CanonicalId::from_bytes([0xEE; 32]),
            Kind::Classifier,
            Lifecycle::Process,
            "demo.MyClassifier".into(),
        );
        assert_eq!(h.kind, Kind::Classifier);
        assert_eq!(h.py_label, "demo.MyClassifier");
    }
}
