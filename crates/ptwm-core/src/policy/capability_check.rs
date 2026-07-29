//! Capability verification against a HostPolicy.
//!
//! Given an ExtensionTableEntry and a HostPolicy, decide whether the
//! entry's declared capabilities are admitted.

use crate::extension::{ExtensionTableEntry, capability::CapabilityValue};

use super::native_deps::{
    CommandRunner, NativeDep, NativeDepVerdict, RealCommandRunner, RunResult, VendorTable,
    verify_native_dep_with,
};

/// Minimal HostPolicy stub used by capability_check::check.
///
/// `policy::resolve::ResolvedPolicy` owns the canonical version of this
/// and feeds it in; this stand-alone struct lets tests here run without
/// needing the file/resolver machinery.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct HostPolicy {
    #[serde(default)]
    pub allow_host_imports: Vec<String>,
    #[serde(default)]
    pub deny_capabilities: Vec<String>,
    #[serde(default)]
    pub available_hardware: Vec<String>,
    #[serde(default)]
    pub denied_sandbox_classes: Vec<String>,
    #[serde(default)]
    pub allow_unknown_capabilities: Vec<String>,
    #[serde(default)]
    pub allow_unverified_native_deps: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CapabilityVerdict {
    Admitted,
    Denied { reason: String },
}

pub fn check(
    entry: &ExtensionTableEntry,
    policy: &HostPolicy,
    table: &VendorTable,
) -> CapabilityVerdict {
    check_with(entry, policy, table, &RealCommandRunner)
}

pub fn check_with(
    entry: &ExtensionTableEntry,
    policy: &HostPolicy,
    table: &VendorTable,
    runner: &dyn CommandRunner,
) -> CapabilityVerdict {
    let caps = &entry.capabilities;

    // 1. Deny explicitly-denied capability keys.
    for k in caps.0.keys() {
        if policy.deny_capabilities.contains(k) {
            return CapabilityVerdict::Denied {
                reason: format!("capability '{k}' explicitly denied by policy"),
            };
        }
    }

    // 2. Unknown keys → deny unless explicitly allowed.
    for key in caps.unknown_keys() {
        if !policy.allow_unknown_capabilities.contains(&key.to_string()) {
            return CapabilityVerdict::Denied {
                reason: format!("unknown capability key: {key}"),
            };
        }
    }

    // 3. host_imports: each declared import must be permitted.
    if let Some(CapabilityValue::List(imports)) = caps.get("host_imports") {
        for imp in imports {
            let CapabilityValue::Text(name) = imp else {
                continue;
            };
            if !policy.allow_host_imports.contains(name) {
                return CapabilityVerdict::Denied {
                    reason: format!("host import '{name}' not permitted"),
                };
            }
        }
    }

    // 4. hardware_class.
    if let Some(CapabilityValue::Text(class)) = caps.get("hardware_class") {
        if !policy.available_hardware.contains(class) {
            return CapabilityVerdict::Denied {
                reason: format!("hardware class '{class}' not available"),
            };
        }
    }

    // 5. sandbox_class: explicit deny list.
    if let Some(CapabilityValue::Text(class)) = caps.get("sandbox_class") {
        if policy.denied_sandbox_classes.contains(class) {
            return CapabilityVerdict::Denied {
                reason: format!("sandbox class '{class}' explicitly denied"),
            };
        }
    }

    // 6. native_deps: each declared dep verified by OS package check.
    if let Some(CapabilityValue::List(deps)) = caps.get("native_deps") {
        for dep_entry in deps {
            let dep = match parse_native_dep(dep_entry) {
                Some(d) => d,
                None => {
                    return CapabilityVerdict::Denied {
                        reason: "malformed native_deps entry".into(),
                    };
                }
            };
            match verify_native_dep_with(&dep, runner, table) {
                NativeDepVerdict::Verified => {}
                NativeDepVerdict::Unverified { reason } => {
                    if !policy.allow_unverified_native_deps {
                        return CapabilityVerdict::Denied {
                            reason: format!("native dep '{}' unverified: {reason}", dep.name),
                        };
                    }
                }
                NativeDepVerdict::PlatformUnsupported => {
                    if !policy.allow_unverified_native_deps {
                        return CapabilityVerdict::Denied {
                            reason: format!(
                                "native dep '{}' not verifiable on this platform",
                                dep.name
                            ),
                        };
                    }
                }
            }
        }
    }

    CapabilityVerdict::Admitted
}

/// Parse a NativeDep from a CapabilityValue::Map.
fn parse_native_dep(value: &CapabilityValue) -> Option<NativeDep> {
    use crate::policy::native_deps::VendorSignature;
    let CapabilityValue::Map(m) = value else {
        return None;
    };
    let name = match m.get("name")? {
        CapabilityValue::Text(s) => s.clone(),
        _ => return None,
    };
    let version_constraint = match m.get("version_constraint") {
        Some(CapabilityValue::Text(s)) => s.clone(),
        _ => "*".to_string(),
    };
    let vendor_signature = match m.get("vendor_signature") {
        Some(CapabilityValue::Text(s)) => match s.as_str() {
            "any" => VendorSignature::Any,
            _ => return None,
        },
        Some(CapabilityValue::Map(vm)) => {
            if let Some(CapabilityValue::Text(name)) = vm.get("vendor") {
                VendorSignature::Vendor(name.clone())
            } else if let Some(CapabilityValue::Text(fp)) = vm.get("key_fingerprint") {
                VendorSignature::KeyFingerprint(fp.clone())
            } else {
                return None;
            }
        }
        None => VendorSignature::Any,
        _ => return None,
    };
    Some(NativeDep {
        name,
        version_constraint,
        vendor_signature,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extension::{
        Attestation, CanonicalId, CapabilityMap, ExtensionTableEntry, Kind, Lifecycle,
        capability::CapabilityValue, table::FLAVOR_WASM,
    };

    fn entry_with_caps(caps: CapabilityMap) -> ExtensionTableEntry {
        ExtensionTableEntry {
            canonical_id: CanonicalId::from_bytes([0xAA; 32]),
            human_label: "io.example.test".into(),
            kind: Kind::PlaneCodec,
            abi_version: 1,
            lifecycle: Lifecycle::Thread,
            flavor_hints: FLAVOR_WASM,
            capabilities: caps,
            attestation: Attestation::PgpSignature(Vec::new()),
            install_hint: None,
            embedded_wasm_offset: None,
            embedded_wasm_length: None,
        }
    }

    #[test]
    fn admits_empty_capabilities() {
        let entry = entry_with_caps(CapabilityMap::new());
        let policy = HostPolicy::default();
        let v = check(&entry, &policy, &VendorTable::default());
        assert_eq!(v, CapabilityVerdict::Admitted);
    }

    #[test]
    fn unknown_key_denied_by_default() {
        let mut caps = CapabilityMap::new();
        caps.set("future_thing", CapabilityValue::Int(42));
        let entry = entry_with_caps(caps);
        let policy = HostPolicy::default();
        match check(&entry, &policy, &VendorTable::default()) {
            CapabilityVerdict::Denied { reason } => assert!(reason.contains("future_thing")),
            _ => panic!("expected Denied"),
        }
    }

    #[test]
    fn unknown_key_admitted_when_explicitly_allowed() {
        let mut caps = CapabilityMap::new();
        caps.set("future_thing", CapabilityValue::Int(42));
        let entry = entry_with_caps(caps);
        let mut policy = HostPolicy::default();
        policy
            .allow_unknown_capabilities
            .push("future_thing".into());
        let v = check(&entry, &policy, &VendorTable::default());
        assert_eq!(v, CapabilityVerdict::Admitted);
    }

    #[test]
    fn host_import_denied_when_not_allowed() {
        let mut caps = CapabilityMap::new();
        caps.set(
            "host_imports",
            CapabilityValue::List(vec![CapabilityValue::Text("clock".into())]),
        );
        let entry = entry_with_caps(caps);
        let policy = HostPolicy::default();
        match check(&entry, &policy, &VendorTable::default()) {
            CapabilityVerdict::Denied { reason } => assert!(reason.contains("clock")),
            _ => panic!("expected Denied"),
        }
    }

    #[test]
    fn hardware_class_must_match_available() {
        let mut caps = CapabilityMap::new();
        caps.set("hardware_class", CapabilityValue::Text("cuda".into()));
        let entry = entry_with_caps(caps);
        let mut policy = HostPolicy::default();
        policy.available_hardware.push("cpu".into());
        match check(&entry, &policy, &VendorTable::default()) {
            CapabilityVerdict::Denied { reason } => assert!(reason.contains("cuda")),
            _ => panic!("expected Denied"),
        }
    }

    #[test]
    fn sandbox_class_explicit_deny() {
        let mut caps = CapabilityMap::new();
        caps.set("sandbox_class", CapabilityValue::Text("network".into()));
        let entry = entry_with_caps(caps);
        let mut policy = HostPolicy::default();
        policy.denied_sandbox_classes.push("network".into());
        match check(&entry, &policy, &VendorTable::default()) {
            CapabilityVerdict::Denied { reason } => assert!(reason.contains("network")),
            _ => panic!("expected Denied"),
        }
    }

    #[test]
    fn deny_capabilities_takes_precedence() {
        let mut caps = CapabilityMap::new();
        caps.set("determinism", CapabilityValue::Bool(true));
        let entry = entry_with_caps(caps);
        let mut policy = HostPolicy::default();
        policy.deny_capabilities.push("determinism".into());
        match check(&entry, &policy, &VendorTable::default()) {
            CapabilityVerdict::Denied { reason } => assert!(reason.contains("determinism")),
            _ => panic!("expected Denied"),
        }
    }

    /// Fake CommandRunner: programmable responses.
    struct FakeRunner {
        which_returns: std::collections::HashMap<String, Option<std::path::PathBuf>>,
        run_returns: std::collections::HashMap<(String, Vec<String>), RunResult>,
    }

    impl FakeRunner {
        fn new() -> Self {
            Self {
                which_returns: Default::default(),
                run_returns: Default::default(),
            }
        }
        fn with_which(mut self, prog: &str, path: Option<&str>) -> Self {
            self.which_returns
                .insert(prog.to_string(), path.map(std::path::PathBuf::from));
            self
        }
        fn with_run(mut self, prog: &str, args: &[&str], result: RunResult) -> Self {
            self.run_returns.insert(
                (
                    prog.to_string(),
                    args.iter().map(|s| s.to_string()).collect(),
                ),
                result,
            );
            self
        }
    }

    impl CommandRunner for FakeRunner {
        fn run(&self, program: &str, args: &[&str]) -> std::io::Result<RunResult> {
            let key = (
                program.to_string(),
                args.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            );
            self.run_returns
                .get(&key)
                .map(|r| RunResult {
                    status_success: r.status_success,
                    stdout: r.stdout.clone(),
                    stderr: r.stderr.clone(),
                })
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("fake runner has no entry for {program} {args:?}"),
                    )
                })
        }
        fn which(&self, program: &str) -> Option<std::path::PathBuf> {
            self.which_returns.get(program).cloned().flatten()
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn hardware_backend_cuda_capability_admitted_when_libcuda_present() {
        let runner = FakeRunner::new()
            .with_which("dpkg", Some("/usr/bin/dpkg"))
            .with_run(
                "dpkg-query",
                &["-S", "libcuda.so*"],
                RunResult {
                    status_success: true,
                    stdout: "libcuda1: /usr/lib/x86_64-linux-gnu/libcuda.so.1\n".into(),
                    stderr: String::new(),
                },
            )
            .with_run(
                "dpkg",
                &["--verify", "libcuda1"],
                RunResult {
                    status_success: true,
                    stdout: String::new(),
                    stderr: String::new(),
                },
            );

        let mut caps = CapabilityMap::new();
        caps.set("hardware_class", CapabilityValue::Text("cuda".into()));
        caps.set(
            "native_deps",
            CapabilityValue::List(vec![CapabilityValue::Map({
                let mut m = std::collections::BTreeMap::new();
                m.insert("name".to_string(), CapabilityValue::Text("cuda".into()));
                m.insert(
                    "version_constraint".to_string(),
                    CapabilityValue::Text(">=12.0".into()),
                );
                m
            })]),
        );
        let entry = entry_with_caps(caps);

        let mut policy = HostPolicy::default();
        policy.available_hardware.push("cuda".into());

        let verdict = check_with(&entry, &policy, &VendorTable::default(), &runner);
        assert!(
            matches!(verdict, CapabilityVerdict::Admitted),
            "got: {verdict:?}"
        );
    }

    #[test]
    fn hardware_backend_cuda_capability_denied_when_hardware_class_not_in_policy() {
        let mut caps = CapabilityMap::new();
        caps.set("hardware_class", CapabilityValue::Text("cuda".into()));
        let entry = entry_with_caps(caps);
        let mut policy = HostPolicy::default();
        policy.available_hardware.push("cpu".into());
        match check(&entry, &policy, &VendorTable::default()) {
            CapabilityVerdict::Denied { reason } => assert!(reason.contains("cuda")),
            other => panic!("expected Denied, got {other:?}"),
        }
    }
}
