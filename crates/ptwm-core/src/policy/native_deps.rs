//! Verify declared native-library dependencies against the OS package
//! manager. Linux: dpkg + rpm. macOS: codesign. Other platforms: fall
//! through to Unverified-PlatformUnsupported.

use std::path::PathBuf;
use std::process::Command;

use serde::{Deserialize, Serialize};

use crate::extension::ExtensionError;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NativeDep {
    pub name: String,
    pub version_constraint: String,
    pub vendor_signature: VendorSignature,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VendorSignature {
    /// Library merely needs to be installed.
    Any,
    /// Match against a vendor name (e.g. "nvidia"). Looked up in
    /// vendor_table.toml.
    Vendor(String),
    /// Pinned fingerprint (e.g., a specific package-signing key blake3
    /// or an Apple team identifier).
    KeyFingerprint(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NativeDepVerdict {
    Verified,
    Unverified { reason: String },
    PlatformUnsupported,
}

/// Abstraction over OS commands so tests can inject fakes.
pub trait CommandRunner: Send + Sync {
    fn run(&self, program: &str, args: &[&str]) -> std::io::Result<RunResult>;
    fn which(&self, program: &str) -> Option<PathBuf>;
}

pub struct RunResult {
    pub status_success: bool,
    pub stdout: String,
    pub stderr: String,
}

pub struct RealCommandRunner;

impl CommandRunner for RealCommandRunner {
    fn run(&self, program: &str, args: &[&str]) -> std::io::Result<RunResult> {
        let out = Command::new(program).args(args).output()?;
        Ok(RunResult {
            status_success: out.status.success(),
            stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        })
    }
    fn which(&self, program: &str) -> Option<PathBuf> {
        which::which(program).ok()
    }
}

/// Vendor table — maps vendor name to (linux dpkg pkg names, linux rpm
/// pkg names, macos team identifiers). Loaded from the bundled
/// vendor_table.toml.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct VendorTable {
    #[serde(default, flatten)]
    pub entries: std::collections::BTreeMap<String, VendorEntry>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct VendorEntry {
    #[serde(default)]
    pub linux_dpkg_keys: Vec<String>,
    #[serde(default)]
    pub linux_rpm_keys: Vec<String>,
    #[serde(default)]
    pub macos_team_ids: Vec<String>,
}

pub const BUNDLED_VENDOR_TABLE_TOML: &str = include_str!("../../data/vendor_table.toml");

pub fn bundled_vendor_table() -> Result<VendorTable, ExtensionError> {
    toml::from_str(BUNDLED_VENDOR_TABLE_TOML)
        .map_err(|e| ExtensionError::ManifestParse(format!("vendor_table.toml: {e}")))
}

/// Verify a single native-dep declaration.
// `return` inside cfg-guarded blocks is intentional: each block covers one
// platform branch and `return` keeps the function well-formed on all targets.
#[allow(clippy::needless_return)]
pub fn verify_native_dep_with(
    dep: &NativeDep,
    runner: &dyn CommandRunner,
    table: &VendorTable,
) -> NativeDepVerdict {
    #[cfg(target_os = "linux")]
    {
        return verify_linux(dep, runner, table);
    }
    #[cfg(target_os = "macos")]
    {
        return verify_macos(dep, runner, table);
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = (dep, runner, table);
        NativeDepVerdict::PlatformUnsupported
    }
}

pub fn verify_native_dep(dep: &NativeDep, table: &VendorTable) -> NativeDepVerdict {
    verify_native_dep_with(dep, &RealCommandRunner, table)
}

#[cfg(target_os = "linux")]
fn verify_linux(
    dep: &NativeDep,
    runner: &dyn CommandRunner,
    table: &VendorTable,
) -> NativeDepVerdict {
    // Try dpkg first, then rpm.
    if runner.which("dpkg").is_some() {
        // dpkg-query -S "lib<name>.so*" → "package: /path/to/lib"
        let pattern = format!("lib{}.so*", dep.name);
        let q = match runner.run("dpkg-query", &["-S", &pattern]) {
            Ok(r) if r.status_success => r,
            Ok(_) => {
                return NativeDepVerdict::Unverified {
                    reason: format!("dpkg-query found no owner for {pattern}"),
                };
            }
            Err(e) => {
                return NativeDepVerdict::Unverified {
                    reason: e.to_string(),
                };
            }
        };
        // Parse the first line — package name before ':'.
        let pkg = q
            .stdout
            .lines()
            .next()
            .and_then(|l| l.split_once(':').map(|(p, _)| p.trim().to_string()))
            .unwrap_or_default();
        if pkg.is_empty() {
            return NativeDepVerdict::Unverified {
                reason: "could not parse dpkg-query output".into(),
            };
        }
        // dpkg --verify <pkg> → silent on success
        let v = match runner.run("dpkg", &["--verify", &pkg]) {
            Ok(r) => r,
            Err(e) => {
                return NativeDepVerdict::Unverified {
                    reason: e.to_string(),
                };
            }
        };
        if !v.status_success || !v.stdout.is_empty() {
            return NativeDepVerdict::Unverified {
                reason: format!("dpkg --verify {pkg}: {}", v.stdout),
            };
        }
        // Match vendor.
        return match_vendor_linux(&pkg, dep, table, /*is_dpkg=*/ true);
    }
    if runner.which("rpm").is_some() {
        // rpm -q --whatprovides "lib<name>.so*" → owning package name
        let provides = format!("lib{}.so*", dep.name);
        let qf = match runner.run("rpm", &["-q", "--whatprovides", &provides]) {
            Ok(r) if r.status_success => r,
            Ok(_) => {
                return NativeDepVerdict::Unverified {
                    reason: format!("rpm --whatprovides found no owner for {provides}"),
                };
            }
            Err(e) => {
                return NativeDepVerdict::Unverified {
                    reason: e.to_string(),
                };
            }
        };
        let pkg = qf
            .stdout
            .lines()
            .next()
            .unwrap_or_default()
            .trim()
            .to_string();
        if pkg.is_empty() {
            return NativeDepVerdict::Unverified {
                reason: "could not parse rpm output".into(),
            };
        }
        let v = match runner.run("rpm", &["-V", &pkg]) {
            Ok(r) => r,
            Err(e) => {
                return NativeDepVerdict::Unverified {
                    reason: e.to_string(),
                };
            }
        };
        if !v.status_success || !v.stdout.is_empty() {
            return NativeDepVerdict::Unverified {
                reason: format!("rpm -V {pkg}: {}", v.stdout),
            };
        }
        return match_vendor_linux(&pkg, dep, table, /*is_dpkg=*/ false);
    }
    NativeDepVerdict::PlatformUnsupported
}

#[cfg(target_os = "linux")]
fn match_vendor_linux(
    pkg: &str,
    dep: &NativeDep,
    table: &VendorTable,
    is_dpkg: bool,
) -> NativeDepVerdict {
    match &dep.vendor_signature {
        VendorSignature::Any => NativeDepVerdict::Verified,
        VendorSignature::Vendor(name) => {
            let entry = match table.entries.get(name) {
                Some(e) => e,
                None => {
                    return NativeDepVerdict::Unverified {
                        reason: format!("vendor '{name}' not in vendor_table"),
                    };
                }
            };
            let candidates = if is_dpkg {
                &entry.linux_dpkg_keys
            } else {
                &entry.linux_rpm_keys
            };
            if candidates
                .iter()
                .any(|c| pkg.contains(c.as_str()) || c == pkg)
            {
                NativeDepVerdict::Verified
            } else {
                NativeDepVerdict::Unverified {
                    reason: format!("pkg {pkg} not associated with vendor {name}"),
                }
            }
        }
        VendorSignature::KeyFingerprint(_) => {
            // Out of scope for v1 — distros don't surface per-package
            // signing-key fingerprints uniformly. Fall through.
            NativeDepVerdict::PlatformUnsupported
        }
    }
}

#[cfg(target_os = "macos")]
fn verify_macos(
    dep: &NativeDep,
    runner: &dyn CommandRunner,
    table: &VendorTable,
) -> NativeDepVerdict {
    // Try common framework paths for the library.
    let candidate_paths = vec![
        PathBuf::from(format!("/usr/lib/lib{}.dylib", dep.name)),
        PathBuf::from(format!("/Library/Frameworks/{}.framework", dep.name)),
        PathBuf::from(format!("/System/Library/Frameworks/{}.framework", dep.name)),
    ];
    let path = match candidate_paths.iter().find(|p| p.exists()) {
        Some(p) => p.clone(),
        None => {
            return NativeDepVerdict::Unverified {
                reason: format!("library {} not found in standard paths", dep.name),
            };
        }
    };
    let r = match runner.run(
        "codesign",
        &["--display", "--verbose=2", &path.to_string_lossy()],
    ) {
        Ok(r) => r,
        Err(e) => {
            return NativeDepVerdict::Unverified {
                reason: e.to_string(),
            };
        }
    };
    // codesign emits "TeamIdentifier=ABCDEFGHIJ" on stderr.
    let team_id = r
        .stderr
        .lines()
        .find_map(|l| l.strip_prefix("TeamIdentifier="))
        .map(str::to_string);
    match &dep.vendor_signature {
        VendorSignature::Any => NativeDepVerdict::Verified,
        VendorSignature::Vendor(name) => {
            let entry = match table.entries.get(name) {
                Some(e) => e,
                None => {
                    return NativeDepVerdict::Unverified {
                        reason: format!("vendor '{name}' not in vendor_table"),
                    };
                }
            };
            match team_id {
                Some(tid) if entry.macos_team_ids.contains(&tid) => NativeDepVerdict::Verified,
                Some(tid) => NativeDepVerdict::Unverified {
                    reason: format!("team {tid} not associated with vendor {name}"),
                },
                None => NativeDepVerdict::Unverified {
                    reason: format!(
                        "no TeamIdentifier emitted by codesign for {}",
                        path.display()
                    ),
                },
            }
        }
        VendorSignature::KeyFingerprint(fp) => {
            if team_id.as_deref() == Some(fp.as_str()) {
                NativeDepVerdict::Verified
            } else {
                NativeDepVerdict::Unverified {
                    reason: format!("team_id mismatch (got {team_id:?}, want {fp})"),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fake CommandRunner — programmable responses.
    struct FakeRunner {
        which_returns: std::collections::HashMap<String, Option<PathBuf>>,
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
                .insert(prog.to_string(), path.map(PathBuf::from));
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
        fn which(&self, program: &str) -> Option<PathBuf> {
            self.which_returns.get(program).cloned().flatten()
        }
    }

    #[test]
    fn bundled_vendor_table_parses() {
        let _ = bundled_vendor_table().unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_any_vendor_verifies_when_dpkg_clean() {
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
        let dep = NativeDep {
            name: "cuda".into(),
            version_constraint: ">=12.0".into(),
            vendor_signature: VendorSignature::Any,
        };
        let table = VendorTable::default();
        let v = verify_native_dep_with(&dep, &runner, &table);
        assert_eq!(v, NativeDepVerdict::Verified);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_vendor_mismatch_yields_unverified() {
        let runner = FakeRunner::new()
            .with_which("dpkg", Some("/usr/bin/dpkg"))
            .with_run(
                "dpkg-query",
                &["-S", "libcuda.so*"],
                RunResult {
                    status_success: true,
                    stdout: "evil-cuda-clone: /opt/evil/libcuda.so.1\n".into(),
                    stderr: String::new(),
                },
            )
            .with_run(
                "dpkg",
                &["--verify", "evil-cuda-clone"],
                RunResult {
                    status_success: true,
                    stdout: String::new(),
                    stderr: String::new(),
                },
            );
        let dep = NativeDep {
            name: "cuda".into(),
            version_constraint: ">=12.0".into(),
            vendor_signature: VendorSignature::Vendor("nvidia".into()),
        };
        let table = VendorTable {
            entries: std::collections::BTreeMap::from([(
                "nvidia".into(),
                VendorEntry {
                    linux_dpkg_keys: vec!["libcuda1".into(), "cuda-12-0".into()],
                    ..Default::default()
                },
            )]),
        };
        match verify_native_dep_with(&dep, &runner, &table) {
            NativeDepVerdict::Unverified { reason } => {
                assert!(reason.contains("evil-cuda-clone"));
            }
            v => panic!("expected Unverified, got {v:?}"),
        }
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    #[test]
    fn unknown_platform_returns_platform_unsupported() {
        let dep = NativeDep {
            name: "foo".into(),
            version_constraint: "*".into(),
            vendor_signature: VendorSignature::Any,
        };
        let v = verify_native_dep_with(&dep, &RealCommandRunner, &VendorTable::default());
        assert_eq!(v, NativeDepVerdict::PlatformUnsupported);
    }
}
