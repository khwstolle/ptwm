//! Pip-package install source. Two paths:
//!
//!   1. If `python -c "import <name>"` already succeeds (or
//!      `importlib.metadata.distribution(name)` succeeds), no-op.
//!   2. Otherwise invoke `python -m pip install <name>` in a subprocess.
//!
//! For both paths, the actual contribution registration happens later
//! at PTWM startup via `python/ptwm/_loader.py::load_all_host_extensions`.
//! The Rust-side `install` doesn't return a bundle directory — pip
//! installs into the site-packages, not into the user extensions dir.

use std::process::Command;

use crate::extension::ExtensionError;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PipInstallOutcome {
    AlreadyInstalled,
    Installed { stdout: String },
}

pub fn install_pip_package(name: &str) -> Result<PipInstallOutcome, ExtensionError> {
    install_pip_package_with("python", name)
}

pub fn install_pip_package_with(
    python: &str,
    name: &str,
) -> Result<PipInstallOutcome, ExtensionError> {
    validate_package_name(name)?;

    if package_is_importable(python, name)? {
        return Ok(PipInstallOutcome::AlreadyInstalled);
    }

    let out = Command::new(python)
        .args(["-m", "pip", "install", "--quiet", name])
        .output()
        .map_err(|e| ExtensionError::ManifestParse(format!("spawn pip install: {e}")))?;

    if !out.status.success() {
        return Err(ExtensionError::ManifestParse(format!(
            "pip install {name} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        )));
    }

    Ok(PipInstallOutcome::Installed {
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
    })
}

fn package_is_importable(python: &str, name: &str) -> Result<bool, ExtensionError> {
    // `validate_package_name` has already guaranteed `name` is a safe
    // identifier (PEP 503 normalized form, no shell metacharacters), so
    // splicing it into the `import` statement is safe.
    let canonical = name.replace('-', "_");
    let out = Command::new(python)
        .args(["-c", &format!("import {canonical}")])
        .output()
        .map_err(|e| ExtensionError::ManifestParse(format!("spawn python: {e}")))?;
    Ok(out.status.success())
}

/// Strict whitelist for pip package names. We accept the PEP 503
/// normalized form — the first character must be alphanumeric or `_`,
/// the rest may also include `.` and `-`. This rejects:
///
/// * empty strings
/// * leading hyphens (`pip install -r evil.txt` argument injection)
/// * shell metacharacters (`;`, `|`, `&`, `` ` ``, `$`, …)
/// * newlines and other control characters (would be appended as a new
///   statement to the `python -c "import ..."` invocation)
/// * spaces (would split into multiple pip arguments)
/// * PEP 508 version specifiers (`>=`, `<`, `==`, `,`, …) — callers
///   that need a version pin should add it explicitly via a separate
///   parameter we don't yet support.
fn validate_package_name(name: &str) -> Result<(), ExtensionError> {
    if name.is_empty() {
        return Err(ExtensionError::ManifestParse(
            "pip package name is empty".into(),
        ));
    }
    let mut chars = name.chars();
    let first = chars.next().unwrap();
    if !(first.is_ascii_alphanumeric() || first == '_') {
        return Err(ExtensionError::ManifestParse(format!(
            "pip package name must start with [A-Za-z0-9_], got: {name}"
        )));
    }
    for c in chars {
        if !(c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '-') {
            return Err(ExtensionError::ManifestParse(format!(
                "pip package name contains unsafe character {c:?}: {name}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unsafe_characters() {
        let res = install_pip_package_with("python", "evil; rm -rf /");
        assert!(res.is_err());
    }

    #[test]
    fn rejects_empty() {
        let res = install_pip_package_with("python", "");
        assert!(res.is_err());
    }

    /// If `python` doesn't exist, the spawn should fail cleanly (not panic).
    #[test]
    fn missing_python_fails_cleanly() {
        let res = install_pip_package_with("/nonexistent/python/binary", "ptwm");
        assert!(res.is_err());
    }

    #[test]
    fn accepts_normal_pep503_names() {
        assert!(validate_package_name("ptwm").is_ok());
        assert!(validate_package_name("numpy").is_ok());
        assert!(validate_package_name("scikit-learn").is_ok());
        assert!(validate_package_name("typing_extensions").is_ok());
        assert!(validate_package_name("torch.distributed").is_ok());
        assert!(validate_package_name("Pillow").is_ok());
        assert!(validate_package_name("foo123").is_ok());
    }

    #[test]
    fn rejects_leading_hyphen_argument_injection() {
        // pip would interpret `-r ...` as a switch (read requirements).
        assert!(validate_package_name("-rfile.txt").is_err());
        assert!(validate_package_name("--index-url=http://evil").is_err());
        assert!(validate_package_name("-c constraints").is_err());
    }

    #[test]
    fn rejects_newlines_and_control_chars() {
        // Newlines would let the attacker terminate `import x` and inject
        // another statement into the `python -c` payload.
        assert!(validate_package_name("foo\nimport os").is_err());
        assert!(validate_package_name("foo\rbar").is_err());
        assert!(validate_package_name("foo\t").is_err());
        assert!(validate_package_name("foo\0bar").is_err());
    }

    #[test]
    fn rejects_shell_metacharacters() {
        for bad in [
            "a;b", "a|b", "a&b", "a`b", "a$b", "a(b", "a)b", "a{b", "a}b", "a*b", "a?b", "a~b",
            "a<b", "a>b", "a b", "a'b", "a\"b", "a\\b",
        ] {
            assert!(
                validate_package_name(bad).is_err(),
                "should reject: {bad:?}"
            );
        }
    }

    #[test]
    fn rejects_pep508_version_specifiers() {
        // Allowed via a future explicit version parameter; not via the
        // bare package-name argument.
        assert!(validate_package_name("foo>=1.0").is_err());
        assert!(validate_package_name("foo==1.0").is_err());
        assert!(validate_package_name("foo<2").is_err());
        assert!(validate_package_name("foo,bar").is_err());
        assert!(validate_package_name("foo[extras]").is_err());
    }
}
