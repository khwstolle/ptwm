//! Compose a PolicyFile + trust state into a ResolvedPolicy.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use super::capability_check::HostPolicy;
use super::file::{PolicyFile, RoleOverride};
use crate::extension::{CanonicalId, ExtensionError};

#[derive(Clone, Debug)]
pub struct ResolvedPolicy {
    pub allowed: HashSet<CanonicalId>,
    pub effective_global: HashSet<CanonicalId>,
    pub per_role_pool: BTreeMap<String, HashSet<CanonicalId>>,
    pub host: HostPolicy,
}

impl ResolvedPolicy {
    /// Emit a compact TOML representation of the resolved policy for
    /// inclusion in benchmark / ablation CSVs.
    pub fn to_toml(&self) -> Result<String, ExtensionError> {
        #[derive(serde::Serialize)]
        struct Out<'a> {
            allowed: Vec<String>,
            effective_global: Vec<String>,
            per_role_pool: BTreeMap<&'a String, Vec<String>>,
            host: &'a HostPolicy,
        }

        fn ids_to_strings(set: &HashSet<CanonicalId>) -> Vec<String> {
            let mut v: Vec<String> = set.iter().map(|id| id.to_string()).collect();
            v.sort();
            v
        }

        let out = Out {
            allowed: ids_to_strings(&self.allowed),
            effective_global: ids_to_strings(&self.effective_global),
            per_role_pool: self
                .per_role_pool
                .iter()
                .map(|(role, set)| (role, ids_to_strings(set)))
                .collect(),
            host: &self.host,
        };
        toml::to_string_pretty(&out).map_err(|e| ExtensionError::ManifestParse(e.to_string()))
    }
}

pub fn resolve(
    file: &PolicyFile,
    trusted: &HashSet<CanonicalId>,
    base_dir: &Path,
) -> Result<ResolvedPolicy, ExtensionError> {
    let merged = merge_includes(file, base_dir, &mut HashSet::new())?;

    let mut allowed: HashSet<CanonicalId> = trusted.iter().copied().collect();
    for s in &merged.allow.extra {
        if let Some(id) = parse_canonical_id(s) {
            allowed.insert(id);
        } else {
            return Err(ExtensionError::ManifestParse(format!(
                "allow.extra contains malformed id: {s}"
            )));
        }
    }

    let mut effective_global = allowed.clone();
    for s in &merged.ignore.extensions {
        if let Some(id) = parse_canonical_id(s) {
            effective_global.remove(&id);
        } else {
            return Err(ExtensionError::ManifestParse(format!(
                "ignore.extensions contains malformed id: {s}"
            )));
        }
    }

    let mut per_role_pool = BTreeMap::new();
    for (role, ovr) in &merged.per_role {
        per_role_pool.insert(role.clone(), apply_role(&effective_global, ovr)?);
    }

    Ok(ResolvedPolicy {
        allowed,
        effective_global,
        per_role_pool,
        host: merged.capabilities,
    })
}

fn apply_role(
    base: &HashSet<CanonicalId>,
    ovr: &RoleOverride,
) -> Result<HashSet<CanonicalId>, ExtensionError> {
    let mut s = base.clone();
    for x in &ovr.allow_extra {
        let id = parse_canonical_id(x).ok_or_else(|| {
            ExtensionError::ManifestParse(format!("per_role.allow_extra malformed: {x}"))
        })?;
        s.insert(id);
    }
    for x in &ovr.ignore {
        let id = parse_canonical_id(x).ok_or_else(|| {
            ExtensionError::ManifestParse(format!("per_role.ignore malformed: {x}"))
        })?;
        s.remove(&id);
    }
    Ok(s)
}

fn merge_includes(
    file: &PolicyFile,
    base_dir: &Path,
    seen: &mut HashSet<PathBuf>,
) -> Result<PolicyFile, ExtensionError> {
    let mut merged = file.clone();
    // Drain the includes so we don't recurse forever.
    let includes = std::mem::take(&mut merged.include);
    for inc in includes {
        let path = expand_path(&inc, base_dir);
        let canon = path.canonicalize().unwrap_or_else(|_| path.clone());
        if !seen.insert(canon.clone()) {
            return Err(ExtensionError::ManifestParse(format!(
                "include cycle: {}",
                canon.display()
            )));
        }
        let src = std::fs::read_to_string(&path)
            .map_err(|e| ExtensionError::ManifestParse(format!("read {path:?}: {e}")))?;
        let inc_file = PolicyFile::from_toml(&src)?;
        let inc_dir = path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| base_dir.to_path_buf());
        let resolved_inc = merge_includes(&inc_file, &inc_dir, seen)?;

        // First-wins merge: prepend resolved_inc's fields, leaving outer
        // file's explicit settings to override.
        let mut prepended = resolved_inc.clone();
        prepended.allow.extra.extend(merged.allow.extra);
        prepended.ignore.extensions.extend(merged.ignore.extensions);
        for (k, v) in merged.per_role {
            prepended.per_role.insert(k, v);
        }
        // capabilities: merged's explicit settings win in a coarse way;
        // we overwrite scalars from merged if non-default.
        let mut caps = prepended.capabilities;
        if !merged.capabilities.allow_host_imports.is_empty() {
            caps.allow_host_imports = merged.capabilities.allow_host_imports;
        }
        if !merged.capabilities.deny_capabilities.is_empty() {
            caps.deny_capabilities = merged.capabilities.deny_capabilities;
        }
        if !merged.capabilities.available_hardware.is_empty() {
            caps.available_hardware = merged.capabilities.available_hardware;
        }
        if !merged.capabilities.denied_sandbox_classes.is_empty() {
            caps.denied_sandbox_classes = merged.capabilities.denied_sandbox_classes;
        }
        if !merged.capabilities.allow_unknown_capabilities.is_empty() {
            caps.allow_unknown_capabilities = merged.capabilities.allow_unknown_capabilities;
        }
        if merged.capabilities.allow_unverified_native_deps {
            caps.allow_unverified_native_deps = true;
        }
        prepended.capabilities = caps;
        prepended.allow.extra.sort();
        prepended.allow.extra.dedup();
        prepended.ignore.extensions.sort();
        prepended.ignore.extensions.dedup();
        merged = prepended;
    }
    Ok(merged)
}

fn expand_path(s: &str, base: &Path) -> PathBuf {
    if let Some(rest) = s.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    let p = PathBuf::from(s);
    if p.is_absolute() { p } else { base.join(p) }
}

fn parse_canonical_id(s: &str) -> Option<CanonicalId> {
    let s = s.strip_prefix("blake3:").unwrap_or(s);
    let bytes = hex::decode(s).ok()?;
    if bytes.len() != 32 {
        return None;
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes);
    Some(CanonicalId::from_bytes(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn id_str(byte: u8) -> String {
        format!("blake3:{}", hex::encode([byte; 32]))
    }
    fn id_bytes(byte: u8) -> CanonicalId {
        CanonicalId::from_bytes([byte; 32])
    }

    #[test]
    fn resolves_allow_extra_into_allowed() {
        let mut pf = PolicyFile::default();
        pf.allow.extra.push(id_str(0x11));
        let trusted = HashSet::new();
        let r = resolve(&pf, &trusted, Path::new(".")).unwrap();
        assert!(r.allowed.contains(&id_bytes(0x11)));
    }

    #[test]
    fn ignore_subtracts_from_effective_global() {
        let mut pf = PolicyFile::default();
        pf.allow.extra.push(id_str(0x22));
        pf.allow.extra.push(id_str(0x33));
        pf.ignore.extensions.push(id_str(0x22));
        let r = resolve(&pf, &HashSet::new(), Path::new(".")).unwrap();
        assert!(r.effective_global.contains(&id_bytes(0x33)));
        assert!(!r.effective_global.contains(&id_bytes(0x22)));
    }

    #[test]
    fn per_role_applies_after_global() {
        let mut pf = PolicyFile::default();
        pf.allow.extra.push(id_str(0x44));
        pf.allow.extra.push(id_str(0x55));
        let mut role = RoleOverride::default();
        role.ignore.push(id_str(0x44));
        role.allow_extra.push(id_str(0x66));
        pf.per_role.insert("attention.q".into(), role);
        let r = resolve(&pf, &HashSet::new(), Path::new(".")).unwrap();
        let pool = &r.per_role_pool["attention.q"];
        assert!(pool.contains(&id_bytes(0x55)));
        assert!(pool.contains(&id_bytes(0x66)));
        assert!(!pool.contains(&id_bytes(0x44)));
    }

    #[test]
    fn include_chain_resolves() {
        let dir = tempdir().unwrap();
        let inc_path = dir.path().join("inc.toml");
        std::fs::write(
            &inc_path,
            format!("[allow]\nextra = [{:?}]\n", id_str(0x77),),
        )
        .unwrap();
        let mut pf = PolicyFile::default();
        pf.include.push(inc_path.to_string_lossy().into_owned());
        let r = resolve(&pf, &HashSet::new(), dir.path()).unwrap();
        assert!(r.allowed.contains(&id_bytes(0x77)));
    }

    #[test]
    fn include_cycle_detected() {
        let dir = tempdir().unwrap();
        let a = dir.path().join("a.toml");
        let b = dir.path().join("b.toml");
        std::fs::write(&a, format!("include = [{:?}]\n", b.to_string_lossy())).unwrap();
        std::fs::write(&b, format!("include = [{:?}]\n", a.to_string_lossy())).unwrap();
        let src = std::fs::read_to_string(&a).unwrap();
        let pf = PolicyFile::from_toml(&src).unwrap();
        let res = resolve(&pf, &HashSet::new(), dir.path());
        assert!(res.is_err());
    }

    #[test]
    fn malformed_id_rejected() {
        let mut pf = PolicyFile::default();
        pf.allow.extra.push("not-an-id".into());
        let res = resolve(&pf, &HashSet::new(), Path::new("."));
        assert!(res.is_err());
    }

    #[test]
    fn resolved_to_toml_round_trips_ids() {
        let mut pf = PolicyFile::default();
        pf.allow.extra.push(id_str(0x88));
        let r = resolve(&pf, &HashSet::new(), Path::new(".")).unwrap();
        let s = r.to_toml().unwrap();
        assert!(s.contains(&hex::encode([0x88; 32])));
    }
}
