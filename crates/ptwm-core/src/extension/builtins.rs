//! Materialize ExtensionTableEntry values for in-tree built-ins, and the
//! static registry that maps built-in canonical ids to their names and
//! dispatch targets.

use std::collections::HashMap;
use std::sync::OnceLock;

use super::capability::CapabilityValue;
use super::dispatch::BuiltinKind;
use super::table::{FLAVOR_HOST, FLAVOR_NATIVE};
use super::{
    Attestation, BUILTIN_PUBKEY, CanonicalId, CapabilityMap, ExtensionTable, ExtensionTableEntry,
    Kind, Lifecycle,
};
use crate::codec::CodecId;
use crate::transforms::op::OpId;

/// Version that the built-in ids and labels of this build derive from.
pub const CURRENT_BUILTIN_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Prefix of every built-in label: `io.ptwm.builtin.<name>@<version>`.
pub const BUILTIN_LABEL_PREFIX: &str = "io.ptwm.builtin.";

/// Name, kind and flavor of every built-in that has an Extension Table
/// entry, in the order of [`builtin_entries`]. Transform names must match
/// [`OpId::builtin_name`].
const BUILTIN_SPECS: &[(&str, Kind, u8)] = &[
    // Plane codecs
    ("identity", Kind::PlaneCodec, FLAVOR_NATIVE),
    ("huffman", Kind::PlaneCodec, FLAVOR_NATIVE),
    ("rans", Kind::PlaneCodec, FLAVOR_NATIVE),
    ("zstd", Kind::PlaneCodec, FLAVOR_NATIVE),
    ("zstd_dict", Kind::PlaneCodec, FLAVOR_NATIVE),
    ("fpc", Kind::PlaneCodec, FLAVOR_NATIVE),
    ("tans", Kind::PlaneCodec, FLAVOR_NATIVE),
    ("order1_scale_ac", Kind::PlaneCodec, FLAVOR_NATIVE),
    ("arithmetic_o0", Kind::PlaneCodec, FLAVOR_NATIVE),
    ("arithmetic_o0_adaptive", Kind::PlaneCodec, FLAVOR_NATIVE),
    ("arithmetic_o1", Kind::PlaneCodec, FLAVOR_NATIVE),
    ("per_group_codebook", Kind::PlaneCodec, FLAVOR_NATIVE),
    ("context_mixing_lite", Kind::PlaneCodec, FLAVOR_NATIVE),
    ("huff_llm_5bit", Kind::PlaneCodec, FLAVOR_NATIVE),
    ("order1_arithmetic", Kind::PlaneCodec, FLAVOR_NATIVE),
    ("neural_predictor", Kind::PlaneCodec, FLAVOR_NATIVE),
    // Transforms
    ("bit_reorder_ieee16", Kind::Transform, FLAVOR_NATIVE),
    ("bit_reorder_ieee32", Kind::Transform, FLAVOR_NATIVE),
    ("bit_reorder_fp8_e4m3", Kind::Transform, FLAVOR_NATIVE),
    ("bit_reorder_fp8_e5m2", Kind::Transform, FLAVOR_NATIVE),
    ("byte_split", Kind::Transform, FLAVOR_NATIVE),
    ("nibble_split", Kind::Transform, FLAVOR_NATIVE),
    ("byte_passthrough", Kind::Transform, FLAVOR_NATIVE),
    ("xor_delta", Kind::Transform, FLAVOR_NATIVE),
    ("int_delta", Kind::Transform, FLAVOR_NATIVE),
    ("predictor_xor", Kind::Transform, FLAVOR_NATIVE),
    ("burrows_wheeler", Kind::Transform, FLAVOR_NATIVE),
    ("move_to_front", Kind::Transform, FLAVOR_NATIVE),
    ("mxfp4_deinterleave", Kind::Transform, FLAVOR_NATIVE),
    ("block_microscaling_repack", Kind::Transform, FLAVOR_NATIVE),
    ("concat", Kind::Transform, FLAVOR_NATIVE),
    ("reshape", Kind::Transform, FLAVOR_NATIVE),
    ("index_bitwidth_pack", Kind::Transform, FLAVOR_NATIVE),
    ("mantissa_zero_strip", Kind::Transform, FLAVOR_NATIVE),
    ("source", Kind::Transform, FLAVOR_NATIVE),
    ("terminal", Kind::Transform, FLAVOR_NATIVE),
    ("spherical_normalize", Kind::Transform, FLAVOR_NATIVE),
    ("alpha_stable_normalize", Kind::Transform, FLAVOR_NATIVE),
    ("float_delta", Kind::Transform, FLAVOR_NATIVE),
    ("entropy_estimate", Kind::Transform, FLAVOR_NATIVE),
    // Encode-only Python-resident contributions (chain builder + classifiers).
    ("production_chains", Kind::ChainBuilder, FLAVOR_HOST),
    ("classifier_heuristic", Kind::Classifier, FLAVOR_HOST),
    ("classifier_hf_quant_config", Kind::Classifier, FLAVOR_HOST),
    ("classifier_ptwm_config", Kind::Classifier, FLAVOR_HOST),
    ("classifier_explicit_flags", Kind::Classifier, FLAVOR_HOST),
];

/// One built-in known to the registry.
#[derive(Debug, Clone, Copy)]
pub struct BuiltinRef {
    /// Built-in name (`io.ptwm.builtin.<name>`).
    pub name: &'static str,
    /// The in-tree op or codec this id dispatches to, if any.
    pub dispatch: Option<BuiltinKind>,
    /// Position in [`builtin_entries`], or `None` for ids that dispatch but
    /// have no Extension Table entry (`huffman_nibble`).
    pub entry_index: Option<usize>,
}

/// Lookup tables over every built-in at [`CURRENT_BUILTIN_VERSION`].
#[derive(Debug)]
pub struct BuiltinRegistry {
    by_id: HashMap<CanonicalId, BuiltinRef>,
    by_name: HashMap<&'static str, CanonicalId>,
    entries: Vec<ExtensionTableEntry>,
}

impl BuiltinRegistry {
    fn build() -> Self {
        let mut by_id: HashMap<CanonicalId, BuiltinRef> = HashMap::new();
        let mut by_name: HashMap<&'static str, CanonicalId> = HashMap::new();
        let mut entries = Vec::with_capacity(BUILTIN_SPECS.len());
        for (i, &(name, kind, flavor)) in BUILTIN_SPECS.iter().enumerate() {
            let id = CanonicalId::derive(&BUILTIN_PUBKEY, name, CURRENT_BUILTIN_VERSION);
            entries.push(make_entry(name, kind, flavor, CURRENT_BUILTIN_VERSION, id));
            by_name.insert(name, id);
            by_id.insert(
                id,
                BuiltinRef {
                    name,
                    dispatch: None,
                    entry_index: Some(i),
                },
            );
        }
        let mut add_dispatch = |name: &'static str, kind: BuiltinKind| {
            let id = CanonicalId::derive(&BUILTIN_PUBKEY, name, CURRENT_BUILTIN_VERSION);
            by_id
                .entry(id)
                .or_insert(BuiltinRef {
                    name,
                    dispatch: None,
                    entry_index: None,
                })
                .dispatch = Some(kind);
        };
        for op in OpId::ALL {
            add_dispatch(op.builtin_name(), BuiltinKind::Op(op));
        }
        for &(name, raw) in crate::codec::REGISTRY {
            if let Some(codec) = CodecId::from_u16(raw) {
                add_dispatch(name, BuiltinKind::Codec(codec));
            }
        }
        Self {
            by_id,
            by_name,
            entries,
        }
    }

    /// The built-in with this canonical id, if any.
    pub fn get(&self, id: &CanonicalId) -> Option<&BuiltinRef> {
        self.by_id.get(id)
    }

    /// Current canonical id of the built-in named `name` that has an
    /// Extension Table entry.
    pub fn id_for_entry_name(&self, name: &str) -> Option<CanonicalId> {
        self.by_name.get(name).copied()
    }
}

/// The process-wide built-in registry, built on first use.
pub fn builtin_registry() -> &'static BuiltinRegistry {
    static REGISTRY: OnceLock<BuiltinRegistry> = OnceLock::new();
    REGISTRY.get_or_init(BuiltinRegistry::build)
}

/// Canonical id of the built-in `name` at the crate version. Uses the
/// registry for known names and derives the id otherwise.
pub(super) fn current_builtin_id(name: &str) -> CanonicalId {
    builtin_registry()
        .by_name
        .get(name)
        .copied()
        // Dispatch-only built-ins (`huffman_nibble`) are not in `by_name`.
        .unwrap_or_else(|| CanonicalId::derive(&BUILTIN_PUBKEY, name, CURRENT_BUILTIN_VERSION))
}

fn make_entry(
    name: &str,
    kind: Kind,
    flavor: u8,
    version: &str,
    canonical_id: CanonicalId,
) -> ExtensionTableEntry {
    let mut caps = CapabilityMap::new();
    caps.set("determinism", CapabilityValue::Bool(true));
    caps.set("hardware_class", CapabilityValue::Text("cpu".into()));

    ExtensionTableEntry {
        canonical_id,
        human_label: format!("{BUILTIN_LABEL_PREFIX}{name}@{version}"),
        kind,
        abi_version: 1,
        lifecycle: Lifecycle::Thread,
        flavor_hints: flavor,
        capabilities: caps,
        // Built-ins are not attested by an external author key. A zero-byte
        // signature marks "no external attestation needed"; the verifier
        // short-circuits when the pubkey is BUILTIN_PUBKEY.
        attestation: Attestation::PgpSignature(Vec::new()),
        install_hint: None,
        embedded_wasm_offset: None,
        embedded_wasm_length: None,
    }
}

pub fn builtin_entry(name: &str, kind: Kind, deterministic: bool) -> ExtensionTableEntry {
    let mut e = make_entry(
        name,
        kind,
        FLAVOR_NATIVE,
        CURRENT_BUILTIN_VERSION,
        current_builtin_id(name),
    );
    e.capabilities
        .set("determinism", CapabilityValue::Bool(deterministic));
    e
}

/// Every built-in Extension Table entry at the crate version.
pub fn builtin_entries() -> Vec<ExtensionTableEntry> {
    builtin_registry().entries.clone()
}

/// The Extension Table entry of the built-in `name` with its id and label
/// derived for `version`, as a writer of that version emits it. Returns
/// `None` if `name` has no built-in entry.
pub fn builtin_entry_at_version(name: &str, version: &str) -> Option<ExtensionTableEntry> {
    let reg = builtin_registry();
    let idx = reg.by_id.get(reg.by_name.get(name)?)?.entry_index?;
    if version == CURRENT_BUILTIN_VERSION {
        return Some(reg.entries[idx].clone());
    }
    let (name, kind, flavor) = BUILTIN_SPECS[idx];
    Some(make_entry(
        name,
        kind,
        flavor,
        version,
        CanonicalId::derive(&BUILTIN_PUBKEY, name, version),
    ))
}

/// Whether `id` is a built-in with an Extension Table entry at the crate
/// version. Ids of other versions are built-in only after
/// [`normalize_builtin_ids`] has mapped them.
pub fn is_builtin(id: &CanonicalId) -> bool {
    builtin_registry()
        .by_id
        .get(id)
        .is_some_and(|r| r.entry_index.is_some())
}

/// Map built-in entries written by another release to the ids and labels of
/// the crate version.
///
/// An entry is changed only when all of these hold:
/// - its label is `io.ptwm.builtin.<name>@<ver>`;
/// - `<name>` has a built-in entry ([`builtin_entries`]);
/// - its id equals `CanonicalId::derive(BUILTIN_PUBKEY, name, ver)`.
///
/// Its id and label are then replaced in memory with the current ones; all
/// other fields stay as read. Entries that fail any check (third-party
/// entries, or an entry with a built-in label and some other id) are left
/// untouched, so they are still treated as non-built-in.
///
/// Returns the built-in version the table was written with: the first
/// version other than the current one, or the current version if every
/// built-in entry already has it. Returns `None` if no entry is a built-in.
/// Writers use one version per table, so a table that mixes versions is
/// not writer output; for such a table the result names one of them.
pub fn normalize_builtin_ids(table: &mut ExtensionTable) -> Option<String> {
    let reg = builtin_registry();
    let mut found_current = false;
    let mut other: Option<String> = None;
    for entry in &mut table.entries {
        let Some(rest) = entry.human_label.strip_prefix(BUILTIN_LABEL_PREFIX) else {
            continue;
        };
        let Some((name, ver)) = rest.split_once('@') else {
            continue;
        };
        let Some(&current_id) = reg.by_name.get(name) else {
            continue;
        };
        if ver == CURRENT_BUILTIN_VERSION {
            found_current |= entry.canonical_id == current_id;
            continue;
        }
        if entry.canonical_id != CanonicalId::derive(&BUILTIN_PUBKEY, name, ver) {
            continue;
        }
        if other.is_none() {
            other = Some(ver.to_owned());
        }
        entry.canonical_id = current_id;
        entry.human_label = format!("{BUILTIN_LABEL_PREFIX}{name}@{CURRENT_BUILTIN_VERSION}");
    }
    other.or_else(|| found_current.then(|| CURRENT_BUILTIN_VERSION.to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extension::builtin_canonical_id;

    fn table_at(version: &str) -> ExtensionTable {
        ExtensionTable {
            entries: ["source", "byte_split", "huffman", "zstd"]
                .iter()
                .map(|n| builtin_entry_at_version(n, version).unwrap())
                .collect(),
        }
    }

    #[test]
    fn registry_covers_every_entry_and_dispatch_target() {
        let reg = builtin_registry();
        for (i, e) in builtin_entries().iter().enumerate() {
            let r = reg.get(&e.canonical_id).expect("entry in registry");
            assert_eq!(r.entry_index, Some(i));
            assert_eq!(
                e.human_label,
                format!("{BUILTIN_LABEL_PREFIX}{}@{CURRENT_BUILTIN_VERSION}", r.name)
            );
            assert_eq!(builtin_canonical_id(r.name), e.canonical_id);
        }
        for op in OpId::ALL {
            assert_eq!(
                reg.get(&op.canonical_id()).and_then(|r| r.dispatch),
                Some(BuiltinKind::Op(op))
            );
        }
        // `huffman_nibble` dispatches but has no entry, so it is not a
        // built-in for `is_builtin`.
        let nibble = builtin_canonical_id("huffman_nibble");
        assert_eq!(
            reg.get(&nibble).and_then(|r| r.dispatch),
            Some(BuiltinKind::Codec(CodecId::HuffmanNibble))
        );
        assert!(!is_builtin(&nibble));
        assert!(builtin_entry_at_version("huffman_nibble", "1.0.0").is_none());
    }

    #[test]
    fn builtin_entry_at_version_derives_id_and_label() {
        let e = builtin_entry_at_version("huffman", "1.0.0").unwrap();
        assert_eq!(e.human_label, "io.ptwm.builtin.huffman@1.0.0");
        assert_eq!(
            e.canonical_id,
            CanonicalId::derive(&BUILTIN_PUBKEY, "huffman", "1.0.0")
        );
        let cur = builtin_entry_at_version("huffman", CURRENT_BUILTIN_VERSION).unwrap();
        assert_eq!(cur, builtin_entries()[1]);
        // Only id and label depend on the version.
        let mut e2 = e.clone();
        e2.canonical_id = cur.canonical_id;
        e2.human_label = cur.human_label.clone();
        assert_eq!(e2, cur);
        // Host-flavored built-ins keep their flavor.
        let host = builtin_entry_at_version("production_chains", "1.0.0").unwrap();
        assert_eq!(host.flavor_hints, FLAVOR_HOST);
        assert!(builtin_entry_at_version("not_a_builtin", "1.0.0").is_none());
    }

    #[test]
    fn ids_of_other_versions_are_builtin_only_after_normalisation() {
        let old = "0.9.0";
        let mut table = table_at(old);
        for e in &table.entries {
            assert!(!is_builtin(&e.canonical_id));
            assert_eq!(crate::extension::dispatch_builtin(&e.canonical_id), None);
        }
        assert_eq!(normalize_builtin_ids(&mut table).as_deref(), Some(old));
        assert_eq!(table, table_at(CURRENT_BUILTIN_VERSION));
        for e in &table.entries {
            assert!(is_builtin(&e.canonical_id));
        }
        assert_eq!(
            crate::extension::dispatch_builtin(&table.entries[2].canonical_id),
            Some(BuiltinKind::Codec(CodecId::Huffman))
        );
        // Normalising again changes nothing and reports the current version.
        assert_eq!(
            normalize_builtin_ids(&mut table).as_deref(),
            Some(CURRENT_BUILTIN_VERSION)
        );
        assert_eq!(table, table_at(CURRENT_BUILTIN_VERSION));
    }

    #[test]
    fn forged_builtin_labels_are_not_normalised() {
        let mut table = table_at("1.0.0");
        // Built-in label, but an id that does not re-derive from it.
        table.entries[0].canonical_id = CanonicalId::from_bytes([0x11; 32]);
        // Built-in label of the current version with a wrong id.
        table.entries[1].human_label =
            format!("io.ptwm.builtin.byte_split@{CURRENT_BUILTIN_VERSION}");
        // Id of one name under the label of another.
        table.entries[2].human_label = "io.ptwm.builtin.zstd@1.0.0".into();
        let before = table.clone();
        let version = normalize_builtin_ids(&mut table);
        assert_eq!(table.entries[..3], before.entries[..3]);
        for e in &table.entries[..3] {
            assert!(!is_builtin(&e.canonical_id));
        }
        // The genuine entry is still mapped.
        assert!(is_builtin(&table.entries[3].canonical_id));
        assert_eq!(version.as_deref(), Some("1.0.0"));
    }

    #[test]
    fn normalisation_ignores_third_party_and_unknown_names() {
        let id = CanonicalId::derive(&[0x22; 32], "huffman", "1.0.0");
        let mut entry = builtin_entry_at_version("huffman", "1.0.0").unwrap();
        entry.canonical_id = id;
        entry.human_label = "com.example.huffman@1.0.0".into();
        let mut unknown = entry.clone();
        unknown.canonical_id = CanonicalId::derive(&BUILTIN_PUBKEY, "nope", "1.0.0");
        unknown.human_label = "io.ptwm.builtin.nope@1.0.0".into();
        let mut no_version = entry.clone();
        no_version.human_label = "io.ptwm.builtin.huffman".into();
        let mut table = ExtensionTable {
            entries: vec![entry, unknown, no_version],
        };
        let before = table.clone();
        assert_eq!(normalize_builtin_ids(&mut table), None);
        assert_eq!(table, before);
        assert_eq!(normalize_builtin_ids(&mut ExtensionTable::empty()), None);
    }

    #[test]
    fn every_builtin_has_unique_id() {
        let entries = builtin_entries();
        let mut ids: Vec<_> = entries.iter().map(|e| e.canonical_id).collect();
        ids.sort_by_key(|id| *id.as_bytes());
        ids.dedup();
        assert_eq!(ids.len(), entries.len(), "duplicate canonical ids");
    }

    #[test]
    fn is_builtin_detects_huffman() {
        let id = builtin_canonical_id("huffman");
        assert!(is_builtin(&id));
        let fake = CanonicalId::from_bytes([0xCC; 32]);
        assert!(!is_builtin(&fake));
    }

    #[test]
    fn every_op_has_a_matching_builtin_entry() {
        // Every OpId variant's canonical_id MUST match an entry in
        // builtin_entries(). Otherwise the writer's interning of a chain
        // referencing that op will succeed only because of the
        // placeholder helper — the *curated* table won't recognize it.
        use crate::transforms::op::OpId;
        for op in [
            OpId::BitReorderIeee16,
            OpId::BitReorderIeee32,
            OpId::BitReorderFp8E4M3,
            OpId::BitReorderFp8E5M2,
            OpId::ByteSplit,
            OpId::NibbleSplit,
            OpId::BytePassthrough,
            OpId::XorDelta,
            OpId::IntDelta,
            OpId::PredictorXor,
            OpId::BurrowsWheeler,
            OpId::MoveToFront,
            OpId::MxFp4Deinterleave,
            OpId::BlockMicroscalingRepack,
            OpId::Concat,
            OpId::Reshape,
            OpId::IndexBitwidthPack,
            OpId::MantissaZeroStrip,
            OpId::Source,
            OpId::Terminal,
            OpId::FloatDelta,
            OpId::EntropyEstimate,
            OpId::SphericalNormalize,
            OpId::AlphaStableNormalize,
        ] {
            let id = op.canonical_id();
            assert!(
                is_builtin(&id),
                "OpId {op:?} canonical_id has no matching builtin entry"
            );
        }
    }

    #[test]
    fn chain_builder_and_classifier_builtins_are_present() {
        let entries = builtin_entries();
        assert_eq!(
            entries
                .iter()
                .filter(|e| e.kind == Kind::ChainBuilder)
                .count(),
            1
        );
        assert_eq!(
            entries
                .iter()
                .filter(|e| e.kind == Kind::Classifier)
                .count(),
            4
        );
        // All host-flavored.
        for e in entries
            .iter()
            .filter(|e| matches!(e.kind, Kind::ChainBuilder | Kind::Classifier))
        {
            assert_eq!(e.flavor_hints, super::super::table::FLAVOR_HOST);
        }
    }

    #[test]
    fn every_plane_codec_has_a_matching_builtin_entry() {
        use crate::codecs::arithmetic::{ArithmeticO0, ArithmeticO0Adaptive, ArithmeticO1};
        use crate::codecs::context_mixing::ContextMixingLite;
        use crate::codecs::huff_llm::HuffLlm5Bit;
        use crate::codecs::huffman::Huffman;
        use crate::codecs::identity::Identity;
        use crate::codecs::order1_scale_ac::Order1ScaleAC;
        use crate::codecs::per_group_codebook::PerGroupCodebook;
        use crate::codecs::rans::Rans;
        use crate::codecs::tans::Tans;
        use crate::codecs::zstd::Zstd;
        for id in [
            Identity.canonical_id(),
            Huffman.canonical_id(),
            Rans.canonical_id(),
            Tans.canonical_id(),
            Zstd.canonical_id(),
            Order1ScaleAC.canonical_id(),
            ArithmeticO0.canonical_id(),
            ArithmeticO0Adaptive.canonical_id(),
            ArithmeticO1.canonical_id(),
            PerGroupCodebook.canonical_id(),
            ContextMixingLite.canonical_id(),
            HuffLlm5Bit.canonical_id(),
        ] {
            assert!(
                is_builtin(&id),
                "codec canonical_id has no matching builtin entry"
            );
        }
    }
}
