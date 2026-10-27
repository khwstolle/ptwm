//! Helpers for building an ExtensionTable while serializing chains.

use std::collections::HashMap;

use super::{
    CURRENT_BUILTIN_VERSION, CanonicalId, ExtensionTableEntry, builtin_entry_at_version,
    builtin_registry,
};

/// Intern `ExtensionTableEntry` values by canonical id, returning a
/// stable `u16` index for each first insertion.
pub struct ExtensionTableBuilder {
    by_id: HashMap<CanonicalId, u16>,
    entries: Vec<ExtensionTableEntry>,
    /// Version whose ids and labels built-in entries are interned with.
    builtin_version: String,
}

impl Default for ExtensionTableBuilder {
    fn default() -> Self {
        Self::with_builtin_version(CURRENT_BUILTIN_VERSION)
    }
}

impl ExtensionTableBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// A builder that interns built-ins with the ids and labels of
    /// `version`, as a writer of that release does. The transcoder uses it
    /// to rebuild tables written by other releases byte-exactly.
    pub fn with_builtin_version(version: &str) -> Self {
        Self {
            by_id: HashMap::new(),
            entries: Vec::new(),
            builtin_version: version.to_owned(),
        }
    }

    /// A builder whose table starts as `entries`. Interning an id already in
    /// `entries` returns its first index. Ids are matched as given, so
    /// callers pass entries whose built-in ids were normalised to the crate
    /// version.
    pub fn from_entries(entries: Vec<ExtensionTableEntry>) -> Self {
        let mut by_id = HashMap::with_capacity(entries.len());
        for (i, e) in entries.iter().enumerate() {
            let idx = u16::try_from(i).expect("extension table index overflow (>= 65536 entries)");
            by_id.entry(e.canonical_id).or_insert(idx);
        }
        Self {
            by_id,
            entries,
            builtin_version: CURRENT_BUILTIN_VERSION.to_owned(),
        }
    }

    /// The version built-in entries are interned with.
    pub fn builtin_version(&self) -> &str {
        &self.builtin_version
    }

    /// Number of entries interned so far.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no entry has been interned.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Intern the built-in `name` at the builder's built-in version.
    ///
    /// Entries are keyed by their current canonical id, so the same built-in
    /// interns once whatever version the builder writes. Returns `None` if
    /// `name` has no built-in entry.
    pub fn intern_builtin(&mut self, name: &str) -> Option<u16> {
        let current = builtin_registry().id_for_entry_name(name)?;
        if let Some(&idx) = self.by_id.get(&current) {
            return Some(idx);
        }
        let entry = builtin_entry_at_version(name, &self.builtin_version)?;
        let idx = u16::try_from(self.entries.len())
            .expect("extension table index overflow (>= 65536 entries)");
        self.by_id.insert(current, idx);
        self.entries.push(entry);
        Some(idx)
    }

    /// Insert `entry` if its `canonical_id` is not yet known, returning the
    /// index. Otherwise return the existing index.
    ///
    /// # Panics
    ///
    /// Panics if the table would grow beyond `u16::MAX` entries.
    pub fn intern(&mut self, entry: ExtensionTableEntry) -> u16 {
        if let Some(&idx) = self.by_id.get(&entry.canonical_id) {
            return idx;
        }
        let idx = u16::try_from(self.entries.len())
            .expect("extension table index overflow (>= 65536 entries)");
        self.by_id.insert(entry.canonical_id, idx);
        self.entries.push(entry);
        idx
    }

    pub fn finish(self) -> Vec<ExtensionTableEntry> {
        self.entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extension::Kind;
    use crate::extension::builtins::builtin_entry;

    fn make_entry(name: &str) -> ExtensionTableEntry {
        builtin_entry(name, Kind::Transform, true)
    }

    #[test]
    fn empty_builder_produces_empty_entries() {
        let b = ExtensionTableBuilder::new();
        assert!(b.finish().is_empty());
    }

    #[test]
    fn two_distinct_ids_produce_two_entries() {
        let mut b = ExtensionTableBuilder::new();
        let idx0 = b.intern(make_entry("source"));
        let idx1 = b.intern(make_entry("terminal"));
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(b.finish().len(), 2);
    }

    #[test]
    fn intern_builtin_uses_the_builder_version() {
        let mut b = ExtensionTableBuilder::with_builtin_version("1.0.0");
        assert_eq!(b.builtin_version(), "1.0.0");
        assert_eq!(b.intern_builtin("source"), Some(0));
        assert_eq!(b.intern_builtin("zstd"), Some(1));
        assert_eq!(b.intern_builtin("source"), Some(0));
        assert_eq!(b.intern_builtin("not_a_builtin"), None);
        let entries = b.finish();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].human_label, "io.ptwm.builtin.source@1.0.0");
        assert_eq!(
            entries[0].canonical_id,
            CanonicalId::derive(&crate::extension::BUILTIN_PUBKEY, "source", "1.0.0")
        );

        // The default version produces the current entries.
        let mut b = ExtensionTableBuilder::new();
        b.intern_builtin("source").unwrap();
        assert_eq!(b.finish(), vec![make_entry("source")]);
    }

    #[test]
    fn from_entries_returns_existing_indices() {
        let entries = vec![
            make_entry("terminal"),
            make_entry("source"),
            make_entry("source"),
        ];
        let mut b = ExtensionTableBuilder::from_entries(entries);
        assert_eq!(b.len(), 3);
        assert_eq!(b.intern(make_entry("source")), 1);
        assert_eq!(b.intern_builtin("terminal"), Some(0));
        assert_eq!(b.intern_builtin("byte_split"), Some(3));
        assert_eq!(b.len(), 4);
    }

    #[test]
    fn same_id_interned_twice_returns_same_index() {
        let mut b = ExtensionTableBuilder::new();
        let idx0 = b.intern(make_entry("source"));
        let idx1 = b.intern(make_entry("source"));
        assert_eq!(idx0, idx1);
        assert_eq!(b.finish().len(), 1);
    }
}
