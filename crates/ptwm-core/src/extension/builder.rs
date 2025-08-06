//! Helpers for building an ExtensionTable while serializing chains.

use std::collections::HashMap;

use super::{CanonicalId, ExtensionTableEntry};

/// Intern `ExtensionTableEntry` values by canonical id, returning a
/// stable `u16` index for each first insertion.
#[derive(Default)]
pub struct ExtensionTableBuilder {
    by_id: HashMap<CanonicalId, u16>,
    entries: Vec<ExtensionTableEntry>,
}

impl ExtensionTableBuilder {
    pub fn new() -> Self {
        Self::default()
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
    fn same_id_interned_twice_returns_same_index() {
        let mut b = ExtensionTableBuilder::new();
        let idx0 = b.intern(make_entry("source"));
        let idx1 = b.intern(make_entry("source"));
        assert_eq!(idx0, idx1);
        assert_eq!(b.finish().len(), 1);
    }
}
