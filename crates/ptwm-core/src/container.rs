//! Container writer and reader orchestration.

use std::io::{Seek, SeekFrom, Write};
use std::sync::Arc;

use rayon::prelude::*;

use xxhash_rust::xxh64::xxh64;

use std::collections::HashMap;

use crate::chain::Chain;
use crate::chain::wire::{read_chain, write_chain};
use crate::error::PtwmCoreError;
use crate::extension::{CanonicalId, ExtensionTable, ExtensionTableBuilder};
use crate::header::{HEADER_LEN, Header, MAGIC_PTWM};
use crate::index::{
    EOF_SENTINEL, IndexEntry, name_hash, parse_index, verify_eof_sentinel, write_index,
};
use crate::prelude::{PreludeEntry, parse_prelude, write_prelude};
use crate::tensor_record::{TensorRecord, write_tensor_record};

// ── Chain registry ────────────────────────────────────────────────────────────

/// In-memory chain registry section.
///
/// Wire format:
/// ```text
/// num_chains: u16
/// [chain_entry] × num_chains:
///     chain_id: u16
///     (chain::wire::write_chain payload: num_nodes, num_edges, num_terminals,
///      lens, nodes_bytes, edges_bytes, terminals_bytes)
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ChainRegistry {
    pub chains: Vec<(u16 /* chain_id */, Chain)>,
}

impl ChainRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Encode the registry into `out`, interning each op's canonical id into
    /// `builder`. The caller must subsequently call `builder.finish()` and
    /// store the resulting entries as the file's Extension Table so the reader
    /// can resolve indices back to op ids.
    pub fn write(
        &self,
        builder: &mut ExtensionTableBuilder,
        out: &mut Vec<u8>,
    ) -> Result<(), PtwmCoreError> {
        if self.chains.len() > u16::MAX as usize {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "ChainRegistry: {} chains exceeds u16::MAX",
                self.chains.len()
            )));
        }
        out.extend_from_slice(&(self.chains.len() as u16).to_le_bytes());
        for (chain_id, chain) in &self.chains {
            out.extend_from_slice(&chain_id.to_le_bytes());
            write_chain(chain, builder, out)?;
        }
        Ok(())
    }

    /// Decode a registry from `buf`, resolving each node's u16 table index
    /// via `table`. Returns `(registry, bytes_consumed)`.
    pub fn read(buf: &[u8], table: &ExtensionTable) -> Result<(Self, usize), PtwmCoreError> {
        if buf.len() < 2 {
            return Err(PtwmCoreError::InvalidContainer(
                "ChainRegistry: buffer truncated (need ≥ 2 bytes for num_chains)".into(),
            ));
        }
        let num_chains = u16::from_le_bytes(buf[..2].try_into().unwrap()) as usize;
        let mut pos = 2usize;
        let mut chains = Vec::with_capacity(num_chains);
        for i in 0..num_chains {
            if pos + 2 > buf.len() {
                return Err(PtwmCoreError::InvalidContainer(format!(
                    "ChainRegistry: truncated reading chain_id for entry {i}"
                )));
            }
            let chain_id = u16::from_le_bytes(buf[pos..pos + 2].try_into().unwrap());
            pos += 2;
            let (chain, consumed) = read_chain(&buf[pos..], table).map_err(|e| {
                PtwmCoreError::InvalidContainer(format!(
                    "ChainRegistry: entry {i} (chain_id={chain_id}): {e}"
                ))
            })?;
            pos += consumed;
            chains.push((chain_id, chain));
        }
        Ok((Self { chains }, pos))
    }

    /// Return a reference to the `Chain` with the given `chain_id`, or `None`.
    pub fn lookup(&self, chain_id: u16) -> Option<&Chain> {
        self.chains
            .iter()
            .find(|(id, _)| *id == chain_id)
            .map(|(_, c)| c)
    }
}

/// Streaming writer for a `.ptwm` container.
pub struct ContainerWriter<W: Write + Seek> {
    writer: W,
    header: Header,
    header_position: u64,
    tensors_start_pos: u64,
    // Offsets tracked internally (not in the v1 header directly).
    // TODO: chain_registry_offset will be encoded via the shared_prelude
    // region once the chain registry is fully wired into the Extension Table.
    chain_registry_offset: u64,
    index_entries: Vec<IndexEntry>,
    prelude_entries: Vec<PreludeEntry>,
    chain_registry: ChainRegistry,
    /// Pre-serialized chain registry bytes (produced in `set_chain_registry`
    /// using the builder that also populated `extension_table`). Stored here
    /// so `flush_prelude` can write exactly the bytes that correspond to the
    /// table indices already committed to the extension table.
    chain_registry_bytes: Vec<u8>,
    extension_table: ExtensionTable,
    /// Maps a builtin codec `CanonicalId` to its index in `extension_table`.
    /// Populated by `set_chain_registry` so the compressor can fill
    /// `PlaneRecord::codec_table_idx` without re-scanning the table.
    codec_idx_map: HashMap<CanonicalId, u16>,
    prelude_written: bool,
    finalized: bool,
    /// Total byte count of the prelude + chain-registry section.
    prelude_section_len: u64,
}

impl<W: Write + Seek> ContainerWriter<W> {
    /// Create a new container writer. Writes a 91-byte header placeholder
    /// immediately.
    pub fn new(mut writer: W) -> Result<Self, PtwmCoreError> {
        let header_position = writer
            .stream_position()
            .map_err(|e| PtwmCoreError::Io(e.to_string()))?;

        // Write placeholder (magic bytes, rest zeros).
        let mut placeholder = [0u8; HEADER_LEN];
        placeholder[..9].copy_from_slice(&MAGIC_PTWM);
        writer
            .write_all(&placeholder)
            .map_err(|e| PtwmCoreError::Io(e.to_string()))?;

        let header = Header::new_ptwm();

        Ok(Self {
            writer,
            header,
            header_position,
            tensors_start_pos: 0,
            chain_registry_offset: 0,
            index_entries: Vec::new(),
            prelude_entries: Vec::new(),
            chain_registry: ChainRegistry::new(),
            chain_registry_bytes: Vec::new(),
            extension_table: ExtensionTable::empty(),
            codec_idx_map: HashMap::new(),
            prelude_written: false,
            finalized: false,
            prelude_section_len: 0,
        })
    }

    /// Store prelude entries to be written at finalize time (before tensors).
    ///
    /// Must be called before any `append_tensor` call.
    pub fn set_prelude(&mut self, entries: &[PreludeEntry]) -> Result<(), PtwmCoreError> {
        if self.prelude_written || !self.index_entries.is_empty() {
            return Err(PtwmCoreError::InvalidContainer(
                "prelude must be set before appending tensors".into(),
            ));
        }
        self.prelude_entries = entries.to_vec();
        Ok(())
    }

    /// Set the chain registry to be written between prelude and tensors.
    ///
    /// This method also builds the Extension Table from the registry's chains
    /// (interning each op's canonical id) **plus all builtin plane-codec
    /// entries**.  Interning codecs here ensures that `PlaneRecord::codec_table_idx`
    /// can be filled for every codec the compressor might use, without a
    /// second table-building pass.  Any previously set extension table is
    /// replaced.
    ///
    /// After this call, use [`codec_table_idx_for`](Self::codec_table_idx_for)
    /// to obtain the stable index for a codec's canonical id.
    ///
    /// Must be called before any `append_tensor` call.
    pub fn set_chain_registry(&mut self, registry: ChainRegistry) -> Result<(), PtwmCoreError> {
        if self.prelude_written || !self.index_entries.is_empty() {
            return Err(PtwmCoreError::InvalidContainer(
                "chain registry must be set before appending tensors".into(),
            ));
        }
        // Build the extension table from the registry chains, producing the
        // chain registry bytes in the same pass so table indices are consistent.
        let mut builder = ExtensionTableBuilder::new();
        let mut registry_bytes = Vec::new();
        registry.write(&mut builder, &mut registry_bytes)?;

        // Additionally intern all builtin plane-codec entries so every codec
        // the compressor selects gets a stable u16 table index for
        // `PlaneRecord::codec_table_idx`.  Ops that are already in the table
        // from chain interning just get their existing index back.
        //
        // Note: we use `builtin_entries()` (the curated list with `is_builtin`
        // coverage) and filter by `Kind::PlaneCodec` rather than hard-coding
        // names.  This avoids interning `huffman_nibble`, which is present in
        // `dispatch_builtin` for legacy wire compat but has no `builtin_entry`
        // and therefore is NOT recognised by `is_builtin` / `collect_missing_flavors`.
        use crate::extension::Kind;
        use crate::extension::builtin_entries;
        let mut codec_idx_map = HashMap::new();
        for entry in builtin_entries()
            .into_iter()
            .filter(|e| e.kind == Kind::PlaneCodec)
        {
            let canonical_id = entry.canonical_id;
            let idx = builder.intern(entry);
            codec_idx_map.insert(canonical_id, idx);
        }

        let entries = builder.finish();
        self.extension_table = ExtensionTable { entries };
        self.chain_registry_bytes = registry_bytes;
        self.chain_registry = registry;
        self.codec_idx_map = codec_idx_map;
        Ok(())
    }

    /// Return the Extension Table index for a codec's canonical id, or
    /// `None` if the id was never interned (e.g. `set_chain_registry` was
    /// not called before this point).
    ///
    /// The compressor uses this to populate `PlaneRecord::codec_table_idx`.
    pub fn codec_table_idx_for(&self, canonical_id: &CanonicalId) -> Option<u16> {
        self.codec_idx_map.get(canonical_id).copied()
    }

    /// Set the Extension Table that will be serialized immediately after the
    /// header at `extension_table_offset = HEADER_LEN`. Callers populate this
    /// with a real table; the default is `ExtensionTable::empty()`.
    pub fn set_extension_table(&mut self, table: ExtensionTable) {
        self.extension_table = table;
    }

    /// Write the extension table, prelude section, and chain registry section,
    /// then record all absolute offsets in the pending header.
    ///
    /// Section order on disk: header → extension_table → prelude → chain_registry → tensors → index.
    fn flush_prelude(&mut self) -> Result<(), PtwmCoreError> {
        if self.prelude_written {
            return Ok(());
        }

        // Extension table sits immediately after the header at HEADER_LEN.
        // Serialize it now (set_extension_table must be called before this point).
        let et_bytes = self
            .extension_table
            .to_bytes()
            .map_err(|e| PtwmCoreError::InvalidContainer(format!("extension table: {e}")))?;
        let et_len = et_bytes.len() as u64;
        let et_hash = if et_len > 0 {
            ExtensionTable::hash(&et_bytes)
        } else {
            [0u8; 32]
        };
        self.header.extension_table_offset = HEADER_LEN as u64;
        self.header.extension_table_length = et_len;
        self.header.extension_table_hash = et_hash;
        self.writer
            .write_all(&et_bytes)
            .map_err(|e| PtwmCoreError::Io(e.to_string()))?;

        // shared_prelude_offset = absolute position right after the extension table.
        let prelude_start = self
            .writer
            .stream_position()
            .map_err(|e| PtwmCoreError::Io(e.to_string()))?;
        self.header.shared_prelude_offset = prelude_start;

        let mut prelude_buf = Vec::new();
        write_prelude(&self.prelude_entries, &mut prelude_buf)?;
        self.writer
            .write_all(&prelude_buf)
            .map_err(|e| PtwmCoreError::Io(e.to_string()))?;

        // Chain registry follows immediately after the prelude.
        // TODO: chain_registry will move into the Extension Table in a
        // follow-up change; for now it is embedded in the shared-prelude region.
        let chain_registry_offset = self
            .writer
            .stream_position()
            .map_err(|e| PtwmCoreError::Io(e.to_string()))?;
        self.chain_registry_offset = chain_registry_offset;

        // Write the pre-serialized chain registry bytes (built in
        // `set_chain_registry` using the same builder that produced the
        // extension table). If the registry was never set, write the zero
        // chain-count prefix for an empty registry.
        let registry_bytes = if self.chain_registry_bytes.is_empty() {
            // Empty registry: just the u16 count of 0.
            let mut empty = Vec::new();
            let mut nop_builder = ExtensionTableBuilder::new();
            self.chain_registry
                .write(&mut nop_builder, &mut empty)
                .map_err(|e| PtwmCoreError::InvalidContainer(format!("chain registry: {e}")))?;
            empty
        } else {
            self.chain_registry_bytes.clone()
        };
        self.writer
            .write_all(&registry_bytes)
            .map_err(|e| PtwmCoreError::Io(e.to_string()))?;

        let after_registry = self
            .writer
            .stream_position()
            .map_err(|e| PtwmCoreError::Io(e.to_string()))?;
        self.prelude_section_len = after_registry - prelude_start;
        self.prelude_written = true;
        Ok(())
    }

    /// Append a tensor record to the container.
    pub fn append_tensor(&mut self, record: &TensorRecord) -> Result<(), PtwmCoreError> {
        if self.finalized {
            return Err(PtwmCoreError::InvalidContainer("already finalized".into()));
        }

        // On first tensor: flush prelude and record tensors_start_pos.
        if !self.prelude_written {
            self.flush_prelude()?;
            self.tensors_start_pos = self.header.shared_prelude_offset + self.prelude_section_len;
        }

        let tensor_offset = self
            .writer
            .stream_position()
            .map_err(|e| PtwmCoreError::Io(e.to_string()))?;

        let mut tensor_buf = Vec::new();
        write_tensor_record(record, &mut tensor_buf)?;
        let tensor_len = tensor_buf.len() as u64;

        self.writer
            .write_all(&tensor_buf)
            .map_err(|e| PtwmCoreError::Io(e.to_string()))?;

        self.index_entries.push(IndexEntry {
            name_hash: name_hash(&record.name),
            tensor_offset,
            tensor_len,
        });

        Ok(())
    }

    /// Append a tensor whose record bytes were serialized ahead of time (e.g.
    /// in parallel via [`write_tensor_record`](crate::tensor_record::write_tensor_record)).
    /// Identical on-disk effect to [`Self::append_tensor`] but skips the
    /// (payload-copy-heavy) serialization, which the caller has already done.
    pub fn append_serialized_tensor(
        &mut self,
        name: &str,
        tensor_buf: &[u8],
    ) -> Result<(), PtwmCoreError> {
        if self.finalized {
            return Err(PtwmCoreError::InvalidContainer("already finalized".into()));
        }
        if !self.prelude_written {
            self.flush_prelude()?;
            self.tensors_start_pos = self.header.shared_prelude_offset + self.prelude_section_len;
        }
        let tensor_offset = self
            .writer
            .stream_position()
            .map_err(|e| PtwmCoreError::Io(e.to_string()))?;
        self.writer
            .write_all(tensor_buf)
            .map_err(|e| PtwmCoreError::Io(e.to_string()))?;
        self.index_entries.push(IndexEntry {
            name_hash: name_hash(name),
            tensor_offset,
            tensor_len: tensor_buf.len() as u64,
        });
        Ok(())
    }

    /// Finalize the container: write the index, patch the header, and append
    /// the EOF sentinel.
    pub fn finalize(mut self) -> Result<(), PtwmCoreError> {
        if self.finalized {
            return Err(PtwmCoreError::InvalidContainer("already finalized".into()));
        }

        // If no tensors were appended, write prelude now so prelude_offset is valid.
        if !self.prelude_written {
            self.flush_prelude()?;
        }

        // Record tensor-index offset and write sorted index.
        let index_offset = self
            .writer
            .stream_position()
            .map_err(|e| PtwmCoreError::Io(e.to_string()))?;
        self.header.tensor_index_offset = index_offset;

        let mut index_buf = Vec::new();
        write_index(&mut self.index_entries, &mut index_buf);
        let index_len = index_buf.len() as u64;
        self.writer
            .write_all(&index_buf)
            .map_err(|e| PtwmCoreError::Io(e.to_string()))?;

        // Record tensor-index length and shared-prelude length.
        self.header.tensor_index_length = index_len;
        self.header.shared_prelude_length = self.prelude_section_len;

        // Seek back to header position and write the finalized header.
        self.writer
            .seek(SeekFrom::Start(self.header_position))
            .map_err(|e| PtwmCoreError::Io(e.to_string()))?;
        let header_bytes = self.header.to_bytes()?;
        self.writer
            .write_all(&header_bytes)
            .map_err(|e| PtwmCoreError::Io(e.to_string()))?;

        // Seek to end of file and write EOF sentinel.
        self.writer
            .seek(SeekFrom::End(0))
            .map_err(|e| PtwmCoreError::Io(e.to_string()))?;
        self.writer
            .write_all(&EOF_SENTINEL)
            .map_err(|e| PtwmCoreError::Io(e.to_string()))?;

        Ok(())
    }
}

// ── Missing-flavor detection ──────────────────────────────────────────────────

/// Inspect the Extension Table and return one [`MissingFlavorEntry`] for every
/// non-builtin contribution whose required flavor is not available on this host.
///
/// Built-in contributions (whose `canonical_id` is derived from
/// [`crate::extension::BUILTIN_PUBKEY`]) are always considered available and
/// are skipped.
///
/// `installed` is the pre-fetched discovery list; callers that also need the
/// list for verifier lookups should scan once and pass it here.
fn collect_missing_flavors(
    table: &ExtensionTable,
    installed: &[crate::discovery::DiscoveredContribution],
) -> Result<Vec<crate::error::MissingFlavorEntry>, PtwmCoreError> {
    use crate::error::MissingFlavorEntry;
    use crate::extension::is_builtin;

    let mut needs = Vec::new();
    for entry in &table.entries {
        if is_builtin(&entry.canonical_id) {
            continue;
        }
        // Match by canonical_id: derive it for each discovered contribution
        // from the bundle manifest's author_pubkey + name + version.
        let detected = installed.iter().find(|d| {
            let computed = decode_pubkey_to_canonical(
                &d.manifest.bundle.author_pubkey,
                &d.manifest.bundle.name,
                &d.manifest.bundle.version,
            );
            computed == entry.canonical_id
        });
        let available_flavors = match detected {
            Some(d) => d.installed_flavors,
            None => 0,
        };
        let needed_flavors = entry.flavor_hints;
        if available_flavors & needed_flavors == 0 {
            needs.push(MissingFlavorEntry {
                id: entry.canonical_id,
                label: entry.human_label.clone(),
                requested_flavors: needed_flavors,
                available_flavors,
                install_hint: entry.install_hint.clone(),
            });
        }
    }
    Ok(needs)
}

/// Look up the author public key for a non-builtin contribution by its
/// `canonical_id`. Iterates the pre-scanned `installed` list and re-derives
/// the canonical id from each manifest's `author_pubkey + name + version`
/// until a match is found. Returns `None` when no installed manifest claims
/// this id (the caller should treat the entry as unverifiable / missing).
fn lookup_author_pubkey(
    canonical_id: &crate::extension::CanonicalId,
    installed: &[crate::discovery::DiscoveredContribution],
) -> Option<(crate::trust::PublicKey, usize)> {
    for (idx, d) in installed.iter().enumerate() {
        let computed = decode_pubkey_to_canonical(
            &d.manifest.bundle.author_pubkey,
            &d.manifest.bundle.name,
            &d.manifest.bundle.version,
        );
        if &computed == canonical_id {
            // Parse the raw pubkey bytes from the "ed25519:<hex>" string.
            let hex_part = d
                .manifest
                .bundle
                .author_pubkey
                .strip_prefix("ed25519:")
                .unwrap_or(&d.manifest.bundle.author_pubkey);
            let bytes = hex::decode(hex_part).unwrap_or_default();
            if bytes.len() == 32 {
                let mut key = [0u8; 32];
                key.copy_from_slice(&bytes);
                return Some((crate::trust::PublicKey(key), idx));
            }
        }
    }
    None
}

/// Parse an `"ed25519:<hex>"` public-key string into a 32-byte array and
/// derive the canonical id for the given name + version. If the string is
/// malformed or the hex doesn't decode to 32 bytes, the key bytes are
/// zero-padded / truncated and the resulting id is deterministic but will
/// not match any well-formed contribution.
fn decode_pubkey_to_canonical(
    pubkey_str: &str,
    name: &str,
    version: &str,
) -> crate::extension::CanonicalId {
    let hex_part = pubkey_str.strip_prefix("ed25519:").unwrap_or(pubkey_str);
    let bytes = hex::decode(hex_part).unwrap_or_else(|_| pubkey_str.as_bytes().to_vec());
    let mut key = [0u8; 32];
    let n = bytes.len().min(32);
    key[..n].copy_from_slice(&bytes[..n]);
    crate::extension::CanonicalId::derive(&key, name, version)
}

/// Zero-copy reader for a `.ptwm` container backed by an in-memory byte slice.
#[derive(Debug)]
pub struct ContainerReader<'a> {
    pub buf: &'a [u8],
    pub header: Header,
    pub extension_table: ExtensionTable,
    pub prelude: Vec<PreludeEntry>,
    /// Codec router: resolves a `CanonicalId` (from the Extension Table) to
    /// a `DispatchedPlaneCodec`.  Used by `decode_tensor` when a
    /// `PlaneRecord::codec_table_idx` is present.
    pub router: crate::flavor::PlaneCodecRouter,
    /// Chain registry parsed from the shared-prelude region.
    ///
    /// The chain registry immediately follows the prelude bytes inside the
    /// shared-prelude section. It is empty when the section has no registry.
    ///
    /// TODO: chain_registry will be replaced by the Extension Table; this
    /// field is kept for backward compat during the transition.
    pub chain_registry: ChainRegistry,
    pub index: Vec<IndexEntry>,
    /// Derived: number of tensors in the index (not stored in the v1 header).
    pub num_tensors: u64,
    /// Derived: absolute file offset of the first tensor record.
    ///
    /// Equals `shared_prelude_offset + shared_prelude_length` once all
    /// sections are fully written. Stored here for test/diagnostic access.
    pub tensors_offset: u64,
    /// Derived: absolute offset where the chain registry starts inside the
    /// shared-prelude region (0 when there is no chain registry).
    pub chain_registry_offset: u64,
    /// Non-builtin extensions referenced by this file that are not available
    /// on the current host (or whose installed flavor doesn't satisfy the
    /// request). Populated by `open_partial`; always empty when `open` returns
    /// `Ok` (because `open` returns `Err` instead).
    pub missing_extensions: Vec<crate::error::MissingFlavorEntry>,
}

impl<'a> ContainerReader<'a> {
    /// Open a container from a byte slice. Verifies the EOF sentinel, parses
    /// the header (magic only — no numeric version check in v1), prelude,
    /// chain registry, and index. All declared offsets are bounds-checked
    /// against `buf.len()` before slicing so that a truncated or adversarial
    /// file returns `InvalidContainer` rather than panicking with
    /// index-out-of-bounds.
    ///
    /// Returns `Err(PtwmCoreError::MissingFlavor { .. })` before touching any
    /// tensor payload if the file references non-builtin extensions that are
    /// not available on this host. Use [`ContainerReader::open_partial`] to
    /// suppress that check and continue with partial availability.
    pub fn open(buf: &'a [u8]) -> Result<Self, PtwmCoreError> {
        Self::open_with(buf, false)
    }

    /// Open a container, skipping the missing-flavor check.
    ///
    /// If the file references non-builtin extensions that are not installed,
    /// the reader is returned anyway and the affected entries are available
    /// via [`ContainerReader::missing_extensions`]. Decoding a tensor that
    /// needs a missing extension will still fail at decode time.
    pub fn open_partial(buf: &'a [u8]) -> Result<Self, PtwmCoreError> {
        Self::open_with(buf, true)
    }

    /// Shared implementation for [`open`](Self::open) and
    /// [`open_partial`](Self::open_partial).
    fn open_with(buf: &'a [u8], skip_missing: bool) -> Result<Self, PtwmCoreError> {
        verify_eof_sentinel(buf)?;
        if buf.len() < HEADER_LEN {
            return Err(PtwmCoreError::InvalidContainer(
                "file shorter than header".into(),
            ));
        }
        let header = Header::from_bytes(&buf[..HEADER_LEN])?;

        // Parse the Extension Table (sits at extension_table_offset for
        // extension_table_length bytes, immediately after the header).
        let extension_table = if header.extension_table_length == 0 {
            ExtensionTable::empty()
        } else {
            let et_start = usize::try_from(header.extension_table_offset).map_err(|_| {
                PtwmCoreError::InvalidContainer("extension_table_offset overflows usize".into())
            })?;
            let et_len = usize::try_from(header.extension_table_length).map_err(|_| {
                PtwmCoreError::InvalidContainer("extension_table_length overflows usize".into())
            })?;
            let et_end = et_start.checked_add(et_len).ok_or_else(|| {
                PtwmCoreError::InvalidContainer(
                    "extension_table_offset + extension_table_length overflows".into(),
                )
            })?;
            if et_end > buf.len() {
                return Err(PtwmCoreError::InvalidContainer(
                    "extension table region exceeds buffer".into(),
                ));
            }
            let table_bytes = &buf[et_start..et_end];
            let actual_hash = ExtensionTable::hash(table_bytes);
            if actual_hash != header.extension_table_hash {
                return Err(PtwmCoreError::ExtensionTableHashMismatch);
            }
            ExtensionTable::from_bytes(table_bytes)
                .map_err(|e| PtwmCoreError::InvalidContainer(format!("extension table: {e}")))?
        };

        // Scan installed extensions once; used by both the missing-flavor check
        // and the trust verifier pass below.
        let installed = crate::discovery::scan_all_cached().unwrap_or_default();

        // Check for missing non-builtin flavors immediately after parsing the
        // Extension Table — before touching any tensor payload.
        let needs = collect_missing_flavors(&extension_table, &installed)?;
        if !needs.is_empty() && !skip_missing {
            return Err(PtwmCoreError::MissingFlavor { needs });
        }

        // ── Trust Verifier pass (spec §6.5: trusted ⇒ allowed) ───────────────
        //
        // For every non-builtin entry in the Extension Table that IS installed
        // (i.e. not flagged as missing above), verify that its author is
        // trusted by the active keyring. Entries with no matching install
        // (already covered by MissingFlavor above) are skipped here.
        {
            // Build the active keyring: start from the bundled keys, merge the
            // user keyring on top.
            let mut keyring = match crate::trust::evaluate_bundled() {
                Ok(crate::trust::BundleStatus::Match { keyring }) => keyring,
                Ok(crate::trust::BundleStatus::FreshInstall { keyring, hash }) => {
                    // First-run convenience: auto-trust the bundled keyring and
                    // persist the lock so subsequent opens see Match.
                    let _ = crate::trust::apply_fresh_install(&hash);
                    keyring
                }
                // Bundled keyring changed since last acceptance — do not
                // activate it until the user runs `ptwm trust update --bundled`.
                Ok(crate::trust::BundleStatus::Mismatch { .. }) => crate::trust::Keyring::default(),
                Err(_) => crate::trust::Keyring::default(),
            };

            // Merge user keyring (keys.toml lives beside bundled.lock).
            let user_keys_path = crate::trust::lock_path().with_file_name("keys.toml");
            if let Ok(user) = crate::trust::Keyring::load_from_path(&user_keys_path) {
                keyring.entries.extend(user.entries);
            }

            let verifier = crate::trust::Verifier::new(keyring);

            let mut untrusted: Vec<(crate::extension::CanonicalId, String)> = Vec::new();
            for entry in &extension_table.entries {
                if crate::extension::is_builtin(&entry.canonical_id) {
                    // Built-ins are trusted by construction; verifier would
                    // short-circuit anyway, but skip the lookup entirely.
                    continue;
                }
                let Some((pubkey, contrib_idx)) =
                    lookup_author_pubkey(&entry.canonical_id, &installed)
                else {
                    // No matching installed manifest — MissingFlavor already
                    // accounts for this entry. If skip_missing is true we keep
                    // going; treat as unverifiable (but not actively untrusted).
                    continue;
                };
                // Recompute the signature's signed material from the on-disk
                // bundle (manifest + binaries, in the order
                // `ext_tooling/sign.py::sign_bundle` writes them). Tampering
                // with either the manifest TOML or any binary after signing
                // changes the digest and trips Ed25519 verification.
                let signed_material = match installed[contrib_idx].signed_digest() {
                    Ok(d) => d,
                    Err(e) => {
                        untrusted.push((entry.canonical_id, e.to_string()));
                        continue;
                    }
                };
                match verifier.verify(entry, &signed_material, &pubkey) {
                    Ok(crate::trust::TrustVerdict::Trusted) => {}
                    Ok(crate::trust::TrustVerdict::Untrusted { reason }) => {
                        untrusted.push((entry.canonical_id, reason));
                    }
                    Err(e) => {
                        untrusted.push((entry.canonical_id, e.to_string()));
                    }
                }
            }
            if !untrusted.is_empty() && !skip_missing {
                let (id, reason) = untrusted.remove(0);
                return Err(PtwmCoreError::ContributionUntrusted { id, reason });
            }
        }

        let prelude_off = usize::try_from(header.shared_prelude_offset).map_err(|_| {
            PtwmCoreError::InvalidContainer("shared_prelude_offset overflows usize".into())
        })?;
        let index_off = usize::try_from(header.tensor_index_offset).map_err(|_| {
            PtwmCoreError::InvalidContainer("tensor_index_offset overflows usize".into())
        })?;

        if prelude_off < HEADER_LEN || prelude_off > buf.len() {
            return Err(PtwmCoreError::InvalidContainer(
                "shared_prelude_offset out of bounds".into(),
            ));
        }
        if index_off < HEADER_LEN || index_off > buf.len() {
            return Err(PtwmCoreError::InvalidContainer(
                "tensor_index_offset out of bounds".into(),
            ));
        }

        let (prelude, prelude_bytes_consumed) = parse_prelude(&buf[prelude_off..])?;

        // The chain registry immediately follows the prelude bytes within the
        // shared-prelude region.
        let cr_off = prelude_off + prelude_bytes_consumed;
        let chain_registry = if cr_off < index_off && cr_off < buf.len() {
            let (registry, _) = ChainRegistry::read(&buf[cr_off..], &extension_table)?;
            registry
        } else {
            ChainRegistry::new()
        };
        let chain_registry_offset = if chain_registry.chains.is_empty() {
            0
        } else {
            cr_off as u64
        };

        let index = parse_index(&buf[index_off..])?;
        let num_tensors = index.len() as u64;
        // tensors_offset: region between prelude section end and index start.
        // shared_prelude_offset + shared_prelude_length = end of prelude section.
        let tensors_offset = if header.shared_prelude_length > 0 {
            header
                .shared_prelude_offset
                .saturating_add(header.shared_prelude_length)
        } else {
            // Fallback: tensors sit right after the chain-registry end.
            cr_off as u64
        };

        // Build the codec router once from the installed extensions snapshot.
        // Using the same `installed` slice that the missing-flavor check and
        // trust verifier already consumed is correct: it is a
        // point-in-time snapshot of the host's extension directory.
        let router = crate::flavor::PlaneCodecRouter::new(installed.clone());

        Ok(Self {
            buf,
            header,
            extension_table,
            prelude,
            chain_registry,
            router,
            index,
            num_tensors,
            tensors_offset,
            chain_registry_offset,
            missing_extensions: needs,
        })
    }

    /// Look up a tensor by name and return a slice covering its serialized
    /// `TensorRecord` bytes. Returns `None` if no tensor with that name
    /// exists. The on-disk index is sorted by `name_hash`, so we use
    /// `binary_search_by_key` for an O(log n) lookup instead of a linear
    /// scan — material on models with thousands of tensors.
    pub fn get_tensor_record_bytes(&self, name: &str) -> Option<&[u8]> {
        let target = name_hash(name);
        let idx = self
            .index
            .binary_search_by_key(&target, |e| e.name_hash)
            .ok()?;
        let e = &self.index[idx];
        let start = e.tensor_offset as usize;
        let end = start.checked_add(e.tensor_len as usize)?;
        if end > self.buf.len() {
            return None;
        }
        Some(&self.buf[start..end])
    }

    /// Retrieve the shape + dtype_name embedded in this tensor's metadata,
    /// or `None` if no metadata is present. Errors if the record itself is
    /// malformed or the metadata isn't the expected shape-convention map.
    pub fn tensor_shape(&self, name: &str) -> Result<Option<(Vec<u64>, String)>, PtwmCoreError> {
        use crate::metadata::decode_shape_metadata;
        use crate::tensor_record::parse_tensor_record;

        let record_bytes = self.get_tensor_record_bytes(name).ok_or_else(|| {
            PtwmCoreError::InvalidContainer(format!("tensor '{}' not found", name))
        })?;
        let (rec, _) = parse_tensor_record(record_bytes)?;
        match rec.tensor_metadata {
            Some(bytes) => Ok(Some(decode_shape_metadata(&bytes)?)),
            None => Ok(None),
        }
    }

    /// Return every tensor name in index order (i.e. sorted by name hash,
    /// which is how the on-disk index is stored). Requires parsing each
    /// tensor record header; use sparingly on very large containers.
    pub fn tensor_names(&self) -> Result<Vec<String>, PtwmCoreError> {
        use crate::tensor_record::parse_tensor_record;

        let mut names = Vec::with_capacity(self.index.len());
        for e in &self.index {
            let start = e.tensor_offset as usize;
            let end = start.checked_add(e.tensor_len as usize).ok_or_else(|| {
                PtwmCoreError::InvalidContainer("tensor_offset + tensor_len overflows usize".into())
            })?;
            if end > self.buf.len() {
                return Err(PtwmCoreError::InvalidContainer(format!(
                    "tensor index entry ({start}..{end}) exceeds buffer ({} bytes)",
                    self.buf.len()
                )));
            }
            let (rec, _) = parse_tensor_record(&self.buf[start..end])?;
            names.push(rec.name);
        }
        Ok(names)
    }

    /// Decode every tensor in the container, in index order, across the rayon
    /// pool. `decode_tensor` is `&self` and reads only shared state, so the
    /// per-tensor decodes are independent — this is the primary multi-core
    /// scaling lever for full-model loads (the per-tensor `decode_tensor` stays
    /// available for random access). Returns `(name, raw_bytes)` per tensor.
    pub fn decode_model(&self) -> Result<Vec<(String, Vec<u8>)>, PtwmCoreError> {
        if self.index.is_empty() {
            return Ok(Vec::new());
        }
        use crate::tensor_record::parse_tensor_record;
        // Parse each record header and decode it in the same parallel pass —
        // no sequential tensor_names() pre-pass.
        self.index
            .par_iter()
            .map(|e| {
                let start = e.tensor_offset as usize;
                let end = start.checked_add(e.tensor_len as usize).ok_or_else(|| {
                    PtwmCoreError::InvalidContainer(
                        "tensor_offset + tensor_len overflows usize".into(),
                    )
                })?;
                if end > self.buf.len() {
                    return Err(PtwmCoreError::InvalidContainer(format!(
                        "tensor index entry ({start}..{end}) exceeds buffer ({} bytes)",
                        self.buf.len()
                    )));
                }
                let (rec, _) = parse_tensor_record(&self.buf[start..end])?;
                let bytes = self.decode_tensor_record(&rec)?;
                Ok((rec.name, bytes))
            })
            .collect::<Result<Vec<_>, _>>()
    }

    /// Decode a tensor by name: resolves `chain_ref` via inline or registry,
    /// decodes every terminal plane, then runs `inverse_chain` on the recovered
    /// terminal planes to reconstruct the source bytes. Validates plane CRCs
    /// (when present) and the optional tensor-level `payload_hash`.
    pub fn decode_tensor(&self, name: &str) -> Result<Vec<u8>, PtwmCoreError> {
        use crate::tensor_record::parse_tensor_record;
        let record_bytes = self.get_tensor_record_bytes(name).ok_or_else(|| {
            PtwmCoreError::InvalidContainer(format!("tensor '{}' not found", name))
        })?;
        let (rec, _consumed) = parse_tensor_record(record_bytes)?;
        self.decode_tensor_record(&rec)
    }

    /// Decode a tensor from an already-parsed record. Shared by `decode_tensor`
    /// (which resolves the record by name) and `decode_model` (which parses
    /// records from the index in parallel) — avoids a redundant by-name lookup
    /// and a second record parse.
    fn decode_tensor_record(
        &self,
        rec: &crate::tensor_record::TensorRecord,
    ) -> Result<Vec<u8>, PtwmCoreError> {
        use crate::chain::runtime::{InverseContext, chain_terminal_descriptors, inverse_chain};
        use crate::chain::validate::validate_chain;
        use crate::codec::codec_for;
        use crate::tensor_record::{CHAIN_REF_INLINE_BIT, TENSOR_FLAG_PAYLOAD_HASH};
        use crate::transforms::op::Plane;

        // Resolve the chain via the inline bit or the chain registry.
        let chain: &Chain = if rec.chain_ref & CHAIN_REF_INLINE_BIT != 0 {
            rec.inline_chain.as_ref().ok_or_else(|| {
                PtwmCoreError::InvalidContainer(
                    "decode_tensor: inline bit set but inline_chain is None".into(),
                )
            })?
        } else {
            let chain_id = rec.chain_ref & !CHAIN_REF_INLINE_BIT;
            self.chain_registry.lookup(chain_id).ok_or_else(|| {
                PtwmCoreError::InvalidContainer(format!(
                    "decode_tensor: chain_id {chain_id} not in registry"
                ))
            })?
        };

        // Validate the chain structurally before handing it to the runtime.
        // A malformed wire-derived chain (out-of-range edge nodes, missing
        // Source, cycles, …) would otherwise reach `topo_sort` /
        // `propagate_descriptors` with unchecked indices.
        let n_dependencies = u8::try_from(rec.dependencies.len()).map_err(|_| {
            PtwmCoreError::InvalidContainer(format!(
                "decode_tensor: tensor record has {} dependencies (max 255)",
                rec.dependencies.len()
            ))
        })?;
        validate_chain(chain, n_dependencies)?;

        // Discover the descriptor of every terminal by walking the chain
        // forward symbolically (no source bytes needed). This is a pure
        // function of the chain structure — see `chain_terminal_descriptors`.
        let term_descriptors = chain_terminal_descriptors(chain)?;
        if term_descriptors.len() != rec.terminals.len() {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "decode_tensor: chain has {} terminals but record has {} planes",
                term_descriptors.len(),
                rec.terminals.len()
            )));
        }

        // Decode each terminal plane to bytes.
        //
        // Dispatch order:
        // 1. Look up the Extension Table entry to obtain the `CanonicalId`, then
        //    resolve it via `self.router` — which confirms the codec is available
        //    on this host.  For built-in codecs `dispatch_builtin` maps back to a
        //    `CodecId` so we can call the full `PlaneCodec` interface (with state,
        //    layout, decoded_len).  For third-party non-builtin codecs the router
        //    currently returns `DispatchedPlaneCodec`, but the flat ABI for those
        //    is a separate concern; reaching a non-builtin here returns an error
        //    until that path is wired.
        use crate::extension::dispatch_builtin;

        let mut terminal_planes: Vec<Plane> = Vec::with_capacity(rec.terminals.len());
        for (plane_rec, descriptor) in rec.terminals.iter().zip(term_descriptors) {
            // Per-plane dispatch yields one of two cases:
            //
            // * `Builtin(codec)` — an in-tree PlaneCodec impl. Use the full
            //   structured interface (with state, layout, decoded_len).
            // * `ThirdParty(dispatched)` — a flat-ABI router codec (native
            //   or wasm). Decode via `decode_with_state`, threading any
            //   plane state through the byte ABI.
            enum DispatchTarget {
                Builtin(Box<dyn crate::codec::PlaneCodec>),
                ThirdParty(std::sync::Arc<dyn crate::flavor::DispatchedPlaneCodec>),
            }

            // Router path: validate via Extension Table + PlaneCodecRouter.
            let entry = self
                .extension_table
                .entries
                .get(plane_rec.codec_table_idx as usize)
                .ok_or_else(|| {
                    PtwmCoreError::InvalidContainer(format!(
                        "plane record references codec_table_idx {} but extension \
                         table has {} entries",
                        plane_rec.codec_table_idx,
                        self.extension_table.entries.len()
                    ))
                })?;

            // Resolve (loads native/wasm if needed).
            let dispatched =
                self.router
                    .get(&entry.canonical_id)
                    .map_err(|e| PtwmCoreError::CodecDecode {
                        codec: "router",
                        msg: e.to_string(),
                    })?;

            let target: DispatchTarget = match dispatch_builtin(&entry.canonical_id) {
                Some(crate::extension::BuiltinKind::Codec(cid)) => {
                    DispatchTarget::Builtin(codec_for(cid).ok_or_else(|| {
                        PtwmCoreError::InvalidContainer(format!(
                            "no PlaneCodec impl for built-in {:?}",
                            cid
                        ))
                    })?)
                }
                Some(crate::extension::BuiltinKind::Op(_)) => {
                    return Err(PtwmCoreError::InvalidContainer(format!(
                        "codec_table_idx {} resolves to a transform op, not a codec",
                        plane_rec.codec_table_idx
                    )));
                }
                None => DispatchTarget::ThirdParty(dispatched),
            };

            // Shim that exposes the dispatch target as a `PlaneCodec` so the
            // existing loop body stays the same. The third-party variant
            // ignores `layout` (the flat ABI doesn't carry it) and uses the
            // router's `decode_with_state` for the actual call.
            let codec: Box<dyn crate::codec::PlaneCodec> = match target {
                DispatchTarget::Builtin(c) => c,
                DispatchTarget::ThirdParty(arc) => {
                    Box::new(crate::flavor::ThirdPartyPlaneCodec::new(arc))
                }
            };

            let plane_bytes = if let Some(chunks) = &plane_rec.chunk_table {
                // Chunked plane: each chunk frame is independent and carries its
                // own state, so decode them across the rayon pool and
                // concatenate in order. (The encoder frames chunks sequentially,
                // so the offsets are contiguous.)
                let total: usize = chunks.iter().map(|e| e.decoded_size as usize).sum();
                let decoded_chunks: Vec<Vec<u8>> = chunks
                    .par_iter()
                    .enumerate()
                    .map(|(i, entry)| -> Result<Vec<u8>, PtwmCoreError> {
                        let start = entry.offset_in_payload as usize;
                        let end = if i + 1 < chunks.len() {
                            chunks[i + 1].offset_in_payload as usize
                        } else {
                            plane_rec.payload_bytes.len()
                        };
                        if start > end || end > plane_rec.payload_bytes.len() {
                            return Err(PtwmCoreError::InvalidContainer(
                                "chunk table offset out of bounds".into(),
                            ));
                        }
                        let frame = &plane_rec.payload_bytes[start..end];
                        if frame.len() < 5 {
                            return Err(PtwmCoreError::InvalidContainer(
                                "chunk frame truncated".into(),
                            ));
                        }
                        let state_version = frame[0];
                        let state_len =
                            u32::from_le_bytes(frame[1..5].try_into().unwrap()) as usize;
                        // frame.len() >= 5 checked above; this form avoids a
                        // 5 + state_len overflow on 32-bit usize.
                        if frame.len() - 5 < state_len {
                            return Err(PtwmCoreError::InvalidContainer(
                                "chunk frame state_len exceeds frame".into(),
                            ));
                        }
                        let state_bytes = &frame[5..5 + state_len];
                        let chunk_payload = &frame[5 + state_len..];
                        let decoded = codec.decode(
                            state_version,
                            state_bytes,
                            chunk_payload,
                            &plane_rec.layout,
                            entry.decoded_size as usize,
                        )?;
                        if decoded.len() != entry.decoded_size as usize {
                            return Err(PtwmCoreError::InvalidContainer(
                                "chunk decoded_size mismatch".into(),
                            ));
                        }
                        Ok(decoded)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let mut out = Vec::with_capacity(total);
                for d in &decoded_chunks {
                    out.extend_from_slice(d);
                }
                out
            } else {
                let (state_bytes, state_version): (Vec<u8>, u8) = match plane_rec.state_source {
                    crate::codec::StateSource::None => (Vec::new(), 0),
                    crate::codec::StateSource::Inline => (
                        plane_rec.inline_state_bytes.clone(),
                        plane_rec.state_version,
                    ),
                    crate::codec::StateSource::Shared => {
                        let entry = self
                            .prelude
                            .iter()
                            .find(|e| e.shared_state_id == plane_rec.state_info)
                            .ok_or_else(|| {
                                PtwmCoreError::InvalidContainer(format!(
                                    "shared state {} not in prelude",
                                    plane_rec.state_info
                                ))
                            })?;
                        if plane_rec.state_version != entry.state_format_version {
                            return Err(PtwmCoreError::InvalidContainer(format!(
                                "shared state {}: plane state_version {} \
                                 does not match prelude state_format_version {}",
                                plane_rec.state_info,
                                plane_rec.state_version,
                                entry.state_format_version
                            )));
                        }
                        let role_n = plane_rec.role as u8;
                        if (1..=8).contains(&role_n) {
                            let role_bit = 1u8 << (role_n - 1);
                            if entry.applies_to_mask & role_bit == 0 {
                                return Err(PtwmCoreError::InvalidContainer(format!(
                                    "shared state {}: applies_to_mask 0b{:08b} \
                                     does not cover plane role {:?}",
                                    plane_rec.state_info, entry.applies_to_mask, plane_rec.role,
                                )));
                            }
                        }
                        let computed = xxh64(&entry.state_bytes, 0);
                        if computed != entry.state_xxhash64 {
                            return Err(PtwmCoreError::InvalidContainer(format!(
                                "shared state xxhash64 mismatch for \
                                 shared_state_id={} (codec={:?}): \
                                 stored={:#018x} computed={:#018x}",
                                plane_rec.state_info,
                                entry.codec_id,
                                entry.state_xxhash64,
                                computed,
                            )));
                        }
                        (entry.state_bytes.clone(), entry.state_format_version)
                    }
                    crate::codec::StateSource::External => {
                        return Err(PtwmCoreError::InvalidContainer(
                            "External state not supported".into(),
                        ));
                    }
                };
                // Decoded length comes from the propagated terminal descriptor.
                let decoded_len = descriptor.length_bytes as usize;
                codec.decode(
                    state_version,
                    &state_bytes,
                    &plane_rec.payload_bytes,
                    &plane_rec.layout,
                    decoded_len,
                )?
            };

            // Plane CRC enforcement (when emitted by writer).
            if let Some(expected_crc) = plane_rec.crc32 {
                let got = crc32fast::hash(&plane_bytes);
                if got != expected_crc {
                    return Err(PtwmCoreError::InvalidContainer("plane CRC mismatch".into()));
                }
            }

            terminal_planes.push(Plane {
                bytes: Arc::from(plane_bytes.into_boxed_slice()),
                descriptor,
            });
        }

        // Run the inverse chain. Cross-tensor `deps` are not wired through
        // this decode path; a chain that contains a delta op will fail with
        // a runtime error here, which is the correct behaviour.
        let inv_ctx = InverseContext {
            terminal_planes: &terminal_planes,
            deps: &[],
        };
        let source_plane = inverse_chain(chain, &inv_ctx)?;
        let raw: Vec<u8> = source_plane.bytes.as_ref().to_vec();

        // Tensor-level payload hash validation.
        if rec.flags & TENSOR_FLAG_PAYLOAD_HASH != 0
            && let Some(expected) = rec.payload_hash
        {
            let got = xxhash_rust::xxh64::xxh64(&raw, 0);
            if got != expected {
                return Err(PtwmCoreError::InvalidContainer(
                    "tensor payload hash mismatch".into(),
                ));
            }
        }
        Ok(raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{CodecId, StateSource};
    use crate::layout::PlaneLayout;
    use crate::plane_record::PlaneRecord;
    use crate::tensor_record::{CHAIN_REF_INLINE_BIT, parse_tensor_record};
    use crate::types::PlaneRole;

    fn sample_tensor_record(name: &str) -> TensorRecord {
        TensorRecord {
            // 0x0010: chain_ref slot used as registry index for these tests.
            chain_ref: 0x0010,
            dtype_code: 0x0010,
            input_format: 0,
            flags: 0,
            orig_size: 16,
            name: name.to_string(),
            payload_hash: None,
            inline_chain: None,
            dependencies: Vec::new(),
            tensor_metadata: None,
            terminals: vec![PlaneRecord {
                role: PlaneRole::Value,
                codec_id: CodecId::Identity,
                codec_table_idx: 0,
                state_source: StateSource::None,
                state_version: 0,
                state_info: 0,
                payload_len: 8,
                crc32: None,
                chunk_table: None,
                external_state: None,
                inline_state_bytes: Vec::new(),
                payload_bytes: vec![0x11; 8],
                layout: PlaneLayout::Flat,
            }],
        }
    }

    fn sample_prelude_entry() -> PreludeEntry {
        PreludeEntry {
            shared_state_id: 0,
            codec_id: CodecId::Huffman,
            state_format_version: 0,
            applies_to_mask: 0b0001,
            state_xxhash64: 0,
            name: String::new(),
            state_bytes: vec![1, 2, 3],
        }
    }

    fn build_container() -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let mut w = ContainerWriter::new(cursor).unwrap();
            w.set_prelude(&[sample_prelude_entry()]).unwrap();
            w.append_tensor(&sample_tensor_record("a")).unwrap();
            w.append_tensor(&sample_tensor_record("b")).unwrap();
            w.finalize().unwrap();
        }
        buf
    }

    #[test]
    fn container_roundtrip() {
        let buf = build_container();
        let reader = ContainerReader::open(&buf).unwrap();

        for name in ["a", "b"] {
            let record_bytes = reader
                .get_tensor_record_bytes(name)
                .expect("tensor not found");
            let (parsed, consumed) = parse_tensor_record(record_bytes).unwrap();
            assert_eq!(consumed, record_bytes.len());

            let expected = sample_tensor_record(name);
            assert_eq!(parsed.chain_ref, expected.chain_ref);
            assert_eq!(parsed.dtype_code, expected.dtype_code);
            assert_eq!(parsed.input_format, expected.input_format);
            assert_eq!(parsed.orig_size, expected.orig_size);
            assert_eq!(parsed.name, expected.name);
            assert_eq!(parsed.terminals.len(), expected.terminals.len());
        }
    }

    #[test]
    fn header_offsets_correct_after_finalize() {
        let buf = build_container();
        let reader = ContainerReader::open(&buf).unwrap();

        // v1 header fields.
        assert!(
            reader.header.shared_prelude_offset > 0,
            "shared_prelude_offset must be non-zero"
        );
        assert!(
            reader.header.tensor_index_offset > 0,
            "tensor_index_offset must be non-zero"
        );
        // Derived reader fields.
        assert!(
            reader.tensors_offset > 0,
            "tensors_offset (derived) must be non-zero"
        );
        assert!(
            reader.header.shared_prelude_offset < reader.tensors_offset,
            "prelude must come before tensors"
        );
        assert!(
            reader.tensors_offset < reader.header.tensor_index_offset,
            "tensors must come before index"
        );
        assert_eq!(reader.num_tensors, 2);
    }

    #[test]
    fn truncated_file_rejected() {
        let buf = build_container();
        let truncated = &buf[..buf.len() - 1];
        let err = ContainerReader::open(truncated).unwrap_err();
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("sentinel") || msg.contains("truncated"),
            "expected sentinel/truncated error, got: {msg}"
        );
    }

    #[test]
    fn corrupt_header_rejected() {
        let mut buf = build_container();
        // Corrupt the magic bytes (byte 1 = 'P' in MAGIC_PTWM).
        // The v1 header has no CRC; the magic is the sole integrity check.
        buf[1] ^= 0xFF;
        let err = ContainerReader::open(&buf).unwrap_err();
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("magic") || msg.contains("header"),
            "expected magic/header error, got: {msg}"
        );
    }

    #[test]
    fn truncated_header_with_sentinel_rejected_cleanly() {
        // Regression: a 9-byte file containing just the EOF sentinel preceded
        // by a single byte used to panic with "range end index 76 out of
        // range for slice of length 9" when Header::from_bytes was called
        // before bounds-checking buf.len(). cargo-fuzz crash:
        // crash-dd18bcff24a8e291256a7de624ce2818bc0890e1.
        let mut buf = vec![0x03];
        buf.extend_from_slice(&crate::index::EOF_SENTINEL);
        let err = ContainerReader::open(&buf).unwrap_err();
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("shorter than header") || msg.contains("header too short"),
            "expected header-length error, got: {msg}"
        );
    }

    #[test]
    fn out_of_range_offsets_rejected_cleanly() {
        // Regression: a header whose declared tensor_index_offset points past
        // the end of the file used to panic at the slice op.
        let mut buf = build_container();
        // In the v1 header, tensor_index_offset lives at bytes 59..67.
        // The v1 header has no self-CRC, so we can patch without recomputing anything.
        let bogus = (buf.len() as u64 + 1).to_le_bytes();
        buf[59..67].copy_from_slice(&bogus);
        let err = ContainerReader::open(&buf).unwrap_err();
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("offset") || msg.contains("out of bounds"),
            "expected offset error, got: {msg}"
        );
    }

    /// Regression guard for hash enforcement: a shared-state entry whose
    /// stored `state_xxhash64` does not match the actual `state_bytes` must
    /// cause `decode_tensor` to return `InvalidContainer` rather than silently
    /// decoding garbage.
    #[test]
    fn shared_state_xxhash64_mismatch_rejected() {
        use crate::codec::CodecId;
        use crate::plane_record::PlaneRecord;
        use crate::types::PlaneRole;

        // Construct a prelude entry with a hashed state but a deliberately wrong hash value.
        let real_state_bytes = vec![0xAA, 0xBB, 0xCC, 0xDD];
        let corrupt_hash: u64 = 0xDEAD_BEEF_DEAD_BEEF;
        // Verify the corrupt hash is actually wrong so the test is meaningful.
        assert_ne!(
            xxh64(&real_state_bytes, 0),
            corrupt_hash,
            "corrupt_hash must differ from the real hash"
        );
        let prelude_entry = PreludeEntry {
            shared_state_id: 0,
            codec_id: CodecId::Identity,
            state_format_version: 1,
            applies_to_mask: 0b0001, // covers PlaneRole::Value (bit 0)
            state_xxhash64: corrupt_hash,
            name: String::new(),
            state_bytes: real_state_bytes,
        };

        // Build a tensor with an inline Source(int8, 4 bytes) → BytePassthrough →
        // Value{IntN{8}} chain so the chain-based decode path can resolve it,
        // then have its single Value plane use StateSource::Shared pointing at
        // shared_state_id=0 so we hit the hash-check on shared state.
        let chain = v3_passthrough_chain(4);

        let mut buf: Vec<u8> = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let mut w = ContainerWriter::new(cursor).unwrap();

            // Setup the registry so we get a valid codec_table_idx for Identity
            let registry = ChainRegistry::new();
            w.set_chain_registry(registry).unwrap();
            let canonical_id = crate::extension::builtin_canonical_id("identity");
            let codec_table_idx = w
                .codec_table_idx_for(&canonical_id)
                .expect("Identity must be in registry");

            let tensor = TensorRecord {
                // High bit set → inline_chain is consulted instead of registry.
                chain_ref: CHAIN_REF_INLINE_BIT,
                dtype_code: 0x0005,
                input_format: 0,
                flags: 0,
                orig_size: 4,
                name: "t".to_string(),
                payload_hash: None,
                inline_chain: Some(chain),
                dependencies: Vec::new(),
                tensor_metadata: None,
                terminals: vec![PlaneRecord {
                    role: PlaneRole::Value,
                    codec_id: CodecId::Identity,
                    codec_table_idx,
                    state_source: StateSource::Shared,
                    state_version: 1, // matches prelude entry's state_format_version
                    state_info: 0,    // shared_state_id = 0
                    payload_len: 4,
                    crc32: None,
                    chunk_table: None,
                    external_state: None,
                    inline_state_bytes: Vec::new(),
                    payload_bytes: vec![0u8; 4],
                    layout: crate::layout::PlaneLayout::Flat,
                }],
            };

            w.set_prelude(&[prelude_entry]).unwrap();
            w.append_tensor(&tensor).unwrap();
            w.finalize().unwrap();
        }

        let reader = ContainerReader::open(&buf).expect("container must open cleanly");
        let err = reader
            .decode_tensor("t")
            .expect_err("decode_tensor must fail when state_xxhash64 does not match state_bytes");
        let msg = err.to_string();
        assert!(
            msg.contains("xxhash64 mismatch"),
            "expected 'xxhash64 mismatch' in error, got: {msg}"
        );
    }

    #[test]
    fn parse_index_rejects_oversized_count() {
        // Regression: parse_index used to compute `expected_len = 8 + count *
        // 24` which wraps in release mode for adversarial counts (e.g.
        // 0x0606_0606_0606_0606), then panic at Vec::with_capacity. Now the
        // count is rejected before allocation.
        use crate::index::parse_index;
        let mut buf = vec![0u8; 32];
        buf[..8].copy_from_slice(&0x0606_0606_0606_0606u64.to_le_bytes());
        let err = parse_index(&buf).unwrap_err();
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("count exceeds") || msg.contains("truncated"),
            "expected count/truncated error, got: {msg}"
        );
    }

    // ── ChainRegistry tests ───────────────────────────────────────────────────

    fn make_minimal_chain(id: u8) -> Chain {
        use crate::chain::{ChainNode, TerminalRef};
        use crate::transforms::op::OpId;
        use crate::types::role::{Role, ValueFormat};
        Chain {
            nodes: vec![ChainNode {
                op: OpId::BytePassthrough,
                params: vec![id], // use id as a distinguishing param byte
            }],
            edges: vec![],
            terminals: vec![TerminalRef {
                node_idx: 0,
                output_idx: 0,
                role: Role::Value {
                    format: ValueFormat::Fp4E2m1,
                },
            }],
        }
    }

    #[test]
    fn chain_registry_roundtrip() {
        use crate::extension::{ExtensionTable, ExtensionTableBuilder};
        let registry = ChainRegistry {
            chains: vec![
                (0, make_minimal_chain(0)),
                (1, make_minimal_chain(1)),
                (42, make_minimal_chain(42)),
            ],
        };
        let mut builder = ExtensionTableBuilder::new();
        let mut buf = Vec::new();
        registry.write(&mut builder, &mut buf).unwrap();
        let table = ExtensionTable {
            entries: builder.finish(),
        };
        let (parsed, consumed) = ChainRegistry::read(&buf, &table).unwrap();
        assert_eq!(consumed, buf.len());
        assert_eq!(parsed.chains.len(), 3);
        assert_eq!(parsed.chains[0].0, 0);
        assert_eq!(parsed.chains[1].0, 1);
        assert_eq!(parsed.chains[2].0, 42);
        assert_eq!(parsed, registry);
    }

    #[test]
    fn chain_registry_lookup() {
        let chain_a = make_minimal_chain(10);
        let chain_b = make_minimal_chain(20);
        let registry = ChainRegistry {
            chains: vec![(5, chain_a.clone()), (99, chain_b.clone())],
        };
        assert_eq!(registry.lookup(5), Some(&chain_a));
        assert_eq!(registry.lookup(99), Some(&chain_b));
        assert_eq!(registry.lookup(0), None);
        assert_eq!(registry.lookup(100), None);
    }

    #[test]
    fn chain_registry_truncated_rejected() {
        use crate::extension::{ExtensionTable, ExtensionTableBuilder};
        let registry = ChainRegistry {
            chains: vec![(0, make_minimal_chain(0)), (1, make_minimal_chain(1))],
        };
        let mut builder = ExtensionTableBuilder::new();
        let mut buf = Vec::new();
        registry.write(&mut builder, &mut buf).unwrap();
        let table = ExtensionTable {
            entries: builder.finish(),
        };
        // Truncating to 1 byte should fail (can't even read num_chains).
        assert!(ChainRegistry::read(&buf[..1], &table).is_err());
        // Truncating partway through should also fail.
        for truncate_at in 2..buf.len() {
            // May or may not succeed depending on where we cut; but if the first
            // chain is intact, a second pass would succeed on a partial second chain.
            // We only assert: if both chains are declared, a truncation mid-chain fails.
            if let Ok((reg, _)) = ChainRegistry::read(&buf[..truncate_at], &table) {
                // A partial read that claims success must have read fewer chains.
                assert!(reg.chains.len() < 2);
            }
        }
    }

    // ── End-to-end container test ──────────────────────────────────────────

    /// Construct a minimal PlaneRecord with explicit payload bytes.
    fn terminal_record(payload: Vec<u8>) -> PlaneRecord {
        PlaneRecord {
            role: PlaneRole::Value,
            codec_id: CodecId::Identity,
            codec_table_idx: 0,
            state_source: StateSource::None,
            state_version: 0,
            state_info: 0,
            payload_len: payload.len() as u64,
            crc32: None,
            chunk_table: None,
            external_state: None,
            inline_state_bytes: Vec::new(),
            payload_bytes: payload,
            layout: PlaneLayout::Flat,
        }
    }

    /// Write a synthetic container with:
    /// - 2 chains in the registry (chain_id 0, chain_id 1).
    /// - Tensor A: chain_ref = 0x0000 (registry ref, chain_id 0), 2 terminals.
    /// - Tensor B: chain_ref = 0x8001 (inline bit set), inline chain, 2 terminals.
    /// Read it back and assert structural equality.
    #[test]
    fn container_v2_end_to_end_roundtrip() {
        let chain_a = make_minimal_chain(0);
        let chain_b = make_minimal_chain(1);

        let registry = ChainRegistry {
            chains: vec![(0, chain_a.clone()), (1, chain_b.clone())],
        };

        let tensor_a = TensorRecord {
            chain_ref: 0x0000, // no inline bit → chain_id 0 from registry
            dtype_code: 0x0010,
            input_format: 0,
            flags: 0,
            orig_size: 16,
            name: "tensor_a".to_string(),
            payload_hash: None,
            inline_chain: None,
            dependencies: Vec::new(),
            tensor_metadata: None,
            terminals: vec![
                terminal_record(vec![0xAA; 8]),
                terminal_record(vec![0xBB; 8]),
            ],
        };

        let tensor_b = TensorRecord {
            chain_ref: 0x8001, // inline bit set
            dtype_code: 0x0011,
            input_format: 0,
            flags: 0,
            orig_size: 24,
            name: "tensor_b".to_string(),
            payload_hash: None,
            inline_chain: Some(chain_b.clone()),
            dependencies: Vec::new(),
            tensor_metadata: None,
            terminals: vec![
                terminal_record(vec![0xCC; 12]),
                terminal_record(vec![0xDD; 12]),
            ],
        };

        // Write the container.
        let mut buf: Vec<u8> = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let mut w = ContainerWriter::new(cursor).unwrap();
            w.set_chain_registry(registry.clone()).unwrap();
            w.append_tensor(&tensor_a).unwrap();
            w.append_tensor(&tensor_b).unwrap();
            w.finalize().unwrap();
        }

        // Read it back.
        let reader = ContainerReader::open(&buf).unwrap();
        let hdr = &reader.header;

        // Header sanity (v1 uses derived reader fields for offsets not in the header).
        assert_eq!(reader.num_tensors, 2);
        assert!(
            reader.chain_registry_offset > 0,
            "chain_registry_offset (derived) must be set"
        );
        assert!(hdr.shared_prelude_offset < reader.chain_registry_offset);
        assert!(reader.chain_registry_offset < reader.tensors_offset);
        assert!(reader.tensors_offset < hdr.tensor_index_offset);

        // Chain registry.
        assert_eq!(reader.chain_registry.chains.len(), 2);
        assert_eq!(reader.chain_registry.lookup(0), Some(&chain_a));
        assert_eq!(reader.chain_registry.lookup(1), Some(&chain_b));
        assert_eq!(reader.chain_registry.lookup(99), None);

        // Tensor A: registry ref, no inline chain.
        let bytes_a = reader.get_tensor_record_bytes("tensor_a").unwrap();
        let (rec_a, consumed_a) = parse_tensor_record(bytes_a).unwrap();
        assert_eq!(consumed_a, bytes_a.len());
        assert_eq!(rec_a.chain_ref, 0x0000);
        assert!(rec_a.inline_chain.is_none());
        assert_eq!(rec_a.name, "tensor_a");
        assert_eq!(rec_a.terminals.len(), 2);
        assert_eq!(rec_a.terminals[0].payload_bytes, vec![0xAA; 8]);
        assert_eq!(rec_a.terminals[1].payload_bytes, vec![0xBB; 8]);

        // Tensor B: inline chain present.
        let bytes_b = reader.get_tensor_record_bytes("tensor_b").unwrap();
        let (rec_b, consumed_b) = parse_tensor_record(bytes_b).unwrap();
        assert_eq!(consumed_b, bytes_b.len());
        assert_eq!(rec_b.chain_ref, 0x8001);
        assert_eq!(rec_b.inline_chain, Some(chain_b));
        assert_eq!(rec_b.name, "tensor_b");
        assert_eq!(rec_b.terminals.len(), 2);
        assert_eq!(rec_b.terminals[0].payload_bytes, vec![0xCC; 12]);
        assert_eq!(rec_b.terminals[1].payload_bytes, vec![0xDD; 12]);
    }

    // ── decode_tensor ────────────────────────────────────────────────────

    /// Build a `Source(int8) → BytePassthrough → Terminal{Value(IntN bits=8)}`
    /// chain.
    fn v3_passthrough_chain(n_bytes: u32) -> Chain {
        use crate::chain::{ChainEdge, ChainNode, TerminalRef};
        use crate::transforms::op::OpId;
        use crate::types::role::{Role, ValueFormat};
        let mut params = vec![1u8];
        params.extend_from_slice(&n_bytes.to_le_bytes());
        params.extend_from_slice(&0x0005u16.to_le_bytes()); // int8 → 1 byte/elem
        Chain {
            nodes: vec![
                ChainNode {
                    op: OpId::Source,
                    params,
                },
                ChainNode {
                    op: OpId::BytePassthrough,
                    params: vec![],
                },
            ],
            edges: vec![ChainEdge {
                src_node: 0,
                src_output_idx: 0,
                dst_node: 1,
                dst_input_idx: 0,
                role_override: Some(Role::Value {
                    format: ValueFormat::IntN { bits: 8 },
                }),
                vendor_bytes: vec![],
            }],
            terminals: vec![TerminalRef {
                node_idx: 1,
                output_idx: 0,
                role: Role::Value {
                    format: ValueFormat::IntN { bits: 8 },
                },
            }],
        }
    }

    /// Build a `Source(fp16) → BitReorderIeee16 → ByteSplit{n=2} → 2× Terminal`
    /// chain (the canonical FP16 split chain).
    fn v3_fp16_split_chain(n_elements: u32) -> Chain {
        use crate::chain::{ChainEdge, ChainNode, TerminalRef};
        use crate::transforms::op::OpId;
        use crate::types::role::Role;
        let mut params = vec![1u8];
        params.extend_from_slice(&n_elements.to_le_bytes());
        params.extend_from_slice(&0x0002u16.to_le_bytes()); // Float16 (chain-internal)
        Chain {
            nodes: vec![
                ChainNode {
                    op: OpId::Source,
                    params,
                },
                ChainNode {
                    op: OpId::BitReorderIeee16,
                    params: vec![],
                },
                ChainNode {
                    op: OpId::ByteSplit,
                    params: vec![2u8],
                },
            ],
            edges: vec![
                ChainEdge {
                    src_node: 0,
                    src_output_idx: 0,
                    dst_node: 1,
                    dst_input_idx: 0,
                    role_override: None,
                    vendor_bytes: vec![],
                },
                ChainEdge {
                    src_node: 1,
                    src_output_idx: 0,
                    dst_node: 2,
                    dst_input_idx: 0,
                    role_override: None,
                    vendor_bytes: vec![],
                },
            ],
            terminals: vec![
                TerminalRef {
                    node_idx: 2,
                    output_idx: 0,
                    role: Role::IntegerByte { index: 0, of: 2 },
                },
                TerminalRef {
                    node_idx: 2,
                    output_idx: 1,
                    role: Role::IntegerByte { index: 1, of: 2 },
                },
            ],
        }
    }

    #[test]
    fn decode_tensor_roundtrip_two_tensors() {
        use crate::compressor::{CompressorOptions, InputTensor, compress_model};
        use std::io::Cursor;

        let raw_a: Vec<u8> = (0u8..32).collect();
        let raw_b: Vec<u8> = (0u8..32).rev().collect();
        let chain = v3_passthrough_chain(32);

        let inputs = vec![
            InputTensor {
                name: "a".into(),
                candidate_chains: vec![chain.clone()],
                dtype_code: 0x0005,
                input_format: 0,
                raw_bytes: &raw_a,
                shape: None,
                dtype_name: None,
                delta_reference_blake3: None,
            },
            InputTensor {
                name: "b".into(),
                candidate_chains: vec![chain.clone()],
                dtype_code: 0x0005,
                input_format: 0,
                raw_bytes: &raw_b,
                shape: None,
                dtype_name: None,
                delta_reference_blake3: None,
            },
        ];

        let mut buf: Vec<u8> = Vec::new();
        compress_model(
            Cursor::new(&mut buf),
            &inputs,
            CompressorOptions {
                method_hint: 1,
                emit_payload_hash: false,
                emit_plane_crc: true,
                forced_codec: None,
                allow_codec_ids: None,
                chunk_size: None,
            },
        )
        .expect("compress succeeds");

        let reader = ContainerReader::open(&buf).expect("open succeeds");
        let got_a = reader.decode_tensor("a").expect("decode a");
        let got_b = reader.decode_tensor("b").expect("decode b");
        assert_eq!(got_a, raw_a, "tensor a must roundtrip bit-exactly");
        assert_eq!(got_b, raw_b, "tensor b must roundtrip bit-exactly");
    }

    #[test]
    fn decode_model_roundtrip_chunked() {
        // Multiple tensors, forced rANS + chunked output, decoded via the
        // inter-tensor parallel decode_model — exercises both the parallel
        // chunk decode and the parallel per-tensor decode at once.
        use crate::codec::CodecId;
        use crate::compressor::{CompressorOptions, InputTensor, compress_model};
        use std::io::Cursor;

        let raws: Vec<Vec<u8>> = (0..6u32)
            .map(|t| (0..2048u32).map(|i| ((i + t * 7) % 251) as u8).collect())
            .collect();
        let chains: Vec<_> = raws
            .iter()
            .map(|r| v3_passthrough_chain(r.len() as u32))
            .collect();
        let inputs: Vec<InputTensor> = raws
            .iter()
            .enumerate()
            .map(|(t, r)| InputTensor {
                name: format!("t{t}"),
                candidate_chains: vec![chains[t].clone()],
                dtype_code: 0x0005,
                input_format: 0,
                raw_bytes: r,
                shape: None,
                dtype_name: None,
                delta_reference_blake3: None,
            })
            .collect();

        let mut buf: Vec<u8> = Vec::new();
        compress_model(
            Cursor::new(&mut buf),
            &inputs,
            CompressorOptions {
                method_hint: 1,
                emit_payload_hash: false,
                emit_plane_crc: true,
                forced_codec: Some(CodecId::Rans),
                allow_codec_ids: None,
                chunk_size: Some(256),
            },
        )
        .expect("compress succeeds");

        let reader = ContainerReader::open(&buf).expect("open succeeds");
        let decoded = reader.decode_model().expect("decode_model");
        assert_eq!(decoded.len(), raws.len());
        for (name, bytes) in &decoded {
            let t: usize = name.trim_start_matches('t').parse().unwrap();
            assert_eq!(bytes, &raws[t], "tensor {name} must roundtrip bit-exactly");
        }
    }

    #[test]
    fn decode_tensor_chain_with_split() {
        use crate::compressor::{CompressorOptions, InputTensor, compress_model};
        use std::io::Cursor;

        // 1024 bytes of fake fp16 data with mild structure (so trial-encode
        // has something to chew on but Identity is still acceptable).
        let n_elements: u32 = 512;
        let mut raw: Vec<u8> = Vec::with_capacity((n_elements * 2) as usize);
        for i in 0..n_elements {
            let bits: u16 = 0x3800 | ((i as u16) & 0x03FF);
            raw.extend_from_slice(&bits.to_le_bytes());
        }

        let chain = v3_fp16_split_chain(n_elements);
        let inputs = vec![InputTensor {
            name: "fp16.weight".into(),
            candidate_chains: vec![chain],
            dtype_code: 0x0002, // Float16 (chain-internal)
            input_format: 0,
            raw_bytes: &raw,
            shape: None,
            dtype_name: None,
            delta_reference_blake3: None,
        }];

        let mut buf: Vec<u8> = Vec::new();
        compress_model(
            Cursor::new(&mut buf),
            &inputs,
            CompressorOptions {
                method_hint: 1,
                emit_payload_hash: true,
                emit_plane_crc: true,
                forced_codec: None,
                allow_codec_ids: None,
                chunk_size: None,
            },
        )
        .expect("compress succeeds");

        let reader = ContainerReader::open(&buf).expect("open succeeds");
        let got = reader
            .decode_tensor("fp16.weight")
            .expect("decode fp16.weight");
        assert_eq!(got, raw, "split-chain roundtrip must be bit-exact");
    }

    // ── Extension Table integration tests ─────────────────────────────────────

    fn sample_extension_table_entry() -> crate::extension::ExtensionTableEntry {
        use crate::extension::capability::{CapabilityMap, CapabilityValue};
        use crate::extension::table::{Attestation, FLAVOR_NATIVE, FLAVOR_WASM};
        use crate::extension::{CanonicalId, Kind, Lifecycle};
        let mut caps = CapabilityMap::new();
        caps.set("determinism", CapabilityValue::Bool(true));
        caps.set("hardware_class", CapabilityValue::Text("cpu".into()));
        crate::extension::ExtensionTableEntry {
            canonical_id: CanonicalId::derive(&[0x22; 32], "bar", "0.1.0"),
            human_label: "io.example.bar".into(),
            kind: Kind::Transform,
            abi_version: 1,
            lifecycle: Lifecycle::Thread,
            flavor_hints: FLAVOR_WASM | FLAVOR_NATIVE,
            capabilities: caps,
            attestation: Attestation::PgpSignature(vec![0xCD; 32]),
            install_hint: Some("https://example.com/bar".into()),
            embedded_wasm_offset: None,
            embedded_wasm_length: None,
        }
    }

    #[test]
    fn write_then_read_yields_identical_extension_table() {
        let table = ExtensionTable {
            entries: vec![sample_extension_table_entry()],
        };

        let mut buf: Vec<u8> = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let mut w = ContainerWriter::new(cursor).unwrap();
            w.set_extension_table(table.clone());
            w.append_tensor(&sample_tensor_record("t")).unwrap();
            w.finalize().unwrap();
        }

        // Use open_partial because the sample entry is non-builtin (pubkey
        // [0x22; 32]) and no extensions are installed in the test environment.
        // The test is concerned with extension table round-trip fidelity, not
        // the missing-flavor check.
        let reader = ContainerReader::open_partial(&buf).expect("open_partial must succeed");
        assert_eq!(
            reader.extension_table, table,
            "extension table must round-trip identically"
        );
        // The missing-extension entry is non-builtin, so it should be reported.
        assert_eq!(
            reader.missing_extensions.len(),
            1,
            "one non-builtin entry should appear in missing_extensions"
        );
        // Also verify header fields are consistent.
        assert_eq!(reader.header.extension_table_offset, HEADER_LEN as u64);
        assert!(reader.header.extension_table_length > 0);
        // Hash stored in the header must match what we'd compute from the table bytes.
        let et_bytes = table.to_bytes().unwrap();
        let expected_hash = ExtensionTable::hash(&et_bytes);
        assert_eq!(reader.header.extension_table_hash, expected_hash);
    }

    #[test]
    fn corrupted_extension_table_hash_is_detected() {
        let table = ExtensionTable {
            entries: vec![sample_extension_table_entry()],
        };

        let mut buf: Vec<u8> = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let mut w = ContainerWriter::new(cursor).unwrap();
            w.set_extension_table(table);
            w.append_tensor(&sample_tensor_record("t")).unwrap();
            w.finalize().unwrap();
        }

        // Read the header to learn the extension table region.
        let header = Header::from_bytes(&buf[..HEADER_LEN]).unwrap();
        assert!(header.extension_table_length > 0, "table must be non-empty");
        let et_start = header.extension_table_offset as usize;
        let et_end = et_start + header.extension_table_length as usize;

        // Flip one byte inside the extension table region.
        buf[et_start] ^= 0xFF;

        let err = ContainerReader::open(&buf)
            .expect_err("open must fail when extension table bytes are corrupted");
        assert!(
            matches!(err, PtwmCoreError::ExtensionTableHashMismatch),
            "expected ExtensionTableHashMismatch, got: {err}"
        );
        // Silence unused-variable warnings for bounds locals.
        let _ = et_end;
    }

    #[test]
    fn decode_tensor_payload_hash_mismatch() {
        use crate::compressor::{CompressorOptions, InputTensor, compress_model};
        use crate::header::HEADER_LEN;
        use std::io::Cursor;

        // Build a 1-tensor container with payload_hash on. The chain is
        // BytePassthrough+Identity codec, so the terminal payload bytes are
        // identical to the raw tensor bytes. Mutating one byte inside the
        // terminal's payload region therefore flips one byte in the
        // reconstruction and the payload-hash check fails.
        let raw: Vec<u8> = (0u8..64).collect();
        let chain = v3_passthrough_chain(raw.len() as u32);
        let inputs = vec![InputTensor {
            name: "t".into(),
            candidate_chains: vec![chain],
            dtype_code: 0x0005,
            input_format: 0,
            raw_bytes: &raw,
            shape: None,
            dtype_name: None,
            delta_reference_blake3: None,
        }];

        let mut buf: Vec<u8> = Vec::new();
        compress_model(
            Cursor::new(&mut buf),
            &inputs,
            CompressorOptions {
                method_hint: 1,
                emit_payload_hash: true,
                emit_plane_crc: false,
                forced_codec: Some(crate::codec::CodecId::Identity),
                allow_codec_ids: None,
                chunk_size: None,
            },
        )
        .expect("compress succeeds");

        // Sanity: the clean container decodes correctly first.
        let clean_reader = ContainerReader::open(&buf).expect("open clean");
        assert_eq!(clean_reader.decode_tensor("t").unwrap(), raw);

        // Locate one of the raw tensor bytes inside `buf`. With Identity codec
        // and BytePassthrough, the terminal's payload contains the exact raw
        // bytes verbatim. We pick a distinctive, non-zero byte (raw[7] = 0x07)
        // and find it in the tensors region (after HEADER + prelude + chain
        // registry). Searching the whole buffer is fine — but we restrict to
        // the tensors region to make sure we hit a payload byte rather than
        // header metadata or index.
        let tensors_start = clean_reader.tensors_offset as usize;
        let tensors_end = clean_reader.header.tensor_index_offset as usize;
        assert!(tensors_start >= HEADER_LEN);
        assert!(tensors_end > tensors_start);

        // The raw bytes 0..64 appear contiguously in the tensor record's
        // payload_bytes. Find that 64-byte sequence in the tensors region and
        // flip a byte right in the middle.
        let target: Vec<u8> = (0u8..64).collect();
        let region = &buf[tensors_start..tensors_end];
        let pos = region
            .windows(target.len())
            .position(|w| w == target.as_slice())
            .expect("raw payload sequence must appear inside the tensor region");
        let mutate_at = tensors_start + pos + 30; // mid-payload
        // Flip the byte (XOR with 0x55) so it definitely differs.
        buf[mutate_at] ^= 0x55;

        let reader = ContainerReader::open(&buf).expect("open mutated container");
        let err = reader
            .decode_tensor("t")
            .expect_err("decode_tensor must fail when payload is corrupted");
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("payload hash mismatch") || msg.contains("crc"),
            "expected payload hash or CRC mismatch error, got: {msg}"
        );
    }

    // ── Missing-flavor tests ───────────────────────────────────────────────

    /// Build a container whose Extension Table references a single non-builtin
    /// entry with a non-zero `flavor_hints`. Since no extension is installed in
    /// the test environment, `open` must return `MissingFlavor` and
    /// `open_partial` must succeed with `missing_extensions` populated.
    fn build_container_with_nonbuiltin_extension() -> Vec<u8> {
        use crate::extension::table::FLAVOR_WASM;
        use crate::extension::{Attestation, CapabilityMap, ExtensionTableEntry, Kind, Lifecycle};

        // Use a non-builtin pubkey so `is_builtin` returns false.
        let fake_id = crate::extension::CanonicalId::derive(&[0xCC; 32], "fake", "1.0.0");
        let fake_entry = ExtensionTableEntry {
            canonical_id: fake_id,
            human_label: "io.example.fake@1.0.0".into(),
            kind: Kind::Transform,
            abi_version: 1,
            lifecycle: Lifecycle::Thread,
            flavor_hints: FLAVOR_WASM,
            capabilities: CapabilityMap::new(),
            attestation: Attestation::PgpSignature(Vec::new()),
            install_hint: Some("https://example.com/fake-1.0.0.tar.zst".into()),
            embedded_wasm_offset: None,
            embedded_wasm_length: None,
        };

        let mut buf: Vec<u8> = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let mut w = ContainerWriter::new(cursor).unwrap();
            // Override the extension table with a table that has our fake entry.
            w.set_extension_table(crate::extension::ExtensionTable {
                entries: vec![fake_entry],
            });
            w.append_tensor(&sample_tensor_record("a")).unwrap();
            w.finalize().unwrap();
        }
        buf
    }

    #[test]
    fn open_fails_when_extension_missing() {
        let buf = build_container_with_nonbuiltin_extension();
        let err = ContainerReader::open(&buf)
            .expect_err("open must fail when a non-builtin extension is missing");
        match err {
            PtwmCoreError::MissingFlavor { needs } => {
                assert_eq!(needs.len(), 1, "expected exactly one missing entry");
                assert_eq!(needs[0].label, "io.example.fake@1.0.0");
                assert!(
                    needs[0].install_hint.is_some(),
                    "install_hint should be propagated"
                );
            }
            other => panic!("expected MissingFlavor, got: {other}"),
        }
    }

    #[test]
    fn open_partial_succeeds_with_missing_extension() {
        let buf = build_container_with_nonbuiltin_extension();
        let reader = ContainerReader::open_partial(&buf)
            .expect("open_partial must succeed even with missing extensions");
        assert_eq!(
            reader.missing_extensions.len(),
            1,
            "expected exactly one missing entry in missing_extensions"
        );
        assert_eq!(reader.missing_extensions[0].label, "io.example.fake@1.0.0");
        // The reader is otherwise functional.
        assert_eq!(reader.num_tensors, 1);
    }

    // ── Verifier / trust tests ─────────────────────────────────────────────

    /// Serialise all XDG-mutating tests in this module so they don't race.
    static XDG_CONFIG_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// RAII guard that points XDG_CONFIG_HOME at a temp directory for the
    /// duration of a test and restores the previous value on drop.
    fn with_clean_xdg_config(dir: &std::path::Path) -> impl Drop + '_ {
        struct Guard {
            prev: Option<String>,
            _lock: std::sync::MutexGuard<'static, ()>,
        }
        impl Drop for Guard {
            fn drop(&mut self) {
                match &self.prev {
                    Some(v) => unsafe { std::env::set_var("XDG_CONFIG_HOME", v) },
                    None => unsafe { std::env::remove_var("XDG_CONFIG_HOME") },
                }
            }
        }
        let lock = XDG_CONFIG_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let prev = std::env::var("XDG_CONFIG_HOME").ok();
        unsafe { std::env::set_var("XDG_CONFIG_HOME", dir) };
        Guard { prev, _lock: lock }
    }

    /// A container whose Extension Table contains only built-in entries must
    /// open cleanly without the verifier raising `ContributionUntrusted`.
    #[test]
    fn open_succeeds_for_builtin_only_table() {
        use tempfile::tempdir;

        let tmp = tempdir().unwrap();
        let _g = with_clean_xdg_config(tmp.path());

        // Build a container with one built-in entry in the Extension Table.
        let mut buf: Vec<u8> = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let mut w = ContainerWriter::new(cursor).unwrap();
            w.set_extension_table(crate::extension::ExtensionTable {
                entries: vec![crate::extension::builtin_entries()[0].clone()],
            });
            w.append_tensor(&sample_tensor_record("t")).unwrap();
            w.finalize().unwrap();
        }

        let reader = ContainerReader::open(&buf)
            .expect("open must succeed for a builtin-only Extension Table");
        assert!(
            reader.missing_extensions.is_empty(),
            "no missing extensions expected for builtin entries"
        );
    }

    /// A container referencing a non-builtin contribution whose author is not
    /// in the active keyring (and is not installed on this host) must return
    /// either `ContributionUntrusted` or `MissingFlavor` — not silently accept
    /// the file. The exact variant depends on whether discovery finds a
    /// matching manifest; in a clean test environment with no extensions
    /// installed, `MissingFlavor` fires first (before the verifier pass).
    #[test]
    fn open_fails_for_untrusted_third_party() {
        use crate::extension::table::FLAVOR_WASM;
        use crate::extension::{Attestation, CapabilityMap, ExtensionTableEntry, Kind, Lifecycle};
        use tempfile::tempdir;

        let tmp = tempdir().unwrap();
        let _g = with_clean_xdg_config(tmp.path());

        let entry = ExtensionTableEntry {
            canonical_id: crate::extension::CanonicalId::derive(
                &[0xDE; 32],
                "third-party",
                "0.1.0",
            ),
            human_label: "io.example.third-party@0.1.0".into(),
            kind: Kind::PlaneCodec,
            abi_version: 1,
            lifecycle: Lifecycle::Thread,
            flavor_hints: FLAVOR_WASM,
            capabilities: CapabilityMap::new(),
            attestation: Attestation::PgpSignature(Vec::new()),
            install_hint: None,
            embedded_wasm_offset: None,
            embedded_wasm_length: None,
        };

        let mut buf: Vec<u8> = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let mut w = ContainerWriter::new(cursor).unwrap();
            w.set_extension_table(crate::extension::ExtensionTable {
                entries: vec![entry],
            });
            w.append_tensor(&sample_tensor_record("t")).unwrap();
            w.finalize().unwrap();
        }

        let res = ContainerReader::open(&buf);
        assert!(
            matches!(
                res,
                Err(PtwmCoreError::ContributionUntrusted { .. })
                    | Err(PtwmCoreError::MissingFlavor { .. })
            ),
            "expected ContributionUntrusted or MissingFlavor, got: {res:?}"
        );
    }

    /// Serialise data-mutating tests on PTWM_EXTENSION_PATH so they don't race.
    static EXT_PATH_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// RAII guard for PTWM_EXTENSION_PATH (separate from XDG_CONFIG_HOME so
    /// callers can compose the two).
    fn with_ext_path<'a>(path: &str) -> impl Drop + 'a {
        struct Guard {
            prev: Option<String>,
            _lock: std::sync::MutexGuard<'static, ()>,
        }
        impl Drop for Guard {
            fn drop(&mut self) {
                match &self.prev {
                    Some(v) => unsafe { std::env::set_var("PTWM_EXTENSION_PATH", v) },
                    None => unsafe { std::env::remove_var("PTWM_EXTENSION_PATH") },
                }
            }
        }
        let lock = EXT_PATH_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let prev = std::env::var("PTWM_EXTENSION_PATH").ok();
        unsafe { std::env::set_var("PTWM_EXTENSION_PATH", path) };
        Guard { prev, _lock: lock }
    }

    /// End-to-end Verifier path: a real signed bundle on disk, an author
    /// key in the user keyring, an extension-table entry that carries the
    /// matching signature → container opens successfully.
    /// Tampering with the binary after signing invalidates the digest and
    /// the same container fails to open with `ContributionUntrusted`.
    #[test]
    fn open_verifies_signed_bundle_end_to_end() {
        use crate::extension::table::FLAVOR_WASM;
        use crate::extension::{Attestation, CapabilityMap, ExtensionTableEntry, Kind, Lifecycle};
        use crate::trust::SecretKey;
        use std::fs;
        use tempfile::tempdir;

        let cfg_tmp = tempdir().unwrap();
        let ext_tmp = tempdir().unwrap();
        let _g_cfg = with_clean_xdg_config(cfg_tmp.path());
        let _g_ext = with_ext_path(ext_tmp.path().to_str().unwrap());

        // Sign a bundle laid out under the PTWM_EXTENSION_PATH root.
        let sk = SecretKey::generate();
        let pubkey = sk.public();
        let bundle_dir = ext_tmp
            .path()
            .join(hex::encode(&pubkey.0[..4]))
            .join("demo@0.1.0");
        fs::create_dir_all(&bundle_dir).unwrap();
        let manifest_text = format!(
            r#"
[bundle]
name = "demo"
version = "0.1.0"
author_pubkey = "ed25519:{}"
"#,
            hex::encode(pubkey.0),
        );
        let manifest_path = bundle_dir.join("manifest.toml");
        fs::write(&manifest_path, &manifest_text).unwrap();
        let wasm_bytes = [0xAA, 0xBB, 0xCC, 0xDD];
        let wasm_path = bundle_dir.join("demo.wasm");
        fs::write(&wasm_path, wasm_bytes).unwrap();

        // Compute the same digest the Verifier will, sign it.
        let mut hasher = blake3::Hasher::new();
        hasher.update(manifest_text.as_bytes());
        hasher.update(&wasm_bytes);
        let mut digest = [0u8; 32];
        digest.copy_from_slice(hasher.finalize().as_bytes());
        let sig = sk.sign(&digest);

        // Trust the author key in the user keyring.
        let user_keys = crate::trust::lock_path().with_file_name("keys.toml");
        fs::create_dir_all(user_keys.parent().unwrap()).unwrap();
        fs::write(
            &user_keys,
            format!(
                r#"
[[entries]]
kind = "author_key"
pubkey = "ed25519:{}"
label = "test-author"
"#,
                hex::encode(pubkey.0),
            ),
        )
        .unwrap();

        // Build the extension-table entry whose canonical id matches the
        // manifest-derived id; the signature is over the bundle's digest.
        let canonical_id = crate::extension::CanonicalId::derive(&pubkey.0, "demo", "0.1.0");
        let entry = ExtensionTableEntry {
            canonical_id,
            human_label: "io.example.demo@0.1.0".into(),
            kind: Kind::PlaneCodec,
            abi_version: 1,
            lifecycle: Lifecycle::Thread,
            flavor_hints: FLAVOR_WASM,
            capabilities: CapabilityMap::new(),
            attestation: Attestation::PgpSignature(sig),
            install_hint: None,
            embedded_wasm_offset: None,
            embedded_wasm_length: None,
        };

        // Pristine bundle → strict `open` succeeds: the wasm flavor is on
        // disk and the digest matches the signature.
        let mut buf: Vec<u8> = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let mut w = ContainerWriter::new(cursor).unwrap();
            w.set_extension_table(crate::extension::ExtensionTable {
                entries: vec![entry.clone()],
            });
            w.append_tensor(&sample_tensor_record("t")).unwrap();
            w.finalize().unwrap();
        }
        let reader = ContainerReader::open(&buf).expect("pristine signed bundle must verify");
        assert!(reader.missing_extensions.is_empty());

        // Tamper with the binary → digest no longer matches the signature.
        // Use strict `open` so the verifier's verdict isn't swallowed by
        // `skip_missing`.
        fs::write(&wasm_path, [0xFFu8; 4]).unwrap();
        let res = ContainerReader::open(&buf);
        assert!(
            matches!(res, Err(PtwmCoreError::ContributionUntrusted { .. })),
            "tampered bundle should fail with ContributionUntrusted, got: {res:?}"
        );
    }

    // ── C2: codec_table_idx + router tests ────────────────────────────────────

    /// End-to-end roundtrip that goes through the `PlaneCodecRouter` in
    /// `decode_tensor`.  The writer fills `codec_table_idx` for each plane;
    /// the reader resolves the canonical id via the Extension Table and
    /// confirms the codec is available through the router before decoding.
    #[test]
    fn round_trip_with_builtin_via_router() {
        use crate::compressor::{CompressorOptions, InputTensor, compress_model};
        use std::io::Cursor;

        let n_elements: u32 = 512;
        let mut original_bytes: Vec<u8> = Vec::with_capacity((n_elements * 2) as usize);
        for i in 0..n_elements {
            let bits: u16 = 0x3800 | ((i as u16) & 0x03FF);
            original_bytes.extend_from_slice(&bits.to_le_bytes());
        }

        let chain = v3_fp16_split_chain(n_elements);
        let inputs = vec![InputTensor {
            name: "weight".into(),
            candidate_chains: vec![chain],
            dtype_code: 0x0002,
            input_format: 0,
            raw_bytes: &original_bytes,
            shape: None,
            dtype_name: None,
            delta_reference_blake3: None,
        }];

        let mut buf: Vec<u8> = Vec::new();
        compress_model(
            Cursor::new(&mut buf),
            &inputs,
            CompressorOptions {
                method_hint: 1,
                emit_payload_hash: true,
                emit_plane_crc: false,
                forced_codec: None,
                allow_codec_ids: None,
                chunk_size: None,
            },
        )
        .expect("compress succeeds");

        // decode_tensor now routes through PlaneCodecRouter for planes that
        // carry a codec_table_idx (which is all planes from this writer).
        let reader = ContainerReader::open(&buf).expect("open succeeds");
        let decoded = reader.decode_tensor("weight").expect("decode via router");
        assert_eq!(
            decoded, original_bytes,
            "round_trip_with_builtin_via_router: bit-exact roundtrip failed"
        );
    }
}
