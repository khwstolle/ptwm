//! Container writer and reader orchestration.

use std::io::{Seek, SeekFrom, Write};
use std::sync::Arc;

use rayon::prelude::*;

use xxhash_rust::xxh64::xxh64;

use std::collections::{HashMap, HashSet};

use crate::chain::Chain;
use crate::chain::wire::{read_chain, write_chain};
use crate::chain::wire_compact::{bind_deferred_source, has_deferred_source};
use crate::codec::CodecId;
use crate::error::PtwmCoreError;
use crate::extension::{CanonicalId, ExtensionTable, ExtensionTableBuilder};
use crate::header::{
    ContainerMagic, HEADER_LEN, Header, MAGIC_PTWC, MAGIC_PTWM, MAGIC_PTWX, sniff,
};
use crate::index::{
    EOF_SENTINEL, IndexEntry, name_hash, parse_index, verify_eof_sentinel, write_index,
};
use crate::prelude::{PreludeEntry, parse_prelude, write_prelude};
use crate::tensor_record::{TensorRecord, write_tensor_record_with_builtin_version};
use crate::tensor_record_compact::{
    RecordContext, parse_compact_record, plane_codec_ids, record_binding, write_compact_record,
};

// ── Chain registry ────────────────────────────────────────────────────────────

/// Largest registry chain id. A tensor record addresses a registry chain
/// through the low 15 bits of `chain_ref`; the high bit marks an inline chain.
pub const MAX_CHAIN_ID: u16 = 0x7FFF;

/// Largest number of chains a registry can hold: one per id in
/// `0..=MAX_CHAIN_ID`.
pub const MAX_REGISTRY_CHAINS: usize = MAX_CHAIN_ID as usize + 1;

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
        if self.chains.len() > MAX_REGISTRY_CHAINS {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "ChainRegistry: {} chains exceeds the limit of {MAX_REGISTRY_CHAINS}",
                self.chains.len()
            )));
        }
        if let Some((id, _)) = self.chains.iter().find(|(id, _)| *id > MAX_CHAIN_ID) {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "ChainRegistry: chain_id {id:#06x} does not fit in the 15-bit chain_ref"
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
        // Each entry takes at least its 2-byte chain_id.
        let mut chains = Vec::with_capacity(num_chains.min((buf.len() - 2) / 2));
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
        // Writers assign ids by position; try that first.
        if let Some((id, chain)) = self.chains.get(usize::from(chain_id))
            && *id == chain_id
        {
            return Some(chain);
        }
        self.chains
            .iter()
            .find(|(id, _)| *id == chain_id)
            .map(|(_, c)| c)
    }
}

/// Which plane codecs an Extension Table interns besides the chain ops.
#[derive(Clone, Copy, Debug)]
pub enum CodecSet<'a> {
    /// Every built-in plane codec (legacy layout, v1.1 and earlier).
    AllBuiltins,
    /// Only the listed codecs, interned in built-in registry order.
    Only(&'a [CanonicalId]),
}

/// Output of [`build_extension_table`].
#[derive(Debug)]
pub struct BuiltExtensionTable {
    pub table: ExtensionTable,
    /// Serialized chain registry whose op indices refer to `table`.
    pub registry_bytes: Vec<u8>,
    /// Table index of every interned plane codec.
    pub codec_idx_map: HashMap<CanonicalId, u16>,
}

/// Build the Extension Table and the serialized chain registry for a
/// container. Chain ops are interned first (in registry order), then the
/// plane codecs selected by `codecs`, in the order of
/// [`crate::extension::builtin_entries`].
///
/// Built-in entries carry the ids and labels of `builtin_version`
/// ([`crate::extension::LEGACY_BUILTIN_VERSION`] for new legacy files). The
/// returned `codec_idx_map` is keyed by the current canonical ids whatever
/// the version.
///
/// The result is a pure function of `registry`, the codec set and the
/// version, so a reader-side tool (the transcoder) can rebuild the same
/// bytes.
pub fn build_extension_table(
    registry: &ChainRegistry,
    codecs: CodecSet<'_>,
    builtin_version: &str,
) -> Result<BuiltExtensionTable, PtwmCoreError> {
    use crate::extension::{Kind, builtin_entries, builtin_registry};

    let mut builder = ExtensionTableBuilder::with_builtin_version(builtin_version)?;
    let mut registry_bytes = Vec::new();
    registry.write(&mut builder, &mut registry_bytes)?;

    // Use `builtin_entries()` (the curated list with `is_builtin` coverage)
    // filtered by `Kind::PlaneCodec` rather than hard-coded names. This
    // excludes `huffman_nibble`, which `dispatch_builtin` keeps for legacy
    // wire compat but which has no `builtin_entry`, so `is_builtin` and
    // `collect_missing_flavors` do not recognise it.
    let codec_ids: Vec<CanonicalId> = builtin_entries()
        .into_iter()
        .filter(|e| e.kind == Kind::PlaneCodec)
        .map(|e| e.canonical_id)
        .collect();
    if let CodecSet::Only(wanted) = codecs {
        for id in wanted {
            if !codec_ids.contains(id) {
                return Err(PtwmCoreError::InvalidContainer(format!(
                    "codec {id:?} has no built-in Extension Table entry"
                )));
            }
        }
    }
    let reg = builtin_registry();
    let mut codec_idx_map = HashMap::new();
    for canonical_id in codec_ids {
        let keep = match codecs {
            CodecSet::AllBuiltins => true,
            CodecSet::Only(wanted) => wanted.contains(&canonical_id),
        };
        if keep {
            let name = reg
                .get(&canonical_id)
                .map(|r| r.name)
                .expect("built-in codec entry is in the registry");
            let idx = builder
                .intern_builtin(name)
                .expect("built-in codec has an Extension Table entry");
            codec_idx_map.insert(canonical_id, idx);
        }
    }

    Ok(BuiltExtensionTable {
        table: ExtensionTable {
            entries: builder.finish(),
        },
        registry_bytes,
        codec_idx_map,
    })
}

/// Validate serialized Extension Table bytes and encode `registry` against
/// them, for [`ContainerWriter::set_raw_extension_table`].
///
/// Checks:
/// - `table_bytes` parse as one Extension Table with no trailing bytes and
///   re-serialize to the same bytes;
/// - no entry references an embedded WASM module;
/// - every op of every registry chain is already in the table (built-in
///   ids of other releases count once normalised).
///
/// Registry ops resolve to the first entry with their id; `codec_idx_map`
/// maps each built-in codec's current id to its first entry. The returned
/// `table` is the table as parsed, without normalisation.
pub fn encode_registry_for_table(
    table_bytes: &[u8],
    registry: &ChainRegistry,
) -> Result<BuiltExtensionTable, PtwmCoreError> {
    use crate::extension::{BuiltinKind, dispatch_builtin, normalize_builtin_ids};

    let bad = |msg: String| PtwmCoreError::InvalidContainer(format!("raw extension table: {msg}"));
    let (table, used) =
        ExtensionTable::from_bytes_prefix(table_bytes).map_err(|e| bad(e.to_string()))?;
    if used != table_bytes.len() {
        return Err(bad(format!("{} trailing bytes", table_bytes.len() - used)));
    }
    if table.to_bytes().map_err(|e| bad(e.to_string()))? != table_bytes {
        return Err(bad("bytes are not in canonical form".into()));
    }
    if table
        .entries
        .iter()
        .any(|e| e.embedded_wasm_offset.is_some() || e.embedded_wasm_length.is_some())
    {
        return Err(bad("embedded WASM entries are not supported".into()));
    }

    let mut normalised = table.clone();
    normalize_builtin_ids(&mut normalised);
    let n_entries = normalised.entries.len();
    let mut codec_idx_map = HashMap::new();
    for (i, e) in normalised.entries.iter().enumerate() {
        if let Some(BuiltinKind::Codec(_)) = dispatch_builtin(&e.canonical_id) {
            let idx = u16::try_from(i).map_err(|_| bad("too many entries".into()))?;
            codec_idx_map.entry(e.canonical_id).or_insert(idx);
        }
    }
    let mut builder = ExtensionTableBuilder::from_entries(normalised.entries);
    let mut registry_bytes = Vec::new();
    registry.write(&mut builder, &mut registry_bytes)?;
    if builder.len() != n_entries {
        return Err(bad("a chain op is missing from the table".into()));
    }
    Ok(BuiltExtensionTable {
        table,
        registry_bytes,
        codec_idx_map,
    })
}

/// Encode `registry` as a compact registry whose op references index
/// `table` (the entry with each op's built-in id).
fn compact_registry_bytes(
    registry: &ChainRegistry,
    table: &ExtensionTable,
) -> Result<Vec<u8>, PtwmCoreError> {
    let reg = crate::extension::builtin_registry();
    let mut index: HashMap<CanonicalId, u64> = HashMap::new();
    for (i, e) in table.entries.iter().enumerate() {
        index.entry(e.canonical_id).or_insert(i as u64);
    }
    crate::ptwc::write_registry(registry, |op| {
        reg.id_for_entry_name(op.builtin_name())
            .and_then(|id| index.get(&id).copied())
            .ok_or_else(|| {
                PtwmCoreError::InvalidContainer(format!("op {op:?} is not in the extension table"))
            })
    })
}

/// On-disk layout of a container.
///
/// `Default` gives `Compact`, the layout of new multi-tensor writes
/// ([`crate::compressor::CompressorOptions`]). [`ContainerWriter::new`]
/// still writes `Legacy`; use [`ContainerWriter::with_layout`] to choose.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ContainerLayout {
    /// `\x89PTWM`: fixed-width header, full Extension Table, hash-sorted
    /// index. Readable by v1.1 and later.
    Legacy,
    /// `\x89PTWC`: varint fields, built-ins stored by id, names in the index
    /// ([`crate::ptwc`]). Readable by v1.2 and later.
    #[default]
    Compact,
}

impl ContainerLayout {
    /// The layout's name as the Python bindings spell it.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::Compact => "compact",
        }
    }

    /// Parse a layout name (`"legacy"` or `"compact"`).
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "legacy" => Some(Self::Legacy),
            "compact" => Some(Self::Compact),
            _ => None,
        }
    }
}

/// Built-in version a new writer of `layout` stamps on built-in entries.
///
/// The legacy layout uses [`crate::extension::LEGACY_BUILTIN_VERSION`], so
/// a v1.1 reader recognises its built-in ids whatever the crate version.
/// The compact layout stores built-ins without a version; its tables use
/// the crate version internally.
fn default_builtin_version(layout: ContainerLayout) -> String {
    match layout {
        ContainerLayout::Legacy => crate::extension::LEGACY_BUILTIN_VERSION.to_owned(),
        ContainerLayout::Compact => crate::extension::CURRENT_BUILTIN_VERSION.to_owned(),
    }
}

/// Serializes tensor records for one container writer's layout and tables.
///
/// [`ContainerWriter::record_encoder`] returns it so that callers can
/// serialize records in parallel and append the bytes with
/// [`ContainerWriter::append_serialized_tensor`].
#[derive(Debug, Clone)]
pub struct RecordEncoder {
    layout: ContainerLayout,
    builtin_version: String,
    codec_ids: Vec<Option<CodecId>>,
    registry_bound: Vec<bool>,
}

impl RecordEncoder {
    /// Serialize `rec` in the writer's layout. A compact record leaves out
    /// the name, which the index stores.
    pub fn encode(&self, rec: &TensorRecord) -> Result<Vec<u8>, PtwmCoreError> {
        let est: usize = rec
            .terminals
            .iter()
            .map(|t| t.payload_bytes.len() + t.inline_state_bytes.len())
            .sum();
        let mut buf = Vec::with_capacity(est + 256);
        self.encode_into(rec, &mut buf)?;
        Ok(buf)
    }

    /// [`encode`](Self::encode) into `out`; on error `out` is unchanged.
    pub fn encode_into(&self, rec: &TensorRecord, out: &mut Vec<u8>) -> Result<(), PtwmCoreError> {
        match self.layout {
            ContainerLayout::Legacy => {
                write_tensor_record_with_builtin_version(rec, out, &self.builtin_version)
            }
            ContainerLayout::Compact => {
                let ctx = RecordContext {
                    codec_ids: &self.codec_ids,
                    registry_bound: &self.registry_bound,
                };
                write_compact_record(rec, &ctx, out).map_err(|e| {
                    PtwmCoreError::InvalidContainer(format!("tensor '{}': {e}", rec.name))
                })
            }
        }
    }
}

/// Streaming writer for a `.ptwm` container.
pub struct ContainerWriter<W: Write + Seek> {
    writer: W,
    layout: ContainerLayout,
    /// Record names and lengths in body order (compact layout).
    compact_index: Vec<(String, u64)>,
    /// Record encoder state for the current tables.
    encoder: RecordEncoder,
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
    /// Names appended so far. A container holds each name at most once.
    names: HashSet<String>,
    /// Version whose ids and labels built-in Extension Table entries are
    /// written with (file-level and inline-chain tables).
    builtin_version: String,
    /// Extension Table bytes to write verbatim instead of serializing
    /// `extension_table` (see `set_raw_extension_table`).
    raw_extension_table: Option<Vec<u8>>,
    /// Whether an Extension Table or chain registry was set.
    table_set: bool,
}

impl<W: Write + Seek> ContainerWriter<W> {
    /// Create a new legacy-layout container writer. Writes a 91-byte header
    /// placeholder immediately.
    pub fn new(writer: W) -> Result<Self, PtwmCoreError> {
        Self::with_layout(writer, ContainerLayout::Legacy)
    }

    /// Create a container writer for `layout`. The legacy layout writes a
    /// header placeholder now and patches it in [`finalize`](Self::finalize);
    /// the compact layout writes its header with the first tensor and never
    /// seeks.
    pub fn with_layout(mut writer: W, layout: ContainerLayout) -> Result<Self, PtwmCoreError> {
        let header_position = writer
            .stream_position()
            .map_err(|e| PtwmCoreError::Io(e.to_string()))?;

        if layout == ContainerLayout::Legacy {
            // Write placeholder (magic bytes, rest zeros).
            let mut placeholder = [0u8; HEADER_LEN];
            placeholder[..9].copy_from_slice(&MAGIC_PTWM);
            writer
                .write_all(&placeholder)
                .map_err(|e| PtwmCoreError::Io(e.to_string()))?;
        }

        let header = Header::new_ptwm();
        let builtin_version = default_builtin_version(layout);

        Ok(Self {
            writer,
            layout,
            compact_index: Vec::new(),
            encoder: RecordEncoder {
                layout,
                builtin_version: builtin_version.clone(),
                codec_ids: Vec::new(),
                registry_bound: Vec::new(),
            },
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
            names: HashSet::new(),
            builtin_version,
            raw_extension_table: None,
            table_set: false,
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
    /// This is the legacy (v1.1 and earlier) table layout. Writers that know
    /// the set of codecs in use should call
    /// [`set_chain_registry_with_codecs`](Self::set_chain_registry_with_codecs)
    /// instead, which omits the unreferenced codec entries (about 125 B each).
    ///
    /// After this call, use [`codec_table_idx_for`](Self::codec_table_idx_for)
    /// to obtain the stable index for a codec's canonical id.
    ///
    /// Must be called before any `append_tensor` call.
    pub fn set_chain_registry(&mut self, registry: ChainRegistry) -> Result<(), PtwmCoreError> {
        self.install_registry(registry, CodecSet::AllBuiltins)
    }

    /// Set the chain registry and build an Extension Table that holds only
    /// the chain ops plus the plane codecs in `codecs`.
    ///
    /// Codec entries are interned in the order of
    /// [`crate::extension::builtin_entries`], not in the order of `codecs`,
    /// so the table is a pure function of the registry and the codec *set*.
    /// The transcoder relies on this to rebuild the same table bytes.
    ///
    /// Returns `InvalidContainer` if a codec in `codecs` has no built-in
    /// Extension Table entry.
    ///
    /// Must be called before any `append_tensor` call.
    pub fn set_chain_registry_with_codecs(
        &mut self,
        registry: ChainRegistry,
        codecs: &[CanonicalId],
    ) -> Result<(), PtwmCoreError> {
        self.install_registry(registry, CodecSet::Only(codecs))
    }

    fn install_registry(
        &mut self,
        registry: ChainRegistry,
        codecs: CodecSet<'_>,
    ) -> Result<(), PtwmCoreError> {
        if self.prelude_written || !self.index_entries.is_empty() {
            return Err(PtwmCoreError::InvalidContainer(
                "chain registry must be set before appending tensors".into(),
            ));
        }
        self.check_registry_for_layout(&registry)?;
        let built = build_extension_table(&registry, codecs, &self.builtin_version)?;
        self.chain_registry_bytes = match self.layout {
            ContainerLayout::Legacy => built.registry_bytes,
            ContainerLayout::Compact => compact_registry_bytes(&registry, &built.table)?,
        };
        self.extension_table = built.table;
        self.raw_extension_table = None;
        self.table_set = true;
        self.chain_registry = registry;
        self.codec_idx_map = built.codec_idx_map;
        self.refresh_encoder();
        Ok(())
    }

    /// Fail on registry chains the layout cannot store: a bound Source (a
    /// Source node with empty params) needs the compact layout.
    fn check_registry_for_layout(&self, registry: &ChainRegistry) -> Result<(), PtwmCoreError> {
        if self.layout == ContainerLayout::Legacy
            && let Some((id, _)) = registry.chains.iter().find(|(_, c)| has_deferred_source(c))
        {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "chain {id} has a bound Source (empty Source params), which needs the \
                 compact layout"
            )));
        }
        Ok(())
    }

    /// Recompute the record encoder's view of the tables.
    fn refresh_encoder(&mut self) {
        self.encoder.builtin_version = self.builtin_version.clone();
        self.encoder.codec_ids = plane_codec_ids(&self.extension_table);
        self.encoder.registry_bound = self
            .chain_registry
            .chains
            .iter()
            .map(|(_, c)| has_deferred_source(c))
            .collect();
    }

    /// The writer's layout.
    pub fn layout(&self) -> ContainerLayout {
        self.layout
    }

    /// A record encoder for the current layout and tables. Take it after the
    /// Extension Table and chain registry are set.
    pub fn record_encoder(&self) -> RecordEncoder {
        self.encoder.clone()
    }

    /// Serialize `rec` the way [`append_tensor`](Self::append_tensor) would,
    /// for [`append_serialized_tensor`](Self::append_serialized_tensor).
    pub fn encode_record(&self, rec: &TensorRecord) -> Result<Vec<u8>, PtwmCoreError> {
        self.encoder.encode(rec)
    }

    /// Use `table` as the Extension Table and encode `registry` against it
    /// (compact layout only). Registry ops resolve to the table entry with
    /// their built-in id, and [`codec_table_idx_for`](Self::codec_table_idx_for)
    /// to the entry of each built-in codec. The transcoder uses this to write
    /// a table and registry exactly as they were read.
    ///
    /// Must be called before any tensor is appended.
    pub fn set_compact_tables(
        &mut self,
        table: ExtensionTable,
        registry: ChainRegistry,
    ) -> Result<(), PtwmCoreError> {
        if self.layout != ContainerLayout::Compact {
            return Err(PtwmCoreError::InvalidContainer(
                "set_compact_tables needs the compact layout".into(),
            ));
        }
        if self.prelude_written || !self.index_entries.is_empty() {
            return Err(PtwmCoreError::InvalidContainer(
                "extension table must be set before appending tensors".into(),
            ));
        }
        // Fails early on entries the compact table cannot store.
        crate::extension::compact_table::compact_table_bytes(&table)?;
        self.chain_registry_bytes = compact_registry_bytes(&registry, &table)?;
        let mut codec_idx_map = HashMap::new();
        for (i, e) in table.entries.iter().enumerate() {
            if let Some(crate::extension::BuiltinKind::Codec(_)) =
                crate::extension::dispatch_builtin(&e.canonical_id)
            {
                codec_idx_map.entry(e.canonical_id).or_insert(i as u16);
            }
        }
        self.extension_table = table;
        self.raw_extension_table = None;
        self.chain_registry = registry;
        self.codec_idx_map = codec_idx_map;
        self.table_set = true;
        self.refresh_encoder();
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
        self.raw_extension_table = None;
        self.table_set = true;
        self.refresh_encoder();
    }

    /// Write built-in Extension Table entries with the ids and labels of
    /// `version` instead of [`crate::extension::LEGACY_BUILTIN_VERSION`]. This applies to the table
    /// built by [`set_chain_registry`](Self::set_chain_registry) and
    /// [`set_chain_registry_with_codecs`](Self::set_chain_registry_with_codecs)
    /// and to inline-chain tables. The transcoder uses it to reproduce files
    /// written by other releases; new files keep the default.
    ///
    /// Must be called before the Extension Table or chain registry is set
    /// and before any tensor is appended.
    pub fn set_builtin_version(&mut self, version: &str) -> Result<(), PtwmCoreError> {
        if self.table_set || self.prelude_written || !self.index_entries.is_empty() {
            return Err(PtwmCoreError::InvalidContainer(
                "builtin version must be set before the extension table and tensors".into(),
            ));
        }
        if self.layout == ContainerLayout::Compact {
            return Err(PtwmCoreError::InvalidContainer(
                "the compact layout stores built-ins without a version".into(),
            ));
        }
        if !crate::extension::is_valid_builtin_version(version) {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "invalid builtin version {version:?}"
            )));
        }
        self.builtin_version = version.to_owned();
        self.refresh_encoder();
        Ok(())
    }

    /// Use `table_bytes` verbatim as the file's Extension Table and encode
    /// `registry` against it.
    ///
    /// The transcoder uses this for tables that no table builder reproduces
    /// (for example a table that mixes built-in versions). The bytes are
    /// validated by [`encode_registry_for_table`] first. Plane codec indices
    /// for [`codec_table_idx_for`](Self::codec_table_idx_for) come from the
    /// first table entry of each codec.
    ///
    /// Must be called before any tensor is appended.
    pub fn set_raw_extension_table(
        &mut self,
        table_bytes: &[u8],
        registry: ChainRegistry,
    ) -> Result<(), PtwmCoreError> {
        if self.prelude_written || !self.index_entries.is_empty() {
            return Err(PtwmCoreError::InvalidContainer(
                "extension table must be set before appending tensors".into(),
            ));
        }
        if self.layout != ContainerLayout::Legacy {
            return Err(PtwmCoreError::InvalidContainer(
                "set_raw_extension_table needs the legacy layout".into(),
            ));
        }
        self.check_registry_for_layout(&registry)?;
        let built = encode_registry_for_table(table_bytes, &registry)?;
        self.extension_table = built.table;
        self.raw_extension_table = Some(table_bytes.to_vec());
        self.chain_registry_bytes = built.registry_bytes;
        self.chain_registry = registry;
        self.codec_idx_map = built.codec_idx_map;
        self.table_set = true;
        self.refresh_encoder();
        Ok(())
    }

    /// Write the extension table, prelude section, and chain registry section,
    /// then record all absolute offsets in the pending header.
    ///
    /// Section order on disk: header → extension_table → prelude → chain_registry → tensors → index.
    fn flush_prelude(&mut self) -> Result<(), PtwmCoreError> {
        if self.prelude_written {
            return Ok(());
        }
        if self.layout == ContainerLayout::Compact {
            return self.flush_compact_prelude();
        }

        // Extension table sits immediately after the header at HEADER_LEN.
        // Serialize it now (set_extension_table must be called before this point).
        let et_bytes = match &self.raw_extension_table {
            Some(raw) => raw.clone(),
            None => self
                .extension_table
                .to_bytes()
                .map_err(|e| PtwmCoreError::InvalidContainer(format!("extension table: {e}")))?,
        };
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

    /// Write the compact header, Extension Table, prelude and registry.
    fn flush_compact_prelude(&mut self) -> Result<(), PtwmCoreError> {
        let et = crate::extension::compact_table::compact_table_bytes(&self.extension_table)?;
        let mut prelude = Vec::new();
        write_prelude(&self.prelude_entries, &mut prelude)?;
        // Encode the registry against the final table: the table may have
        // been replaced after the registry was set.
        let registry = compact_registry_bytes(&self.chain_registry, &self.extension_table)?;
        let header = crate::ptwc::header_bytes(&et, &prelude, &registry);
        for part in [&header, &et, &prelude, &registry] {
            self.writer
                .write_all(part)
                .map_err(|e| PtwmCoreError::Io(e.to_string()))?;
        }
        self.prelude_written = true;
        Ok(())
    }

    /// Append a tensor record to the container.
    pub fn append_tensor(&mut self, record: &TensorRecord) -> Result<(), PtwmCoreError> {
        if self.finalized {
            return Err(PtwmCoreError::InvalidContainer("already finalized".into()));
        }
        self.check_new_name(&record.name)?;
        if self.layout == ContainerLayout::Compact {
            let bytes = self.encoder.encode(record)?;
            return self.append_serialized_tensor(&record.name, &bytes);
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
        write_tensor_record_with_builtin_version(record, &mut tensor_buf, &self.builtin_version)?;
        let tensor_len = tensor_buf.len() as u64;

        self.writer
            .write_all(&tensor_buf)
            .map_err(|e| PtwmCoreError::Io(e.to_string()))?;

        self.index_entries.push(IndexEntry {
            name_hash: name_hash(&record.name),
            tensor_offset,
            tensor_len,
            name: None,
        });
        self.names.insert(record.name.clone());

        Ok(())
    }

    /// Fail if a tensor named `name` was already appended. Readers look
    /// tensors up by name, so a second record with the same name could never
    /// be reached.
    fn check_new_name(&self, name: &str) -> Result<(), PtwmCoreError> {
        if self.names.contains(name) {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "duplicate tensor name '{name}'"
            )));
        }
        Ok(())
    }

    /// Append a tensor whose record bytes were serialized ahead of time by
    /// [`encode_record`](Self::encode_record) or a [`RecordEncoder`] (for
    /// example in parallel). Identical on-disk effect to
    /// [`Self::append_tensor`] but skips the (payload-copy-heavy)
    /// serialization, which the caller has already done. The bytes must be
    /// in this writer's layout.
    pub fn append_serialized_tensor(
        &mut self,
        name: &str,
        tensor_buf: &[u8],
    ) -> Result<(), PtwmCoreError> {
        if self.finalized {
            return Err(PtwmCoreError::InvalidContainer("already finalized".into()));
        }
        self.check_new_name(name)?;
        if self.layout == ContainerLayout::Compact {
            if name.is_empty() || name.len() > crate::ptwc::MAX_NAME_LEN {
                return Err(PtwmCoreError::InvalidContainer(format!(
                    "tensor name of {} bytes (the compact layout needs 1..={})",
                    name.len(),
                    crate::ptwc::MAX_NAME_LEN
                )));
            }
            if (tensor_buf.len() as u64) < crate::tensor_record_compact::MIN_RECORD_LEN {
                return Err(PtwmCoreError::InvalidContainer(format!(
                    "tensor '{name}': {}-byte record is not a compact record",
                    tensor_buf.len()
                )));
            }
            self.flush_prelude()?;
            self.writer
                .write_all(tensor_buf)
                .map_err(|e| PtwmCoreError::Io(e.to_string()))?;
            self.compact_index
                .push((name.to_owned(), tensor_buf.len() as u64));
            self.names.insert(name.to_owned());
            return Ok(());
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
            name: None,
        });
        self.names.insert(name.to_owned());
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
        if self.layout == ContainerLayout::Compact {
            let index = crate::ptwc::write_index(&self.compact_index)?;
            self.writer
                .write_all(&index)
                .map_err(|e| PtwmCoreError::Io(e.to_string()))?;
            self.writer
                .write_all(&crate::ptwc::footer_bytes(index.len() as u64))
                .map_err(|e| PtwmCoreError::Io(e.to_string()))?;
            self.finalized = true;
            return Ok(());
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

/// Discovery, missing-flavor and trust pass over an Extension Table (spec
/// §6.5: trusted ⇒ allowed). Returns the installed-extension snapshot and
/// the missing entries. With `skip_missing`, missing or untrusted entries
/// do not fail the call.
#[allow(clippy::type_complexity)]
fn check_extensions(
    extension_table: &ExtensionTable,
    skip_missing: bool,
) -> Result<
    (
        Vec<crate::discovery::DiscoveredContribution>,
        Vec<crate::error::MissingFlavorEntry>,
    ),
    PtwmCoreError,
> {
    // A table of built-ins only needs no discovery, keyring or trust
    // pass: built-ins are trusted by construction and always available.
    // Skipping it keeps opening such a file free of filesystem access.
    let all_builtin = extension_table
        .entries
        .iter()
        .all(|e| crate::extension::is_builtin(&e.canonical_id));

    // Scan installed extensions once; used by both the missing-flavor check
    // and the trust verifier pass below.
    let installed = if all_builtin {
        Vec::new()
    } else {
        crate::discovery::scan_all_cached().unwrap_or_default()
    };

    // Check for missing non-builtin flavors immediately after parsing the
    // Extension Table — before touching any tensor payload.
    let needs = collect_missing_flavors(extension_table, &installed)?;
    if !needs.is_empty() && !skip_missing {
        return Err(PtwmCoreError::MissingFlavor { needs });
    }

    // ── Trust Verifier pass (spec §6.5: trusted ⇒ allowed) ───────────────
    //
    // For every non-builtin entry in the Extension Table that IS installed
    // (i.e. not flagged as missing above), verify that its author is
    // trusted by the active keyring. Entries with no matching install
    // (already covered by MissingFlavor above) are skipped here.
    if !all_builtin {
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
            let Some((pubkey, contrib_idx)) = lookup_author_pubkey(&entry.canonical_id, &installed)
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
    Ok((installed, needs))
}

/// Parse the Extension Table of a legacy container whose header is
/// `header`. The region is bounds-checked and its BLAKE3 hash verified.
/// Built-in ids are not normalized.
fn legacy_extension_table(buf: &[u8], header: &Header) -> Result<ExtensionTable, PtwmCoreError> {
    if header.extension_table_length == 0 {
        return Ok(ExtensionTable::empty());
    }
    let table_bytes = usize::try_from(header.extension_table_offset)
        .ok()
        .zip(usize::try_from(header.extension_table_length).ok())
        .and_then(|(start, len)| buf.get(start..start.checked_add(len)?))
        .ok_or_else(|| {
            PtwmCoreError::InvalidContainer(format!(
                "extension table region at offset {} with length {} exceeds the buffer ({} bytes)",
                header.extension_table_offset,
                header.extension_table_length,
                buf.len()
            ))
        })?;
    if ExtensionTable::hash(table_bytes) != header.extension_table_hash {
        return Err(PtwmCoreError::ExtensionTableHashMismatch);
    }
    ExtensionTable::from_bytes(table_bytes)
        .map_err(|e| PtwmCoreError::InvalidContainer(format!("extension table: {e}")))
}

/// Read the Extension Table of a container in either layout without opening
/// the rest of it: no extension check, prelude, registry or index parse.
///
/// A legacy (`\x89PTWM` / `\x89PTWX`) file has its table region
/// bounds-checked and its hash verified, and built-in ids written by other
/// releases are mapped to the current ones (see
/// [`crate::extension::normalize_builtin_ids`]). A compact (`\x89PTWC`) file
/// has its EOF sentinel, sections and `meta_xxh64` checked, and its compact
/// entries expanded to full entries. Any other magic is an error.
pub fn read_extension_table(buf: &[u8]) -> Result<ExtensionTable, PtwmCoreError> {
    if sniff(buf) == Some(ContainerMagic::Ptwc) {
        verify_eof_sentinel(buf)?;
        let sec = crate::ptwc::locate_sections(buf)?;
        return crate::extension::compact_table::read_compact_table(&buf[sec.extension_table]);
    }
    if buf.len() < HEADER_LEN {
        return Err(PtwmCoreError::InvalidContainer(
            "file shorter than header".into(),
        ));
    }
    let header = Header::from_bytes(&buf[..HEADER_LEN])?;
    let mut table = legacy_extension_table(buf, &header)?;
    crate::extension::normalize_builtin_ids(&mut table);
    Ok(table)
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
    /// Built-in version of the file's Extension Table, as returned by
    /// [`crate::extension::normalize_builtin_ids`]: the release whose
    /// built-in ids the writer used. `None` when the table holds no built-in
    /// entry. `extension_table` holds the ids mapped to the crate version.
    /// Always `None` for the compact layout, which stores built-ins without
    /// a version.
    pub builtin_version: Option<String>,
    /// The file's layout, from its magic.
    pub layout: ContainerLayout,
    /// Compact layout: the codec id of each Extension Table entry (see
    /// [`plane_codec_ids`]). Empty for the legacy layout.
    codec_ids: Vec<Option<CodecId>>,
    /// Compact layout: whether each registry chain has a bound Source.
    registry_bound: Vec<bool>,
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
        // A compact single-tensor frame has no EOF sentinel. Name it here so
        // the sentinel check does not report a valid frame as truncated.
        if buf.starts_with(&crate::compact::MAGIC_COMPACT) {
            return Err(PtwmCoreError::InvalidContainer(
                "this is a compact single-tensor frame (\\x89PTWS), not a container; \
                 decode it with decode_compact"
                    .into(),
            ));
        }
        verify_eof_sentinel(buf)?;
        if sniff(buf) == Some(ContainerMagic::Ptwc) {
            return Self::open_compact(buf, skip_missing);
        }
        if buf.len() < HEADER_LEN {
            return Err(PtwmCoreError::InvalidContainer(
                "file shorter than header".into(),
            ));
        }
        let header = Header::from_bytes(&buf[..HEADER_LEN])?;
        if header.is_ptwx() != (header.magic == MAGIC_PTWX) {
            return Err(PtwmCoreError::InvalidContainer(
                "the PTWX header flag disagrees with the magic".into(),
            ));
        }

        let mut extension_table = legacy_extension_table(buf, &header)?;
        // Map built-in ids written by other releases to the current ones.
        // This runs after the hash check, which covers the on-disk bytes.
        let builtin_version = crate::extension::normalize_builtin_ids(&mut extension_table);

        let (installed, needs) = check_extensions(&extension_table, skip_missing)?;

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

        // The prelude and the chain registry both live inside the
        // shared-prelude section; parse them from that slice only.
        let prelude_end = usize::try_from(header.shared_prelude_length)
            .ok()
            .and_then(|len| prelude_off.checked_add(len))
            .filter(|&end| end <= buf.len())
            .ok_or_else(|| {
                PtwmCoreError::InvalidContainer("shared prelude section exceeds buffer".into())
            })?;
        let prelude_section = &buf[prelude_off..prelude_end];
        let (prelude, prelude_bytes_consumed) = parse_prelude(prelude_section)?;

        // The chain registry immediately follows the prelude bytes within the
        // shared-prelude section.
        let cr_off = prelude_off + prelude_bytes_consumed;
        let chain_registry = if prelude_bytes_consumed < prelude_section.len() {
            let (registry, _) =
                ChainRegistry::read(&prelude_section[prelude_bytes_consumed..], &extension_table)?;
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
        // Tensor records start at the end of the shared-prelude section.
        // prelude_end is bounds-checked above, and parse_prelude has already
        // rejected a section too short to hold the prelude (including
        // shared_prelude_length == 0).
        let tensors_offset = prelude_end as u64;

        // Build the codec router once from the installed extensions snapshot.
        // Using the same `installed` slice that the missing-flavor check and
        // trust verifier already consumed is correct: it is a
        // point-in-time snapshot of the host's extension directory.
        let router = crate::flavor::PlaneCodecRouter::new(installed.clone());

        let reader = Self {
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
            builtin_version,
            layout: ContainerLayout::Legacy,
            codec_ids: Vec::new(),
            registry_bound: Vec::new(),
        };
        reader.check_legacy_record_names()?;
        Ok(reader)
    }

    /// Check that every legacy record's name hashes to its index entry's
    /// `name_hash` and that no two records share a name.
    ///
    /// The legacy index stores only the hash, and nothing else covers the
    /// name bytes, so without this check a bit flip in a record name renames
    /// the tensor: two records then report the same name and the original
    /// name is not found. The compact index stores the names and rejects
    /// duplicates in [`crate::ptwc::read_index`].
    fn check_legacy_record_names(&self) -> Result<(), PtwmCoreError> {
        use crate::tensor_record::peek_tensor_record_name;

        let mut names: HashSet<&str> = HashSet::with_capacity(self.index.len());
        for e in &self.index {
            let name = peek_tensor_record_name(self.entry_bytes(e)?)?;
            if name_hash(name) != e.name_hash {
                return Err(PtwmCoreError::InvalidContainer(format!(
                    "tensor record name {name:?} does not match its index entry \
                     (name hash {:#018x}, index has {:#018x})",
                    name_hash(name),
                    e.name_hash
                )));
            }
            if !names.insert(name) {
                return Err(PtwmCoreError::InvalidContainer(format!(
                    "duplicate tensor name {name:?}"
                )));
            }
        }
        Ok(())
    }

    /// [`open_with`](Self::open_with) for the compact layout.
    fn open_compact(buf: &'a [u8], skip_missing: bool) -> Result<Self, PtwmCoreError> {
        use crate::extension::compact_table::read_compact_table;

        let sec = crate::ptwc::locate_sections(buf)?;
        let extension_table = read_compact_table(&buf[sec.extension_table.clone()])?;
        let (installed, needs) = check_extensions(&extension_table, skip_missing)?;

        let prelude_bytes = &buf[sec.prelude.clone()];
        let (prelude, used) = parse_prelude(prelude_bytes)?;
        if used != prelude_bytes.len() {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "compact container: {} trailing bytes after the prelude",
                prelude_bytes.len() - used
            )));
        }
        // Reserved fields and padding must be zero: one encoding per prelude.
        let mut rewritten = Vec::with_capacity(prelude_bytes.len());
        write_prelude(&prelude, &mut rewritten)?;
        if rewritten != prelude_bytes {
            return Err(PtwmCoreError::InvalidContainer(
                "compact container: prelude is not in canonical form".into(),
            ));
        }

        let chain_registry =
            crate::ptwc::read_registry(&buf[sec.registry.clone()], &extension_table)?;
        let index = crate::ptwc::read_index(&buf[sec.index.clone()], sec.body.clone())?;
        let registry_bound = chain_registry
            .chains
            .iter()
            .map(|(_, c)| has_deferred_source(c))
            .collect();
        let codec_ids = plane_codec_ids(&extension_table);
        let router = crate::flavor::PlaneCodecRouter::new(installed);

        // The compact header has no fixed fields; this view gives code that
        // reads offsets from `header` the section positions.
        let header = Header {
            magic: MAGIC_PTWC,
            flags: 0,
            extension_table_offset: sec.extension_table.start as u64,
            extension_table_length: sec.extension_table.len() as u64,
            extension_table_hash: [0u8; 32],
            tensor_index_offset: sec.index.start as u64,
            tensor_index_length: sec.index.len() as u64,
            shared_prelude_offset: sec.prelude.start as u64,
            shared_prelude_length: (sec.registry.end - sec.prelude.start) as u64,
        };
        Ok(Self {
            buf,
            header,
            extension_table,
            prelude,
            router,
            num_tensors: index.len() as u64,
            index,
            tensors_offset: sec.body.start as u64,
            chain_registry_offset: if chain_registry.chains.is_empty() {
                0
            } else {
                sec.registry.start as u64
            },
            chain_registry,
            missing_extensions: needs,
            builtin_version: None,
            layout: ContainerLayout::Compact,
            codec_ids,
            registry_bound,
        })
    }

    /// The context compact records are parsed against.
    fn record_context(&self) -> RecordContext<'_> {
        RecordContext {
            codec_ids: &self.codec_ids,
            registry_bound: &self.registry_bound,
        }
    }

    /// Parse serialized record bytes of this container's layout. A compact
    /// record does not hold its name, so the returned `name` is empty; use
    /// [`tensor_record`](Self::tensor_record) for a record with its name.
    pub fn parse_record(&self, bytes: &[u8]) -> Result<TensorRecord, PtwmCoreError> {
        match self.layout {
            ContainerLayout::Legacy => {
                crate::tensor_record::parse_tensor_record(bytes).map(|(rec, _)| rec)
            }
            ContainerLayout::Compact => parse_compact_record(bytes, &self.record_context()),
        }
    }

    /// The record bytes an index entry points at.
    fn entry_bytes(&self, e: &IndexEntry) -> Result<&'a [u8], PtwmCoreError> {
        let buf: &'a [u8] = self.buf;
        usize::try_from(e.tensor_offset)
            .ok()
            .zip(usize::try_from(e.tensor_len).ok())
            .and_then(|(start, len)| buf.get(start..start.checked_add(len)?))
            .ok_or_else(|| {
                PtwmCoreError::InvalidContainer(format!(
                    "tensor index entry ({}+{}) exceeds buffer ({} bytes)",
                    e.tensor_offset,
                    e.tensor_len,
                    buf.len()
                ))
            })
    }

    /// Parse the record of an index entry, with its name.
    fn entry_record(&self, e: &IndexEntry) -> Result<TensorRecord, PtwmCoreError> {
        let mut rec = self.parse_record(self.entry_bytes(e)?)?;
        if let Some(name) = &e.name {
            rec.name = name.to_string();
        }
        Ok(rec)
    }

    /// Look up a tensor by name and parse its record.
    pub fn tensor_record(&self, name: &str) -> Result<TensorRecord, PtwmCoreError> {
        let bytes = self.get_tensor_record_bytes(name).ok_or_else(|| {
            PtwmCoreError::InvalidContainer(format!("tensor '{}' not found", name))
        })?;
        let mut rec = self.parse_record(bytes)?;
        if self.layout == ContainerLayout::Compact {
            rec.name = name.to_owned();
        }
        Ok(rec)
    }

    /// Every tensor record with its name, in body (file) order.
    pub fn tensor_records(&self) -> Result<Vec<TensorRecord>, PtwmCoreError> {
        self.entries_in_body_order()
            .into_iter()
            .map(|e| self.entry_record(e))
            .collect()
    }

    /// Index entries in body (file) order.
    fn entries_in_body_order(&self) -> Vec<&IndexEntry> {
        let mut entries: Vec<&IndexEntry> = self.index.iter().collect();
        entries.sort_by_key(|e| e.tensor_offset);
        entries
    }

    /// Look up a tensor by name and return a slice covering its serialized
    /// `TensorRecord` bytes. Returns `None` if no tensor with that name
    /// exists.
    ///
    /// The on-disk index is sorted by `name_hash`, so a binary search finds
    /// the run of entries with the target hash in O(log n). Two names can
    /// share a hash, so every entry in the run has its record name compared
    /// against `name`. Entries whose range or name cannot be read do not
    /// match.
    pub fn get_tensor_record_bytes(&self, name: &str) -> Option<&[u8]> {
        self.record_bytes_with_hash(name, name_hash(name))
    }

    /// [`get_tensor_record_bytes`](Self::get_tensor_record_bytes) with the
    /// index hash given explicitly, so tests can force a hash collision.
    fn record_bytes_with_hash(&self, name: &str, target: u64) -> Option<&[u8]> {
        use crate::tensor_record::peek_tensor_record_name;

        let lo = self.index.partition_point(|e| e.name_hash < target);
        let hi = lo + self.index[lo..].partition_point(|e| e.name_hash == target);
        self.index[lo..hi].iter().find_map(|e| {
            let bytes = self.entry_bytes(e).ok()?;
            let found = match &e.name {
                Some(n) => &**n,
                None => peek_tensor_record_name(bytes).ok()?,
            };
            (found == name).then_some(bytes)
        })
    }

    /// Retrieve the shape + dtype_name embedded in this tensor's metadata,
    /// or `None` if no metadata is present. Errors if the record itself is
    /// malformed or the metadata isn't the expected shape-convention map.
    pub fn tensor_shape(&self, name: &str) -> Result<Option<(Vec<u64>, String)>, PtwmCoreError> {
        use crate::metadata::decode_shape_metadata;

        let rec = self.tensor_record(name)?;
        match rec.tensor_metadata {
            Some(bytes) => Ok(Some(decode_shape_metadata(&bytes)?)),
            None => Ok(None),
        }
    }

    /// Return every tensor name in body (write) order: the order of the
    /// records in the file, which is the order the writer appended them. The
    /// compact layout reads the names from the index alone; the legacy layout
    /// parses the name at the start of each record.
    pub fn tensor_names(&self) -> Result<Vec<String>, PtwmCoreError> {
        use crate::tensor_record::peek_tensor_record_name;

        self.entries_in_body_order()
            .into_iter()
            .map(|e| match &e.name {
                Some(name) => Ok(name.to_string()),
                None => peek_tensor_record_name(self.entry_bytes(e)?).map(str::to_owned),
            })
            .collect()
    }

    /// Decode every tensor in the container across the rayon pool. Returns
    /// `(name, raw_bytes)` per tensor in body (write) order, the order of
    /// [`tensor_names`](Self::tensor_names). `decode_tensor` is `&self` and
    /// reads only shared state, so the per-tensor decodes are independent;
    /// this is the main multi-core lever for full-model loads (the
    /// per-tensor `decode_tensor` stays available for random access).
    pub fn decode_model(&self) -> Result<Vec<(String, Vec<u8>)>, PtwmCoreError> {
        if self.index.is_empty() {
            return Ok(Vec::new());
        }
        // Parse each record header and decode it in the same parallel pass,
        // with no sequential tensor_names() pre-pass.
        self.entries_in_body_order()
            .into_par_iter()
            .map(|e| {
                let rec = self.entry_record(e)?;
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
        let rec = self.tensor_record(name)?;
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
        use crate::chain::runtime::{
            InverseContext, chain_source_descriptor, inverse_chain, max_plane_len,
            propagate_terminal_descriptors_bounded,
        };
        use crate::chain::validate::validate_chain;
        use crate::codec::codec_for;
        use crate::tensor_record::{CHAIN_REF_INLINE_BIT, TENSOR_FLAG_PAYLOAD_HASH};
        use crate::transforms::op::Plane;

        // The compact layout requires exact sizes throughout: the Source
        // equals the raw size and each plane decodes to its descriptor
        // length. The legacy layout tolerates older writers' slack.
        let strict = self.layout == ContainerLayout::Compact;

        // Resolve the chain via the inline bit or the chain registry.
        let bound_chain: Chain;
        let chain: &Chain = if rec.chain_ref & CHAIN_REF_INLINE_BIT != 0 {
            rec.inline_chain.as_ref().ok_or_else(|| {
                PtwmCoreError::InvalidContainer(
                    "decode_tensor: inline bit set but inline_chain is None".into(),
                )
            })?
        } else {
            let chain_id = rec.chain_ref & !CHAIN_REF_INLINE_BIT;
            let shared = self.chain_registry.lookup(chain_id).ok_or_else(|| {
                PtwmCoreError::InvalidContainer(format!(
                    "decode_tensor: chain_id {chain_id} not in registry"
                ))
            })?;
            if strict && has_deferred_source(shared) {
                // A bound Source takes its params from the record's shape.
                let params = record_binding(rec.tensor_metadata.as_deref(), rec.dtype_code)
                    .ok_or_else(|| {
                        PtwmCoreError::InvalidContainer(format!(
                            "decode_tensor: chain {chain_id} has a bound Source but the \
                             record has no shape that binds it"
                        ))
                    })?;
                let mut c = shared.clone();
                bind_deferred_source(&mut c, &params);
                bound_chain = c;
                &bound_chain
            } else {
                shared
            }
        };

        // Bound the chain by the record's `orig_size` before any plane is
        // sized from it, and before `validate_chain`, which calls every op's
        // `propagate_descriptors` on lengths derived from the Source. The
        // Source may overstate the raw size by up to 8×: the Python
        // byte-input path wrote such chains before the P4 fix, and those
        // files must keep decoding. A Source declaring more is corrupt;
        // rejecting it here keeps codecs from pre-allocating gigabytes for a
        // small tensor. `chain_source_descriptor` checks node 0 itself and
        // needs no prior validation.
        let source_desc = chain_source_descriptor(chain)?;
        if strict && source_desc.length_bytes != rec.orig_size {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "decode_tensor: chain Source declares {} bytes for a {}-byte tensor",
                source_desc.length_bytes, rec.orig_size
            )));
        }
        let max_source = rec.orig_size.saturating_mul(8);
        if source_desc.length_bytes > max_source {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "decode_tensor: chain Source declares {} bytes for a {}-byte tensor \
                 (at most {max_source} accepted)",
                source_desc.length_bytes, rec.orig_size
            )));
        }

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
        let plane_bound = max_plane_len(rec.orig_size);

        // Discover the descriptor of every terminal by walking the chain's
        // descriptors symbolically (no source bytes, no plane buffers). Every
        // node's output is bounded by `plane_bound`.
        let term_descriptors = propagate_terminal_descriptors_bounded(chain, plane_bound)?;
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
                // The chunk table is file data: bound every decoded size and
                // their sum before any chunk is decoded or allocated.
                let total = chunks
                    .iter()
                    .try_fold(0u64, |acc, e| acc.checked_add(u64::from(e.decoded_size)))
                    .filter(|&t| t <= plane_bound)
                    .filter(|&t| !strict || t == descriptor.length_bytes)
                    .ok_or_else(|| {
                        PtwmCoreError::InvalidContainer(format!(
                            "chunk table decodes to more than {plane_bound} bytes (or, in \
                             a compact container, not to the plane's {} bytes) for a \
                             {}-byte tensor",
                            descriptor.length_bytes, rec.orig_size
                        ))
                    })? as usize;
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
                        // The table's decoded size is bounded above; the codec
                        // stops at it, so a corrupt chunk cannot decode past it.
                        let decoded = codec.decode_bounded(
                            state_version,
                            state_bytes,
                            chunk_payload,
                            &plane_rec.layout,
                            entry.decoded_size as usize,
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
                // Decoded length comes from the propagated terminal descriptor,
                // which `plane_bound` already limits.
                let decoded_len = usize::try_from(descriptor.length_bytes).map_err(|_| {
                    PtwmCoreError::InvalidContainer("plane length overflows usize".into())
                })?;
                // Self-delimiting codecs may return a length other than the
                // descriptor's: older files carry Sources that overstate
                // (Python byte input) or understate (byte passthrough over an
                // element shape) the raw size. So the codec's hard limit is
                // `plane_bound`, derived from `orig_size`, not `decoded_len`;
                // the length check after `inverse_chain` and the payload hash
                // cover the rest. The codec stops at the limit, so a corrupt
                // payload cannot allocate past it.
                let limit = if strict {
                    decoded_len
                } else {
                    usize::try_from(plane_bound).unwrap_or(usize::MAX)
                };
                let plane = codec.decode_bounded(
                    state_version,
                    &state_bytes,
                    &plane_rec.payload_bytes,
                    &plane_rec.layout,
                    decoded_len,
                    limit,
                )?;
                if strict && plane.len() != decoded_len {
                    return Err(PtwmCoreError::InvalidContainer(format!(
                        "decoded plane is {} bytes, its descriptor says {decoded_len}",
                        plane.len()
                    )));
                }
                plane
            };
            // The codecs enforce the bound; this check keeps it for any
            // codec that does not.
            if plane_bytes.len() as u64 > plane_bound {
                return Err(PtwmCoreError::InvalidContainer(format!(
                    "decoded plane is {} bytes, bound is {plane_bound} for a {}-byte tensor",
                    plane_bytes.len(),
                    rec.orig_size
                )));
            }

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
        if source_plane.bytes.len() as u64 != rec.orig_size {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "decode_tensor: chain decoded {} bytes, record says {}",
                source_plane.bytes.len(),
                rec.orig_size
            )));
        }
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
    use crate::tensor_record::write_tensor_record;
    use crate::tensor_record::{CHAIN_REF_INLINE_BIT, parse_tensor_record};
    use crate::types::PlaneRole;
    use std::io::Cursor;

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
    fn chain_registry_write_rejects_ids_beyond_15_bits() {
        let mut builder = ExtensionTableBuilder::new();
        let mut out = Vec::new();
        let reg = ChainRegistry {
            chains: vec![(MAX_CHAIN_ID + 1, make_minimal_chain(0))],
        };
        let err = reg.write(&mut builder, &mut out).unwrap_err().to_string();
        assert!(err.contains("15-bit"), "{err}");

        let reg = ChainRegistry {
            chains: vec![(MAX_CHAIN_ID, make_minimal_chain(0))],
        };
        reg.write(&mut builder, &mut out).unwrap();
    }

    #[test]
    fn chain_registry_write_rejects_more_than_0x8000_chains() {
        let chain = make_minimal_chain(0);
        let reg = ChainRegistry {
            chains: (0..=MAX_REGISTRY_CHAINS)
                .map(|i| ((i & 0x7FFF) as u16, chain.clone()))
                .collect(),
        };
        let mut builder = ExtensionTableBuilder::new();
        let mut out = Vec::new();
        let err = reg.write(&mut builder, &mut out).unwrap_err().to_string();
        assert!(err.contains("exceeds the limit"), "{err}");
        assert!(out.is_empty());
    }

    #[test]
    fn set_builtin_version_must_come_first() {
        let mut buf = Cursor::new(Vec::new());
        let mut w = ContainerWriter::new(&mut buf).unwrap();
        for bad in ["", "1@2", &"9".repeat(65)] {
            assert!(w.set_builtin_version(bad).is_err(), "{bad:?}");
            // The table builder applies the same rule.
            let registry = ChainRegistry {
                chains: vec![(0, make_minimal_chain(1))],
            };
            assert!(
                build_extension_table(&registry, CodecSet::AllBuiltins, bad).is_err(),
                "{bad:?}"
            );
        }
        w.set_builtin_version("1.0.0").unwrap();
        w.set_chain_registry(ChainRegistry {
            chains: vec![(0, make_minimal_chain(1))],
        })
        .unwrap();
        let err = w.set_builtin_version("1.0.1").unwrap_err().to_string();
        assert!(err.contains("must be set before"), "{err}");
        assert!(
            w.extension_table
                .entries
                .iter()
                .all(|e| e.human_label.ends_with("@1.0.0"))
        );
        // Codec indices are keyed by the current ids whatever the version.
        assert!(
            w.codec_table_idx_for(&crate::extension::builtin_canonical_id("zstd"))
                .is_some()
        );
    }

    /// A legacy writer stamps built-ins with the ids and labels of `1.1.0`,
    /// which a v1.1 reader recognises, whatever the crate version. The
    /// compact layout keeps the crate version internally.
    #[test]
    fn legacy_writer_stamps_legacy_builtin_version() {
        use crate::extension::{BUILTIN_LABEL_PREFIX, BUILTIN_PUBKEY, LEGACY_BUILTIN_VERSION};
        assert_eq!(LEGACY_BUILTIN_VERSION, "1.1.0");
        let mut buf = Cursor::new(Vec::new());
        let mut w = ContainerWriter::new(&mut buf).unwrap();
        assert_eq!(w.builtin_version, "1.1.0");
        assert_eq!(w.encoder.builtin_version, "1.1.0");
        w.set_chain_registry(ChainRegistry {
            chains: vec![(0, make_minimal_chain(1))],
        })
        .unwrap();
        assert!(!w.extension_table.entries.is_empty());
        for e in &w.extension_table.entries {
            let rest = e.human_label.strip_prefix(BUILTIN_LABEL_PREFIX).unwrap();
            let (name, ver) = rest.split_once('@').unwrap();
            assert_eq!(ver, "1.1.0", "{}", e.human_label);
            assert_eq!(
                e.canonical_id,
                CanonicalId::derive(&BUILTIN_PUBKEY, name, "1.1.0")
            );
        }

        let mut buf = Cursor::new(Vec::new());
        let w = ContainerWriter::with_layout(&mut buf, ContainerLayout::Compact).unwrap();
        assert_eq!(w.builtin_version, crate::extension::CURRENT_BUILTIN_VERSION);
    }

    /// Serialized Extension Table of a container written with `registry`.
    fn table_bytes_for(registry: &ChainRegistry, codecs: CodecSet<'_>) -> Vec<u8> {
        build_extension_table(registry, codecs, crate::extension::CURRENT_BUILTIN_VERSION)
            .unwrap()
            .table
            .to_bytes()
            .unwrap()
    }

    #[test]
    fn raw_extension_table_is_validated() {
        let registry = ChainRegistry {
            chains: vec![(0, v3_passthrough_chain(4))],
        };
        let good = table_bytes_for(&registry, CodecSet::AllBuiltins);
        let built = encode_registry_for_table(&good, &registry).unwrap();
        let reference = build_extension_table(
            &registry,
            CodecSet::AllBuiltins,
            crate::extension::CURRENT_BUILTIN_VERSION,
        )
        .unwrap();
        assert_eq!(built.registry_bytes, reference.registry_bytes);
        assert_eq!(built.codec_idx_map, reference.codec_idx_map);

        // Trailing bytes.
        let mut trailing = good.clone();
        trailing.push(0);
        let e = encode_registry_for_table(&trailing, &registry).unwrap_err();
        assert!(e.to_string().contains("trailing bytes"), "{e}");

        // A table that lacks a chain op.
        let other = ChainRegistry {
            chains: vec![(0, make_minimal_chain(0))],
        };
        let only_passthrough = table_bytes_for(&other, CodecSet::Only(&[]));
        let e = encode_registry_for_table(&only_passthrough, &registry).unwrap_err();
        assert!(e.to_string().contains("missing from the table"), "{e}");

        // Embedded WASM entries.
        let mut table = ExtensionTable::from_bytes(&good).unwrap();
        table.entries[0].embedded_wasm_offset = Some(0);
        table.entries[0].embedded_wasm_length = Some(1);
        let wasm = table.to_bytes().unwrap();
        let e = encode_registry_for_table(&wasm, &registry).unwrap_err();
        assert!(e.to_string().contains("embedded WASM"), "{e}");

        // Garbage.
        assert!(encode_registry_for_table(b"\xff\xff\xff\xff", &registry).is_err());
    }

    /// A raw table is written verbatim; the registry indices resolve
    /// against it, here a table whose ops come after its codecs.
    #[test]
    fn raw_extension_table_is_written_verbatim() {
        use crate::extension::{Kind, builtin_entries};
        let registry = ChainRegistry {
            chains: vec![(0, v3_passthrough_chain(4))],
        };
        let mut entries: Vec<_> = builtin_entries()
            .into_iter()
            .filter(|e| e.kind == Kind::PlaneCodec)
            .collect();
        entries
            .push(crate::extension::builtin_entry_at_version("byte_passthrough", "1.0.0").unwrap());
        entries.push(crate::extension::builtin_entry_at_version("source", "1.0.0").unwrap());
        let raw = ExtensionTable { entries }.to_bytes().unwrap();

        let mut buf = Cursor::new(Vec::new());
        {
            let mut w = ContainerWriter::new(&mut buf).unwrap();
            w.set_raw_extension_table(&raw, registry.clone()).unwrap();
            let identity = crate::extension::builtin_canonical_id("identity");
            assert_eq!(w.codec_table_idx_for(&identity), Some(0));
            let mut rec = sample_tensor_record("t");
            rec.chain_ref = 0;
            rec.orig_size = 4;
            rec.terminals[0].codec_id = CodecId::Identity;
            rec.terminals[0].codec_table_idx = 0;
            rec.terminals[0].payload_bytes = vec![1, 2, 3, 4];
            rec.terminals[0].payload_len = 4;
            rec.terminals[0].crc32 = None;
            w.append_tensor(&rec).unwrap();
            w.finalize().unwrap();
        }
        let blob = buf.into_inner();
        let reader = ContainerReader::open(&blob).unwrap();
        let et = &blob[reader.header.extension_table_offset as usize..][..raw.len()];
        assert_eq!(et, raw.as_slice());
        assert_eq!(reader.chain_registry, registry);
        assert_eq!(reader.decode_tensor("t").unwrap(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn writer_rejects_duplicate_tensor_names() {
        let mut buf = Cursor::new(Vec::new());
        let mut w = ContainerWriter::new(&mut buf).unwrap();
        w.append_tensor(&sample_tensor_record("a")).unwrap();
        let err = w
            .append_tensor(&sample_tensor_record("a"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("duplicate tensor name"), "{err}");

        let mut bytes = Vec::new();
        write_tensor_record(&sample_tensor_record("b"), &mut bytes).unwrap();
        w.append_serialized_tensor("b", &bytes).unwrap();
        let err = w
            .append_serialized_tensor("b", &bytes)
            .unwrap_err()
            .to_string();
        assert!(err.contains("duplicate tensor name"), "{err}");
        let err = w
            .append_serialized_tensor("a", &bytes)
            .unwrap_err()
            .to_string();
        assert!(err.contains("duplicate tensor name"), "{err}");
        w.finalize().unwrap();
    }

    /// A compact frame passed to the container reader is named as such, not
    /// reported as a truncated file (it has no EOF sentinel).
    #[test]
    fn container_open_names_a_compact_frame() {
        let mut frame = crate::compact::MAGIC_COMPACT.to_vec();
        frame.extend_from_slice(&[0u8; 40]);
        for open in [ContainerReader::open, ContainerReader::open_partial] {
            let Err(err) = open(&frame) else {
                panic!("compact frame opened as a container")
            };
            let err = err.to_string();
            assert!(err.contains("compact single-tensor frame"), "{err}");
            assert!(!err.contains("truncated"), "{err}");
        }
    }

    /// A legacy record whose name does not hash to its index entry, or two
    /// records with one name, are rejected when the container opens. A bit
    /// flip in a record name once renamed the tensor silently.
    #[test]
    fn legacy_open_rejects_renamed_and_duplicate_records() {
        let mut buf = Cursor::new(Vec::new());
        {
            let mut w = ContainerWriter::new(&mut buf).unwrap();
            for name in ["layer.0.weight", "layer.1.weight"] {
                w.append_tensor(&sample_tensor_record(name)).unwrap();
            }
            w.finalize().unwrap();
        }
        let good = buf.into_inner();
        ContainerReader::open(&good).unwrap();
        let find = |hay: &[u8], needle: &[u8]| {
            let at: Vec<usize> = hay
                .windows(needle.len())
                .enumerate()
                .filter(|(_, w)| *w == needle)
                .map(|(i, _)| i)
                .collect();
            assert_eq!(at.len(), 1, "{needle:?} must occur once");
            at[0]
        };

        // One bit flip turns the '0' of "layer.0.weight" into '1'.
        let mut renamed = good.clone();
        let at = find(&renamed, b"layer.0.weight");
        renamed[at + 6] ^= 1;
        for open in [ContainerReader::open, ContainerReader::open_partial] {
            let Err(err) = open(&renamed) else {
                panic!("renamed record accepted")
            };
            let err = err.to_string();
            assert!(err.contains("does not match its index entry"), "{err}");
        }

        // The same rename with the index hash patched to match.
        let mut dup = renamed.clone();
        let at = find(&dup, &name_hash("layer.0.weight").to_le_bytes());
        dup[at..at + 8].copy_from_slice(&name_hash("layer.1.weight").to_le_bytes());
        let Err(err) = ContainerReader::open(&dup) else {
            panic!("duplicate name accepted")
        };
        let err = err.to_string();
        assert!(err.contains("duplicate tensor name"), "{err}");
    }

    /// Two names with the same index hash: lookups compare the record name
    /// and pick the right entry, whatever their order in the hash run.
    #[test]
    fn lookup_resolves_name_hash_collision() {
        let names = ["first", "second", "third"];
        let mut buf = Cursor::new(Vec::new());
        {
            let mut w = ContainerWriter::new(&mut buf).unwrap();
            for name in names {
                w.append_tensor(&sample_tensor_record(name)).unwrap();
            }
            w.finalize().unwrap();
        }
        let bytes = buf.into_inner();
        let mut reader = ContainerReader::open(&bytes).unwrap();
        let expected: Vec<Vec<u8>> = names
            .iter()
            .map(|n| reader.get_tensor_record_bytes(n).unwrap().to_vec())
            .collect();

        // Give "first" and "second" one hash, then try both run orders.
        let forced = name_hash("first");
        for e in reader.index.iter_mut() {
            if e.name_hash == name_hash("second") {
                e.name_hash = forced;
            }
        }
        reader.index.sort_by_key(|e| e.name_hash);
        let run: Vec<usize> = (0..reader.index.len())
            .filter(|&i| reader.index[i].name_hash == forced)
            .collect();
        assert_eq!(run.len(), 2);
        for _ in 0..2 {
            for (name, want) in names[..2].iter().zip(&expected) {
                let got = reader.record_bytes_with_hash(name, forced).unwrap();
                assert_eq!(got, want.as_slice(), "lookup of {name}");
            }
            assert_eq!(
                reader.get_tensor_record_bytes("third").unwrap(),
                expected[2].as_slice()
            );
            // A name in the run's hash with no record of its own.
            assert!(reader.record_bytes_with_hash("absent", forced).is_none());
            reader.index.swap(run[0], run[1]);
        }
        // The real hash of "second" no longer has an entry.
        assert!(reader.get_tensor_record_bytes("second").is_none());
        assert!(reader.decode_tensor("second").is_err());
    }

    /// The prelude and the chain registry are parsed from the shared-prelude
    /// section only. A shrunk shared_prelude_length cuts the registry off
    /// instead of letting the parser read into the tensor records.
    #[test]
    fn prelude_and_registry_are_bounded_by_shared_prelude_length() {
        let mut buf = Cursor::new(Vec::new());
        {
            let mut w = ContainerWriter::new(&mut buf).unwrap();
            w.set_chain_registry(ChainRegistry {
                chains: vec![(0, make_minimal_chain(7))],
            })
            .unwrap();
            w.append_tensor(&sample_tensor_record("t")).unwrap();
            w.finalize().unwrap();
        }
        let bytes = buf.into_inner();
        let reader = ContainerReader::open(&bytes).unwrap();
        assert_eq!(reader.chain_registry.chains.len(), 1);
        let header = reader.header;

        let patch = |len: u64| {
            let mut h = header;
            h.shared_prelude_length = len;
            let mut b = bytes.clone();
            b[..HEADER_LEN].copy_from_slice(&h.to_bytes().unwrap());
            b
        };

        // Section ends one byte early: the registry is truncated.
        let short = patch(header.shared_prelude_length - 1);
        let err = ContainerReader::open(&short).expect_err("must fail");
        assert!(err.to_string().contains("ChainRegistry"), "{err}");

        // Section holds only the prelude: no registry is read.
        let prelude_only = patch(4);
        let reader = ContainerReader::open(&prelude_only).unwrap();
        assert!(reader.chain_registry.chains.is_empty());

        // Section shorter than the prelude count.
        let err = ContainerReader::open(&patch(2)).expect_err("must fail");
        assert!(err.to_string().contains("prelude"), "{err}");

        // A zero-length section has no prelude at all and is rejected.
        let err = ContainerReader::open(&patch(0)).expect_err("must fail");
        assert!(err.to_string().contains("prelude"), "{err}");

        // An intact section puts the tensor records right after it.
        assert_eq!(
            ContainerReader::open(&bytes).unwrap().tensors_offset,
            header.shared_prelude_offset + header.shared_prelude_length
        );

        // Section runs past the end of the buffer.
        let err = ContainerReader::open(&patch(u64::MAX)).expect_err("must fail");
        assert!(err.to_string().contains("shared prelude"), "{err}");
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
                ..Default::default()
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
                ..Default::default()
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
                ..Default::default()
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
                // Reads the legacy index and records, so the layout is pinned.
                container_layout: ContainerLayout::Legacy,
                ..Default::default()
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
                ..Default::default()
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

    // ── Decode bounds (P5) ───────────────────────────────────────────────

    /// Compress `raw` through a passthrough chain with the Identity codec
    /// (so plane bytes equal tensor bytes), then rewrite the container with
    /// the record's chain inline and `edit` applied to the record.
    fn identity_container_with(
        raw: &[u8],
        chunk_size: Option<u32>,
        edit: impl FnOnce(&mut TensorRecord),
    ) -> Vec<u8> {
        identity_container_with_chain(
            raw,
            v3_passthrough_chain(raw.len() as u32),
            chunk_size,
            edit,
        )
    }

    /// [`identity_container_with`] for a given single-terminal chain.
    fn identity_container_with_chain(
        raw: &[u8],
        chain: Chain,
        chunk_size: Option<u32>,
        edit: impl FnOnce(&mut TensorRecord),
    ) -> Vec<u8> {
        use crate::compressor::{CompressorOptions, InputTensor, compress_model};
        use crate::tensor_record::{CHAIN_REF_INLINE_BIT, parse_tensor_record};
        use std::io::Cursor;

        let inputs = vec![InputTensor {
            name: "t".into(),
            candidate_chains: vec![chain.clone()],
            dtype_code: 0x0005,
            input_format: 0,
            raw_bytes: raw,
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
                emit_payload_hash: false,
                emit_plane_crc: false,
                forced_codec: Some(CodecId::Identity),
                allow_codec_ids: None,
                chunk_size,
                // Parses legacy records, so the layout is pinned.
                container_layout: ContainerLayout::Legacy,
                ..Default::default()
            },
        )
        .unwrap();
        let reader = ContainerReader::open(&buf).unwrap();
        assert_eq!(reader.decode_tensor("t").unwrap(), raw);
        let (mut rec, _) =
            parse_tensor_record(reader.get_tensor_record_bytes("t").unwrap()).unwrap();
        rec.chain_ref = CHAIN_REF_INLINE_BIT;
        rec.inline_chain = Some(chain);
        edit(&mut rec);

        let mut out: Vec<u8> = Vec::new();
        {
            let mut w = ContainerWriter::new(std::io::Cursor::new(&mut out)).unwrap();
            w.set_extension_table(reader.extension_table.clone());
            w.append_tensor(&rec).unwrap();
            w.finalize().unwrap();
        }
        out
    }

    fn set_source_shape(rec: &mut TensorRecord, dims: &[u32], dtype_code: u16) {
        let mut params = vec![dims.len() as u8];
        for d in dims {
            params.extend_from_slice(&d.to_le_bytes());
        }
        params.extend_from_slice(&dtype_code.to_le_bytes());
        rec.inline_chain.as_mut().unwrap().nodes[0].params = params;
    }

    #[test]
    fn rewritten_identity_container_still_decodes() {
        let raw: Vec<u8> = (0..64u8).collect();
        let buf = identity_container_with(&raw, None, |_| {});
        assert_eq!(
            ContainerReader::open(&buf)
                .unwrap()
                .decode_tensor("t")
                .unwrap(),
            raw
        );
    }

    #[test]
    fn huge_source_fails_fast_without_allocating() {
        // A 16-byte tensor whose chain declares [1<<20, 1<<20] fp32 (4 TiB),
        // and one that declares 4 GiB: both fail before any plane buffer is
        // sized from the chain.
        let raw = vec![7u8; 16];
        for (dims, dtype) in [
            (vec![1u32 << 20, 1 << 20], 0x0003u16),
            (vec![u32::MAX], 0x0005),
        ] {
            let buf =
                identity_container_with(&raw, None, |rec| set_source_shape(rec, &dims, dtype));
            let reader = ContainerReader::open(&buf).unwrap();
            let t0 = std::time::Instant::now();
            let err = reader.decode_tensor("t").unwrap_err().to_string();
            assert!(err.contains("Source declares"), "{err}");
            assert!(t0.elapsed() < std::time::Duration::from_secs(1));
        }
    }

    #[test]
    fn huge_source_is_rejected_before_op_descriptor_arithmetic() {
        // Source [u32::MAX, u32::MAX] int8 (about 2^64 bytes) feeding
        // BurrowsWheeler, whose output length adds a block header per
        // 256 KiB. The Source bound must reject the chain before
        // `validate_chain` walks the op descriptors; before the fix, a build
        // with overflow checks (debug, cargo-fuzz) panicked inside
        // `BurrowsWheeler::propagate_descriptors`.
        use crate::transforms::op::OpId;
        let raw: Vec<u8> = (0..64u8).collect();
        let mut chain = v3_passthrough_chain(raw.len() as u32);
        chain.nodes[1].op = OpId::BurrowsWheeler;
        let buf = identity_container_with_chain(&raw, chain, None, |rec| {
            set_source_shape(rec, &[u32::MAX, u32::MAX], 0x0005)
        });
        let err = ContainerReader::open(&buf)
            .unwrap()
            .decode_tensor("t")
            .unwrap_err()
            .to_string();
        assert!(err.contains("Source declares"), "{err}");
    }

    #[test]
    fn overstated_source_within_8x_still_decodes() {
        // Files written by the Python byte-input path before the P4 fix
        // declare fp32 elements for a byte count: 4× the raw size.
        let raw: Vec<u8> = (0..64u8).collect();
        let buf = identity_container_with(&raw, None, |rec| set_source_shape(rec, &[64], 0x0003));
        assert_eq!(
            ContainerReader::open(&buf)
                .unwrap()
                .decode_tensor("t")
                .unwrap(),
            raw
        );
        // Above 8× is rejected.
        let buf = identity_container_with(&raw, None, |rec| set_source_shape(rec, &[129], 0x0003));
        let err = ContainerReader::open(&buf)
            .unwrap()
            .decode_tensor("t")
            .unwrap_err()
            .to_string();
        assert!(err.contains("Source declares"), "{err}");
    }

    #[test]
    fn understated_source_still_decodes() {
        // Byte passthrough over an element shape (Python fallback for dtypes
        // without a production chain) declares fewer bytes than the tensor.
        let raw: Vec<u8> = (0..64u8).collect();
        let buf = identity_container_with(&raw, None, |rec| set_source_shape(rec, &[16], 0x0005));
        assert_eq!(
            ContainerReader::open(&buf)
                .unwrap()
                .decode_tensor("t")
                .unwrap(),
            raw
        );
    }

    #[test]
    fn decoded_length_must_equal_orig_size() {
        let raw: Vec<u8> = (0..64u8).collect();
        let buf = identity_container_with(&raw, None, |rec| rec.orig_size = 63);
        let err = ContainerReader::open(&buf)
            .unwrap()
            .decode_tensor("t")
            .unwrap_err()
            .to_string();
        assert!(err.contains("record says 63"), "{err}");
    }

    #[test]
    fn chunk_table_beyond_bound_is_rejected_before_decode() {
        let raw: Vec<u8> = (0..2048u32).map(|i| (i % 251) as u8).collect();
        let clean = identity_container_with(&raw, Some(256), |rec| {
            assert!(rec.terminals[0].chunk_table.is_some());
        });
        assert_eq!(
            ContainerReader::open(&clean)
                .unwrap()
                .decode_tensor("t")
                .unwrap(),
            raw
        );
        // One chunk claims u32::MAX decoded bytes for a 2 KiB tensor.
        let buf = identity_container_with(&raw, Some(256), |rec| {
            rec.terminals[0].chunk_table.as_mut().unwrap()[0].decoded_size = u32::MAX;
        });
        let t0 = std::time::Instant::now();
        let err = ContainerReader::open(&buf)
            .unwrap()
            .decode_tensor("t")
            .unwrap_err()
            .to_string();
        assert!(err.contains("chunk table decodes to more than"), "{err}");
        assert!(t0.elapsed() < std::time::Duration::from_secs(1));
    }
}
