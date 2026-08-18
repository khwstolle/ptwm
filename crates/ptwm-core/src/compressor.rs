//! PPG compressor entry point.
//!
//! `compress_model` takes per-tensor candidate `Chain`s, runs each forward
//! DAG via [`forward_chain`], routes every terminal plane through the
//! capability-keyed [`dispatch`] menu, fits Pass-2 shared state on the
//! chosen terminals, and writes a chain-registry-backed `.ptwm` container.

use std::io::{Seek, Write};
use std::sync::Arc;

use rayon::prelude::*;
use xxhash_rust::xxh64::xxh64;

use crate::chain::Chain;
use crate::chain::runtime::{ForwardContext, forward_chain};
use crate::chain::validate::validate_chain;
use crate::codec::{CodecId, StateSource, codec_for};
use crate::codecs::order1_scale_ac;
use crate::container::{ChainRegistry, ContainerWriter};
use crate::dispatch::dispatch;
use crate::error::PtwmCoreError;
use crate::extension::CanonicalId;
use crate::fit::fit_shared;
use crate::layout::PlaneLayout;
use crate::metadata::encode_shape_metadata;
use crate::plane_record::PlaneRecord;
use crate::prelude::PreludeEntry;
use crate::select::{MenuItem, TrialResult, trial_encode_plane};
use crate::tensor_record::{Dependency, RefKind, TENSOR_FLAG_PAYLOAD_HASH, TensorRecord};
use crate::transforms::op::Plane;
use crate::types::PlaneRole;
use crate::types::descriptor::{ElementWidth, Layout, PlaneDescriptor};
use crate::types::role::{NibbleKind, Role};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Map a builtin [`CodecId`] to its [`CanonicalId`] so the compressor can
/// intern the codec into the Extension Table and fill `codec_table_idx`.
fn codec_canonical_id(id: CodecId) -> CanonicalId {
    use crate::extension::builtin_canonical_id;
    match id {
        CodecId::Identity => builtin_canonical_id("identity"),
        CodecId::Huffman => builtin_canonical_id("huffman"),
        CodecId::HuffmanNibble => builtin_canonical_id("huffman_nibble"),
        CodecId::Rans => builtin_canonical_id("rans"),
        CodecId::Zstd => builtin_canonical_id("zstd"),
        CodecId::ZstdDict => builtin_canonical_id("zstd_dict"),
        CodecId::Fpc => builtin_canonical_id("fpc"),
        CodecId::Tans => builtin_canonical_id("tans"),
        CodecId::PerGroupCodebook => builtin_canonical_id("per_group_codebook"),
        CodecId::Order1ScaleAC => builtin_canonical_id("order1_scale_ac"),
        CodecId::ArithmeticO0 => builtin_canonical_id("arithmetic_o0"),
        CodecId::ArithmeticO0Adaptive => builtin_canonical_id("arithmetic_o0_adaptive"),
        CodecId::ArithmeticO1 => builtin_canonical_id("arithmetic_o1"),
        CodecId::ContextMixingLite => builtin_canonical_id("context_mixing_lite"),
        CodecId::HuffLlm5Bit => builtin_canonical_id("huff_llm_5bit"),
        CodecId::Order1Arithmetic => builtin_canonical_id("order1_arithmetic"),
        CodecId::NeuralPredictor => builtin_canonical_id("neural_predictor"),
    }
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Per-tensor input for the compressor. The caller supplies one or more
/// candidate [`Chain`]s; the compressor runs each forward+trial-encode and
/// keeps the chain with the smallest output (multi-chain trial-encode).
pub struct InputTensor<'a> {
    pub name: String,
    /// At least one chain. When `len == 1` the compressor uses that chain
    /// unconditionally.
    pub candidate_chains: Vec<Chain>,
    pub dtype_code: u16,
    pub input_format: u8,
    pub raw_bytes: &'a [u8],
    /// Optional row-major shape. When both `shape` and `dtype_name` are
    /// supplied, the tensor record carries encoded shape metadata.
    pub shape: Option<Vec<u64>>,
    /// Canonical dtype name. Paired with `shape` for tensor metadata.
    pub dtype_name: Option<String>,
    /// When `Some`, attaches a `Dependency::ExternalSafetensors` entry
    /// carrying the supplied 32-byte BLAKE3 digest. The decoder verifies
    /// the supplied reference against this hash.
    pub delta_reference_blake3: Option<[u8; 32]>,
}

/// Compressor options.
pub struct CompressorOptions {
    /// Header hint to readers about the dominant codec.
    pub method_hint: u16,
    pub emit_payload_hash: bool,
    pub emit_plane_crc: bool,
    /// When `Some`, bypass dispatcher selection and force every plane
    /// through this codec. The forced codec must accept any descriptor it
    /// sees; `Identity` is always safe.
    pub forced_codec: Option<CodecId>,
    /// When `Some`, restrict the trial-encode menu to codecs whose
    /// `CodecId` appears in the list. Empty intersection on a plane
    /// returns `InvalidContainer` rather than falling back to Identity
    /// (silent fallback would falsify ablation measurements). Ignored
    /// when `forced_codec` is `Some`.
    pub allow_codec_ids: Option<Vec<CodecId>>,
    /// When `Some`, encode each plane in fixed-size chunks.
    pub chunk_size: Option<u32>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Map a typed [`Role`] to the legacy [`PlaneRole`] (used for the on-disk
/// `PlaneRecord.role` byte). Best-effort: unmapped variants fall to `Value`.
fn legacy_plane_role(role: &Role) -> PlaneRole {
    match role {
        Role::Value { .. } => PlaneRole::Value,
        Role::Scale { .. } => PlaneRole::Scale,
        Role::GlobalScale { .. } => PlaneRole::Scale,
        Role::ExponentByte => PlaneRole::Exponent,
        Role::MantissaByte { .. } => PlaneRole::Mantissa,
        Role::Nibble { kind } => match kind {
            NibbleKind::Exponent => PlaneRole::Exponent,
            NibbleKind::SignMantissa => PlaneRole::Sign,
            NibbleKind::Value => PlaneRole::Value,
        },
        Role::IntegerByte { index, .. } => match index {
            0 => PlaneRole::Byte0,
            1 => PlaneRole::Byte1,
            2 => PlaneRole::Byte2,
            _ => PlaneRole::Byte3,
        },
        Role::Residual { .. } => PlaneRole::Value,
        Role::Raw => PlaneRole::Value,
        Role::Index => PlaneRole::Value,
        Role::Vendor { .. } => PlaneRole::Value,
    }
}

/// Map a typed [`Layout`] to the on-disk [`PlaneLayout`].
fn legacy_plane_layout(layout: Layout) -> PlaneLayout {
    match layout {
        Layout::Flat => PlaneLayout::Flat,
        Layout::Rows { row_len } => PlaneLayout::rows(row_len).unwrap_or(PlaneLayout::Flat),
    }
}

/// Build a default source `PlaneDescriptor` keyed on dtype code.
///
/// `element_width` and `is_nibble_packed` come from `transforms::source`,
/// which owns the dtype-code mapping. This function used to inline its own
/// copy of that match. The copy drifted: it kept packed FP4 in the byte
/// catch-all, so every plane reaching the trial encode was byte-width and
/// not nibble-packed, and `PerGroupCodebook` could accept none of them.
/// Call the mapping, do not restate it.
///
/// `length_bytes` is the raw byte count; `Layout::Rows{row_len}` applies
/// when `shape` is supplied, using the row-major last dimension.
///
/// Public so the PyO3 binding (`ptwm-py`) can build the same source
/// descriptor without duplicating the dtype-code → element-width mapping.
pub fn source_descriptor_for(
    dtype_code: u16,
    raw_byte_count: u64,
    shape: Option<&[u64]>,
) -> PlaneDescriptor {
    let element_width = crate::transforms::source::element_width_for(dtype_code);
    let layout = match shape {
        Some(s) if !s.is_empty() => match u32::try_from(*s.last().unwrap()) {
            Ok(row_len) => Layout::rows(row_len).unwrap_or(Layout::Flat),
            Err(_) => {
                tracing::warn!(
                    "source_descriptor_for: last dim {} > u32::MAX; falling back to \
                     Layout::Flat (Order1ScaleAC and other row-aware codecs disabled)",
                    s.last().unwrap()
                );
                Layout::Flat
            }
        },
        _ => Layout::Flat,
    };
    PlaneDescriptor {
        role: Role::Raw,
        element_width,
        length_bytes: raw_byte_count,
        layout,
        derives_from_tensor: None,
        residual_of: None,
        is_nibble_packed: crate::transforms::source::is_nibble_packed_dtype(dtype_code),
        vendor_bytes: vec![],
    }
}

/// Build a trial-encode menu for a terminal plane keyed on its descriptor.
///
/// Returns a `Vec<MenuItem>` whose entries cover every dispatcher-accepted
/// codec, with `state_source = None` for [`CodecId::Identity`] and `Inline`
/// for everything else. Shared-state variants are appended separately by the
/// caller (after Pass-2 fitting) — keeping the menu builder free of shared
/// state lets the per-chain trial in Pass-1 use a fixed lower-bound menu.
fn menu_for_descriptor(descriptor: &PlaneDescriptor) -> Vec<MenuItem> {
    let candidates = dispatch(descriptor);
    candidates
        .into_iter()
        .map(|codec_id| MenuItem {
            codec_id,
            state_source: if codec_id == CodecId::Identity {
                StateSource::None
            } else {
                StateSource::Inline
            },
            shared_state_bytes: None,
            shared_state_id: None,
        })
        .collect()
}

/// Drop menu entries whose codec returns `false` from `should_attempt`.
/// This is the content-aware second-stage filter that runs after the
/// descriptor-only [`menu_for_descriptor`]: it lets expensive codecs
/// (notably `Order1ScaleAC`) short-circuit their fit when a cheap
/// statistical sample says they won't beat a cheaper menu item.
///
/// **Only inline-state entries** are subject to the gate: shared-state
/// variants amortize their table overhead across every plane that
/// references the prelude entry, so the per-plane cost model the gate
/// assumes does not apply. Pruning shared-state entries here would
/// silently drop Pass-2 fits from the trial menu and break the path
/// that promotes good candidates into the prelude.
///
/// Entries always survive when their codec's `should_attempt` returns
/// `true`, so this can never make a chain fail to encode — at minimum
/// `Identity` (whose default `should_attempt` is `true`) remains.
fn prune_menu_by_should_attempt(
    menu: &mut Vec<MenuItem>,
    plane: &[u8],
    descriptor: &PlaneDescriptor,
    layout: &PlaneLayout,
) {
    menu.retain(|item| {
        if item.state_source != StateSource::Inline {
            return true;
        }
        codec_for(item.codec_id)
            .map(|codec| codec.should_attempt(plane, descriptor, layout))
            .unwrap_or(true)
    });
}

/// Append `Shared` MenuItems for every shared-state entry whose codec id is
/// already in the inline menu. Caller must supply the full prelude entries
/// list and the candidate menu produced by [`menu_for_descriptor`].
fn append_shared_menu_items(
    base: &mut Vec<MenuItem>,
    descriptor: &PlaneDescriptor,
    prelude_entries: &[PreludeEntry],
) {
    for entry in prelude_entries {
        // Only consider shared states whose codec is in the inline menu —
        // they are the ones where the dispatcher already approved this codec
        // for this descriptor.
        let codec_in_menu = base.iter().any(|m| m.codec_id == entry.codec_id);
        if !codec_in_menu {
            continue;
        }
        // Specialist codecs (PGC, O1SAC) only accept their own descriptor
        // shapes; for other codecs the dispatcher already filtered, so the
        // `codec_in_menu` check above is sufficient.
        if entry.codec_id == CodecId::Order1ScaleAC
            && !matches!(
                descriptor.role,
                Role::Scale { .. } | Role::GlobalScale { .. }
            )
        {
            continue;
        }
        if entry.codec_id == CodecId::PerGroupCodebook && !descriptor.is_nibble_packed {
            continue;
        }
        base.push(MenuItem {
            codec_id: entry.codec_id,
            state_source: StateSource::Shared,
            shared_state_bytes: Some(entry.state_bytes.clone()),
            shared_state_id: Some(entry.shared_state_id),
        });
    }
}

/// One terminal plane chosen by the trial-encoder, ready to be written.
struct ChosenTerminal {
    role: PlaneRole,
    codec_id: CodecId,
    state_source: StateSource,
    shared_state_id: Option<u16>,
    encoded_state_bytes: Vec<u8>,
    encoded_state_version: u8,
    payload_bytes: Vec<u8>,
    // Arc-shared with the terminal plane: an O(1) clone, avoiding a full copy
    // of every plane (~the whole model) in the hot Pass-1 loop.
    raw_plane_for_crc: Arc<[u8]>,
    layout: PlaneLayout,
}

/// Frame a single encoded chunk for storage inside a plane's payload_bytes.
/// Layout: `[state_version u8][state_len u32 LE][state_bytes][payload]`.
fn frame_chunk(out: &mut Vec<u8>, encoded: &crate::codec::Encoded) {
    out.push(encoded.state_format_version);
    out.extend_from_slice(&(encoded.state_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&encoded.state_bytes);
    out.extend_from_slice(&encoded.payload);
}

/// Encode a single plane in fixed-size chunks.
fn encode_plane_chunked(
    plane_raw: &[u8],
    codec: &dyn crate::codec::PlaneCodec,
    chunk_size: u32,
) -> Result<(Vec<u8>, Vec<crate::plane_record::ChunkEntry>), PtwmCoreError> {
    use crate::plane_record::ChunkEntry;

    let chunk_size = (chunk_size as usize).max(1);

    if plane_raw.is_empty() {
        let encoded = codec.encode(&[], None, &PlaneLayout::Flat)?;
        let mut payload_bytes: Vec<u8> = Vec::new();
        frame_chunk(&mut payload_bytes, &encoded);
        return Ok((
            payload_bytes,
            vec![ChunkEntry {
                offset_in_payload: 0,
                decoded_size: 0,
            }],
        ));
    }

    // Chunks are entropy-coded independently (no shared state), so they are
    // embarrassingly parallel. Encode every chunk on the rayon global pool
    // (sized by RAYON_NUM_THREADS), then frame them sequentially in order so
    // the on-disk layout is byte-identical to the serial path — the decoder
    // is unchanged. This is the primary multi-core scaling lever for a single
    // large tensor.
    let encoded: Vec<crate::codec::Encoded> = plane_raw
        .par_chunks(chunk_size)
        .map(|chunk| codec.encode(chunk, None, &PlaneLayout::Flat))
        .collect::<Result<Vec<_>, _>>()?;

    // Pre-size to the exact framed length (frame_chunk writes
    // 1 + 4 + state_bytes + payload per chunk) so sequential framing never
    // reallocates.
    let payload_capacity: usize = encoded
        .iter()
        .map(|enc| 1 + 4 + enc.state_bytes.len() + enc.payload.len())
        .sum();
    let mut payload_bytes: Vec<u8> = Vec::with_capacity(payload_capacity);
    let mut entries: Vec<ChunkEntry> = Vec::with_capacity(encoded.len());
    for (chunk, enc) in plane_raw.chunks(chunk_size).zip(encoded.iter()) {
        // ChunkEntry.offset_in_payload is u32; reject a plane whose framed
        // payload would exceed the u32 offset range rather than silently
        // truncating (reachable for multi-GB planes in very large models).
        let offset_in_payload = u32::try_from(payload_bytes.len()).map_err(|_| {
            PtwmCoreError::InvalidContainer("chunked plane payload exceeds u32 offset range".into())
        })?;
        frame_chunk(&mut payload_bytes, enc);
        entries.push(ChunkEntry {
            offset_in_payload,
            decoded_size: chunk.len() as u32,
        });
    }

    Ok((payload_bytes, entries))
}

/// Restrict the menu to `allow`. When `allow` is `None` the menu passes
/// through unchanged. Beyond the subtractive filter, any codec named in
/// `allow` that the default dispatch did not produce is **added** when its
/// `accepts(descriptor)` is true — this is how an opt-in codec (e.g.
/// `context_mixing_lite`, absent from the default menu) enters selection.
/// An empty menu after filtering returns `InvalidContainer` — a silent
/// `Identity` fallback would falsify ablation measurements.
fn filter_menu_by_allow_list(
    menu: &mut Vec<MenuItem>,
    allow: Option<&[CodecId]>,
    descriptor: &PlaneDescriptor,
) -> Result<(), PtwmCoreError> {
    let Some(allow) = allow else {
        return Ok(());
    };
    menu.retain(|item| allow.contains(&item.codec_id));
    for &id in allow {
        if menu.iter().any(|m| m.codec_id == id) {
            continue;
        }
        if let Some(codec) = crate::codec::codec_for(id) {
            if codec.accepts(descriptor) {
                // Use the same state-source convention as `menu_for_descriptor`
                // (`None` only for Identity, `Inline` otherwise). This is what
                // makes `prune_menu_by_should_attempt` consult the codec's
                // `should_attempt` gate — without it an opt-in codec like CM
                // would skip its size gate and trial-encode every plane.
                menu.push(MenuItem {
                    codec_id: id,
                    state_source: if id == CodecId::Identity {
                        StateSource::None
                    } else {
                        StateSource::Inline
                    },
                    shared_state_bytes: None,
                    shared_state_id: None,
                });
            }
        }
    }
    if menu.is_empty() {
        return Err(PtwmCoreError::InvalidContainer(
            "trial-encode menu empty after applying allow_codec_ids; \
             policy excludes every codec the plane's role accepts"
                .into(),
        ));
    }
    Ok(())
}

/// Run the dispatch + trial-encode for one terminal plane. Returns the chosen
/// `MenuItem` plus the encoded result, and the byte-count signal used by the
/// multi-chain ranker (`encoded.payload.len()` — chosen as the dominant
/// signal; state bytes are folded in if present).
fn trial_encode_terminal(
    plane: &Plane,
    forced_codec: Option<CodecId>,
    extra_shared: &[PreludeEntry],
    allow_codec_ids: Option<&[CodecId]>,
) -> Result<(TrialResult, usize), PtwmCoreError> {
    let layout = legacy_plane_layout(plane.descriptor.layout);
    let menu = if let Some(forced) = forced_codec {
        vec![MenuItem {
            codec_id: forced,
            state_source: if forced == CodecId::Identity {
                StateSource::None
            } else {
                StateSource::Inline
            },
            shared_state_bytes: None,
            shared_state_id: None,
        }]
    } else {
        let mut m = menu_for_descriptor(&plane.descriptor);
        append_shared_menu_items(&mut m, &plane.descriptor, extra_shared);
        filter_menu_by_allow_list(&mut m, allow_codec_ids, &plane.descriptor)?;
        prune_menu_by_should_attempt(&mut m, &plane.bytes, &plane.descriptor, &layout);
        m
    };
    let trial = trial_encode_plane(&plane.bytes, &menu, &layout)?;
    let total = trial.encoded.state_bytes.len() + trial.encoded.payload.len();
    Ok((trial, total))
}

/// For a single tensor, run forward(chain) + Pass-1 inline trial-encode for
/// every candidate chain. Returns the chosen chain index, the chain's
/// terminal planes, and the per-terminal `TrialResult`s under inline
/// menus only — Pass 2 (shared-state fit) re-trials terminals that may
/// benefit from a `Shared` variant.
struct Pass1Outcome {
    chain_idx: usize,
    terminals: Vec<(Plane, Role)>,
    trials: Vec<TrialResult>,
    /// Forced+chunked skip path only: the per-terminal `(payload, chunk_table)`
    /// chunk-encoded in this same pass, while the freshly-transformed planes are
    /// still hot in cache — avoids storing 3 GB of planes and reading them back
    /// in a separate Pass-4 encode. `None` on the general path.
    chunked_terminals: Option<Vec<(Vec<u8>, Vec<crate::plane_record::ChunkEntry>)>>,
}

/// Placeholder trial for the deferred-encode fast path: records the forced
/// codec id with no payload. The real (chunk-parallel) encode happens in
/// Pass 4. `state_source` is `None` to match the chunked write path, which is
/// self-contained per chunk.
fn placeholder_trial(forced: CodecId) -> TrialResult {
    TrialResult {
        chosen: MenuItem {
            codec_id: forced,
            state_source: StateSource::None,
            shared_state_bytes: None,
            shared_state_id: None,
        },
        encoded: crate::codec::Encoded {
            state_bytes: Vec::new(),
            state_format_version: 0,
            payload: Vec::new(),
        },
    }
}

fn pass1_select_chain(
    candidates: &[Chain],
    raw_bytes: &[u8],
    source_descriptor: &PlaneDescriptor,
    forced_codec: Option<CodecId>,
    allow_codec_ids: Option<&[CodecId]>,
    // When true, skip the whole-plane selection encode and emit placeholder
    // trials — the caller guarantees a single candidate chain, a forced codec,
    // and chunked output, so Pass 4 re-encodes in parallel chunks anyway and
    // the selection encode would be pure (serial) waste. This is the
    // intra-tensor scaling path for large tensors.
    skip_trial_encode: bool,
    // Set together with `skip_trial_encode`: chunk-encode the terminals in this
    // same (cache-hot) pass instead of a separate Pass-4 read-back.
    chunk_size: Option<u32>,
) -> Result<Pass1Outcome, PtwmCoreError> {
    if candidates.is_empty() {
        return Err(PtwmCoreError::InvalidContainer(
            "compress_model: tensor has no candidate chains".into(),
        ));
    }

    let mut best: Option<(usize, Vec<(Plane, Role)>, Vec<TrialResult>, usize)> = None;
    let mut last_err: Option<PtwmCoreError> = None;
    let mut n_failed: usize = 0;

    for (idx, chain) in candidates.iter().enumerate() {
        let ctx = ForwardContext {
            source_bytes: raw_bytes,
            source_descriptor: source_descriptor.clone(),
            deps: &[],
        };
        let fwd = match forward_chain(chain, &ctx) {
            Ok(f) => f,
            Err(e) => {
                tracing::warn!(
                    "pass1_select_chain: candidate chain {idx} forward failed ({e}); \
                     skipping and trying remaining candidates"
                );
                last_err = Some(e);
                n_failed += 1;
                continue;
            }
        };

        let mut total: usize = 0;
        let mut trials: Vec<TrialResult> = Vec::with_capacity(fwd.terminal_planes.len());
        let mut trial_failed = false;
        for (plane, _role) in &fwd.terminal_planes {
            if skip_trial_encode {
                trials.push(placeholder_trial(
                    forced_codec.expect("skip_trial_encode implies a forced codec"),
                ));
                continue;
            }
            match trial_encode_terminal(plane, forced_codec, &[], allow_codec_ids) {
                Ok((trial, payload_size)) => {
                    total = total.saturating_add(payload_size);
                    trials.push(trial);
                }
                Err(e) => {
                    tracing::warn!(
                        "pass1_select_chain: candidate chain {idx} trial-encode failed ({e}); \
                         skipping and trying remaining candidates"
                    );
                    last_err = Some(e);
                    trial_failed = true;
                    break;
                }
            }
        }
        if trial_failed {
            n_failed += 1;
            continue;
        }

        let take = match &best {
            None => true,
            Some((_, _, _, prev_total)) => total < *prev_total,
        };
        if take {
            best = Some((idx, fwd.terminal_planes, trials, total));
        }
    }

    let (chain_idx, terminals, trials, _) = match best {
        Some(b) => b,
        None => {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "compress_model: all {} candidate chains failed (last error: {})",
                n_failed,
                last_err
                    .map(|e| e.to_string())
                    .unwrap_or_else(|| "<none>".into())
            )));
        }
    };
    // Fused encode: chunk-encode the just-produced terminal planes here, while
    // they are still hot in cache, rather than re-reading 3 GB of planes in a
    // separate Pass-4 loop. Only on the skip path (forced codec + single chain
    // + chunked output); the general path leaves this None and Pass 4 encodes.
    let chunked_terminals = if skip_trial_encode {
        let cs = chunk_size.expect("skip_trial_encode implies chunk_size");
        let codec = codec_for(forced_codec.expect("skip_trial_encode implies a forced codec"))
            .ok_or_else(|| {
                PtwmCoreError::InvalidContainer("no codec registered for forced codec".into())
            })?;
        let encoded = terminals
            .iter()
            .map(|(plane, _)| encode_plane_chunked(&plane.bytes, codec.as_ref(), cs))
            .collect::<Result<Vec<_>, _>>()?;
        Some(encoded)
    } else {
        None
    };
    Ok(Pass1Outcome {
        chain_idx,
        terminals,
        trials,
        chunked_terminals,
    })
}

/// Build the deduped chain registry. Returns the registry and a per-tensor
/// vector of registry indices (each in 0..registry.chains.len()).
fn build_chain_registry(chains: &[Chain]) -> (ChainRegistry, Vec<u16>) {
    let mut registry: Vec<(u16, Chain)> = Vec::new();
    let mut tensor_to_idx: Vec<u16> = Vec::with_capacity(chains.len());
    for c in chains {
        let existing = registry.iter().position(|(_, rc)| rc == c);
        match existing {
            Some(pos) => tensor_to_idx.push(pos as u16),
            None => {
                let new_id = registry.len() as u16;
                registry.push((new_id, c.clone()));
                tensor_to_idx.push(new_id);
            }
        }
    }
    (ChainRegistry { chains: registry }, tensor_to_idx)
}

// ---------------------------------------------------------------------------
// compress_model
// ---------------------------------------------------------------------------

/// PPG entry point. Each input tensor carries one or more candidate
/// [`Chain`]s; the compressor selects the chain that produces the smallest
/// total compressed payload (multi-chain trial-encode), then writes a
/// chain-registry-backed `.ptwm` container.
///
/// Returns the per-tensor chosen chain *registry index* (after dedup) for
/// audit. The order matches the input slice.
///
/// **Pass-2 ordering note.** Shared-state fitting is performed *after*
/// Pass-1 multi-chain selection. This means the multi-chain ranker sees only
/// inline trial-encode payloads. For monotone shared-state codecs (PGC,
/// O1SAC), Inline is the lower bound on what the chain can achieve, so the
/// multi-chain ranking remains correct; per-terminal final encoding may
/// later switch to a `Shared` variant if it wins the re-trial.
pub fn compress_model<W: Write + Seek>(
    writer: W,
    inputs: &[InputTensor<'_>],
    opts: CompressorOptions,
) -> Result<Vec<u16>, PtwmCoreError> {
    // ---- Pass 0: structural validation of every candidate chain. ----------
    // Reject malformed chains at the API boundary so downstream stages can
    // assume well-formed graphs (Source at 0, no cycles, all output indices
    // consumed, terminals in bounds). Without this guard a malformed chain
    // blob from PyO3 would reach `forward_chain` directly and surface a far
    // less actionable error.
    for inp in inputs {
        for (idx, chain) in inp.candidate_chains.iter().enumerate() {
            // `n_dependencies` is the number of cross-tensor refs available;
            // candidate chains are single-tensor here (no XorDelta/FloatDelta
            // dep wiring at this entry point), so 0 is correct.
            validate_chain(chain, 0).map_err(|e| {
                PtwmCoreError::InvalidContainer(format!(
                    "compress_model: tensor '{}' candidate chain {idx} is invalid: {e}",
                    inp.name
                ))
            })?;
        }
    }

    // Optional per-pass timing (set PTWM_PROFILE to emit to stderr) — localizes
    // serial-framework overhead versus the parallel passes.
    let profile = std::env::var_os("PTWM_PROFILE").is_some();
    macro_rules! prof {
        ($t:expr, $label:expr) => {
            if profile {
                eprintln!("[ptwm-profile] {}: {:?}", $label, $t.elapsed());
                $t = std::time::Instant::now();
            }
        };
    }
    let mut prof_t = std::time::Instant::now();

    // ---- Pass 1: per-tensor multi-chain trial-encode. ---------------------
    // Per-tensor forward+trial-encode is the bulk of compression and every
    // tensor is independent, so run them across the rayon global pool. For a
    // multi-tensor model this is the primary multi-core scaling lever
    // (near-linear in cores); nested menu/chunk par_iters compose via
    // rayon work-stealing. Order is preserved by par_iter().collect().
    // Largest-first (LPT) scheduling: hand rayon the biggest tensors first so a
    // long-pole tensor is not stranded in the tail while other cores idle.
    // Results are placed back into input order.
    let mut order: Vec<usize> = (0..inputs.len()).collect();
    order.sort_unstable_by_key(|&i| std::cmp::Reverse(inputs[i].raw_bytes.len()));
    let computed: Vec<(usize, Pass1Outcome)> = order
        .par_iter()
        .map(|&i| {
            let inp = &inputs[i];
            let source_descriptor = source_descriptor_for(
                inp.dtype_code,
                inp.raw_bytes.len() as u64,
                inp.shape.as_deref(),
            );
            // Fast path: a forced codec with a single candidate chain and
            // chunked output needs no selection encode — Pass 4 re-encodes in
            // parallel chunks, so defer all encoding (lets a big tensor's
            // entropy coding scale across cores instead of one serial pass).
            let skip_trial_encode = opts.forced_codec.is_some()
                && opts.chunk_size.is_some()
                && inp.candidate_chains.len() == 1;
            pass1_select_chain(
                &inp.candidate_chains,
                inp.raw_bytes,
                &source_descriptor,
                opts.forced_codec,
                opts.allow_codec_ids.as_deref(),
                skip_trial_encode,
                opts.chunk_size,
            )
            .map(|o| (i, o))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut pass1_slots: Vec<Option<Pass1Outcome>> = (0..inputs.len()).map(|_| None).collect();
    for (i, o) in computed {
        pass1_slots[i] = Some(o);
    }
    let mut pass1: Vec<Pass1Outcome> = pass1_slots
        .into_iter()
        .map(|o| o.expect("every tensor index computed in pass1"))
        .collect();

    prof!(prof_t, "pass1_forward+select");

    // ---- Pass 2: shared-state fit over chosen terminals. ------------------
    // Collect references keyed on the typed PlaneDescriptor: PGC fits on
    // Value-role nibble-packed planes; O1SAC fits on Scale (any
    // ScaleFormat) + Byte + Rows.
    let mut value_refs: Vec<&[u8]> = Vec::new();
    let mut scale_refs: Vec<(&[u8], u32)> = Vec::new();
    for outcome in &pass1 {
        for (plane, _role) in &outcome.terminals {
            let d = &plane.descriptor;
            // PGC eligibility: nibble-packed Value plane (PGC's `accepts`).
            if matches!(d.role, Role::Value { .. } | Role::Nibble { .. }) && d.is_nibble_packed {
                value_refs.push(&plane.bytes);
            }
            // O1SAC eligibility: Scale/GlobalScale + Byte + Rows.
            if matches!(d.role, Role::Scale { .. } | Role::GlobalScale { .. })
                && d.element_width == ElementWidth::Byte
                && let Layout::Rows { row_len } = d.layout
            {
                scale_refs.push((&plane.bytes, row_len));
            }
        }
    }

    let shared = fit_shared(&value_refs, &scale_refs)?;

    // Build prelude candidates from shared-state entries (codec-agnostic).
    let prelude_candidates: Vec<PreludeEntry> = shared
        .entries
        .iter()
        .enumerate()
        .map(|(idx, entry)| {
            let state_bytes = entry.state_bytes.clone();
            let state_format_version = if entry.codec_id == CodecId::Order1ScaleAC {
                order1_scale_ac::STATE_FORMAT_VERSION
            } else {
                entry.state_format_version
            };
            PreludeEntry {
                shared_state_id: idx as u16,
                codec_id: entry.codec_id,
                state_format_version,
                applies_to_mask: entry.applies_to_mask,
                state_xxhash64: xxh64(&state_bytes, 0),
                name: entry.name.clone(),
                state_bytes,
            }
        })
        .collect();

    // ---- Pass 3: per-terminal re-trial with shared-state variants. --------
    //
    // For each terminal in the chosen chain, build a final menu via the
    // dispatcher AND the shared-state additions. If the shared variant beats
    // the inline trial, we adopt it. Otherwise we keep the inline trial we
    // already have from Pass 1.
    let mut chosen_per_tensor: Vec<Vec<ChosenTerminal>> = Vec::with_capacity(inputs.len());
    let mut used_shared_ids: std::collections::HashSet<u16> = std::collections::HashSet::new();

    for outcome in &pass1 {
        let mut chosen_planes: Vec<ChosenTerminal> = Vec::with_capacity(outcome.terminals.len());
        for ((plane, role), inline_trial) in outcome.terminals.iter().zip(outcome.trials.iter()) {
            let layout = legacy_plane_layout(plane.descriptor.layout);

            let final_trial: TrialResult = if opts.forced_codec.is_some() {
                // Forced: keep Pass-1 result; no shared variants apply.
                clone_trial(inline_trial)
            } else if prelude_candidates.is_empty() {
                clone_trial(inline_trial)
            } else {
                // Re-trial with shared variants in the menu.
                let mut full_menu = menu_for_descriptor(&plane.descriptor);
                append_shared_menu_items(&mut full_menu, &plane.descriptor, &prelude_candidates);
                filter_menu_by_allow_list(
                    &mut full_menu,
                    opts.allow_codec_ids.as_deref(),
                    &plane.descriptor,
                )?;
                prune_menu_by_should_attempt(
                    &mut full_menu,
                    &plane.bytes,
                    &plane.descriptor,
                    &layout,
                );
                trial_encode_plane(&plane.bytes, &full_menu, &layout)?
            };

            if final_trial.chosen.state_source == StateSource::Shared
                && let Some(sid) = final_trial.chosen.shared_state_id
            {
                used_shared_ids.insert(sid);
            }

            chosen_planes.push(ChosenTerminal {
                role: legacy_plane_role(role),
                codec_id: final_trial.chosen.codec_id,
                state_source: final_trial.chosen.state_source,
                shared_state_id: final_trial.chosen.shared_state_id,
                encoded_state_bytes: final_trial.encoded.state_bytes,
                encoded_state_version: final_trial.encoded.state_format_version,
                payload_bytes: final_trial.encoded.payload,
                raw_plane_for_crc: plane.bytes.clone(),
                layout,
            });
        }
        chosen_per_tensor.push(chosen_planes);
    }

    prof!(prof_t, "pass2+3_fit+retrial");

    // Drop unreferenced prelude entries.
    let prelude_entries: Vec<PreludeEntry> = prelude_candidates
        .into_iter()
        .filter(|e| used_shared_ids.contains(&e.shared_state_id))
        .collect();

    // ---- Pass 4: write container. -----------------------------------------
    // Build a deduped chain registry from the chosen chain per tensor.
    let chosen_chains: Vec<Chain> = pass1
        .iter()
        .zip(inputs.iter())
        .map(|(outcome, inp)| inp.candidate_chains[outcome.chain_idx].clone())
        .collect();
    let (registry, tensor_chain_indices) = build_chain_registry(&chosen_chains);

    let mut writer = ContainerWriter::new(writer)?;
    writer.set_prelude(&prelude_entries)?;
    writer.set_chain_registry(registry)?;

    // Pre-compute the chunked-path payloads in parallel before the sequential
    // write loop. Otherwise encode_plane_chunked runs inside the writer's
    // per-tensor loop — serial across tensors — so a forced+chunked encode
    // keeps only intra-chunk parallelism and loses the inter-tensor scaling.
    // par_iter across tensors composes with par_chunks within each plane, so
    // every chunk of every tensor is a stealable work item and all cores stay
    // busy regardless of tensor-size skew. The writer then just moves the
    // finished payloads into records. Empty when not chunking.
    let mut chunked_payloads: Vec<Vec<(Vec<u8>, Vec<crate::plane_record::ChunkEntry>)>> =
        if let Some(chunk_size) = opts.chunk_size {
            // Take the fused (cache-hot) payloads encoded during Pass 1 on the
            // skip path; fall back to encoding here on the general chunked path.
            let fused: Vec<Option<Vec<(Vec<u8>, Vec<crate::plane_record::ChunkEntry>)>>> = pass1
                .iter_mut()
                .map(|o| o.chunked_terminals.take())
                .collect();
            fused
                .into_par_iter()
                .zip(chosen_per_tensor.par_iter())
                .map(|(pre, chosen_planes)| match pre {
                    Some(p) => Ok(p),
                    None => chosen_planes
                        .iter()
                        .map(|cp| {
                            let codec = crate::codec::codec_for(cp.codec_id).ok_or_else(|| {
                                PtwmCoreError::InvalidContainer(format!(
                                    "no codec registered for {:?}",
                                    cp.codec_id
                                ))
                            })?;
                            encode_plane_chunked(&cp.raw_plane_for_crc, codec.as_ref(), chunk_size)
                        })
                        .collect::<Result<Vec<_>, _>>(),
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            Vec::new()
        };

    prof!(prof_t, "pass4_chunked_encode");

    let mut tensor_records: Vec<TensorRecord> = Vec::with_capacity(inputs.len());
    for (idx, inp) in inputs.iter().enumerate() {
        let chosen_planes = &chosen_per_tensor[idx];
        let mut plane_records: Vec<PlaneRecord> = Vec::with_capacity(chosen_planes.len());
        for (plane_idx, cp) in chosen_planes.iter().enumerate() {
            let (
                payload_bytes,
                chunk_table,
                state_source,
                inline_state_bytes,
                state_version,
                state_info,
            ) = if opts.chunk_size.is_some() {
                // Take the payload pre-computed in parallel above.
                let (payload_bytes, entries) =
                    std::mem::take(&mut chunked_payloads[idx][plane_idx]);
                (
                    payload_bytes,
                    Some(entries),
                    StateSource::None,
                    Vec::new(),
                    0u8,
                    0u16,
                )
            } else {
                (
                    cp.payload_bytes.clone(),
                    None,
                    cp.state_source,
                    cp.encoded_state_bytes.clone(),
                    cp.encoded_state_version,
                    cp.shared_state_id.unwrap_or(0),
                )
            };

            let stored_layout = if chunk_table.is_some() {
                PlaneLayout::Flat
            } else {
                cp.layout
            };
            let canonical_id = codec_canonical_id(cp.codec_id);
            let codec_table_idx = writer
                .codec_table_idx_for(&canonical_id)
                .expect("codec must be in extension table");
            plane_records.push(PlaneRecord {
                role: cp.role,
                codec_id: cp.codec_id,
                codec_table_idx,
                state_source,
                state_version,
                state_info,
                payload_len: payload_bytes.len() as u64,
                crc32: if opts.emit_plane_crc {
                    Some(crc32fast::hash(&cp.raw_plane_for_crc))
                } else {
                    None
                },
                chunk_table,
                external_state: None,
                inline_state_bytes,
                payload_bytes,
                layout: stored_layout,
            });
        }

        let mut flags: u8 = 0;
        let payload_hash = if opts.emit_payload_hash {
            flags |= TENSOR_FLAG_PAYLOAD_HASH;
            Some(xxh64(inp.raw_bytes, 0))
        } else {
            None
        };

        let tensor_metadata = match (&inp.shape, &inp.dtype_name) {
            (Some(shape), Some(name)) => Some(encode_shape_metadata(shape, name)),
            _ => None,
        };

        let dependencies: Vec<Dependency> = match inp.delta_reference_blake3 {
            Some(hash) => vec![Dependency {
                ref_kind: RefKind::ExternalSafetensors,
                expected_hash: Some(hash),
                ref_bytes: Vec::new(),
            }],
            None => Vec::new(),
        };

        let chain_ref = tensor_chain_indices[idx];
        let rec = TensorRecord {
            chain_ref, // high bit clear → registry index
            dtype_code: inp.dtype_code,
            input_format: inp.input_format,
            flags,
            orig_size: inp.raw_bytes.len() as u64,
            name: inp.name.clone(),
            payload_hash,
            inline_chain: None,
            dependencies,
            tensor_metadata,
            terminals: plane_records,
        };
        tensor_records.push(rec);
    }

    // Serialize tensor records (the payload-copy-heavy step) in parallel —
    // write_tensor_record is pure per record — then append the finished byte
    // blocks sequentially, which is all the ordered writer needs to do.
    let serialized: Vec<Vec<u8>> = tensor_records
        .par_iter()
        .map(|rec| {
            // Pre-size to the plane payloads (the dominant term) plus headroom
            // for record/plane metadata, so the large payload copies don't
            // repeatedly reallocate.
            let est: usize = rec.terminals.iter().map(|t| t.payload_bytes.len()).sum();
            let mut buf = Vec::with_capacity(est + 1024);
            crate::tensor_record::write_tensor_record(rec, &mut buf)?;
            Ok::<Vec<u8>, PtwmCoreError>(buf)
        })
        .collect::<Result<Vec<_>, _>>()?;
    for (rec, buf) in tensor_records.iter().zip(serialized.iter()) {
        writer.append_serialized_tensor(&rec.name, buf)?;
    }

    prof!(prof_t, "pass4_write_loop");
    writer.finalize()?;
    if profile {
        eprintln!("[ptwm-profile] finalize: {:?}", prof_t.elapsed());
    }
    Ok(tensor_chain_indices)
}

/// `TrialResult` does not derive `Clone` upstream; provide a shallow clone
/// helper local to this module so we don't have to widen the public API.
fn clone_trial(t: &TrialResult) -> TrialResult {
    TrialResult {
        chosen: t.chosen.clone(),
        encoded: crate::codec::Encoded {
            state_bytes: t.encoded.state_bytes.clone(),
            state_format_version: t.encoded.state_format_version,
            payload: t.encoded.payload.clone(),
        },
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::graph::{ChainEdge, ChainNode, TerminalRef};
    use crate::container::ContainerReader;
    use crate::transforms::op::OpId;
    use crate::types::role::ScaleFormat;
    use std::io::Cursor;

    #[test]
    fn allow_list_additively_includes_opt_in_codec() {
        // Default dispatch menu for a Raw plane never contains CM.
        let descriptor = PlaneDescriptor {
            role: Role::Raw,
            element_width: ElementWidth::Byte,
            length_bytes: 4096,
            layout: Layout::Flat,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        };
        let mut menu = menu_for_descriptor(&descriptor);
        assert!(
            !menu
                .iter()
                .any(|m| m.codec_id == CodecId::ContextMixingLite)
        );

        // Allow-listing CM adds it (it accepts Raw), keeps the named subset.
        let allow = [CodecId::Identity, CodecId::ContextMixingLite];
        filter_menu_by_allow_list(&mut menu, Some(&allow), &descriptor).unwrap();
        assert!(
            menu.iter()
                .any(|m| m.codec_id == CodecId::ContextMixingLite)
        );
        assert!(menu.iter().all(|m| allow.contains(&m.codec_id)));
        // The added item uses the same state-source convention as the default
        // menu (Inline for non-Identity), so `prune_menu_by_should_attempt`
        // consults its `should_attempt` gate.
        let cm = menu
            .iter()
            .find(|m| m.codec_id == CodecId::ContextMixingLite)
            .unwrap();
        assert_eq!(cm.state_source, StateSource::Inline);
    }

    // ── Fixtures ────────────────────────────────────────────────────────────

    fn source_params_for(n_elements: u32, dtype_code: u16) -> Vec<u8> {
        // 1-D shape encoding (matches `Source::write_params`).
        let mut p = vec![1u8];
        p.extend_from_slice(&n_elements.to_le_bytes());
        p.extend_from_slice(&dtype_code.to_le_bytes());
        p
    }

    fn fp16_split_chain(n_elements: u32) -> Chain {
        // Source(fp16) → BitReorderIeee16 → ByteSplit{n=2} → 2× IntegerByte terminals
        Chain {
            nodes: vec![
                ChainNode {
                    op: OpId::Source,
                    params: source_params_for(n_elements, 0x0002), // Float16 (chain-internal)
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

    fn passthrough_value_chain(n_bytes: u32) -> Chain {
        // Source(int8) → BytePassthrough → Terminal{Value(IntN bits=8)}
        Chain {
            nodes: vec![
                ChainNode {
                    op: OpId::Source,
                    params: source_params_for(n_bytes, 17),
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
                    format: crate::types::role::ValueFormat::IntN { bits: 8 },
                }),
                vendor_bytes: vec![],
            }],
            terminals: vec![TerminalRef {
                node_idx: 1,
                output_idx: 0,
                role: Role::Value {
                    format: crate::types::role::ValueFormat::IntN { bits: 8 },
                },
            }],
        }
    }

    fn passthrough_scale_chain(n_bytes: u32, row_len: u32) -> Chain {
        // Source(int8) → BytePassthrough → Terminal{Scale(E8M0)} (Rows layout
        // injected via role_override; descriptor layout follows source).
        // We override with Scale role so the dispatcher routes via O1SAC menu
        // *if* the descriptor is Rows + Byte; without explicit Rows layout in
        // the source descriptor, this terminal stays in Flat layout. The
        // helper exists primarily for descriptor-shaping tests.
        let _ = row_len;
        Chain {
            nodes: vec![
                ChainNode {
                    op: OpId::Source,
                    params: source_params_for(n_bytes, 17),
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
                role_override: Some(Role::Scale {
                    format: ScaleFormat::E4M3,
                }),
                vendor_bytes: vec![],
            }],
            terminals: vec![TerminalRef {
                node_idx: 1,
                output_idx: 0,
                role: Role::Scale {
                    format: ScaleFormat::E4M3,
                },
            }],
        }
    }

    // ── single chain roundtrip ────────────────────────────────────────────────────

    #[test]
    fn compress_single_chain_roundtrip() {
        // Build a synthetic FP16 tensor with mild structure (mostly small
        // exponents → byte-split exponent plane should compress).
        let n_elements: u32 = 64;
        let mut raw: Vec<u8> = Vec::with_capacity((n_elements * 2) as usize);
        for i in 0..n_elements {
            // Crude fp16 values: sign 0, exponent biased ~14, mantissa varies.
            let bits: u16 = 0x3800 | (i as u16 & 0x03FF);
            raw.extend_from_slice(&bits.to_le_bytes());
        }

        let chain = fp16_split_chain(n_elements);
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
        let chain_indices = compress_model(
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

        assert_eq!(chain_indices, vec![0]);

        // Inspect the container at the structural level. We verify:
        //   - Container opens successfully.
        //   - Chain registry has exactly one entry (matches the input).
        //   - The tensor record's chain_ref is 0 (registry index, no inline).
        //   - The tensor has the expected number of terminals (2 for byte-split).
        //   - Terminal CRCs validate when their planes are decoded inline.
        let reader = ContainerReader::open(&buf).expect("open succeeds");
        assert_eq!(reader.chain_registry.chains.len(), 1);
        assert_eq!(reader.chain_registry.chains[0].0, 0);

        // Structural decode: parse the tensor record bytes and verify
        // chain_ref + terminal count.
        use crate::tensor_record::parse_tensor_record;
        let rec_bytes = reader
            .get_tensor_record_bytes("fp16.weight")
            .expect("tensor present");
        let (rec, _) = parse_tensor_record(rec_bytes).expect("parse tensor record");
        assert_eq!(rec.chain_ref, 0);
        assert!(rec.inline_chain.is_none());
        assert_eq!(rec.terminals.len(), 2);
        assert_eq!(rec.orig_size, raw.len() as u64);

        // Per-terminal: decode inline-state codec, verify CRC matches the
        // raw plane bytes from a fresh forward_chain run.
        use crate::codec::codec_for;
        let source_descriptor = source_descriptor_for(0x0002, raw.len() as u64, None);
        let fwd = forward_chain(
            &inputs[0].candidate_chains[0],
            &ForwardContext {
                source_bytes: &raw,
                source_descriptor,
                deps: &[],
            },
        )
        .unwrap();
        for (i, plane_rec) in rec.terminals.iter().enumerate() {
            let expected_crc = plane_rec.crc32.expect("CRC present");
            let computed_crc = crc32fast::hash(&fwd.terminal_planes[i].0.bytes);
            assert_eq!(
                computed_crc, expected_crc,
                "terminal {} CRC must match recomputed forward plane",
                i
            );
            // Also verify decoder roundtrip on the terminal's payload.
            let codec = codec_for(plane_rec.codec_id).unwrap();
            let decoded = codec
                .decode(
                    plane_rec.state_version,
                    &plane_rec.inline_state_bytes,
                    &plane_rec.payload_bytes,
                    &plane_rec.layout,
                    fwd.terminal_planes[i].0.bytes.len(),
                )
                .unwrap();
            assert_eq!(decoded, fwd.terminal_planes[i].0.bytes.as_ref());
        }
    }

    #[test]
    fn compress_forced_chunked_skip_roundtrip() {
        // Exercises the deferred-encode fast path: a forced codec + single
        // candidate chain + chunk_size makes Pass 1 emit placeholder trials and
        // Pass 4 do the (parallel) chunked encode. Use enough elements that
        // each terminal plane spans several chunks, then assert a bit-exact
        // full decode roundtrip.
        let n_elements: u32 = 4096;
        let mut raw: Vec<u8> = Vec::with_capacity((n_elements * 2) as usize);
        for i in 0..n_elements {
            let bits: u16 = 0x3800 | (i as u16 & 0x03FF);
            raw.extend_from_slice(&bits.to_le_bytes());
        }

        let chain = fp16_split_chain(n_elements);
        let inputs = vec![InputTensor {
            name: "fp16.weight".into(),
            candidate_chains: vec![chain],
            dtype_code: 0x0002,
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
                method_hint: 4,
                emit_payload_hash: true,
                emit_plane_crc: true,
                forced_codec: Some(CodecId::Rans),
                allow_codec_ids: None,
                chunk_size: Some(512), // small → multiple chunks per plane
            },
        )
        .expect("compress succeeds");

        let reader = ContainerReader::open(&buf).expect("open succeeds");
        let got = reader.decode_tensor("fp16.weight").expect("decode");
        assert_eq!(
            got, raw,
            "forced+chunked skip-path roundtrip must be bit-exact"
        );
    }

    // ── multi-chain trial-encode picks smaller ────────────────────────────────────

    #[test]
    fn compress_multi_chain_picks_smaller() {
        // Build a 64-byte int8 tensor whose bytes are highly skewed (mostly
        // zeros) so Huffman compresses dramatically. We give two candidate
        // chains:
        //   (a) Source → BytePassthrough → Terminal[Value]   (full menu)
        //   (b) Source → BytePassthrough → Terminal[Value]   (forced Identity)
        //
        // We force-disable trial-encode for chain (b) by giving it a
        // descriptor that the dispatcher only routes to Identity. Easier
        // path: build chain (b) so its terminal plane is *larger* — e.g.,
        // duplicate-encode via two terminals where one is the byte-split
        // half. We instead use the simpler approach: chain (a) leaves bytes
        // as-is (one terminal); chain (b) uses ByteSplit n=2 (two terminals,
        // each half — both highly skewed and Huffman-compressible).
        //
        // Both are valid; the compressor picks the smaller-output chain.
        // Because the trial-encode picks Huffman for both, the chain with
        // smaller per-terminal Huffman headers wins. We verify the audit
        // index identifies *some* chain — the key invariant is that the
        // ranker runs and picks one deterministically.
        let raw: Vec<u8> = (0..64u32)
            .map(|i| if i % 8 == 0 { 0xFFu8 } else { 0u8 })
            .collect();

        let chain_a = passthrough_value_chain(raw.len() as u32);
        let chain_b = Chain {
            nodes: vec![
                ChainNode {
                    op: OpId::Source,
                    params: source_params_for(raw.len() as u32, 17),
                },
                ChainNode {
                    op: OpId::ByteSplit,
                    params: vec![2u8],
                },
            ],
            edges: vec![ChainEdge {
                src_node: 0,
                src_output_idx: 0,
                dst_node: 1,
                dst_input_idx: 0,
                role_override: None,
                vendor_bytes: vec![],
            }],
            terminals: vec![
                TerminalRef {
                    node_idx: 1,
                    output_idx: 0,
                    role: Role::IntegerByte { index: 0, of: 2 },
                },
                TerminalRef {
                    node_idx: 1,
                    output_idx: 1,
                    role: Role::IntegerByte { index: 1, of: 2 },
                },
            ],
        };

        let inputs = vec![InputTensor {
            name: "int8.t".into(),
            candidate_chains: vec![chain_a.clone(), chain_b.clone()],
            dtype_code: 17,
            input_format: 0,
            raw_bytes: &raw,
            shape: None,
            dtype_name: None,
            delta_reference_blake3: None,
        }];

        // Independently verify which chain has smaller compressed total.
        let source_descriptor = source_descriptor_for(17, raw.len() as u64, None);

        let trial_total = |chain: &Chain| -> usize {
            let fwd = forward_chain(
                chain,
                &ForwardContext {
                    source_bytes: &raw,
                    source_descriptor: source_descriptor.clone(),
                    deps: &[],
                },
            )
            .unwrap();
            let mut total = 0usize;
            for (plane, _) in &fwd.terminal_planes {
                let (_, sz) = trial_encode_terminal(plane, None, &[], None).unwrap();
                total += sz;
            }
            total
        };
        let total_a = trial_total(&chain_a);
        let total_b = trial_total(&chain_b);
        let expected_winner = if total_a <= total_b { 0u16 } else { 1u16 };

        let mut buf: Vec<u8> = Vec::new();
        let chain_indices = compress_model(
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
        .unwrap();

        // The audit index points into the *registry* (after dedup). With two
        // distinct chains the registry index equals the chosen candidate
        // chain index in the original input.
        assert_eq!(chain_indices.len(), 1);
        assert_eq!(chain_indices[0], expected_winner);
    }

    // ── Pass 2 shared-state O1SAC fitting ─────────────────────────────────────────

    #[test]
    fn compress_pass2_shared_o1sac() {
        // Two scale tensors, each with Rows layout + Byte width + Scale role.
        // Pass-2 fit_shared should produce ≥1 Order1ScaleAC entry that lands
        // in the prelude AND is used by ≥1 terminal (i.e., the Shared variant
        // wins the re-trial). Use ≥4096 bytes so the inline-state overhead is
        // comfortably amortised.
        let row_len: u32 = 32;
        let n_rows: u32 = 256;
        let n_bytes = (row_len * n_rows) as usize; // 8192
        // Build a strongly auto-correlated byte stream so O1SAC beats Identity/Huffman.
        let mut raw: Vec<u8> = Vec::with_capacity(n_bytes);
        let mut rng_state: u64 = 0xCAFEF00D;
        for _ in 0..n_rows {
            let mut prev: u8 = 0x80;
            for _ in 0..row_len {
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let delta = ((rng_state >> 32) as u8) & 0x07;
                let next = prev.wrapping_add(delta).wrapping_sub(3);
                raw.push(next);
                prev = next;
            }
        }

        // Build a chain that hands a single terminal with descriptor:
        // role=Scale, width=Byte, layout=Rows{row_len} so the dispatcher
        // routes to O1SAC and Pass-2 fit picks it up.
        // The Source descriptor needs Rows layout for the terminal to inherit
        // it via BytePassthrough's propagate_descriptors. We'll rely on the
        // descriptor we feed via ForwardContext.source_descriptor (this is
        // what compressor builds from `shape`).
        // Provide shape = [n_rows, row_len] so source_descriptor_for returns
        // Rows{row_len = row_len}.
        //
        // After BytePassthrough + role_override Scale, the terminal carries
        // role=Scale + width=Byte (from source) + layout=Rows{row_len}.

        let chain = passthrough_scale_chain(n_bytes as u32, row_len);
        let inputs = vec![
            InputTensor {
                name: "scale.a".into(),
                candidate_chains: vec![chain.clone()],
                dtype_code: 17, // int8 → ElementWidth::Byte
                input_format: 0,
                raw_bytes: &raw,
                shape: Some(vec![n_rows as u64, row_len as u64]),
                dtype_name: None,
                delta_reference_blake3: None,
            },
            InputTensor {
                name: "scale.b".into(),
                candidate_chains: vec![chain.clone()],
                dtype_code: 17,
                input_format: 0,
                raw_bytes: &raw,
                shape: Some(vec![n_rows as u64, row_len as u64]),
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
                emit_plane_crc: false,
                forced_codec: None,
                allow_codec_ids: None,
                chunk_size: None,
            },
        )
        .unwrap();

        let reader = ContainerReader::open(&buf).expect("open succeeds");
        // The prelude must contain at least one O1SAC entry — Pass 2 fit
        // should have produced one for the matching Scale+Byte+Rows planes.
        let has_o1sac = reader
            .prelude
            .iter()
            .any(|e| e.codec_id == CodecId::Order1ScaleAC);
        assert!(
            has_o1sac,
            "Pass-2 must produce an Order1ScaleAC prelude entry for matching scale planes; \
             got {} entries: {:?}",
            reader.prelude.len(),
            reader
                .prelude
                .iter()
                .map(|e| e.codec_id)
                .collect::<Vec<_>>()
        );

        // At least one terminal must reference the shared state.
        use crate::tensor_record::parse_tensor_record;
        let mut shared_ref_count = 0usize;
        for name in ["scale.a", "scale.b"] {
            let bytes = reader.get_tensor_record_bytes(name).unwrap();
            let (rec, _) = parse_tensor_record(bytes).unwrap();
            for t in &rec.terminals {
                if t.state_source == StateSource::Shared {
                    shared_ref_count += 1;
                }
            }
        }
        assert!(
            shared_ref_count > 0,
            "expected ≥1 terminal to reference a shared state"
        );
    }

    // ── chain registry dedup ──────────────────────────────────────────────────────

    #[test]
    fn compress_chain_registry_dedup() {
        let raw: Vec<u8> = (0..32u32).map(|i| i as u8).collect();
        let chain = passthrough_value_chain(raw.len() as u32);

        let inputs = vec![
            InputTensor {
                name: "t0".into(),
                candidate_chains: vec![chain.clone()],
                dtype_code: 17,
                input_format: 0,
                raw_bytes: &raw,
                shape: None,
                dtype_name: None,
                delta_reference_blake3: None,
            },
            InputTensor {
                name: "t1".into(),
                candidate_chains: vec![chain.clone()],
                dtype_code: 17,
                input_format: 0,
                raw_bytes: &raw,
                shape: None,
                dtype_name: None,
                delta_reference_blake3: None,
            },
        ];

        let mut buf: Vec<u8> = Vec::new();
        let chain_indices = compress_model(
            Cursor::new(&mut buf),
            &inputs,
            CompressorOptions {
                method_hint: 1,
                emit_payload_hash: false,
                emit_plane_crc: false,
                forced_codec: None,
                allow_codec_ids: None,
                chunk_size: None,
            },
        )
        .unwrap();

        assert_eq!(chain_indices, vec![0, 0]);

        let reader = ContainerReader::open(&buf).expect("open succeeds");
        assert_eq!(
            reader.chain_registry.chains.len(),
            1,
            "registry must dedupe identical chains"
        );

        use crate::tensor_record::parse_tensor_record;
        for name in ["t0", "t1"] {
            let bytes = reader.get_tensor_record_bytes(name).unwrap();
            let (rec, _) = parse_tensor_record(bytes).unwrap();
            assert_eq!(rec.chain_ref, 0);
            assert!(rec.inline_chain.is_none());
        }
    }
}

#[cfg(test)]
mod source_descriptor_agreement_tests {
    use super::*;
    use crate::transforms::op::Op;
    use crate::transforms::source::Source;

    /// The compress side builds the source descriptor from the raw byte
    /// count; the decompress side rebuilds it from the chain's Source node
    /// (shape + dtype code). The two must agree on `length_bytes` for every
    /// dtype, or a plane decodes at the wrong length.
    fn assert_paths_agree(dtype_code: u16, shape: &[u64], raw_len: u64) {
        let from_compressor = source_descriptor_for(dtype_code, raw_len, Some(shape));
        let src = Source {
            shape: shape.iter().map(|&d| d as u32).collect(),
            dtype_code,
        };
        let from_chain = &src.propagate_descriptors(&[]).unwrap()[0];
        assert_eq!(
            from_compressor.length_bytes, from_chain.length_bytes,
            "dtype 0x{dtype_code:04X}: compress path says {} bytes, decompress path says {}",
            from_compressor.length_bytes, from_chain.length_bytes
        );
        assert_eq!(from_compressor.element_width, from_chain.element_width);
        assert_eq!(
            from_compressor.is_nibble_packed,
            from_chain.is_nibble_packed
        );
    }

    #[test]
    fn packed_fp4_source_descriptor_agrees_across_both_paths() {
        // A packed-FP4 tensor's shape counts packed bytes (one byte per
        // element, two fp4 values), matching `Dtype::element_size` and the
        // shape a torch `float4_e2m1fn_x2` tensor reports.
        assert_paths_agree(0x001F, &[128, 128], 128 * 128);
    }

    #[test]
    fn byte_and_word_source_descriptors_agree_across_both_paths() {
        assert_paths_agree(0x0006, &[10, 10], 100);
        assert_paths_agree(0x0002, &[10, 10], 200);
        assert_paths_agree(0x0003, &[10, 10], 400);
    }
}
