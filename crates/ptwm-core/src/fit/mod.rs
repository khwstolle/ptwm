//! Shared codec state fitters. Each codec contributes its own fitter
//! module; the top-level `fit_shared` aggregates their outputs into a
//! flat `SharedStates::entries` list (codec-agnostic prelude allocation).

mod order1;
mod per_group_codebook;

use crate::codec::CodecId;
use crate::error::PtwmCoreError;

/// One entry destined for the container prelude. Codec-agnostic — the
/// compressor allocates `shared_state_id` from this entry's index in
/// `SharedStates::entries`.
///
/// Construct via [`SharedStateEntry::new`] so `state_bytes` are validated
/// against the codec's parser before they reach the wire. The fields are
/// `pub` for serialization convenience but the constructor is the single
/// supported entry point.
#[derive(Debug, Clone)]
pub struct SharedStateEntry {
    pub codec_id: CodecId,
    pub state_format_version: u8,
    pub applies_to_mask: u8,
    pub name: String,
    pub state_bytes: Vec<u8>,
}

impl SharedStateEntry {
    /// Validate `state_bytes` against the codec's parser, then construct.
    /// Catches in-process desync (wrong byte layout per `codec_id`) before
    /// the entry reaches the prelude writer; on-disk validation runs again
    /// at decode via the same parser.
    pub fn new(
        codec_id: CodecId,
        state_format_version: u8,
        applies_to_mask: u8,
        name: String,
        state_bytes: Vec<u8>,
    ) -> Result<Self, PtwmCoreError> {
        validate_shared_state_bytes(codec_id, state_format_version, &state_bytes)?;
        Ok(Self {
            codec_id,
            state_format_version,
            applies_to_mask,
            name,
            state_bytes,
        })
    }
}

/// Per-codec validator dispatch. Each codec verifies its own state byte
/// layout — adding a new codec to the shared-state path requires adding
/// an arm here so cross-codec contamination fails loudly at construction
/// rather than silently at decode.
fn validate_shared_state_bytes(
    codec_id: CodecId,
    state_format_version: u8,
    state_bytes: &[u8],
) -> Result<(), PtwmCoreError> {
    match codec_id {
        CodecId::PerGroupCodebook => crate::codecs::per_group_codebook::validate_state_bytes(
            state_format_version,
            state_bytes,
        ),
        CodecId::Order1ScaleAC => {
            if state_format_version != crate::codecs::order1_scale_ac::STATE_FORMAT_VERSION {
                return Err(PtwmCoreError::CodecDecode {
                    codec: "Order1ScaleAC",
                    msg: format!("unknown state_format_version {state_format_version}"),
                });
            }
            crate::codecs::order1_scale_ac::deserialize_state(state_bytes).map(|_| ())
        }
        other => Err(PtwmCoreError::InvalidContainer(format!(
            "codec {other:?} not registered for shared state"
        ))),
    }
}

#[derive(Debug, Default)]
pub struct SharedStates {
    pub entries: Vec<SharedStateEntry>,
}

/// Fit shared codec states across all matching planes. PGC fits on
/// `planes_value`; Order1ScaleAC fits on `planes_scale`. Returns a flat
/// `Vec<SharedStateEntry>` — the compressor allocates `shared_state_id`
/// sequentially from the entry index.
pub fn fit_shared(
    planes_value: &[&[u8]],
    planes_scale: &[(&[u8], u32)],
) -> Result<SharedStates, PtwmCoreError> {
    let mut entries = Vec::new();
    if let Some(entry) = per_group_codebook::fit(planes_value)? {
        entries.push(entry);
    }
    let order1_entries = order1::fit(planes_scale)?;
    entries.extend(order1_entries);
    Ok(SharedStates { entries })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_shared_with_pgc_only() {
        let plane: Vec<u8> = (0..1024u32).map(|i| (i % 16) as u8).collect();
        let s = fit_shared(&[&plane], &[]).unwrap();
        assert_eq!(s.entries.len(), 1);
        assert_eq!(s.entries[0].codec_id, CodecId::PerGroupCodebook);
        assert_eq!(s.entries[0].name, "per_group_codebook");
    }

    #[test]
    fn fit_shared_empty_inputs_returns_no_entries() {
        let s = fit_shared(&[], &[]).unwrap();
        assert!(s.entries.is_empty());
    }

    #[test]
    fn fit_shared_returns_pgc_then_order1_entries() {
        let value_plane: Vec<u8> = (0..1024u32).map(|i| (i % 16) as u8).collect();
        let scale_plane_a: Vec<u8> = (0..256u32).map(|i| (i % 32) as u8).collect();
        let scale_plane_b: Vec<u8> = (0..256u32).map(|i| (200 + i % 32) as u8).collect();
        let s = fit_shared(
            &[&value_plane],
            &[(&scale_plane_a, 16), (&scale_plane_b, 16)],
        )
        .unwrap();
        assert!(
            s.entries
                .iter()
                .any(|e| e.codec_id == CodecId::PerGroupCodebook)
        );
        assert!(
            s.entries
                .iter()
                .any(|e| e.codec_id == CodecId::Order1ScaleAC)
        );
    }

    #[test]
    fn shared_state_entry_new_rejects_wrong_codec_bytes() {
        // PGC state must be exactly K * ALPHABET bytes; passing
        // O1SAC-shaped bytes (much larger, different sums) under the PGC
        // codec id must fail at construction, not at decode.
        let bogus = vec![0u8; 32 + 512]; // shaped like an empty O1SAC state
        let err = SharedStateEntry::new(
            CodecId::PerGroupCodebook,
            0,
            0b0001,
            "bogus".to_string(),
            bogus,
        )
        .unwrap_err();
        assert!(format!("{err}").contains("PerGroupCodebook"));
    }

    #[test]
    fn shared_state_entry_new_rejects_unknown_o1sac_version() {
        // Even an empty/zero state slips past the byte parser unless the
        // version mismatch is caught first.
        let err = SharedStateEntry::new(
            CodecId::Order1ScaleAC,
            99,
            0b0010,
            "bad-version".to_string(),
            vec![0u8; 32 + 512],
        )
        .unwrap_err();
        assert!(format!("{err}").contains("state_format_version"));
    }

    #[test]
    fn fit_shared_no_scale_planes_returns_only_pgc() {
        let value_plane: Vec<u8> = (0..1024u32).map(|i| (i % 16) as u8).collect();
        let s = fit_shared(&[&value_plane], &[]).unwrap();
        assert_eq!(s.entries.len(), 1);
        assert_eq!(s.entries[0].codec_id, CodecId::PerGroupCodebook);
    }
}
