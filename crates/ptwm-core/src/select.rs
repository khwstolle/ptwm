//! Per-plane trial-encode: try every menu item in parallel, pick smallest.

use rayon::prelude::*;

use crate::codec::{CodecId, Encoded, StateSource, codec_for};
use crate::error::PtwmCoreError;
use crate::layout::PlaneLayout;

#[derive(Debug, Clone)]
pub struct MenuItem {
    pub codec_id: CodecId,
    pub state_source: StateSource,
    /// For `Shared` state: the bytes of the shared state to pass in.
    /// For others: `None`.
    pub shared_state_bytes: Option<Vec<u8>>,
    /// For `Shared` state: the shared_state_id the writer will emit in
    /// the plane descriptor. For others: `None`.
    pub shared_state_id: Option<u16>,
}

#[derive(Debug)]
pub struct TrialResult {
    pub chosen: MenuItem,
    pub encoded: Encoded,
}

pub fn trial_encode_plane(
    plane: &[u8],
    menu: &[MenuItem],
    layout: &PlaneLayout,
) -> Result<TrialResult, PtwmCoreError> {
    if menu.is_empty() {
        return Err(PtwmCoreError::InvalidContainer("empty codec menu".into()));
    }

    // Encode every menu item in parallel; collect successful (item, encoded)
    // pairs along with each one's total byte count for the min comparison.
    // Errors from any codec encode are propagated via the Result.
    let trials: Vec<(MenuItem, Encoded, usize)> = menu
        .par_iter()
        .map(
            |item| -> Result<(MenuItem, Encoded, usize), PtwmCoreError> {
                let codec = codec_for(item.codec_id).ok_or_else(|| {
                    PtwmCoreError::InvalidContainer(format!(
                        "no codec registered for {:?}",
                        item.codec_id
                    ))
                })?;
                let encoded = codec.encode(plane, item.shared_state_bytes.as_deref(), layout)?;
                let total_size = encoded.state_bytes.len() + encoded.payload.len();
                Ok((item.clone(), encoded, total_size))
            },
        )
        .collect::<Result<Vec<_>, _>>()?;

    // Pick the smallest. Stable tie-break: the menu's iteration order
    // (.min_by_key keeps the first occurrence on equal keys).
    let (chosen, encoded, _) = trials
        .into_iter()
        .min_by_key(|(_, _, size)| *size)
        .ok_or_else(|| PtwmCoreError::InvalidContainer("empty codec menu".into()))?;

    Ok(TrialResult { chosen, encoded })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{CodecId, StateSource};

    #[test]
    fn identity_wins_on_high_entropy() {
        // PRNG-like bytes are effectively uniform; Identity (no tag, no
        // table overhead) wins over Huffman (1-byte tag + table) on a
        // 1 KB plane with near-uniform distribution.
        let plane: Vec<u8> = (0..1024u32)
            .map(|i| (i.wrapping_mul(12347) ^ 0x5A5A5A5A) as u8)
            .collect();
        let menu = vec![
            MenuItem {
                codec_id: CodecId::Identity,
                state_source: StateSource::None,
                shared_state_bytes: None,
                shared_state_id: None,
            },
            MenuItem {
                codec_id: CodecId::Huffman,
                state_source: StateSource::Inline,
                shared_state_bytes: None,
                shared_state_id: None,
            },
        ];
        let r = trial_encode_plane(&plane, &menu, &PlaneLayout::Flat).unwrap();
        assert_eq!(r.chosen.codec_id, CodecId::Identity);
    }

    #[test]
    fn huffman_wins_on_skewed() {
        // All zeros — Huffman compresses massively.
        let plane = vec![0u8; 4096];
        let menu = vec![
            MenuItem {
                codec_id: CodecId::Identity,
                state_source: StateSource::None,
                shared_state_bytes: None,
                shared_state_id: None,
            },
            MenuItem {
                codec_id: CodecId::Huffman,
                state_source: StateSource::Inline,
                shared_state_bytes: None,
                shared_state_id: None,
            },
        ];
        let r = trial_encode_plane(&plane, &menu, &PlaneLayout::Flat).unwrap();
        assert_eq!(r.chosen.codec_id, CodecId::Huffman);
    }

    #[test]
    fn empty_menu_is_error() {
        let plane = vec![0u8; 16];
        let r = trial_encode_plane(&plane, &[], &PlaneLayout::Flat);
        assert!(r.is_err());
    }
}
