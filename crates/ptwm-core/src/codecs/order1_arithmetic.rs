//! Order1Arithmetic: static order-1 (lag-1) arithmetic coder over the
//! exponent byte plane — a learned dynamic-length float coder. Fits a
//! conditional table once, serializes it into codec state (the static
//! table is what makes a future GPU decode kernel feasible), and range-codes
//! against the frozen model. Self-delimiting: it frames the true plane length
//! and ignores the container's source-derived `decoded_len`.

use crate::codec::{CodecId, Encoded, PlaneCodec};
use crate::codecs::arithmetic::{frame_payload, unframe_payload};
use crate::entropy::arithmetic::{Order1Static, decode_bytes, encode_bytes};
use crate::error::PtwmCoreError;
use crate::layout::PlaneLayout;
use crate::types::descriptor::PlaneDescriptor;
use crate::types::role::Role;

/// Below this the serialized conditional table (≥ 1 KiB) isn't amortized;
/// `arithmetic_o1` / `rans` win and we skip the heavier attempt.
const MIN_PLANE_BYTES: usize = 4096;
const STATE_FORMAT_VERSION: u8 = 1;

pub struct Order1Arithmetic;

impl Order1Arithmetic {
    pub fn canonical_id(&self) -> crate::extension::CanonicalId {
        crate::extension::builtin_canonical_id("order1_arithmetic")
    }
}

impl PlaneCodec for Order1Arithmetic {
    fn id(&self) -> CodecId {
        CodecId::Order1Arithmetic
    }

    fn accepts(&self, descriptor: &PlaneDescriptor) -> bool {
        matches!(descriptor.role, Role::ExponentByte)
    }

    fn priority_for(&self, descriptor: &PlaneDescriptor) -> i8 {
        if self.accepts(descriptor) {
            10
        } else {
            i8::MIN
        }
    }

    fn should_attempt(
        &self,
        plane: &[u8],
        _descriptor: &PlaneDescriptor,
        _layout: &PlaneLayout,
    ) -> bool {
        plane.len() >= MIN_PLANE_BYTES
    }

    fn encode(
        &self,
        plane: &[u8],
        shared_state: Option<&[u8]>,
        _layout: &PlaneLayout,
    ) -> Result<Encoded, PtwmCoreError> {
        if shared_state.is_some() {
            return Err(PtwmCoreError::CodecEncode {
                codec: "order1_arithmetic",
                msg: "shared state not supported".into(),
            });
        }
        if plane.is_empty() {
            return Ok(Encoded {
                state_bytes: Vec::new(),
                state_format_version: STATE_FORMAT_VERSION,
                payload: Vec::new(),
            });
        }
        let mut model = Order1Static::fit(plane);
        let payload = frame_payload(plane.len(), &encode_bytes(&mut model, plane));
        Ok(Encoded {
            state_bytes: model.serialize(),
            state_format_version: STATE_FORMAT_VERSION,
            payload,
        })
    }

    fn decode(
        &self,
        state_format_version: u8,
        state_bytes: &[u8],
        payload: &[u8],
        _layout: &PlaneLayout,
        _decoded_len: usize,
    ) -> Result<Vec<u8>, PtwmCoreError> {
        // Validate the version before the empty short-circuit so a future
        // v2 empty-payload blob can't decode as empty under a v1 reader.
        if state_format_version != STATE_FORMAT_VERSION {
            return Err(PtwmCoreError::CodecDecode {
                codec: "order1_arithmetic",
                msg: format!("unsupported state_format_version {state_format_version}"),
            });
        }
        if payload.is_empty() {
            return Ok(Vec::new());
        }
        let mut model = Order1Static::deserialize(state_bytes)?;
        let (n, body) = unframe_payload(payload, "order1_arithmetic")?;
        Ok(decode_bytes(&mut model, body, n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codecs::arithmetic::ArithmeticO1;
    use crate::types::descriptor::{ElementWidth, Layout, PlaneDescriptor};

    fn exp_desc(len: usize) -> PlaneDescriptor {
        PlaneDescriptor {
            role: Role::ExponentByte,
            element_width: ElementWidth::Byte,
            length_bytes: len as u64,
            layout: Layout::Flat,
            derives_from_tensor: None,
            residual_of: None,
            is_nibble_packed: false,
            vendor_bytes: vec![],
        }
    }

    fn rt(data: &[u8]) {
        let enc = Order1Arithmetic
            .encode(data, None, &PlaneLayout::Flat)
            .unwrap();
        let dec = Order1Arithmetic
            .decode(
                enc.state_format_version,
                &enc.state_bytes,
                &enc.payload,
                &PlaneLayout::Flat,
                data.len(),
            )
            .unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn roundtrip_skewed_exponent() {
        let data: Vec<u8> = (0..16384).map(|i| (120 + ((i * 7) % 9)) as u8).collect();
        rt(&data);
    }

    #[test]
    fn roundtrip_empty() {
        rt(&[]);
    }

    #[test]
    fn decode_ignores_wrong_decoded_len() {
        let data: Vec<u8> = (0..8192).map(|i| (127 + (i % 5)) as u8).collect();
        let enc = Order1Arithmetic
            .encode(&data, None, &PlaneLayout::Flat)
            .unwrap();
        let dec = Order1Arithmetic
            .decode(
                enc.state_format_version,
                &enc.state_bytes,
                &enc.payload,
                &PlaneLayout::Flat,
                data.len() * 4, // hostile / source-derived over-estimate
            )
            .unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn accepts_only_exponent_byte() {
        assert!(Order1Arithmetic.accepts(&exp_desc(4096)));
        let mut d = exp_desc(4096);
        d.role = Role::Raw;
        assert!(!Order1Arithmetic.accepts(&d));
    }

    #[test]
    fn bad_state_version_rejected() {
        let data: Vec<u8> = (0..4096).map(|i| (127 + (i % 3)) as u8).collect();
        let enc = Order1Arithmetic
            .encode(&data, None, &PlaneLayout::Flat)
            .unwrap();
        assert!(
            Order1Arithmetic
                .decode(
                    99,
                    &enc.state_bytes,
                    &enc.payload,
                    &PlaneLayout::Flat,
                    data.len()
                )
                .is_err()
        );
    }

    #[test]
    fn compresses_exponent_plane_far_below_raw_and_stays_competitive() {
        // A static learned coder's value proposition is lossless *size
        // reduction vs the raw float* — not superiority over an adaptive
        // arithmetic coder. The transmitted static table (a dense 512 B per
        // present context) is the GPU-decode enabler, and it costs more than
        // the adaptive order-1 coder's near-free online warm-up, so on a
        // stationary plane static does NOT beat adaptive on ratio (measured
        // here: ~36 KB static vs ~30 KB adaptive on this 256 KB plane). Its
        // value is the static, GPU-decodable table, hence the research
        // framing. This test pins the two true, useful properties:
        //   (a) it compresses the exponent plane far below raw, and
        //   (b) it stays within a small factor of the adaptive coder (the
        //       table overhead, not a pathological loss).
        let data: Vec<u8> = (0..262144)
            .map(|i| {
                let base = 126u8;
                base + ((i / 32) % 4) as u8 + (i % 2) as u8
            })
            .collect();
        let stat = Order1Arithmetic
            .encode(&data, None, &PlaneLayout::Flat)
            .unwrap();
        let stat_total = stat.state_bytes.len() + stat.payload.len();
        let adpt = ArithmeticO1
            .encode(&data, None, &PlaneLayout::Flat)
            .unwrap();
        let adpt_total = adpt.state_bytes.len() + adpt.payload.len();
        assert!(
            stat_total * 2 < data.len(),
            "static {stat_total} should be < half of raw {}",
            data.len()
        );
        assert!(
            stat_total <= adpt_total * 3 / 2,
            "static {stat_total} should stay within 1.5x of adaptive {adpt_total}"
        );
    }
}
