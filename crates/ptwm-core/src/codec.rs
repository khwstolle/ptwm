/// Stable codec identifier in the `.ptwm` registry. Values carry
/// documented stewardship ranges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum CodecId {
    Identity = 0x0000,
    Huffman = 0x0001,
    HuffmanNibble = 0x0002,
    Rans = 0x0003,
    Zstd = 0x0004,
    ZstdDict = 0x0005,
    Fpc = 0x0006,
    Tans = 0x0007,
    PerGroupCodebook = 0x0010,
    Order1ScaleAC = 0x0011,
    ArithmeticO0 = 0x0012,
    ArithmeticO0Adaptive = 0x0013,
    ArithmeticO1 = 0x0014,
    ContextMixingLite = 0x0015,
    HuffLlm5Bit = 0x0016,
    Order1Arithmetic = 0x0017,
    NeuralPredictor = 0x0018,
}

/// Every wire id paired with its canonical lowercase name. Single source
/// of truth for the Python-side `CodecId` IntEnum mirror — the Python
/// test asserts membership against this list at runtime.
pub const REGISTRY: &[(&str, u16)] = &[
    ("identity", 0x0000),
    ("huffman", 0x0001),
    ("huffman_nibble", 0x0002),
    ("rans", 0x0003),
    ("zstd", 0x0004),
    ("zstd_dict", 0x0005),
    ("fpc", 0x0006),
    ("tans", 0x0007),
    ("per_group_codebook", 0x0010),
    ("order1_scale_ac", 0x0011),
    ("arithmetic_o0", 0x0012),
    ("arithmetic_o0_adaptive", 0x0013),
    ("arithmetic_o1", 0x0014),
    ("context_mixing_lite", 0x0015),
    ("huff_llm_5bit", 0x0016),
    ("order1_arithmetic", 0x0017),
    ("neural_predictor", 0x0018),
];

impl CodecId {
    pub fn from_u16(raw: u16) -> Option<Self> {
        match raw {
            0x0000 => Some(Self::Identity),
            0x0001 => Some(Self::Huffman),
            0x0002 => Some(Self::HuffmanNibble),
            0x0003 => Some(Self::Rans),
            0x0004 => Some(Self::Zstd),
            0x0005 => Some(Self::ZstdDict),
            0x0006 => Some(Self::Fpc),
            0x0007 => Some(Self::Tans),
            0x0010 => Some(Self::PerGroupCodebook),
            0x0011 => Some(Self::Order1ScaleAC),
            0x0012 => Some(Self::ArithmeticO0),
            0x0013 => Some(Self::ArithmeticO0Adaptive),
            0x0014 => Some(Self::ArithmeticO1),
            0x0015 => Some(Self::ContextMixingLite),
            0x0016 => Some(Self::HuffLlm5Bit),
            0x0017 => Some(Self::Order1Arithmetic),
            0x0018 => Some(Self::NeuralPredictor),
            _ => None,
        }
    }

    pub fn as_u16(&self) -> u16 {
        *self as u16
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum StateSource {
    None = 0x00,
    Inline = 0x01,
    Shared = 0x02,
    External = 0x03,
}

impl StateSource {
    pub fn from_u8(raw: u8) -> Option<Self> {
        match raw {
            0 => Some(Self::None),
            1 => Some(Self::Inline),
            2 => Some(Self::Shared),
            3 => Some(Self::External),
            _ => None,
        }
    }
}

use crate::error::PtwmCoreError;
use crate::layout::PlaneLayout;
use crate::types::descriptor::PlaneDescriptor;

/// Output of an encode call: the encoded state bytes (if any) plus the
/// encoded payload.
#[derive(Debug)]
pub struct Encoded {
    pub state_bytes: Vec<u8>,
    pub state_format_version: u8,
    pub payload: Vec<u8>,
}

/// Plane codec contract.
pub trait PlaneCodec: Send + Sync {
    fn id(&self) -> CodecId;

    /// Returns `true` when this codec can handle a plane described by
    /// `descriptor`. The central dispatcher filters by this predicate
    /// before ranking by [`Self::priority_for`]. Default: always accept.
    fn accepts(&self, _descriptor: &PlaneDescriptor) -> bool {
        true
    }

    /// Priority score for ranking accepted codecs. Higher wins. The
    /// dispatcher sorts candidates in descending order and picks the
    /// first. Default: 1 (above Identity's 0).
    fn priority_for(&self, _descriptor: &PlaneDescriptor) -> i8 {
        1
    }

    /// Cheap pre-encode gate. Returns `false` when a content-aware signal
    /// predicts this codec will lose to a cheaper menu item; the trial-
    /// encode caller may then skip [`Self::encode`] entirely. Default:
    /// always attempt. Codecs with expensive `encode` (e.g.
    /// `Order1ScaleAC`'s per-row PMF fit) override this to short-circuit
    /// when a fast statistical estimate predicts no gain.
    ///
    /// Must stay cheap (`O(sample_size)`) and conservative: a false-`false`
    /// (skipping incorrectly) costs ratio; a false-`true` only costs CPU.
    fn should_attempt(
        &self,
        _plane: &[u8],
        _descriptor: &PlaneDescriptor,
        _layout: &PlaneLayout,
    ) -> bool {
        true
    }

    /// Encode a single plane. The caller passes `shared_state` for
    /// `StateSource::Shared` / `External`; for `Inline`, the writer stores
    /// `Encoded::state_bytes` alongside the payload; for `None`, the
    /// returned `state_bytes` must stay empty.
    fn encode(
        &self,
        plane: &[u8],
        shared_state: Option<&[u8]>,
        layout: &PlaneLayout,
    ) -> Result<Encoded, PtwmCoreError>;

    /// Decode a plane given the state's format version, state bytes, and
    /// payload bytes. The decoder MUST handle every state version ever
    /// emitted by this codec id.
    ///
    /// `decoded_len` is the expected decoded byte length. Self-delimiting
    /// codecs (Huffman, Rans, Zstd, Identity, PerGroupCodebook) ignore
    /// it; non-self-delimiting range coders (Order1ScaleAC) require it.
    fn decode(
        &self,
        state_format_version: u8,
        state_bytes: &[u8],
        payload: &[u8],
        layout: &PlaneLayout,
        decoded_len: usize,
    ) -> Result<Vec<u8>, PtwmCoreError>;
}

/// Look up a codec by id. Returns `None` for unregistered ids; the
/// container-open path should have rejected those already.
pub fn codec_for(id: CodecId) -> Option<Box<dyn PlaneCodec>> {
    match id {
        CodecId::Identity => Some(Box::new(crate::codecs::identity::Identity)),
        CodecId::Huffman => Some(Box::new(crate::codecs::huffman::Huffman)),
        CodecId::PerGroupCodebook => Some(Box::new(
            crate::codecs::per_group_codebook::PerGroupCodebook,
        )),
        CodecId::Order1ScaleAC => Some(Box::new(crate::codecs::order1_scale_ac::Order1ScaleAC)),
        CodecId::Zstd => Some(Box::new(crate::codecs::zstd::Zstd)),
        CodecId::ZstdDict => Some(Box::new(crate::codecs::zstd_dict::ZstdDict)),
        CodecId::Fpc => Some(Box::new(crate::codecs::fpc::Fpc)),
        CodecId::Rans => Some(Box::new(crate::codecs::rans::Rans)),
        CodecId::Tans => Some(Box::new(crate::codecs::tans::Tans)),
        CodecId::ArithmeticO0 => Some(Box::new(crate::codecs::arithmetic::ArithmeticO0)),
        CodecId::ArithmeticO0Adaptive => {
            Some(Box::new(crate::codecs::arithmetic::ArithmeticO0Adaptive))
        }
        CodecId::ArithmeticO1 => Some(Box::new(crate::codecs::arithmetic::ArithmeticO1)),
        CodecId::ContextMixingLite => {
            Some(Box::new(crate::codecs::context_mixing::ContextMixingLite))
        }
        CodecId::HuffLlm5Bit => Some(Box::new(crate::codecs::huff_llm::HuffLlm5Bit)),
        CodecId::Order1Arithmetic => {
            Some(Box::new(crate::codecs::order1_arithmetic::Order1Arithmetic))
        }
        CodecId::NeuralPredictor => {
            Some(Box::new(crate::codecs::neural_predictor::NeuralPredictor))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_for_returns_order1_scale_ac() {
        let c = codec_for(CodecId::Order1ScaleAC).expect("Order1ScaleAC must dispatch");
        assert_eq!(c.id(), CodecId::Order1ScaleAC);
    }

    #[test]
    fn codec_id_roundtrip() {
        for id in [
            CodecId::Identity,
            CodecId::Huffman,
            CodecId::HuffmanNibble,
            CodecId::Rans,
            CodecId::Zstd,
            CodecId::ZstdDict,
            CodecId::Fpc,
            CodecId::Tans,
            CodecId::PerGroupCodebook,
            CodecId::Order1ScaleAC,
            CodecId::ArithmeticO0,
            CodecId::ArithmeticO0Adaptive,
            CodecId::ArithmeticO1,
            CodecId::ContextMixingLite,
            CodecId::HuffLlm5Bit,
            CodecId::Order1Arithmetic,
            CodecId::NeuralPredictor,
        ] {
            assert_eq!(CodecId::from_u16(id.as_u16()), Some(id));
        }
        assert_eq!(CodecId::from_u16(0x0099), None);
    }

    #[test]
    fn state_source_roundtrip() {
        for s in [
            StateSource::None,
            StateSource::Inline,
            StateSource::Shared,
            StateSource::External,
        ] {
            assert_eq!(StateSource::from_u8(s as u8), Some(s));
        }
        assert_eq!(StateSource::from_u8(0x04), None);
    }
}
