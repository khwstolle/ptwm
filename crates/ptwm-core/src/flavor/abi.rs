//! Per-kind ABI: Rust traits the host dispatcher invokes.
//!
//! C ABI symbols and WASM exports for each kind are documented in
//! `docs/architecture/extension-system.md` and implemented by
//! reference contributions under `extensions/`.
//!
//! Per the design, every kind has its own ABI version. The encode-only
//! kinds expose a single `apply` / `score` / etc. method; the both-side
//! kinds expose `encode` + `decode` (or `forward` + `inverse`) pairs.

use crate::extension::{CanonicalId, Lifecycle};

/// Errors returned across the ABI boundary. The wire encoding maps these
/// to negative i64 return codes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CodecError {
    InvalidInput,
    BufferTooSmall {
        needed: u64,
    },
    MissingCapability {
        name: String,
    },
    MissingNativeDep {
        name: String,
        version_constraint: String,
    },
    Unsupported {
        feature: String,
    },
    InternalError,
}

impl std::fmt::Display for CodecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CodecError::InvalidInput => write!(f, "invalid input"),
            CodecError::BufferTooSmall { needed } => {
                write!(f, "buffer too small: needed {needed}")
            }
            CodecError::MissingCapability { name } => {
                write!(f, "missing capability: {name}")
            }
            CodecError::MissingNativeDep {
                name,
                version_constraint,
            } => {
                write!(f, "missing native dep: {name} {version_constraint}")
            }
            CodecError::Unsupported { feature } => write!(f, "unsupported: {feature}"),
            CodecError::InternalError => write!(f, "internal error"),
        }
    }
}

impl std::error::Error for CodecError {}

// ----- Per-kind state types ---------------------------------------------

/// Trait-object marker for the per-kind state. Each kind defines its
/// own `*State: Send` trait alias inheriting this; the host owns the
/// boxed state for the lifetime of the contribution instance.
pub trait AnyState: Send {}

// Per-kind trait definitions below. Each method that needs a state
// handle takes a `&mut dyn KindState`. Stateless kinds get a default
// `NoState` placeholder.

#[derive(Default)]
pub struct NoState;
impl AnyState for NoState {}

pub trait TransformState: AnyState {}
impl<T: AnyState + ?Sized> TransformState for T {}

pub trait PlaneCodecState: AnyState {}
impl<T: AnyState + ?Sized> PlaneCodecState for T {}

pub trait DeltaSchemeState: AnyState {}
impl<T: AnyState + ?Sized> DeltaSchemeState for T {}

pub trait ContainerLayoutState: AnyState {}
impl<T: AnyState + ?Sized> ContainerLayoutState for T {}

pub trait HardwareBackendState: AnyState {}
impl<T: AnyState + ?Sized> HardwareBackendState for T {}

pub trait RawBinaryState: AnyState {}
impl<T: AnyState + ?Sized> RawBinaryState for T {}

// Encode-only kinds (no shared state across calls — usually pure).

pub trait ChainBuilderState: AnyState {}
impl<T: AnyState + ?Sized> ChainBuilderState for T {}

pub trait ChainExplorerState: AnyState {}
impl<T: AnyState + ?Sized> ChainExplorerState for T {}

pub trait ClassifierState: AnyState {}
impl<T: AnyState + ?Sized> ClassifierState for T {}

pub trait ScorerState: AnyState {}
impl<T: AnyState + ?Sized> ScorerState for T {}

pub trait TrainingHookState: AnyState {}
impl<T: AnyState + ?Sized> TrainingHookState for T {}

pub trait IntegrationAdapterState: AnyState {}
impl<T: AnyState + ?Sized> IntegrationAdapterState for T {}

pub trait BenchmarkMetricState: AnyState {}
impl<T: AnyState + ?Sized> BenchmarkMetricState for T {}

// ----- Trait surface, one per kind --------------------------------------

pub trait Transform: Send + Sync {
    fn canonical_id(&self) -> CanonicalId;
    fn lifecycle(&self) -> Lifecycle;
    fn init(&self) -> Box<dyn TransformState>;
    fn forward(
        &self,
        state: &mut dyn TransformState,
        input: &[u8],
        output: &mut [u8],
    ) -> Result<usize, CodecError>;
    fn inverse(
        &self,
        state: &mut dyn TransformState,
        input: &[u8],
        output: &mut [u8],
    ) -> Result<usize, CodecError>;
}

pub trait PlaneCodec: Send + Sync {
    fn canonical_id(&self) -> CanonicalId;
    fn lifecycle(&self) -> Lifecycle;
    fn init(&self) -> Box<dyn PlaneCodecState>;
    fn encode(
        &self,
        state: &mut dyn PlaneCodecState,
        input: &[u8],
        output: &mut [u8],
    ) -> Result<usize, CodecError>;
    fn decode(
        &self,
        state: &mut dyn PlaneCodecState,
        input: &[u8],
        output: &mut [u8],
    ) -> Result<usize, CodecError>;
}

pub trait DeltaScheme: Send + Sync {
    fn canonical_id(&self) -> CanonicalId;
    fn lifecycle(&self) -> Lifecycle;
    fn init(&self) -> Box<dyn DeltaSchemeState>;
    fn encode(
        &self,
        state: &mut dyn DeltaSchemeState,
        base: &[u8],
        target: &[u8],
        output: &mut [u8],
    ) -> Result<usize, CodecError>;
    fn decode(
        &self,
        state: &mut dyn DeltaSchemeState,
        base: &[u8],
        delta: &[u8],
        output: &mut [u8],
    ) -> Result<usize, CodecError>;
}

pub trait ContainerLayout: Send + Sync {
    fn canonical_id(&self) -> CanonicalId;
    fn lifecycle(&self) -> Lifecycle;
    fn init(&self) -> Box<dyn ContainerLayoutState>;
    fn write(
        &self,
        state: &mut dyn ContainerLayoutState,
        payload: &[u8],
        out: &mut [u8],
    ) -> Result<usize, CodecError>;
    fn read(
        &self,
        state: &mut dyn ContainerLayoutState,
        input: &[u8],
        out: &mut [u8],
    ) -> Result<usize, CodecError>;
}

pub trait HardwareBackend: Send + Sync {
    fn canonical_id(&self) -> CanonicalId;
    fn lifecycle(&self) -> Lifecycle;
    fn init(&self) -> Box<dyn HardwareBackendState>;
    fn dispatch_decode(
        &self,
        state: &mut dyn HardwareBackendState,
        plane_codec_id: &CanonicalId,
        input: &[u8],
        output: &mut [u8],
    ) -> Result<usize, CodecError>;
}

pub trait RawBinary: Send + Sync {
    fn canonical_id(&self) -> CanonicalId;
    fn lifecycle(&self) -> Lifecycle;
    fn init(&self) -> Box<dyn RawBinaryState>;
    /// Validates input/output against the contribution's schema.
    fn encode(
        &self,
        state: &mut dyn RawBinaryState,
        input: &[u8],
        output: &mut [u8],
    ) -> Result<usize, CodecError>;
    fn decode(
        &self,
        state: &mut dyn RawBinaryState,
        input: &[u8],
        output: &mut [u8],
    ) -> Result<usize, CodecError>;
}

// Encode-only kinds (apply / score / etc.)

pub trait ChainBuilder: Send + Sync {
    fn canonical_id(&self) -> CanonicalId;
    fn lifecycle(&self) -> Lifecycle;
    fn init(&self) -> Box<dyn ChainBuilderState>;
    /// Produce candidate chains for a (dtype_code, role) pair as
    /// serialized chain blobs.
    fn build_for(
        &self,
        state: &mut dyn ChainBuilderState,
        dtype_code: u16,
        role: &str,
        out: &mut Vec<Vec<u8>>,
    ) -> Result<(), CodecError>;
}

pub trait ChainExplorer: Send + Sync {
    fn canonical_id(&self) -> CanonicalId;
    fn lifecycle(&self) -> Lifecycle;
    fn init(&self) -> Box<dyn ChainExplorerState>;
    /// Search starting from `seed_candidates`; append discoveries to
    /// `out_candidates`. Returning Ok with no new entries is a valid
    /// outcome.
    fn explore(
        &self,
        state: &mut dyn ChainExplorerState,
        dtype_code: u16,
        role: &str,
        seed_candidates: &[Vec<u8>],
        out_candidates: &mut Vec<Vec<u8>>,
    ) -> Result<(), CodecError>;
}

pub trait Classifier: Send + Sync {
    fn canonical_id(&self) -> CanonicalId;
    fn lifecycle(&self) -> Lifecycle;
    fn init(&self) -> Box<dyn ClassifierState>;
    fn classify(
        &self,
        state: &mut dyn ClassifierState,
        tensor_name: &str,
        dtype_code: u16,
        shape: &[u32],
    ) -> Result<String, CodecError>;
}

pub trait Scorer: Send + Sync {
    fn canonical_id(&self) -> CanonicalId;
    fn lifecycle(&self) -> Lifecycle;
    fn init(&self) -> Box<dyn ScorerState>;
    /// Score `bytes` produced by a candidate chain. Lower is better.
    fn score(&self, state: &mut dyn ScorerState, bytes: &[u8]) -> Result<f64, CodecError>;
}

pub trait TrainingHook: Send + Sync {
    fn canonical_id(&self) -> CanonicalId;
    fn lifecycle(&self) -> Lifecycle;
    fn init(&self) -> Box<dyn TrainingHookState>;
    /// Pre-encode hook: examines / mutates the bytes that would be
    /// written. Lifetime-bounded to one tensor.
    fn pre_encode(
        &self,
        state: &mut dyn TrainingHookState,
        tensor_name: &str,
        bytes: &mut Vec<u8>,
    ) -> Result<(), CodecError>;
}

pub trait IntegrationAdapter: Send + Sync {
    fn canonical_id(&self) -> CanonicalId;
    fn lifecycle(&self) -> Lifecycle;
    fn init(&self) -> Box<dyn IntegrationAdapterState>;
    /// Convert a host-library byte payload into a ptwm-shaped tensor
    /// payload (or vice versa, via `mode`).
    fn convert(
        &self,
        state: &mut dyn IntegrationAdapterState,
        mode: ConvertMode,
        input: &[u8],
        output: &mut [u8],
    ) -> Result<usize, CodecError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConvertMode {
    ImportIntoPtwm,
    ExportFromPtwm,
}

pub trait BenchmarkMetric: Send + Sync {
    fn canonical_id(&self) -> CanonicalId;
    fn lifecycle(&self) -> Lifecycle;
    fn init(&self) -> Box<dyn BenchmarkMetricState>;
    /// Measure a numeric metric over the produced bytes. Caller picks
    /// the unit semantics by canonical_id.
    fn measure(
        &self,
        state: &mut dyn BenchmarkMetricState,
        bytes: &[u8],
    ) -> Result<f64, CodecError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compilation check: every trait can be put behind a `Box<dyn ...>`.
    #[allow(dead_code)]
    fn _trait_objects_compile() {
        let _: Option<Box<dyn Transform>> = None;
        let _: Option<Box<dyn PlaneCodec>> = None;
        let _: Option<Box<dyn DeltaScheme>> = None;
        let _: Option<Box<dyn ContainerLayout>> = None;
        let _: Option<Box<dyn HardwareBackend>> = None;
        let _: Option<Box<dyn RawBinary>> = None;
        let _: Option<Box<dyn ChainBuilder>> = None;
        let _: Option<Box<dyn ChainExplorer>> = None;
        let _: Option<Box<dyn Classifier>> = None;
        let _: Option<Box<dyn Scorer>> = None;
        let _: Option<Box<dyn TrainingHook>> = None;
        let _: Option<Box<dyn IntegrationAdapter>> = None;
        let _: Option<Box<dyn BenchmarkMetric>> = None;
    }

    #[test]
    fn codec_error_display_works() {
        let e = CodecError::BufferTooSmall { needed: 1024 };
        assert!(e.to_string().contains("1024"));
    }
}
