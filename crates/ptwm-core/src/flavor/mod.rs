//! PTWM flavor system: WASM (Wasmtime), native cdylib (dlopen), and
//! host-language entry points (Python). Each flavor materializes a
//! `Box<dyn AnyContribution>` from a disk path + manifest entry.
//!
//! This file re-exports the submodules that make up the flavor system.

pub mod abi;
pub mod host;
pub mod native;
pub mod router;
pub mod third_party;
pub mod wasm;

pub use abi::{
    AnyState, BenchmarkMetric, ChainBuilder, ChainExplorer, Classifier, CodecError,
    ContainerLayout, ConvertMode, DeltaScheme, HardwareBackend, IntegrationAdapter, NoState,
    PlaneCodec, RawBinary, Scorer, TrainingHook, Transform,
};
pub use host::HostExtension;
pub use native::{NativeExtension, NativeSymbols, PlaneCodecFn, TransformFn, VerifiedToken};
pub use router::{DispatchedPlaneCodec, PlaneCodecRouter};
pub use third_party::ThirdPartyPlaneCodec;
pub use wasm::{WasmExtension, WasmState, invoke_plane_codec_decode, invoke_plane_codec_encode};
