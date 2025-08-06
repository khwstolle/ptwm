//! Serde data model for the canonical PTWM-JSON manifest (Layer 1).
//!
//! These structs are a lossless projection of the binary `.ptwm` wire format.
//! Every field needed to reconstruct the container byte-for-byte is captured;
//! large/opaque binaries (compressed plane payloads, externalized codec state,
//! embedded modules) are referenced by member key and live outside the JSON.
//!
//! The schema restricts scalars to integers, strings, booleans, and `null`;
//! opaque byte fields are carried as base64 (`*_b64`). See [`super::jcs`] for
//! the canonicalization that makes manifest hashes reproducible.

use serde::{Deserialize, Serialize};

pub const PTWM_MANIFEST_FORMAT: &str = "ptwm-manifest";
pub const MANIFEST_VERSION: u32 = 1;

/// Archive-scoped registry manifest. Lifts the binary `extension_table` and
/// `shared_prelude` to archive scope so codecs/ops/codebooks/chains are defined
/// once and referenced by id.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegistryManifest {
    pub ptwm_format: String,
    pub manifest_version: u32,
    pub container_flags: ContainerFlags,
    /// Hex of the container header's `extension_table_hash` (blake3).
    pub registry_hash_blake3: String,
    pub registry: Registry,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContainerFlags {
    pub embedded_wasm: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Registry {
    pub ops: Vec<OpEntry>,
    pub codecs: Vec<CodecEntry>,
    pub shared_state: Vec<SharedStateEntry>,
    pub chains: Vec<ChainEntry>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpEntry {
    /// Stable wire op id (`OpId::as_u16`).
    pub id: u16,
    /// Human-readable op name.
    pub op: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CodecEntry {
    /// Wire-stable `CodecId` value.
    pub id: u16,
    /// Human-readable codec name.
    pub codec: String,
    /// Member key of an embedded WASM module, when this is a `wasm` codec.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub module: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SharedStateEntry {
    pub id: u16,
    pub codec: u16,
    pub state_format_version: u8,
    pub applies_to_mask: u8,
    /// Hex of the xxhash64 of the state bytes.
    pub hash_xxh64: String,
    pub name: String,
    /// Member key holding the shared state bytes.
    pub payload: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChainEntry {
    pub id: u16,
    pub nodes: Vec<NodeJson>,
    pub edges: Vec<EdgeJson>,
    pub terminals: Vec<TerminalJson>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeJson {
    /// Stable wire op id (`OpId::as_u16`).
    pub op: u16,
    pub params_b64: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EdgeJson {
    pub src: u8,
    pub src_out: u8,
    pub dst: u8,
    pub dst_in: u8,
    pub role_override_b64: Option<String>,
    pub vendor_b64: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TerminalJson {
    pub node: u8,
    pub out: u8,
    /// Human-readable role label.
    pub role: String,
    /// Exact serialized role bytes (base64), used for reconstruction.
    pub role_b64: String,
}

/// Per-tensor manifest — a direct projection of `TensorRecord`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorManifest {
    pub name: String,
    /// Canonical dtype name from the CBOR metadata; `null` when absent.
    pub dtype: Option<String>,
    pub dtype_code: u16,
    /// Tensor shape from the CBOR metadata; `null` when absent.
    pub shape: Option<Vec<u64>>,
    /// Verbatim CBOR metadata bytes (base64), populated only when the metadata
    /// is not the standard `{shape, dtype}` encoding. Guarantees byte-exact
    /// reconstruction for non-standard metadata; `null` for the common case
    /// where `shape`/`dtype` fully determine the bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tensor_metadata_b64: Option<String>,
    pub input_format: u8,
    pub orig_size: u64,
    /// Hex of the xxhash64 payload hash; `null` when not emitted.
    pub payload_hash_xxh64: Option<String>,
    pub chain: ChainSpec,
    pub dependencies: Vec<DependencyJson>,
    pub planes: Vec<PlaneJson>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChainSpec {
    /// Registry chain index when shared; `null` when the chain is inlined.
    #[serde(rename = "ref")]
    pub chain_ref: Option<u16>,
    /// Inlined chain definition (present iff `ref` is `null`).
    pub nodes: Option<Vec<NodeJson>>,
    pub edges: Option<Vec<EdgeJson>>,
    pub terminals: Option<Vec<TerminalJson>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DependencyJson {
    pub ref_kind: String,
    pub ref_b64: String,
    pub expected_hash_blake3: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlaneJson {
    pub index: u32,
    /// Human-readable plane role label.
    pub role: String,
    /// Exact `PlaneRole` discriminant, used for reconstruction.
    pub role_code: u8,
    /// `CodecId` wire value.
    pub codec: u16,
    /// Extension-table index for the codec (`0xFFFF` = absent).
    pub codec_table_idx: u16,
    /// Where the codec state lives in the manifest:
    /// `none` | `inline` | `member` | `shared` | `external`.
    pub state_source: String,
    /// Exact `StateSource` discriminant, used for reconstruction.
    pub state_source_code: u8,
    pub state_version: u8,
    pub state_info: u16,
    /// Inline codec state (base64), present for bounded-small codec tables.
    pub state_b64: Option<String>,
    /// Member key holding externalized codec state (unbounded state).
    pub state_payload: Option<String>,
    /// External state reference (present iff `state_source_code` == external).
    pub external_state: Option<ExternalStateJson>,
    pub crc32: Option<u32>,
    pub chunk_table: Option<Vec<ChunkJson>>,
    pub layout_kind: u8,
    pub layout_row_len: Option<u32>,
    /// Compressed payload length in bytes (`payload_len`).
    pub comp_len: u64,
    /// Member key holding the compressed plane payload.
    pub payload: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExternalStateJson {
    pub ref_kind: u8,
    pub ref_b64: String,
    pub state_format_version: u8,
    pub expected_hash_blake3: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChunkJson {
    pub offset_in_payload: u32,
    pub decoded_size: u32,
}
