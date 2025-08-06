//! Extension-system error type.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ExtensionError {
    #[error("invalid contribution reference: {0}")]
    InvalidRef(String),

    #[error("unknown kind: 0x{0:04X}")]
    UnknownKind(u16),

    #[error("unknown lifecycle: {0}")]
    UnknownLifecycle(u8),

    #[error("manifest parse error: {0}")]
    ManifestParse(String),

    #[error("capability parse error: {0}")]
    CapabilityParse(String),

    #[error("capability serialization error: {0}")]
    CapabilitySer(String),

    #[error("extension table parse error: {0}")]
    TableParse(String),

    #[error("extension table serialization error: {0}")]
    TableSer(String),

    #[error("flavor bitset contains undefined bits: 0x{0:02X}")]
    UnknownFlavorBits(u8),

    #[error("attestation parse error: {0}")]
    AttestationParse(String),
}
