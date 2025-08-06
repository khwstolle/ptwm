use std::fmt;

#[cfg(feature = "pyo3")]
use pyo3::PyErr;
#[cfg(feature = "pyo3")]
use pyo3::exceptions::{PyMemoryError, PyRuntimeError, PyValueError};

/// One entry in a [`PtwmCoreError::MissingFlavor`] error: describes a single
/// non-builtin extension that is required by the file but not available on
/// this host.
#[derive(Clone, Debug)]
pub struct MissingFlavorEntry {
    /// Canonical id of the missing contribution.
    pub id: crate::extension::CanonicalId,
    /// Human-readable label from the file's Extension Table entry.
    pub label: String,
    /// Flavor bits requested by the file (`flavor_hints` in the table entry).
    pub requested_flavors: u8,
    /// Flavor bits detected on disk / in the host registry. `0` means the
    /// contribution was not found at all.
    pub available_flavors: u8,
    /// Optional install hint copied from the Extension Table entry.
    pub install_hint: Option<String>,
}

impl fmt::Display for MissingFlavorEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} (id={}, requested=0b{:03b}, available=0b{:03b})",
            self.label, self.id, self.requested_flavors, self.available_flavors,
        )?;
        if let Some(hint) = &self.install_hint {
            write!(f, " → install: {hint}")?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum PtwmCoreError {
    InvalidNumBuf(u32),
    InvalidByteMode(i32),
    InvalidBitMode(i32),
    HuffmanCompress(String),
    HuffmanDecompress(String),
    RansCompress(String),
    RansDecompress(String),
    CodecEncode {
        codec: &'static str,
        msg: String,
    },
    CodecDecode {
        codec: &'static str,
        msg: String,
    },
    BufferTooSmall {
        expected: usize,
        got: usize,
    },
    Io(String),
    InvalidContainer(String),
    InvalidHeaderField(String),
    InvalidWireFormat(String),
    PayloadHashMismatch {
        expected: u64,
        actual: u64,
    },
    ExtensionTableHashMismatch,
    Extension(crate::extension::ExtensionError),
    /// One or more non-builtin extensions referenced by the file are not
    /// installed (or the installed flavor doesn't satisfy the request).
    MissingFlavor {
        needs: Vec<MissingFlavorEntry>,
    },
    /// A contribution was found but could not be trusted (e.g. invalid
    /// signature, revoked key).
    ContributionUntrusted {
        id: crate::extension::CanonicalId,
        reason: String,
    },
}

impl From<crate::extension::ExtensionError> for PtwmCoreError {
    fn from(e: crate::extension::ExtensionError) -> Self {
        PtwmCoreError::Extension(e)
    }
}

impl fmt::Display for PtwmCoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidNumBuf(n) => write!(f, "Invalid numBuf: {n}. Must be 1, 2, 4, or 8"),
            Self::InvalidByteMode(m) => {
                write!(f, "Invalid bytes_mode: {m}. Must be 1, 8, 10, or 220")
            }
            Self::InvalidBitMode(m) => write!(f, "Invalid bits_mode: {m}. Must be 0 or 1"),
            Self::HuffmanCompress(msg) => write!(f, "Huffman compression error: {msg}"),
            Self::HuffmanDecompress(msg) => write!(f, "Huffman decompression error: {msg}"),
            Self::RansCompress(msg) => write!(f, "rANS compression error: {msg}"),
            Self::RansDecompress(msg) => write!(f, "rANS decompression error: {msg}"),
            Self::CodecEncode { codec, msg } => write!(f, "{codec} encode error: {msg}"),
            Self::CodecDecode { codec, msg } => write!(f, "{codec} decode error: {msg}"),
            Self::Io(msg) => write!(f, "I/O error: {msg}"),
            Self::InvalidContainer(msg) => write!(f, "invalid container: {msg}"),
            Self::BufferTooSmall { expected, got } => {
                write!(f, "Buffer too small: expected {expected}, got {got}")
            }
            Self::InvalidHeaderField(msg) => write!(f, "Invalid header field: {msg}"),
            Self::InvalidWireFormat(msg) => write!(f, "Invalid wire format: {msg}"),
            Self::ExtensionTableHashMismatch => write!(f, "extension table hash mismatch"),
            Self::PayloadHashMismatch { expected, actual } => {
                write!(
                    f,
                    "Payload hash mismatch: expected {expected:#018x}, actual {actual:#018x}"
                )
            }
            Self::Extension(e) => write!(f, "extension error: {e}"),
            Self::MissingFlavor { needs } => {
                write!(f, "missing extension flavors: {} entries", needs.len())?;
                for entry in needs {
                    write!(f, "\n  - {entry}")?;
                }
                Ok(())
            }
            Self::ContributionUntrusted { id, reason } => {
                write!(f, "contribution {id} is untrusted: {reason}")
            }
        }
    }
}

impl std::error::Error for PtwmCoreError {}

impl From<std::io::Error> for PtwmCoreError {
    fn from(e: std::io::Error) -> Self {
        PtwmCoreError::Io(e.to_string())
    }
}

#[cfg(feature = "pyo3")]
impl From<PtwmCoreError> for PyErr {
    fn from(e: PtwmCoreError) -> PyErr {
        match &e {
            PtwmCoreError::HuffmanCompress(_)
            | PtwmCoreError::HuffmanDecompress(_)
            | PtwmCoreError::RansCompress(_)
            | PtwmCoreError::RansDecompress(_)
            | PtwmCoreError::CodecEncode { .. }
            | PtwmCoreError::CodecDecode { .. } => PyRuntimeError::new_err(e.to_string()),
            PtwmCoreError::BufferTooSmall { .. } => PyMemoryError::new_err(e.to_string()),
            PtwmCoreError::InvalidHeaderField(_)
            | PtwmCoreError::PayloadHashMismatch { .. }
            | PtwmCoreError::ExtensionTableHashMismatch
            | PtwmCoreError::InvalidContainer(_)
            | PtwmCoreError::InvalidWireFormat(_) => PyValueError::new_err(e.to_string()),
            PtwmCoreError::Io(_) => pyo3::exceptions::PyOSError::new_err(e.to_string()),
            PtwmCoreError::MissingFlavor { .. } | PtwmCoreError::ContributionUntrusted { .. } => {
                PyValueError::new_err(e.to_string())
            }
            _ => PyValueError::new_err(e.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_header_field_formats() {
        let e = PtwmCoreError::InvalidHeaderField("bad magic".into());
        assert!(e.to_string().contains("bad magic"));
    }

    #[test]
    fn payload_hash_mismatch_formats() {
        let e = PtwmCoreError::PayloadHashMismatch {
            expected: 1,
            actual: 2,
        };
        assert!(e.to_string().contains("Payload hash"));
    }

    #[test]
    fn invalid_num_buf_formats() {
        let e = PtwmCoreError::InvalidNumBuf(3);
        let s = e.to_string();
        assert!(s.contains("3"));
        assert!(s.contains("numBuf"));
    }

    #[test]
    fn invalid_byte_mode_formats() {
        let e = PtwmCoreError::InvalidByteMode(99);
        let s = e.to_string();
        assert!(s.contains("99"));
        assert!(s.contains("bytes_mode"));
    }

    #[test]
    fn invalid_bit_mode_formats() {
        let e = PtwmCoreError::InvalidBitMode(2);
        let s = e.to_string();
        assert!(s.contains("2"));
        assert!(s.contains("bits_mode"));
    }

    #[test]
    fn huffman_compress_error_formats() {
        let e = PtwmCoreError::HuffmanCompress("dst too small".into());
        assert!(e.to_string().contains("Huffman"));
        assert!(e.to_string().contains("dst too small"));
    }

    #[test]
    fn rans_compress_error_formats() {
        let e = PtwmCoreError::RansCompress("state overflow".into());
        assert!(e.to_string().contains("rANS"));
        assert!(e.to_string().contains("state overflow"));
    }

    #[test]
    fn codec_encode_formats() {
        let e = PtwmCoreError::CodecEncode {
            codec: "zstd",
            msg: "level out of range".into(),
        };
        assert!(e.to_string().contains("zstd"));
        assert!(e.to_string().contains("encode"));
    }

    #[test]
    fn buffer_too_small_formats() {
        let e = PtwmCoreError::BufferTooSmall {
            expected: 128,
            got: 64,
        };
        let s = e.to_string();
        assert!(s.contains("128"));
        assert!(s.contains("64"));
    }

    #[test]
    fn invalid_container_formats() {
        let e = PtwmCoreError::InvalidContainer("bad footer magic".into());
        assert!(e.to_string().contains("container"));
        assert!(e.to_string().contains("bad footer magic"));
    }

    #[test]
    fn invalid_wire_format_formats() {
        let e = PtwmCoreError::InvalidWireFormat("non-monotonic cum".into());
        let s = e.to_string();
        assert!(s.contains("wire format"));
        assert!(s.contains("non-monotonic cum"));
    }

    #[test]
    fn io_error_formats() {
        let e = PtwmCoreError::Io("disk full".into());
        assert!(e.to_string().contains("I/O"));
        assert!(e.to_string().contains("disk full"));
    }

    #[test]
    fn io_error_from_std() {
        let std_err = std::io::Error::new(std::io::ErrorKind::NotFound, "nope");
        let e: PtwmCoreError = std_err.into();
        assert!(matches!(e, PtwmCoreError::Io(_)));
        assert!(e.to_string().contains("nope"));
    }
}
