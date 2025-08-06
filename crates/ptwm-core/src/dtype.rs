//! First-class dtype enum for the compression pipeline.
//!
//! The Python side has historically owned the dtype registry
//! (`ptwm.preprocessing.DTYPE_REGISTRY`). This module is the
//! Rust-authoritative equivalent: the wire-format `dtype_code`, the
//! preprocessing modes (`num_buf`, `bit_reorder`, `byte_reorder`), and the
//! name-based lookup used by the standalone CLI all funnel through
//! [`Dtype`].
//!
//! Code values align with the Python `DType`. Only dtypes
//! that have a supported preprocessing recipe are enumerated here —
//! complex, quantized, and 64-bit float variants that exist in the Python
//! enum but have no compression spec are intentionally absent. Such codes
//! will surface as [`PtwmCoreError::InvalidHeaderField`] from
//! [`Dtype::from_code`].

use crate::error::PtwmCoreError;

/// Supported dtypes for the compression pipeline.
///
/// Numeric values match the Python-side `DType.code` for the
/// same dtype; wire-format `dtype_code` values are exactly these discriminants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Dtype {
    Float32 = 1,
    /// Python alias for `Float32` (identical behaviour on the wire).
    Float = 2,
    Float16 = 4,
    /// Python alias for `Float16`.
    Half = 5,
    BFloat16 = 6,
    Uint8 = 13,
    Uint32 = 15,
    Int8 = 17,
    Int16 = 18,
    /// Python alias for `Int16`.
    Short = 19,
    Int32 = 20,
    /// Python alias for `Int32`.
    Int = 21,
    Int64 = 22,
    /// Python alias for `Int64`.
    Long = 23,
    Bool = 24,
    Float8E4M3FN = 29,
    Float8E5M2 = 30,
    /// Packed FP4: two 4-bit values per byte.
    Float4E2M1FNx2 = 31,
}

/// Preprocessing modes for one dtype: the trio that `compress_chunk` accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreprocessingModes {
    pub num_buf: u32,
    pub bit_reorder: i32,
    pub byte_reorder: i32,
}

impl Dtype {
    /// Wire `dtype_code` for this dtype (matches Python
    /// `DType.code`).
    pub const fn code(self) -> u8 {
        self as u8
    }

    /// Recover the dtype from a wire `dtype_code`. Codes outside the supported
    /// set yield [`PtwmCoreError::InvalidHeaderField`].
    pub fn from_code(code: u8) -> Result<Self, PtwmCoreError> {
        match code {
            1 => Ok(Self::Float32),
            2 => Ok(Self::Float),
            4 => Ok(Self::Float16),
            5 => Ok(Self::Half),
            6 => Ok(Self::BFloat16),
            13 => Ok(Self::Uint8),
            15 => Ok(Self::Uint32),
            17 => Ok(Self::Int8),
            18 => Ok(Self::Int16),
            19 => Ok(Self::Short),
            20 => Ok(Self::Int32),
            21 => Ok(Self::Int),
            22 => Ok(Self::Int64),
            23 => Ok(Self::Long),
            24 => Ok(Self::Bool),
            29 => Ok(Self::Float8E4M3FN),
            30 => Ok(Self::Float8E5M2),
            31 => Ok(Self::Float4E2M1FNx2),
            _ => Err(PtwmCoreError::InvalidHeaderField(format!(
                "unsupported dtype code: {code}"
            ))),
        }
    }

    /// Parse a case-insensitive dtype name. Accepts canonical names
    /// (`"float32"`, `"bfloat16"`), short aliases (`"fp32"`, `"bf16"`,
    /// `"f16"`, `"i32"`, `"u8"`), and Python-style aliases (`"half"`,
    /// `"short"`, `"long"`).
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "float32" | "fp32" | "f32" | "float" => Some(Self::Float32),
            "float16" | "fp16" | "f16" | "half" => Some(Self::Float16),
            "bfloat16" | "bf16" => Some(Self::BFloat16),
            "float8_e4m3fn" | "fp8-e4m3fn" | "fp8_e4m3fn" | "fp8e4m3fn" => Some(Self::Float8E4M3FN),
            "float8_e5m2" | "fp8-e5m2" | "fp8_e5m2" | "fp8e5m2" => Some(Self::Float8E5M2),
            "float4_e2m1fn_x2" | "fp4-e2m1fn-x2" | "fp4_e2m1fn_x2" | "fp4" => {
                Some(Self::Float4E2M1FNx2)
            }
            "int8" | "i8" => Some(Self::Int8),
            "uint8" | "u8" => Some(Self::Uint8),
            "uint32" | "u32" => Some(Self::Uint32),
            "int16" | "i16" | "short" => Some(Self::Int16),
            "int32" | "i32" | "int" => Some(Self::Int32),
            "int64" | "i64" | "long" => Some(Self::Int64),
            "bool" | "boolean" => Some(Self::Bool),
            _ => None,
        }
    }

    /// Canonical name (lowercase, underscore-separated).
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::Float32 => "float32",
            Self::Float => "float",
            Self::Float16 => "float16",
            Self::Half => "half",
            Self::BFloat16 => "bfloat16",
            Self::Uint8 => "uint8",
            Self::Uint32 => "uint32",
            Self::Int8 => "int8",
            Self::Int16 => "int16",
            Self::Short => "short",
            Self::Int32 => "int32",
            Self::Int => "int",
            Self::Int64 => "int64",
            Self::Long => "long",
            Self::Bool => "bool",
            Self::Float8E4M3FN => "float8_e4m3fn",
            Self::Float8E5M2 => "float8_e5m2",
            Self::Float4E2M1FNx2 => "float4_e2m1fn_x2",
        }
    }

    /// Element size in bytes.
    pub const fn element_size(self) -> u32 {
        match self {
            Self::Int8
            | Self::Uint8
            | Self::Bool
            | Self::Float8E4M3FN
            | Self::Float8E5M2
            | Self::Float4E2M1FNx2 => 1,
            Self::Int16 | Self::Short | Self::Float16 | Self::Half | Self::BFloat16 => 2,
            Self::Int32 | Self::Int | Self::Uint32 | Self::Float32 | Self::Float => 4,
            Self::Int64 | Self::Long => 8,
        }
    }

    /// Whether this dtype is floating-point (float or FP8/FP4 variants).
    pub const fn is_floating(self) -> bool {
        matches!(
            self,
            Self::Float32
                | Self::Float
                | Self::Float16
                | Self::Half
                | Self::BFloat16
                | Self::Float8E4M3FN
                | Self::Float8E5M2
                | Self::Float4E2M1FNx2
        )
    }

    /// Default preprocessing modes for this dtype. These are the wire values
    /// a compressor uses by default; callers may override by layering their
    /// own modes on top.
    pub const fn preprocessing_modes(self) -> PreprocessingModes {
        let (num_buf, bit_reorder, byte_reorder) = match self {
            Self::Float32 | Self::Float => (4, 1, 220),
            Self::BFloat16 => (2, 1, 10),
            Self::Float16 | Self::Half => (2, 0, 10),
            Self::Float8E4M3FN => (2, 0, 20),
            Self::Float8E5M2 => (2, 0, 22),
            Self::Float4E2M1FNx2 => (1, 0, 10),
            Self::Int8 | Self::Uint8 | Self::Bool => (1, 0, 10),
            Self::Int16 | Self::Short => (2, 0, 10),
            Self::Int32 | Self::Int | Self::Uint32 => (4, 0, 220),
            Self::Int64 | Self::Long => (8, 0, 10),
        };
        PreprocessingModes {
            num_buf,
            bit_reorder,
            byte_reorder,
        }
    }

    /// Recover `num_buf` for decompression given the stored `byte_reorder`.
    ///
    /// Handles two wire-format special cases that the dtype alone can't
    /// resolve:
    /// - `byte_reorder ∈ {20, 22}`: FP8 nibble-split, always 2 planes.
    /// - FP8 dtype with legacy `byte_reorder ∈ {10, 11, 12}`: pre-nibble-split
    ///   blobs used a single plane.
    pub fn num_buf_for_decompress(self, byte_reorder: i32) -> u32 {
        if matches!(byte_reorder, 20 | 22) {
            return 2;
        }
        if matches!(self, Self::Float8E4M3FN | Self::Float8E5M2) && matches!(byte_reorder, 10..=12)
        {
            return 1;
        }
        self.preprocessing_modes().num_buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_code() {
        for dt in [
            Dtype::Float32,
            Dtype::BFloat16,
            Dtype::Float16,
            Dtype::Int8,
            Dtype::Int32,
            Dtype::Int64,
            Dtype::Bool,
            Dtype::Float8E4M3FN,
            Dtype::Float8E5M2,
            Dtype::Float4E2M1FNx2,
        ] {
            assert_eq!(Dtype::from_code(dt.code()).unwrap(), dt);
        }
    }

    #[test]
    fn unsupported_code_errors() {
        assert!(Dtype::from_code(0).is_err()); // NONE
        assert!(Dtype::from_code(3).is_err()); // Float64 (no spec)
        assert!(Dtype::from_code(100).is_err());
    }

    #[test]
    fn from_name_accepts_aliases() {
        assert_eq!(Dtype::from_name("float32"), Some(Dtype::Float32));
        assert_eq!(Dtype::from_name("fp32"), Some(Dtype::Float32));
        assert_eq!(Dtype::from_name("f32"), Some(Dtype::Float32));
        assert_eq!(Dtype::from_name("FLOAT32"), Some(Dtype::Float32));
        assert_eq!(Dtype::from_name("bf16"), Some(Dtype::BFloat16));
        assert_eq!(Dtype::from_name("fp8-e4m3fn"), Some(Dtype::Float8E4M3FN));
        assert_eq!(Dtype::from_name("unknown"), None);
    }

    #[test]
    fn canonical_names_match_python_registry() {
        // Spot-check that canonical names match the Python registry strings.
        assert_eq!(Dtype::Float32.canonical_name(), "float32");
        assert_eq!(Dtype::BFloat16.canonical_name(), "bfloat16");
        assert_eq!(Dtype::Float8E4M3FN.canonical_name(), "float8_e4m3fn");
    }

    #[test]
    fn fp32_preprocessing_modes_match_plan() {
        let m = Dtype::Float32.preprocessing_modes();
        assert_eq!(m.num_buf, 4);
        assert_eq!(m.bit_reorder, 1);
        assert_eq!(m.byte_reorder, 220);
    }

    #[test]
    fn fp8_legacy_decompress_uses_single_plane() {
        // byte_reorder 10/11/12 on an FP8 blob pre-dates nibble split and
        // used num_buf=1.
        assert_eq!(Dtype::Float8E4M3FN.num_buf_for_decompress(10), 1);
        assert_eq!(Dtype::Float8E4M3FN.num_buf_for_decompress(11), 1);
        assert_eq!(Dtype::Float8E4M3FN.num_buf_for_decompress(12), 1);
        // 20/22 is the new nibble-split path.
        assert_eq!(Dtype::Float8E4M3FN.num_buf_for_decompress(20), 2);
        assert_eq!(Dtype::Float8E5M2.num_buf_for_decompress(22), 2);
    }

    #[test]
    fn element_sizes() {
        assert_eq!(Dtype::Int8.element_size(), 1);
        assert_eq!(Dtype::Float16.element_size(), 2);
        assert_eq!(Dtype::Float32.element_size(), 4);
        assert_eq!(Dtype::Int64.element_size(), 8);
    }

    #[test]
    fn is_floating_classification() {
        assert!(Dtype::Float32.is_floating());
        assert!(Dtype::BFloat16.is_floating());
        assert!(Dtype::Float8E4M3FN.is_floating());
        assert!(!Dtype::Int32.is_floating());
        assert!(!Dtype::Bool.is_floating());
    }
}
