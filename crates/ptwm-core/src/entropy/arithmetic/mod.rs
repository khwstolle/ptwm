//! Continuous-precision arithmetic coding over plane bytes.
//!
//! A generic [`encode_bytes`]/[`decode_bytes`] driver runs any
//! [`model::ByteModel`] against the shared [`crate::range_coder`]. Four
//! models (order-0 static, order-0 adaptive, order-1 adaptive, order-1
//! static) are wrapped as peer plane codecs in `crate::codecs::arithmetic`
//! and `crate::codecs::order1_arithmetic`.

pub mod model;

pub use model::{
    ByteModel, Order0Adaptive, Order0Static, Order1Adaptive, Order1Static, decode_bytes,
    encode_bytes,
};
