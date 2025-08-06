//! Opt-in NNCP-style neural-predictor plane codec. A tiny online-learning
//! fixed-point MLP predicts each bit; a binary range coder writes the residual.
//! Fully integer/deterministic; stateless on the wire (the model trains from
//! scratch identically on both sides).

pub mod coder;
pub mod net;

pub use coder::{decode_plane, encode_plane};
