//! Opt-in context-mixing plane codec (cmix/lpaq lineage). Bit-level
//! prediction from byte-history models, context-gated logistic mixing,
//! binary range coding. Fully integer/deterministic; stateless on the wire.

pub mod coder;
pub mod mixer;
pub mod model;
pub mod squash;

pub use coder::{decode_plane, encode_plane};
