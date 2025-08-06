//! PPG chain construction, wire encoding, validation, runtime.

pub mod dispatch;
pub mod graph;
pub mod runtime;
pub mod validate;
pub mod wire;

pub use graph::{Chain, ChainEdge, ChainNode, TerminalRef};
