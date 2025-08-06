//! PPG type system: plane descriptors and supporting types.

pub mod descriptor;
pub mod plane_role;
pub mod role;

pub use descriptor::{ElementWidth, Layout, PlaneDescriptor, PlaneRef, TensorRef};
pub use plane_role::PlaneRole;
pub use role::{NibbleKind, ResidualFormat, Role, ScaleFormat, ValueFormat};
