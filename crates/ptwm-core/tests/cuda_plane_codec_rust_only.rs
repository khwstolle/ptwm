//! Decode through the CUDA plane-codec router from Rust alone.
//!
//! This test exists to pin a layering property: nothing on the decode
//! path may require the Python bindings. It links `ptwm-core` only.

use ptwm_core::flavor::PlaneCodecCudaRouter;

#[test]
fn cuda_plane_codec_router_is_constructible_without_python() {
    // Constructing and querying the router must not need an interpreter.
    let router = PlaneCodecCudaRouter::new(Vec::new());
    let id = ptwm_core::extension::builtin_canonical_id("identity");
    // Default policy denies CUDA, so this resolves to an error rather
    // than a panic -- the point is that it runs at all.
    assert!(router.get(&id).is_err());
}
