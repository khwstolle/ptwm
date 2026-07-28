//! Requires extensions/ref_hardware_backend built for native flavor first:
//!   cd extensions/ref_hardware_backend/rust && cargo build --release
//! Run with: cargo test -p ptwm-core --test hardware_backend_router_native -- --ignored
//!
//! This is the first test in the repo's history exercising the entire
//! `hardware_backend` dispatch path: router resolution, capability
//! check, dlopen, native-symbol probing, and an FFI call, against a
//! real compiled `.so`, not a mock.
//!
//! `ref_hardware_backend`'s manifest declares `hardware_class = "cpu"`.
//! `HardwareBackendRouter::new` gates resolution through the default
//! (empty-`available_hardware`) `HostPolicy`, which denies every
//! `hardware_class`-declaring contribution by design (see
//! `HardwareBackendRouter::resolve`'s doc comment in `router.rs`). This
//! test therefore uses `HardwareBackendRouter::new_with_policy` with a
//! policy that explicitly lists `"cpu"` in `available_hardware`,
//! mirroring what an operator's policy file would need to grant for this
//! contribution to load on a real host.

use ptwm_core::discovery::{DiscoveredContribution, FLAVOR_NATIVE};
use ptwm_core::extension::{CanonicalId, Manifest};
use ptwm_core::flavor::HardwareBackendRouter;
use ptwm_core::policy::HostPolicy;

/// Locate the compiled reference fixture's shared library.
///
/// Falls back to a nonexistent path (rather than panicking at collection
/// time) so `cargo test` can still enumerate this file's tests when the
/// fixture hasn't been built; the `assert!` inside the test itself gives
/// the actionable error message.
fn native_so_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../extensions/ref_hardware_backend/rust/target/release")
        .canonicalize()
        .ok()
        .and_then(|dir| {
            std::fs::read_dir(&dir).ok()?.find_map(|e| {
                let p = e.ok()?.path();
                let ext = p.extension()?.to_str()?;
                (ext == "so" || ext == "dylib").then_some(p)
            })
        })
        .unwrap_or_else(|| std::path::PathBuf::from("/nonexistent"))
}

/// Path to the fixture's bundle directory (where `manifest.toml` and the
/// compiled `.so` both live).
fn fixture_bundle_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../extensions/ref_hardware_backend/rust")
}

/// Load the real `ref_hardware_backend` manifest and wrap it in a
/// `DiscoveredContribution` mirroring what filesystem discovery would
/// produce for an installed bundle.
///
/// `manifest.toml` lives in `extensions/ref_hardware_backend/rust/`, but
/// the compiled `.so` lands in that crate's own `target/release/` (a
/// plain `cargo build` output layout, not the packaged
/// `<bundle_dir>/manifest.toml` + `<bundle_dir>/<name>.so` layout real
/// installed bundles use (see `discovery/mod.rs`'s module doc comment).
/// `DiscoveredContribution::bundle_dir` is what `find_native_path`
/// actually scans, so it must point at the directory holding the `.so`;
/// `manifest_path` stays informational and can point at the manifest's
/// real, separate location.
fn discover_fixture() -> DiscoveredContribution {
    let manifest_path = fixture_bundle_dir().join("manifest.toml");
    let manifest_src = std::fs::read_to_string(&manifest_path).unwrap_or_else(|e| {
        panic!("read {manifest_path:?}: {e} (build ref_hardware_backend first?)")
    });
    let manifest = Manifest::from_toml(&manifest_src)
        .unwrap_or_else(|e| panic!("parse {manifest_path:?}: {e:?}"));

    let bundle_dir = native_so_path()
        .parent()
        .unwrap_or_else(|| panic!("build ref_hardware_backend --release first"))
        .to_path_buf();

    DiscoveredContribution {
        manifest,
        manifest_path,
        bundle_dir,
        installed_flavors: FLAVOR_NATIVE,
    }
}

#[test]
#[ignore = "requires extensions/ref_hardware_backend built for native flavor first"]
fn native_hardware_backend_round_trips_through_real_dispatch_path() {
    let so_path = native_so_path();
    assert!(
        so_path.exists(),
        "build ref_hardware_backend --release first: {so_path:?}"
    );

    let install = discover_fixture();
    let canonical_id_str = install
        .manifest
        .contributions
        .first()
        .expect("ref_hardware_backend manifest declares one contribution")
        .id
        .clone();
    let canonical_id = CanonicalId::parse(&canonical_id_str)
        .unwrap_or_else(|e| panic!("parse canonical id {canonical_id_str:?}: {e:?}"));

    // The fixture declares hardware_class = "cpu"; grant exactly that
    // class so the capability check admits it, mirroring what an
    // operator's policy file would need to configure on a real host.
    // See this file's module doc comment for why HardwareBackendRouter::new
    // (the default-deny policy) cannot be used here.
    let policy = HostPolicy {
        available_hardware: vec!["cpu".into()],
        ..HostPolicy::default()
    };
    let router = HardwareBackendRouter::new_with_policy(vec![install], policy);

    let backend = router
        .get(&canonical_id)
        .expect("should resolve native flavor");

    let stream = backend
        .cuda_stream_handle(0)
        .expect("scaffold returns nonzero sentinel");
    assert_ne!(stream, 0);

    // The CPU-passthrough body does a real memcpy between two "device"
    // pointers that are actually host allocations here: allocate two
    // Vec<u8> buffers, pass their .as_ptr() as u64, and confirm the
    // output buffer received the input bytes unchanged.
    let input = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
    let mut output = vec![0u8; input.len()];
    let n = backend
        .dispatch_decode_cuda(
            &[],
            &canonical_id,
            input.as_ptr() as u64,
            input.len(),
            output.as_mut_ptr() as u64,
            output.len(),
            0,
        )
        .expect("dispatch should succeed");
    assert_eq!(n, input.len());
    assert_eq!(output, input);
}

#[test]
#[ignore = "requires extensions/ref_hardware_backend built for native flavor first"]
fn concurrent_first_access_resolves_exactly_once_and_all_threads_succeed() {
    let so_path = native_so_path();
    assert!(
        so_path.exists(),
        "build ref_hardware_backend --release first: {so_path:?}"
    );

    let install = discover_fixture();
    let canonical_id_str = install
        .manifest
        .contributions
        .first()
        .expect("ref_hardware_backend manifest declares one contribution")
        .id
        .clone();
    let canonical_id = CanonicalId::parse(&canonical_id_str)
        .unwrap_or_else(|e| panic!("parse canonical id {canonical_id_str:?}: {e:?}"));

    let policy = HostPolicy {
        available_hardware: vec!["cpu".into()],
        ..HostPolicy::default()
    };
    let router = std::sync::Arc::new(HardwareBackendRouter::new_with_policy(
        vec![install],
        policy,
    ));

    let handles: Vec<_> = (0..16)
        .map(|i| {
            let router = std::sync::Arc::clone(&router);
            let id = canonical_id.clone();
            std::thread::spawn(move || {
                let backend = router
                    .get(&id)
                    .expect("resolve under concurrent first access");
                let input = vec![i as u8; 64];
                let mut output = vec![0u8; 64];
                let n = backend
                    .dispatch_decode_cuda(
                        &[],
                        &id,
                        input.as_ptr() as u64,
                        input.len(),
                        output.as_mut_ptr() as u64,
                        output.len(),
                        0,
                    )
                    .expect("dispatch under concurrent access");
                assert_eq!(n, input.len());
                assert_eq!(output, input);
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread panicked");
    }
}

