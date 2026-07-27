//! `PlaneCodecRouter`: canonical-id-keyed dispatch for both built-in and
//! third-party plane codecs.
//!
//! # Dispatch order
//!
//! 1. **Built-ins** — resolved via [`dispatch_builtin`] and wrapped in
//!    `BuiltinAdapter`.
//! 2. **Native** (`.so` / `.dylib` / `.dll`) — resolved by scanning `bundle_dir`
//!    for a native artifact, then dlopen-loaded via [`NativeExtension`].
//! 3. **WASM** — resolved by scanning `bundle_dir` for a `.wasm` artifact, then
//!    loaded via [`WasmExtension`].
//! 4. **Host** (Python) — not supported in this Rust-only router; returns
//!    [`CodecError::Unsupported`] with a clear message.
//!
//! The router caches resolved codecs by `CanonicalId` so repeated calls for the
//! same id are free after the first lookup.
//!
//! # Wire-format note
//!
//! `PlaneRecord` now carries both `codec_id: CodecId` (closed enum, legacy)
//! and `codec_table_idx: u16` (Extension Table index, C2 wire-format addition).
//! When `codec_table_idx` is parsed, `decode_tensor` in
//! `container.rs` resolves the `CanonicalId` via the Extension Table and
//! validates availability through this router before dispatching the actual
//! decode to the in-tree `PlaneCodec` implementation (for built-ins) or, for
//! third-party codecs in a future task, to a dynamically-loaded extension.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::discovery::DiscoveredContribution;
use crate::extension::{
    Attestation, BuiltinKind, CanonicalId, ExtensionTableEntry, dispatch_builtin,
};
use crate::flavor::abi::CodecError;
use crate::flavor::native::{NativeExtension, VerifiedToken};
use crate::flavor::wasm::{
    WasmExtension, invoke_delta_scheme_decode, invoke_delta_scheme_encode,
    invoke_plane_codec_decode, invoke_plane_codec_decode_stateful, invoke_plane_codec_encode,
};
use crate::policy::capability_check::HostPolicy;

/// Uniform encode/decode surface exposed by the router.
///
/// The buffer ABI mirrors the flat ABI used by native and WASM extensions:
/// input is a byte slice, output is caller-allocated. Returns the number of
/// bytes written on success.
///
/// `decode_with_state` is the state-aware variant used by the third-party
/// branch of the container decode loop. Stateless codecs can ignore the
/// state parameters; stateful codecs (per-group codebook, predictive
/// scale coder, …) need them to round-trip planes whose state was
/// captured in `PlaneRecord.{inline,shared}_state_bytes`.
pub trait DispatchedPlaneCodec: Send + Sync {
    fn encode(&self, input: &[u8], output: &mut [u8]) -> Result<usize, CodecError>;
    fn decode(&self, input: &[u8], output: &mut [u8]) -> Result<usize, CodecError>;

    /// Decode `input` using the format-versioned `state_bytes` previously
    /// emitted by the matching encode call.
    ///
    /// Default implementation forwards to the stateless [`Self::decode`]
    /// and ignores the state — appropriate for codecs that don't need
    /// per-call state. Stateful codecs override this.
    fn decode_with_state(
        &self,
        _state_version: u8,
        _state_bytes: &[u8],
        input: &[u8],
        output: &mut [u8],
    ) -> Result<usize, CodecError> {
        self.decode(input, output)
    }
}

// ── Built-in adapter ──────────────────────────────────────────────────────────

/// Wraps a `Box<dyn crate::codec::PlaneCodec>` (the in-tree trait) in the
/// `DispatchedPlaneCodec` surface.
///
/// The in-tree codec's `encode` and `decode` signatures differ from the flat
/// ABI: they take structured arguments and return a `Vec<u8>`. The adapter
/// calls them with no shared state / `PlaneLayout::Flat` defaults, which is
/// correct for the router's simple encode/decode contract.
struct BuiltinAdapter {
    inner: Box<dyn crate::codec::PlaneCodec>,
}

impl DispatchedPlaneCodec for BuiltinAdapter {
    fn encode(&self, input: &[u8], output: &mut [u8]) -> Result<usize, CodecError> {
        let encoded = self
            .inner
            .encode(input, None, &crate::layout::PlaneLayout::Flat)
            .map_err(|_| CodecError::InternalError)?;
        let n = encoded.payload.len();
        if n > output.len() {
            return Err(CodecError::BufferTooSmall { needed: n as u64 });
        }
        output[..n].copy_from_slice(&encoded.payload);
        Ok(n)
    }

    fn decode(&self, input: &[u8], output: &mut [u8]) -> Result<usize, CodecError> {
        self.decode_with_state(0, &[], input, output)
    }

    fn decode_with_state(
        &self,
        state_version: u8,
        state_bytes: &[u8],
        input: &[u8],
        output: &mut [u8],
    ) -> Result<usize, CodecError> {
        let decoded = self
            .inner
            .decode(
                state_version,
                state_bytes,
                input,
                &crate::layout::PlaneLayout::Flat,
                output.len(),
            )
            .map_err(|_| CodecError::InternalError)?;
        let n = decoded.len();
        if n > output.len() {
            return Err(CodecError::BufferTooSmall { needed: n as u64 });
        }
        output[..n].copy_from_slice(&decoded);
        Ok(n)
    }
}

// ── Native adapter ────────────────────────────────────────────────────────────

/// Wraps an `Arc<NativeExtension>`.
struct NativeAdapter {
    inner: Arc<NativeExtension>,
}

impl DispatchedPlaneCodec for NativeAdapter {
    fn encode(&self, input: &[u8], output: &mut [u8]) -> Result<usize, CodecError> {
        self.inner.invoke_plane_codec_encode(input, output)
    }

    fn decode(&self, input: &[u8], output: &mut [u8]) -> Result<usize, CodecError> {
        self.inner.invoke_plane_codec_decode(input, output)
    }

    fn decode_with_state(
        &self,
        state_version: u8,
        state_bytes: &[u8],
        input: &[u8],
        output: &mut [u8],
    ) -> Result<usize, CodecError> {
        self.inner
            .invoke_plane_codec_decode_stateful(state_version, state_bytes, input, output)
    }
}

// ── WASM adapter ──────────────────────────────────────────────────────────────

/// Wraps a `WasmExtension`. A fresh store + instance is created per call
/// (matching the Wasmtime lifecycle model where each invocation spins up its
/// own store so fuel is reset).
struct WasmAdapter {
    ext: WasmExtension,
}

impl DispatchedPlaneCodec for WasmAdapter {
    fn encode(&self, input: &[u8], output: &mut [u8]) -> Result<usize, CodecError> {
        let policy = HostPolicy::default();
        let mut store = self.ext.make_store(&policy)?;
        let instance = self.ext.instantiate(&mut store)?;
        invoke_plane_codec_encode(&instance, &mut store, input, output)
    }

    fn decode(&self, input: &[u8], output: &mut [u8]) -> Result<usize, CodecError> {
        let policy = HostPolicy::default();
        let mut store = self.ext.make_store(&policy)?;
        let instance = self.ext.instantiate(&mut store)?;
        invoke_plane_codec_decode(&instance, &mut store, input, output)
    }

    fn decode_with_state(
        &self,
        state_version: u8,
        state_bytes: &[u8],
        input: &[u8],
        output: &mut [u8],
    ) -> Result<usize, CodecError> {
        let policy = HostPolicy::default();
        let mut store = self.ext.make_store(&policy)?;
        let instance = self.ext.instantiate(&mut store)?;
        invoke_plane_codec_decode_stateful(
            &instance,
            &mut store,
            state_version,
            state_bytes,
            input,
            output,
        )
    }
}

// ── Router ────────────────────────────────────────────────────────────────────

/// Resolves a `CanonicalId` to a `Box<dyn DispatchedPlaneCodec>`.
///
/// Construct once per decode session (or shared) and call [`Self::get`] for
/// each plane. Results are internally cached so expensive dynamic loads
/// happen at most once per id.
pub struct PlaneCodecRouter {
    installed: Vec<DiscoveredContribution>,
    cache: Mutex<HashMap<CanonicalId, Arc<dyn DispatchedPlaneCodec>>>,
}

impl std::fmt::Debug for PlaneCodecRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlaneCodecRouter")
            .field("installed_count", &self.installed.len())
            .finish_non_exhaustive()
    }
}

impl PlaneCodecRouter {
    /// Create a new router with the given set of discovered installed
    /// extensions. Pass an empty `Vec` to restrict dispatch to built-ins only.
    pub fn new(installed: Vec<DiscoveredContribution>) -> Self {
        Self {
            installed,
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Resolve `canonical_id` to a `DispatchedPlaneCodec`.
    ///
    /// Returns:
    /// - `Ok(codec)` for built-in codecs and successfully loaded extensions.
    /// - `Err(CodecError::Unsupported)` when the id is recognized as a built-in
    ///   *op* (transform) rather than a codec, or when no flavor can be loaded.
    /// - `Err(CodecError::Unsupported)` for completely unknown ids.
    pub fn get(
        &self,
        canonical_id: &CanonicalId,
    ) -> Result<Arc<dyn DispatchedPlaneCodec>, CodecError> {
        // Fast path: cache hit.
        {
            let cache = self.cache.lock().unwrap();
            if let Some(codec) = cache.get(canonical_id) {
                return Ok(Arc::clone(codec));
            }
        }

        // Slow path: resolve, insert, return.
        let codec = self.resolve(canonical_id)?;
        let arc = Arc::from(codec);
        {
            let mut cache = self.cache.lock().unwrap();
            cache.insert(*canonical_id, Arc::clone(&arc));
        }
        Ok(arc)
    }

    fn resolve(
        &self,
        canonical_id: &CanonicalId,
    ) -> Result<Box<dyn DispatchedPlaneCodec>, CodecError> {
        // 1. Try built-ins first.
        if let Some(kind) = dispatch_builtin(canonical_id) {
            return match kind {
                BuiltinKind::Codec(codec_id) => {
                    let inner =
                        crate::codec::codec_for(codec_id).ok_or(CodecError::Unsupported {
                            feature: format!("built-in codec {codec_id:?} has no impl"),
                        })?;
                    Ok(Box::new(BuiltinAdapter { inner }))
                }
                BuiltinKind::Op(_) => Err(CodecError::Unsupported {
                    feature: format!(
                        "canonical id {canonical_id} is a transform op, not a plane codec"
                    ),
                }),
            };
        }

        // 2. Search installed extensions.
        if let Some(install) = find_install(canonical_id, &self.installed) {
            // Prefer native > wasm. Host is not supported in this router.
            let bundle_dir = &install.bundle_dir;

            if let Ok(native_path) = find_native_path(bundle_dir) {
                let entry = manifest_entry_for(install, canonical_id)?;
                let token = VerifiedToken::new_unchecked();
                let ext = NativeExtension::load(&native_path, &entry, token)?;
                let adapter = NativeAdapter {
                    inner: Arc::new(ext),
                };
                return Ok(Box::new(adapter));
            }

            if let Ok(wasm_path) = find_wasm_path(bundle_dir) {
                let entry = manifest_entry_for(install, canonical_id)?;
                let engine = WasmExtension::make_engine();
                let ext = WasmExtension::load_from_path(&engine, &wasm_path, &entry)?;
                let adapter = WasmAdapter { ext };
                return Ok(Box::new(adapter));
            }

            // Installed but no loadable flavor available.
            return Err(CodecError::Unsupported {
                feature: format!(
                    "extension for {canonical_id} is installed but no native or wasm \
                     artifact was found in {}",
                    bundle_dir.display()
                ),
            });
        }

        // 3. Unknown id.
        Err(CodecError::Unsupported {
            feature: format!("no codec registered for canonical id {canonical_id}"),
        })
    }
}

// ── DeltaScheme router ──────────────────────────────────────────────────────

/// Uniform encode/decode surface for a resolved `delta_scheme` contribution.
///
/// Mirrors [`DispatchedPlaneCodec`], but over the two-buffer delta ABI:
/// `encode(base, target) -> delta`, `decode(base, delta) -> target`.
pub trait DispatchedDeltaScheme: Send + Sync {
    fn encode(&self, base: &[u8], target: &[u8], output: &mut [u8]) -> Result<usize, CodecError>;
    fn decode(&self, base: &[u8], delta: &[u8], output: &mut [u8]) -> Result<usize, CodecError>;
}

struct NativeDeltaSchemeAdapter {
    inner: Arc<NativeExtension>,
}

impl DispatchedDeltaScheme for NativeDeltaSchemeAdapter {
    fn encode(&self, base: &[u8], target: &[u8], output: &mut [u8]) -> Result<usize, CodecError> {
        self.inner.invoke_delta_scheme_encode(base, target, output)
    }

    fn decode(&self, base: &[u8], delta: &[u8], output: &mut [u8]) -> Result<usize, CodecError> {
        self.inner.invoke_delta_scheme_decode(base, delta, output)
    }
}

struct WasmDeltaSchemeAdapter {
    ext: WasmExtension,
}

impl DispatchedDeltaScheme for WasmDeltaSchemeAdapter {
    fn encode(&self, base: &[u8], target: &[u8], output: &mut [u8]) -> Result<usize, CodecError> {
        let policy = HostPolicy::default();
        let mut store = self.ext.make_store(&policy)?;
        let instance = self.ext.instantiate(&mut store)?;
        invoke_delta_scheme_encode(&instance, &mut store, base, target, output)
    }

    fn decode(&self, base: &[u8], delta: &[u8], output: &mut [u8]) -> Result<usize, CodecError> {
        let policy = HostPolicy::default();
        let mut store = self.ext.make_store(&policy)?;
        let instance = self.ext.instantiate(&mut store)?;
        invoke_delta_scheme_decode(&instance, &mut store, base, delta, output)
    }
}

/// Resolves a `CanonicalId` to a `Box<dyn DispatchedDeltaScheme>`.
///
/// Same shape as [`PlaneCodecRouter`], minus the built-in step: `ptwm`
/// ships zero built-in `delta_scheme` contributions today, so resolution
/// goes straight to the installed-extension search (native preferred over
/// WASM). Results are cached by `CanonicalId`, same as `PlaneCodecRouter`.
pub struct DeltaSchemeRouter {
    installed: Vec<DiscoveredContribution>,
    cache: Mutex<HashMap<CanonicalId, Arc<dyn DispatchedDeltaScheme>>>,
}

impl std::fmt::Debug for DeltaSchemeRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeltaSchemeRouter")
            .field("installed_count", &self.installed.len())
            .finish_non_exhaustive()
    }
}

impl DeltaSchemeRouter {
    /// Create a new router with the given set of discovered installed
    /// extensions. Pass an empty `Vec` to get `Unsupported` for every id
    /// (there are no built-in delta schemes to fall back to).
    pub fn new(installed: Vec<DiscoveredContribution>) -> Self {
        Self {
            installed,
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Resolve `canonical_id` to a `DispatchedDeltaScheme`.
    pub fn get(
        &self,
        canonical_id: &CanonicalId,
    ) -> Result<Arc<dyn DispatchedDeltaScheme>, CodecError> {
        {
            let cache = self.cache.lock().unwrap();
            if let Some(scheme) = cache.get(canonical_id) {
                return Ok(Arc::clone(scheme));
            }
        }

        let scheme = self.resolve(canonical_id)?;
        let arc = Arc::from(scheme);
        {
            let mut cache = self.cache.lock().unwrap();
            cache.insert(*canonical_id, Arc::clone(&arc));
        }
        Ok(arc)
    }

    fn resolve(
        &self,
        canonical_id: &CanonicalId,
    ) -> Result<Box<dyn DispatchedDeltaScheme>, CodecError> {
        if let Some(install) = find_install(canonical_id, &self.installed) {
            let bundle_dir = &install.bundle_dir;

            if let Ok(native_path) = find_native_path(bundle_dir) {
                let entry = manifest_entry_for(install, canonical_id)?;
                let token = VerifiedToken::new_unchecked();
                let ext = NativeExtension::load(&native_path, &entry, token)?;
                let adapter = NativeDeltaSchemeAdapter {
                    inner: Arc::new(ext),
                };
                return Ok(Box::new(adapter));
            }

            if let Ok(wasm_path) = find_wasm_path(bundle_dir) {
                let entry = manifest_entry_for(install, canonical_id)?;
                let engine = WasmExtension::make_engine();
                let ext = WasmExtension::load_from_path(&engine, &wasm_path, &entry)?;
                let adapter = WasmDeltaSchemeAdapter { ext };
                return Ok(Box::new(adapter));
            }

            return Err(CodecError::Unsupported {
                feature: format!(
                    "extension for {canonical_id} is installed but no native or wasm \
                     artifact was found in {}",
                    bundle_dir.display()
                ),
            });
        }

        Err(CodecError::Unsupported {
            feature: format!("no delta_scheme registered for canonical id {canonical_id}"),
        })
    }
}

// ── HardwareBackend router ─────────────────────────────────────────────────────

/// Uniform decode surface for GPU-accelerated decode operations.
///
/// Provides two methods: `dispatch_decode_cuda` for decoding on CUDA devices,
/// and `cuda_stream_handle` to acquire a stream for a given device ordinal.
pub trait DispatchedHardwareBackendCuda: Send + Sync {
    fn dispatch_decode_cuda(
        &self,
        state_bytes: &[u8],
        codec_id: &CanonicalId,
        in_dev_ptr: u64,
        in_len: usize,
        out_dev_ptr: u64,
        out_len: usize,
        device_ordinal: u32,
    ) -> Result<usize, CodecError>;

    fn cuda_stream_handle(&self, device_ordinal: u32) -> Result<u64, CodecError>;
}

struct NativeHardwareBackendCudaAdapter {
    inner: Arc<NativeExtension>,
}

impl DispatchedHardwareBackendCuda for NativeHardwareBackendCudaAdapter {
    fn dispatch_decode_cuda(
        &self,
        state_bytes: &[u8],
        codec_id: &CanonicalId,
        in_dev_ptr: u64,
        in_len: usize,
        out_dev_ptr: u64,
        out_len: usize,
        device_ordinal: u32,
    ) -> Result<usize, CodecError> {
        self.inner.invoke_hardware_backend_dispatch_decode_cuda(
            state_bytes, codec_id, in_dev_ptr, in_len, out_dev_ptr, out_len, device_ordinal,
        )
    }

    fn cuda_stream_handle(&self, device_ordinal: u32) -> Result<u64, CodecError> {
        self.inner.invoke_hardware_backend_cuda_stream_handle(device_ordinal)
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Find the `DiscoveredContribution` whose manifest declares a contribution
/// matching `canonical_id`.
///
/// The manifest's `ContributionDecl.id` field is `"blake3:<hex>"`. We parse
/// the 32 hex bytes and compare against `canonical_id`.
fn find_install<'a>(
    canonical_id: &CanonicalId,
    installed: &'a [DiscoveredContribution],
) -> Option<&'a DiscoveredContribution> {
    for install in installed {
        for contrib in &install.manifest.contributions {
            if let Some(id) = parse_canonical_id_str(&contrib.id) {
                if &id == canonical_id {
                    return Some(install);
                }
            }
        }
    }
    None
}

/// Parse a `"blake3:<64-hex-chars>"` string into a `CanonicalId`.
fn parse_canonical_id_str(s: &str) -> Option<CanonicalId> {
    let hex = s.strip_prefix("blake3:")?;
    let bytes = decode_hex32(hex)?;
    Some(CanonicalId::from_bytes(bytes))
}

/// Scan `dir` for a native shared library (`.so` / `.dylib` / `.dll`) and
/// return the first one found.
pub fn find_native_path(dir: &Path) -> Result<PathBuf, CodecError> {
    for entry in std::fs::read_dir(dir)
        .map_err(|_| CodecError::InvalidInput)?
        .flatten()
    {
        let path = entry.path();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if matches!(ext, "so" | "dylib" | "dll") {
                return Ok(path);
            }
        }
    }
    Err(CodecError::Unsupported {
        feature: format!("no native artifact in {}", dir.display()),
    })
}

/// Scan `dir` for a WASM module (`.wasm`) and return the first one found.
pub fn find_wasm_path(dir: &Path) -> Result<PathBuf, CodecError> {
    for entry in std::fs::read_dir(dir)
        .map_err(|_| CodecError::InvalidInput)?
        .flatten()
    {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("wasm") {
            return Ok(path);
        }
    }
    Err(CodecError::Unsupported {
        feature: format!("no wasm artifact in {}", dir.display()),
    })
}

/// Synthesize a minimal `ExtensionTableEntry` from a `DiscoveredContribution`
/// and a target `CanonicalId`.
///
/// The entry is used only to satisfy the `WasmExtension::load_from_path` /
/// `NativeExtension::load` constructors; fields that are not required by those
/// constructors are filled with safe defaults.
pub fn manifest_entry_for(
    install: &DiscoveredContribution,
    canonical_id: &CanonicalId,
) -> Result<ExtensionTableEntry, CodecError> {
    // Find the matching ContributionDecl.
    let decl = install
        .manifest
        .contributions
        .iter()
        .find(|c| parse_canonical_id_str(&c.id).as_ref() == Some(canonical_id))
        .ok_or_else(|| CodecError::Unsupported {
            feature: format!(
                "contribution {canonical_id} not found in manifest {:?}",
                install.manifest_path
            ),
        })?;

    let flavor_hints = {
        let mut h = 0u8;
        for f in &decl.flavors {
            match f.as_str() {
                "wasm" => h |= crate::extension::table::FLAVOR_WASM,
                "native" => h |= crate::extension::table::FLAVOR_NATIVE,
                "host" => h |= crate::extension::table::FLAVOR_HOST,
                _ => {}
            }
        }
        h
    };

    Ok(ExtensionTableEntry {
        canonical_id: *canonical_id,
        human_label: decl.label.clone(),
        kind: decl.kind,
        abi_version: decl.abi_version,
        lifecycle: decl.lifecycle,
        flavor_hints,
        capabilities: decl.capabilities.clone(),
        attestation: Attestation::PgpSignature(Vec::new()),
        install_hint: decl.install_hint.clone(),
        embedded_wasm_offset: None,
        embedded_wasm_length: None,
    })
}

/// Decode a 64-character hex string into a 32-byte array. Returns `None` on
/// any parse error (wrong length or non-hex character).
pub fn decode_hex32(hex: &str) -> Option<[u8; 32]> {
    if hex.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, chunk) in hex.as_bytes().chunks(2).enumerate() {
        let hi = hex_nibble(chunk[0])?;
        let lo = hex_nibble(chunk[1])?;
        out[i] = (hi << 4) | lo;
    }
    Some(out)
}

fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extension::{CanonicalId, builtin_canonical_id};

    #[test]
    fn router_returns_codec_for_builtin_identity() {
        let router = PlaneCodecRouter::new(Vec::new());
        let id = builtin_canonical_id("identity");
        let codec = router.get(&id).expect("identity is a built-in plane codec");
        let input = b"hello";
        let mut output = vec![0u8; 64];
        let n = codec.encode(input, &mut output).unwrap();
        assert_eq!(&output[..n], input);
    }

    #[test]
    fn router_identity_roundtrips() {
        let router = PlaneCodecRouter::new(Vec::new());
        let id = builtin_canonical_id("identity");
        let codec = router.get(&id).unwrap();
        let input = b"roundtrip test data";
        let mut encoded = vec![0u8; 128];
        let n = codec.encode(input, &mut encoded).unwrap();
        let mut decoded = vec![0u8; 128];
        let m = codec.decode(&encoded[..n], &mut decoded).unwrap();
        assert_eq!(&decoded[..m], input);
    }

    #[test]
    fn router_errors_for_unknown_canonical_id() {
        let router = PlaneCodecRouter::new(Vec::new());
        let id = CanonicalId::from_bytes([0xCCu8; 32]);
        assert!(
            matches!(router.get(&id), Err(CodecError::Unsupported { .. })),
            "expected Unsupported for unknown canonical id"
        );
    }

    #[test]
    fn router_errors_for_op_canonical_id() {
        // BitReorderIeee32 is a transform, not a plane codec.
        let router = PlaneCodecRouter::new(Vec::new());
        let id = crate::transforms::op::OpId::BitReorderIeee32.canonical_id();
        assert!(
            matches!(router.get(&id), Err(CodecError::Unsupported { .. })),
            "expected Unsupported for transform op canonical id"
        );
    }

    #[test]
    fn router_caches_builtin_across_calls() {
        let router = PlaneCodecRouter::new(Vec::new());
        let id = builtin_canonical_id("huffman");
        let c1 = router.get(&id).unwrap();
        let c2 = router.get(&id).unwrap();
        // Arc pointer equality — both must be the same cached allocation.
        assert!(Arc::ptr_eq(&c1, &c2));
    }

    #[test]
    fn all_builtin_plane_codecs_resolve() {
        let router = PlaneCodecRouter::new(Vec::new());
        for name in [
            "identity",
            "huffman",
            "rans",
            "zstd",
            "per_group_codebook",
            "order1_scale_ac",
        ] {
            let id = builtin_canonical_id(name);
            assert!(
                router.get(&id).is_ok(),
                "built-in codec '{name}' failed to resolve"
            );
        }
    }

    #[test]
    fn decode_hex32_roundtrips() {
        let bytes = [0xABu8; 32];
        let hex: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
        let decoded = decode_hex32(&hex).unwrap();
        assert_eq!(decoded, bytes);
    }

    #[test]
    fn decode_hex32_rejects_bad_length() {
        assert!(decode_hex32("ab").is_none());
        assert!(decode_hex32("").is_none());
    }

    #[test]
    fn parse_canonical_id_str_accepts_valid() {
        let id = CanonicalId::from_bytes([0x11u8; 32]);
        let hex: String = [0x11u8; 32].iter().map(|b| format!("{:02x}", b)).collect();
        let s = format!("blake3:{}", hex);
        let parsed = parse_canonical_id_str(&s).unwrap();
        assert_eq!(&parsed, &id);
    }

    #[test]
    fn parse_canonical_id_str_rejects_wrong_prefix() {
        assert!(parse_canonical_id_str("sha256:aabb").is_none());
        assert!(parse_canonical_id_str("aabbcc").is_none());
    }

    #[test]
    fn delta_scheme_router_errors_for_unknown_canonical_id() {
        let router = DeltaSchemeRouter::new(Vec::new());
        let id = CanonicalId::from_bytes([0xEEu8; 32]);
        assert!(
            matches!(router.get(&id), Err(CodecError::Unsupported { .. })),
            "expected Unsupported for unknown delta_scheme canonical id"
        );
    }

    #[test]
    fn native_hardware_backend_adapter_forwards_to_extension() {
        fn assert_impl<T: DispatchedHardwareBackendCuda>() {}
        assert_impl::<NativeHardwareBackendCudaAdapter>();
    }
}
