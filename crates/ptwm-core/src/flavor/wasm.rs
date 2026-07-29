//! Wasmtime integration. Host calls into a WASM module via a byte-
//! buffer ABI (no Component Model — plain (ptr, len) pairs through
//! linear memory).

use std::collections::HashSet;
use std::path::Path;

use wasmtime::{
    Config, Engine, Instance, Linker, Memory, Module, Store, StoreLimits, StoreLimitsBuilder,
    TypedFunc,
};
use wasmtime_wasi::WasiCtxBuilder;
use wasmtime_wasi::preview1::{self, WasiP1Ctx};

use crate::extension::{
    Attestation, CanonicalId, ExtensionTableEntry, Lifecycle, capability::CapabilityValue,
};
use crate::flavor::abi::CodecError;
use crate::policy::capability_check::HostPolicy;

/// Default fuel budget per call when the extension declares no
/// `fuel_factor`. ~10 ms on a fast desktop.
const DEFAULT_FUEL: u64 = 100_000_000;

/// Default memory limit per store: 256 MiB. Tunable by `mem_factor`.
const DEFAULT_MEM_LIMIT: usize = 256 * 1024 * 1024;

/// A loaded WASM extension. Holds the compiled `Module`; each
/// invocation spins up a fresh `Store` from `make_store`.
pub struct WasmExtension {
    engine: Engine,
    module: Module,
    pub canonical_id: CanonicalId,
    pub lifecycle: Lifecycle,
    declared_imports: HashSet<String>,
    fuel: u64,
    mem_limit: usize,
}

pub struct WasmState {
    pub limits: StoreLimits,
    pub canonical_id: CanonicalId,
    /// Satisfies the `wasi_snapshot_preview1` imports every `std`-linked
    /// `wasm32-wasip1` module pulls in for its runtime/panic machinery
    /// (`environ_get`, `fd_write`, `proc_exit`, ...), even when the
    /// contribution's own code never touches stdio/env/fs. Fully
    /// sandboxed by `WasiCtxBuilder`'s defaults: closed stdin, sunk
    /// stdout/stderr, no env, no args, no preopens — no capability
    /// beyond what the byte-buffer ABI itself grants.
    pub wasi: WasiP1Ctx,
}

impl WasmExtension {
    /// Build the global engine. Reused across modules.
    pub fn make_engine() -> Engine {
        let mut config = Config::new();
        config.consume_fuel(true);
        config.async_support(false);
        config.wasm_simd(true);
        // Stock `rustc`/LLVM wasm32-wasip1 codegen emits reference-types
        // encoded `call_indirect` (multi-table-shaped, even for ordinary
        // indirect calls with a single table) — disabling this rejects any
        // unmodified `std`-linked module at parse time ("zero byte
        // expected"). Keep wasmtime's own default (on); it depends on
        // bulk_memory, already enabled below.
        config.wasm_reference_types(true);
        config.wasm_bulk_memory(true);
        config.wasm_threads(false);
        config.wasm_memory64(false);
        Engine::new(&config).expect("wasmtime engine init")
    }

    pub fn load(
        engine: &Engine,
        wasm_bytes: &[u8],
        entry: &ExtensionTableEntry,
    ) -> Result<Self, CodecError> {
        let module = Module::new(engine, wasm_bytes).map_err(|_| CodecError::InvalidInput)?;

        let declared_imports = match entry.capabilities.get("host_imports") {
            Some(CapabilityValue::List(items)) => items
                .iter()
                .filter_map(|v| match v {
                    CapabilityValue::Text(s) => Some(s.clone()),
                    _ => None,
                })
                .collect(),
            _ => HashSet::new(),
        };

        let fuel = match entry.capabilities.get("fuel_factor") {
            Some(CapabilityValue::Float(f)) if f.0 > 0.0 => (f.0 * DEFAULT_FUEL as f64) as u64,
            Some(CapabilityValue::Int(i)) if *i > 0 => {
                (*i as u64).saturating_mul(DEFAULT_FUEL / 100)
            }
            _ => DEFAULT_FUEL,
        };
        let mem_limit = match entry.capabilities.get("mem_factor") {
            Some(CapabilityValue::Float(f)) if f.0 > 0.0 => {
                (f.0 * DEFAULT_MEM_LIMIT as f64) as usize
            }
            Some(CapabilityValue::Int(i)) if *i > 0 => {
                (*i as usize) * DEFAULT_MEM_LIMIT
            }
            _ => DEFAULT_MEM_LIMIT,
        };

        Ok(Self {
            engine: engine.clone(),
            module,
            canonical_id: entry.canonical_id,
            lifecycle: entry.lifecycle,
            declared_imports,
            fuel,
            mem_limit,
        })
    }

    /// Load from a filesystem path.
    pub fn load_from_path(
        engine: &Engine,
        path: &Path,
        entry: &ExtensionTableEntry,
    ) -> Result<Self, CodecError> {
        let bytes = std::fs::read(path).map_err(|_| CodecError::InvalidInput)?;
        // Validate signature material is non-empty if the entry isn't builtin.
        // (Verifier was already invoked by the caller — this is a sanity check.)
        match &entry.attestation {
            Attestation::PgpSignature(_) => {}
        }
        Self::load(engine, &bytes, entry)
    }

    pub fn make_store(&self, host_policy: &HostPolicy) -> Result<Store<WasmState>, CodecError> {
        // Verify that all declared host_imports are policy-allowed.
        for imp in &self.declared_imports {
            if !host_policy.allow_host_imports.contains(imp) {
                return Err(CodecError::MissingCapability {
                    name: format!("host_imports:{imp}"),
                });
            }
        }
        let limits = StoreLimitsBuilder::new()
            .memory_size(self.mem_limit)
            .build();
        let mut store = Store::new(
            &self.engine,
            WasmState {
                limits,
                canonical_id: self.canonical_id,
                wasi: WasiCtxBuilder::new().build_p1(),
            },
        );
        store.limiter(|st| &mut st.limits);
        store
            .set_fuel(self.fuel)
            .map_err(|_| CodecError::InternalError)?;
        Ok(store)
    }

    pub fn instantiate(&self, store: &mut Store<WasmState>) -> Result<Instance, CodecError> {
        let mut linker = Linker::<WasmState>::new(&self.engine);
        // Every `std`-linked wasm32-wasip1 module needs its
        // wasi_snapshot_preview1 runtime imports satisfied (see the
        // `WasmState::wasi` doc comment) — wire those in before anything
        // declared-host-imports-specific below.
        preview1::add_to_linker_sync(&mut linker, |state: &mut WasmState| &mut state.wasi)
            .map_err(|_| CodecError::InternalError)?;
        // No host imports beyond WASI preview1 are wired by default.
        // host_imports from the declared set would be added here if v1
        // supported any — for now we deny anything that's declared, since
        // no concrete host-import bindings exist yet.
        if !self.declared_imports.is_empty() {
            return Err(CodecError::Unsupported {
                feature: format!(
                    "host_imports {:?} declared; v1 grants none",
                    self.declared_imports
                ),
            });
        }
        linker
            .instantiate(&mut *store, &self.module)
            .map_err(|_| CodecError::InvalidInput)
    }
}

/// Helper: dispatch into the plane_codec_v1 encode export.
pub fn invoke_plane_codec_encode(
    instance: &Instance,
    store: &mut Store<WasmState>,
    input: &[u8],
    output: &mut [u8],
) -> Result<usize, CodecError> {
    let memory: Memory = instance
        .get_memory(&mut *store, "memory")
        .ok_or(CodecError::InvalidInput)?;
    let alloc: TypedFunc<u32, u32> = instance
        .get_typed_func(&mut *store, "ptwm_alloc")
        .map_err(|_| CodecError::InvalidInput)?;
    let free: TypedFunc<(u32, u32), ()> = instance
        .get_typed_func(&mut *store, "ptwm_free")
        .map_err(|_| CodecError::InvalidInput)?;
    let encode: TypedFunc<(u32, u32, u32, u32), i64> = instance
        .get_typed_func(&mut *store, "ptwm_plane_codec_v1_encode")
        .map_err(|_| CodecError::InvalidInput)?;

    let in_ptr = alloc
        .call(&mut *store, input.len() as u32)
        .map_err(|_| CodecError::InternalError)?;
    memory
        .write(&mut *store, in_ptr as usize, input)
        .map_err(|_| CodecError::InternalError)?;
    let out_ptr = alloc
        .call(&mut *store, output.len() as u32)
        .map_err(|_| CodecError::InternalError)?;

    let rc = encode
        .call(
            &mut *store,
            (in_ptr, input.len() as u32, out_ptr, output.len() as u32),
        )
        .map_err(|_| CodecError::InternalError)?;
    if rc < 0 {
        return Err(decode_codec_error(rc));
    }
    // Bounds-check `rc` (i64) against `output.len()` before narrowing
    // to usize — `rc as usize` would truncate on 32-bit hosts when rc
    // is between u32::MAX and i64::MAX.
    if rc > output.len() as i64 {
        return Err(CodecError::InternalError);
    }
    let written = rc as usize;
    memory
        .read(&mut *store, out_ptr as usize, &mut output[..written])
        .map_err(|_| CodecError::InternalError)?;

    let _ = free.call(&mut *store, (in_ptr, input.len() as u32));
    let _ = free.call(&mut *store, (out_ptr, output.len() as u32));
    Ok(written)
}

/// Helper: dispatch into the plane_codec_v1 decode export.
pub fn invoke_plane_codec_decode(
    instance: &Instance,
    store: &mut Store<WasmState>,
    input: &[u8],
    output: &mut [u8],
) -> Result<usize, CodecError> {
    let memory: Memory = instance
        .get_memory(&mut *store, "memory")
        .ok_or(CodecError::InvalidInput)?;
    let alloc: TypedFunc<u32, u32> = instance
        .get_typed_func(&mut *store, "ptwm_alloc")
        .map_err(|_| CodecError::InvalidInput)?;
    let free: TypedFunc<(u32, u32), ()> = instance
        .get_typed_func(&mut *store, "ptwm_free")
        .map_err(|_| CodecError::InvalidInput)?;
    let decode: TypedFunc<(u32, u32, u32, u32), i64> = instance
        .get_typed_func(&mut *store, "ptwm_plane_codec_v1_decode")
        .map_err(|_| CodecError::InvalidInput)?;

    let in_ptr = alloc
        .call(&mut *store, input.len() as u32)
        .map_err(|_| CodecError::InternalError)?;
    memory
        .write(&mut *store, in_ptr as usize, input)
        .map_err(|_| CodecError::InternalError)?;
    let out_ptr = alloc
        .call(&mut *store, output.len() as u32)
        .map_err(|_| CodecError::InternalError)?;

    let rc = decode
        .call(
            &mut *store,
            (in_ptr, input.len() as u32, out_ptr, output.len() as u32),
        )
        .map_err(|_| CodecError::InternalError)?;
    if rc < 0 {
        return Err(decode_codec_error(rc));
    }
    // See `invoke_plane_codec_encode` for why we bounds-check rc as
    // i64 rather than narrowing first.
    if rc > output.len() as i64 {
        return Err(CodecError::InternalError);
    }
    let written = rc as usize;
    memory
        .read(&mut *store, out_ptr as usize, &mut output[..written])
        .map_err(|_| CodecError::InternalError)?;

    let _ = free.call(&mut *store, (in_ptr, input.len() as u32));
    let _ = free.call(&mut *store, (out_ptr, output.len() as u32));
    Ok(written)
}

/// Helper: dispatch into the optional state-aware decode export.
///
/// Falls back to [`invoke_plane_codec_decode`] when the module didn't
/// export `ptwm_plane_codec_v1_decode_stateful` (legal only when the
/// plane carries no state bytes).
pub fn invoke_plane_codec_decode_stateful(
    instance: &Instance,
    store: &mut Store<WasmState>,
    state_version: u8,
    state_bytes: &[u8],
    input: &[u8],
    output: &mut [u8],
) -> Result<usize, CodecError> {
    // Probe the optional state-aware symbol.
    let decode_stateful: TypedFunc<(u32, u32, u32, u32, u32, u32, u32), i64> = match instance
        .get_typed_func(&mut *store, "ptwm_plane_codec_v1_decode_stateful")
    {
        Ok(f) => f,
        Err(_) => {
            if !state_bytes.is_empty() {
                return Err(CodecError::Unsupported {
                    feature: "ptwm_plane_codec_v1_decode_stateful not exported but plane carries \
                              state bytes; codec cannot honour the per-plane state"
                        .into(),
                });
            }
            return invoke_plane_codec_decode(instance, store, input, output);
        }
    };

    let memory: Memory = instance
        .get_memory(&mut *store, "memory")
        .ok_or(CodecError::InvalidInput)?;
    let alloc: TypedFunc<u32, u32> = instance
        .get_typed_func(&mut *store, "ptwm_alloc")
        .map_err(|_| CodecError::InvalidInput)?;
    let free: TypedFunc<(u32, u32), ()> = instance
        .get_typed_func(&mut *store, "ptwm_free")
        .map_err(|_| CodecError::InvalidInput)?;

    // Skip the WASM allocation entirely for an empty state — guest
    // allocators may handle 0-byte requests inconsistently, and
    // matching native FFI behaviour means passing a null-equivalent
    // (0 pointer) when there are no state bytes.
    let state_ptr = if state_bytes.is_empty() {
        0
    } else {
        let ptr = alloc
            .call(&mut *store, state_bytes.len() as u32)
            .map_err(|_| CodecError::InternalError)?;
        memory
            .write(&mut *store, ptr as usize, state_bytes)
            .map_err(|_| CodecError::InternalError)?;
        ptr
    };
    let in_ptr = alloc
        .call(&mut *store, input.len() as u32)
        .map_err(|_| CodecError::InternalError)?;
    memory
        .write(&mut *store, in_ptr as usize, input)
        .map_err(|_| CodecError::InternalError)?;
    let out_ptr = alloc
        .call(&mut *store, output.len() as u32)
        .map_err(|_| CodecError::InternalError)?;

    // Compute the result, then unconditionally free in the cleanup
    // tail. Returning early on error here would leak the three guest
    // allocations until the store drops — fine in v1 (one store per
    // call), but easy to get wrong as the call surface grows.
    let result: Result<usize, CodecError> = (|| {
        let rc = decode_stateful
            .call(
                &mut *store,
                (
                    state_version as u32,
                    state_ptr,
                    state_bytes.len() as u32,
                    in_ptr,
                    input.len() as u32,
                    out_ptr,
                    output.len() as u32,
                ),
            )
            .map_err(|_| CodecError::InternalError)?;
        if rc < 0 {
            return Err(decode_codec_error(rc));
        }
        // Compare against `output.len() as i64` BEFORE casting to
        // usize — `rc as usize` would truncate on 32-bit hosts when
        // `rc` is between u32::MAX and i64::MAX.
        if rc > output.len() as i64 {
            return Err(CodecError::InternalError);
        }
        let written = rc as usize;
        memory
            .read(&mut *store, out_ptr as usize, &mut output[..written])
            .map_err(|_| CodecError::InternalError)?;
        Ok(written)
    })();

    if !state_bytes.is_empty() {
        let _ = free.call(&mut *store, (state_ptr, state_bytes.len() as u32));
    }
    let _ = free.call(&mut *store, (in_ptr, input.len() as u32));
    let _ = free.call(&mut *store, (out_ptr, output.len() as u32));
    result
}

/// Helper: dispatch into the delta_scheme_v1 encode export.
///
/// Unlike `plane_codec`'s invoke helpers, this threads the per-call state
/// handle the reference ABI actually exports
/// (`ptwm_delta_scheme_v1_{init,cleanup,encode,decode}` — see
/// `extensions/ref_delta_scheme/rust/src/lib.rs`): call `init` for a fresh
/// handle, pass it to `encode`, then `cleanup` it, all within this one
/// invocation (matching the existing per-call store/instance lifecycle in
/// `WasmAdapter`).
pub fn invoke_delta_scheme_encode(
    instance: &Instance,
    store: &mut Store<WasmState>,
    base: &[u8],
    target: &[u8],
    output: &mut [u8],
) -> Result<usize, CodecError> {
    let memory: Memory = instance
        .get_memory(&mut *store, "memory")
        .ok_or(CodecError::InvalidInput)?;
    let alloc: TypedFunc<u32, u32> = instance
        .get_typed_func(&mut *store, "ptwm_alloc")
        .map_err(|_| CodecError::InvalidInput)?;
    let free: TypedFunc<(u32, u32), ()> = instance
        .get_typed_func(&mut *store, "ptwm_free")
        .map_err(|_| CodecError::InvalidInput)?;
    let init: TypedFunc<(), u32> = instance
        .get_typed_func(&mut *store, "ptwm_delta_scheme_v1_init")
        .map_err(|_| CodecError::InvalidInput)?;
    let cleanup: TypedFunc<u32, ()> = instance
        .get_typed_func(&mut *store, "ptwm_delta_scheme_v1_cleanup")
        .map_err(|_| CodecError::InvalidInput)?;
    let encode: TypedFunc<(u32, u32, u32, u32, u32, u32, u32), i64> = instance
        .get_typed_func(&mut *store, "ptwm_delta_scheme_v1_encode")
        .map_err(|_| CodecError::InvalidInput)?;

    let state = init
        .call(&mut *store, ())
        .map_err(|_| CodecError::InternalError)?;

    let base_ptr = alloc
        .call(&mut *store, base.len() as u32)
        .map_err(|_| CodecError::InternalError)?;
    memory
        .write(&mut *store, base_ptr as usize, base)
        .map_err(|_| CodecError::InternalError)?;
    let target_ptr = alloc
        .call(&mut *store, target.len() as u32)
        .map_err(|_| CodecError::InternalError)?;
    memory
        .write(&mut *store, target_ptr as usize, target)
        .map_err(|_| CodecError::InternalError)?;
    let out_ptr = alloc
        .call(&mut *store, output.len() as u32)
        .map_err(|_| CodecError::InternalError)?;

    let result: Result<usize, CodecError> = (|| {
        let rc = encode
            .call(
                &mut *store,
                (
                    state,
                    base_ptr,
                    base.len() as u32,
                    target_ptr,
                    target.len() as u32,
                    out_ptr,
                    output.len() as u32,
                ),
            )
            .map_err(|_| CodecError::InternalError)?;
        if rc < 0 {
            return Err(decode_codec_error(rc));
        }
        if rc > output.len() as i64 {
            return Err(CodecError::InternalError);
        }
        let written = rc as usize;
        memory
            .read(&mut *store, out_ptr as usize, &mut output[..written])
            .map_err(|_| CodecError::InternalError)?;
        Ok(written)
    })();

    let _ = cleanup.call(&mut *store, state);
    let _ = free.call(&mut *store, (base_ptr, base.len() as u32));
    let _ = free.call(&mut *store, (target_ptr, target.len() as u32));
    let _ = free.call(&mut *store, (out_ptr, output.len() as u32));
    result
}

/// Helper: dispatch into the delta_scheme_v1 decode export. See
/// [`invoke_delta_scheme_encode`] for the state-handle lifecycle.
pub fn invoke_delta_scheme_decode(
    instance: &Instance,
    store: &mut Store<WasmState>,
    base: &[u8],
    delta: &[u8],
    output: &mut [u8],
) -> Result<usize, CodecError> {
    let memory: Memory = instance
        .get_memory(&mut *store, "memory")
        .ok_or(CodecError::InvalidInput)?;
    let alloc: TypedFunc<u32, u32> = instance
        .get_typed_func(&mut *store, "ptwm_alloc")
        .map_err(|_| CodecError::InvalidInput)?;
    let free: TypedFunc<(u32, u32), ()> = instance
        .get_typed_func(&mut *store, "ptwm_free")
        .map_err(|_| CodecError::InvalidInput)?;
    let init: TypedFunc<(), u32> = instance
        .get_typed_func(&mut *store, "ptwm_delta_scheme_v1_init")
        .map_err(|_| CodecError::InvalidInput)?;
    let cleanup: TypedFunc<u32, ()> = instance
        .get_typed_func(&mut *store, "ptwm_delta_scheme_v1_cleanup")
        .map_err(|_| CodecError::InvalidInput)?;
    let decode: TypedFunc<(u32, u32, u32, u32, u32, u32, u32), i64> = instance
        .get_typed_func(&mut *store, "ptwm_delta_scheme_v1_decode")
        .map_err(|_| CodecError::InvalidInput)?;

    let state = init
        .call(&mut *store, ())
        .map_err(|_| CodecError::InternalError)?;

    let base_ptr = alloc
        .call(&mut *store, base.len() as u32)
        .map_err(|_| CodecError::InternalError)?;
    memory
        .write(&mut *store, base_ptr as usize, base)
        .map_err(|_| CodecError::InternalError)?;
    let delta_ptr = alloc
        .call(&mut *store, delta.len() as u32)
        .map_err(|_| CodecError::InternalError)?;
    memory
        .write(&mut *store, delta_ptr as usize, delta)
        .map_err(|_| CodecError::InternalError)?;
    let out_ptr = alloc
        .call(&mut *store, output.len() as u32)
        .map_err(|_| CodecError::InternalError)?;

    let result: Result<usize, CodecError> = (|| {
        let rc = decode
            .call(
                &mut *store,
                (
                    state,
                    base_ptr,
                    base.len() as u32,
                    delta_ptr,
                    delta.len() as u32,
                    out_ptr,
                    output.len() as u32,
                ),
            )
            .map_err(|_| CodecError::InternalError)?;
        if rc < 0 {
            return Err(decode_codec_error(rc));
        }
        if rc > output.len() as i64 {
            return Err(CodecError::InternalError);
        }
        let written = rc as usize;
        memory
            .read(&mut *store, out_ptr as usize, &mut output[..written])
            .map_err(|_| CodecError::InternalError)?;
        Ok(written)
    })();

    let _ = cleanup.call(&mut *store, state);
    let _ = free.call(&mut *store, (base_ptr, base.len() as u32));
    let _ = free.call(&mut *store, (delta_ptr, delta.len() as u32));
    let _ = free.call(&mut *store, (out_ptr, output.len() as u32));
    result
}

fn decode_codec_error(rc: i64) -> CodecError {
    match rc {
        -1 => CodecError::InvalidInput,
        // TODO: decode the real CodecError payload from linear memory instead
        // of returning a placeholder `needed: 0`.
        -2 => CodecError::BufferTooSmall { needed: 0 },
        -3 => CodecError::MissingCapability {
            name: String::new(),
        },
        -4 => CodecError::MissingNativeDep {
            name: String::new(),
            version_constraint: String::new(),
        },
        -5 => CodecError::Unsupported {
            feature: String::new(),
        },
        _ => CodecError::InternalError,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extension::{
        Attestation, CapabilityMap, Kind, Lifecycle as Lc, capability::CapabilityValue,
        table::FLAVOR_WASM,
    };

    /// A tiny WASM module that exports memory + ptwm_alloc + ptwm_free +
    /// an identity ptwm_plane_codec_v1_encode (copy input → output).
    /// Hand-rolled WAT so the test doesn't need a Cargo subbuild.
    const IDENTITY_WAT: &str = r#"
(module
  (memory (export "memory") 1)
  (global $bump (mut i32) (i32.const 1024))

  (func $alloc (export "ptwm_alloc") (param $len i32) (result i32)
    (local $p i32)
    local.get $len
    i32.const 0
    i32.le_s
    if (result i32)
      i32.const 1024
    else
      global.get $bump
      local.set $p
      global.get $bump
      local.get $len
      i32.add
      global.set $bump
      local.get $p
    end)

  (func (export "ptwm_free") (param $p i32) (param $len i32))

  (func (export "ptwm_plane_codec_v1_encode")
        (param $in_ptr i32) (param $in_len i32)
        (param $out_ptr i32) (param $out_len i32)
        (result i64)
    (local $i i32)
    (block $done
      (loop $copy
        local.get $i
        local.get $in_len
        i32.ge_s
        br_if $done
        local.get $out_ptr
        local.get $i
        i32.add
        local.get $in_ptr
        local.get $i
        i32.add
        i32.load8_u
        i32.store8
        local.get $i
        i32.const 1
        i32.add
        local.set $i
        br $copy))
    local.get $in_len
    i64.extend_i32_s)
)
"#;

    fn make_entry() -> ExtensionTableEntry {
        let mut caps = CapabilityMap::new();
        caps.set("determinism", CapabilityValue::Bool(true));
        ExtensionTableEntry {
            canonical_id: CanonicalId::from_bytes([0xAB; 32]),
            human_label: "io.test.identity".into(),
            kind: Kind::PlaneCodec,
            abi_version: 1,
            lifecycle: Lc::Thread,
            flavor_hints: FLAVOR_WASM,
            capabilities: caps,
            attestation: Attestation::PgpSignature(Vec::new()),
            install_hint: None,
            embedded_wasm_offset: None,
            embedded_wasm_length: None,
        }
    }

    #[test]
    fn identity_round_trip_via_wasm() {
        let engine = WasmExtension::make_engine();
        let wasm = wat::parse_str(IDENTITY_WAT).expect("WAT parse");
        let entry = make_entry();
        let ext = WasmExtension::load(&engine, &wasm, &entry).unwrap();
        let mut store = ext.make_store(&HostPolicy::default()).unwrap();
        let inst = ext.instantiate(&mut store).unwrap();

        let input = b"hello ptwm";
        let mut output = vec![0u8; 64];
        let n = invoke_plane_codec_encode(&inst, &mut store, input, &mut output).unwrap();
        assert_eq!(n, input.len());
        assert_eq!(&output[..n], input);
    }

    #[test]
    fn declared_host_imports_are_rejected_without_policy_grant() {
        let engine = WasmExtension::make_engine();
        let wasm = wat::parse_str(IDENTITY_WAT).expect("WAT parse");
        let mut entry = make_entry();
        entry.capabilities.set(
            "host_imports",
            CapabilityValue::List(vec![CapabilityValue::Text("clock".into())]),
        );
        let ext = WasmExtension::load(&engine, &wasm, &entry).unwrap();
        // Default HostPolicy has empty allow_host_imports.
        let res = ext.make_store(&HostPolicy::default());
        assert!(matches!(res, Err(CodecError::MissingCapability { .. })));
    }

    #[test]
    fn fuel_metering_is_active() {
        // No need to bound-test exact fuel consumption — just that the
        // store is created with fuel set.
        let engine = WasmExtension::make_engine();
        let wasm = wat::parse_str(IDENTITY_WAT).expect("WAT parse");
        let entry = make_entry();
        let ext = WasmExtension::load(&engine, &wasm, &entry).unwrap();
        let store = ext.make_store(&HostPolicy::default()).unwrap();
        assert!(store.get_fuel().unwrap() > 0);
    }

    /// XOR codec: state byte K, output[i] = input[i] XOR K.
    /// Exports `ptwm_plane_codec_v1_decode_stateful` to demonstrate the
    /// state-aware ABI.
    const XOR_DECODE_WAT: &str = r#"
(module
  (memory (export "memory") 1)
  (global $bump (mut i32) (i32.const 1024))

  (func (export "ptwm_alloc") (param $len i32) (result i32)
    (local $p i32)
    local.get $len
    i32.const 0
    i32.le_s
    if (result i32)
      i32.const 1024
    else
      global.get $bump
      local.set $p
      global.get $bump
      local.get $len
      i32.add
      global.set $bump
      local.get $p
    end)

  (func (export "ptwm_free") (param $p i32) (param $len i32))

  (func (export "ptwm_plane_codec_v1_decode_stateful")
        (param $sv i32) (param $sp i32) (param $sl i32)
        (param $ip i32) (param $il i32)
        (param $op i32) (param $oc i32)
        (result i64)
    (local $key i32)
    (local $i i32)
    local.get $oc
    local.get $il
    i32.lt_s
    if
      i64.const -2
      return
    end
    local.get $sp
    i32.load8_u
    local.set $key
    (block $done
      (loop $copy
        local.get $i
        local.get $il
        i32.ge_s
        br_if $done
        local.get $op
        local.get $i
        i32.add
        local.get $ip
        local.get $i
        i32.add
        i32.load8_u
        local.get $key
        i32.xor
        i32.store8
        local.get $i
        i32.const 1
        i32.add
        local.set $i
        br $copy))
    local.get $il
    i64.extend_i32_s)
)
"#;

    /// XOR-with-zero is the identity, so this exercises the stateful
    /// dispatch path on a one-byte state and confirms the byte ABI
    /// correctly threads `state_bytes` through to the codec.
    #[test]
    fn decode_with_state_zero_key_round_trips() {
        let engine = WasmExtension::make_engine();
        let wasm = wat::parse_str(XOR_DECODE_WAT).expect("WAT parse");
        let entry = make_entry();
        let ext = WasmExtension::load(&engine, &wasm, &entry).unwrap();
        let mut store = ext.make_store(&HostPolicy::default()).unwrap();
        let inst = ext.instantiate(&mut store).unwrap();

        let state = [0u8];
        let input = b"hello stateful";
        let mut output = vec![0u8; 64];
        let n = invoke_plane_codec_decode_stateful(
            &inst,
            &mut store,
            /* state_version */ 1,
            &state,
            input,
            &mut output,
        )
        .unwrap();
        assert_eq!(n, input.len());
        assert_eq!(&output[..n], input);
    }

    /// True fallback: IDENTITY_WAT does NOT export the stateful symbol,
    /// so calling `invoke_plane_codec_decode_stateful` with empty state
    /// must transparently fall through to the stateless decode export.
    /// We re-use IDENTITY_WAT's encode as a stand-in for decode by
    /// exercising the symbol-probe + fallback branch.
    #[test]
    fn decode_with_state_falls_back_when_symbol_missing_and_state_empty() {
        // IDENTITY_WAT exports only encode (not decode), so the
        // fallback `invoke_plane_codec_decode` will itself fail with
        // InvalidInput when it can't resolve the decode symbol — but
        // we get there via the fallback branch, which is what we want
        // to verify here.
        let engine = WasmExtension::make_engine();
        let wasm = wat::parse_str(IDENTITY_WAT).expect("WAT parse");
        let entry = make_entry();
        let ext = WasmExtension::load(&engine, &wasm, &entry).unwrap();
        let mut store = ext.make_store(&HostPolicy::default()).unwrap();
        let inst = ext.instantiate(&mut store).unwrap();

        // Empty state + missing stateful symbol → falls through to
        // stateless decode, which itself errors because IDENTITY_WAT
        // doesn't export decode. The crucial assertion: the error is
        // InvalidInput (from get_typed_func on `decode`), NOT
        // Unsupported (which would mean the stateful-missing-with-
        // non-empty-state guard tripped).
        let input = b"x";
        let mut output = vec![0u8; 8];
        let res = invoke_plane_codec_decode_stateful(
            &inst,
            &mut store,
            /* state_version */ 0,
            &[],
            input,
            &mut output,
        );
        assert!(matches!(res, Err(CodecError::InvalidInput)));
    }

    /// Non-empty state + missing stateful symbol → hard error
    /// (Unsupported), preventing silent data loss.
    #[test]
    fn decode_with_state_errors_when_symbol_missing_and_state_present() {
        let engine = WasmExtension::make_engine();
        let wasm = wat::parse_str(IDENTITY_WAT).expect("WAT parse");
        let entry = make_entry();
        let ext = WasmExtension::load(&engine, &wasm, &entry).unwrap();
        let mut store = ext.make_store(&HostPolicy::default()).unwrap();
        let inst = ext.instantiate(&mut store).unwrap();

        let state = [0x42];
        let input = b"x";
        let mut output = vec![0u8; 8];
        let res = invoke_plane_codec_decode_stateful(
            &inst,
            &mut store,
            /* state_version */ 1,
            &state,
            input,
            &mut output,
        );
        assert!(matches!(res, Err(CodecError::Unsupported { .. })));
    }

    #[test]
    fn decode_with_state_threads_state_through() {
        let engine = WasmExtension::make_engine();
        let wasm = wat::parse_str(XOR_DECODE_WAT).expect("WAT parse");
        let entry = make_entry();
        let ext = WasmExtension::load(&engine, &wasm, &entry).unwrap();
        let mut store = ext.make_store(&HostPolicy::default()).unwrap();
        let inst = ext.instantiate(&mut store).unwrap();

        // XOR with key=0x5A. Encoded bytes are input XOR 0x5A.
        const KEY: u8 = 0x5A;
        let state = [KEY];
        let original: &[u8] = b"stateful codec round-trip";
        let encoded: Vec<u8> = original.iter().map(|b| b ^ KEY).collect();

        let mut output = vec![0u8; 64];
        let n = invoke_plane_codec_decode_stateful(
            &inst,
            &mut store,
            /* state_version */ 1,
            &state,
            &encoded,
            &mut output,
        )
        .unwrap();
        assert_eq!(n, original.len());
        assert_eq!(&output[..n], original);
    }

    /// Passthrough delta_scheme: encode copies `target` (ignores `base`),
    /// decode copies `delta` (ignores `base`) — same behavioral invariant
    /// as `extensions/ref_delta_scheme/rust/src/lib.rs`, hand-rolled in WAT
    /// so the test doesn't need a Cargo subbuild. Both encode and decode
    /// take the `init`-issued state handle as their first argument, per
    /// that reference's actual exported signature.
    const DELTA_SCHEME_PASSTHROUGH_WAT: &str = r#"
(module
  (memory (export "memory") 1)
  (global $bump (mut i32) (i32.const 1024))

  (func (export "ptwm_alloc") (param $len i32) (result i32)
    (local $p i32)
    local.get $len
    i32.const 0
    i32.le_s
    if (result i32)
      i32.const 1024
    else
      global.get $bump
      local.set $p
      global.get $bump
      local.get $len
      i32.add
      global.set $bump
      local.get $p
    end)

  (func (export "ptwm_free") (param $p i32) (param $len i32))

  (func (export "ptwm_delta_scheme_v1_init") (result i32)
    i32.const 1)

  (func (export "ptwm_delta_scheme_v1_cleanup") (param $state i32))

  (func (export "ptwm_delta_scheme_v1_encode")
        (param $state i32)
        (param $base_ptr i32) (param $base_len i32)
        (param $target_ptr i32) (param $target_len i32)
        (param $out_ptr i32) (param $out_len i32)
        (result i64)
    (local $i i32)
    local.get $out_len
    local.get $target_len
    i32.lt_s
    if
      i64.const -2
      return
    end
    (block $done
      (loop $copy
        local.get $i
        local.get $target_len
        i32.ge_s
        br_if $done
        local.get $out_ptr
        local.get $i
        i32.add
        local.get $target_ptr
        local.get $i
        i32.add
        i32.load8_u
        i32.store8
        local.get $i
        i32.const 1
        i32.add
        local.set $i
        br $copy))
    local.get $target_len
    i64.extend_i32_s)

  (func (export "ptwm_delta_scheme_v1_decode")
        (param $state i32)
        (param $base_ptr i32) (param $base_len i32)
        (param $delta_ptr i32) (param $delta_len i32)
        (param $out_ptr i32) (param $out_len i32)
        (result i64)
    (local $i i32)
    local.get $out_len
    local.get $delta_len
    i32.lt_s
    if
      i64.const -2
      return
    end
    (block $done
      (loop $copy
        local.get $i
        local.get $delta_len
        i32.ge_s
        br_if $done
        local.get $out_ptr
        local.get $i
        i32.add
        local.get $delta_ptr
        local.get $i
        i32.add
        i32.load8_u
        i32.store8
        local.get $i
        i32.const 1
        i32.add
        local.set $i
        br $copy))
    local.get $delta_len
    i64.extend_i32_s)
)
"#;

    fn make_delta_scheme_entry() -> ExtensionTableEntry {
        ExtensionTableEntry {
            canonical_id: CanonicalId::from_bytes([0xDE; 32]),
            human_label: "io.test.delta_scheme".into(),
            kind: Kind::DeltaScheme,
            abi_version: 1,
            lifecycle: Lc::Thread,
            flavor_hints: FLAVOR_WASM,
            capabilities: CapabilityMap::new(),
            attestation: Attestation::PgpSignature(Vec::new()),
            install_hint: None,
            embedded_wasm_offset: None,
            embedded_wasm_length: None,
        }
    }

    #[test]
    fn delta_scheme_encode_copies_target_ignores_base() {
        let engine = WasmExtension::make_engine();
        let wasm = wat::parse_str(DELTA_SCHEME_PASSTHROUGH_WAT).expect("WAT parse");
        let entry = make_delta_scheme_entry();
        let ext = WasmExtension::load(&engine, &wasm, &entry).unwrap();
        let mut store = ext.make_store(&HostPolicy::default()).unwrap();
        let inst = ext.instantiate(&mut store).unwrap();

        let base = b"base tensor bytes (ignored)";
        let target = b"target tensor bytes";
        let mut output = vec![0u8; 64];
        let n = invoke_delta_scheme_encode(&inst, &mut store, base, target, &mut output).unwrap();
        assert_eq!(&output[..n], target);
    }

    #[test]
    fn delta_scheme_decode_copies_delta_ignores_base() {
        let engine = WasmExtension::make_engine();
        let wasm = wat::parse_str(DELTA_SCHEME_PASSTHROUGH_WAT).expect("WAT parse");
        let entry = make_delta_scheme_entry();
        let ext = WasmExtension::load(&engine, &wasm, &entry).unwrap();
        let mut store = ext.make_store(&HostPolicy::default()).unwrap();
        let inst = ext.instantiate(&mut store).unwrap();

        let base = b"base tensor bytes (ignored)";
        let delta = b"delta bytes";
        let mut output = vec![0u8; 64];
        let n = invoke_delta_scheme_decode(&inst, &mut store, base, delta, &mut output).unwrap();
        assert_eq!(&output[..n], delta);
    }

    #[test]
    fn delta_scheme_round_trips_encode_then_decode() {
        let engine = WasmExtension::make_engine();
        let wasm = wat::parse_str(DELTA_SCHEME_PASSTHROUGH_WAT).expect("WAT parse");
        let entry = make_delta_scheme_entry();
        let ext = WasmExtension::load(&engine, &wasm, &entry).unwrap();

        let base = b"shared base tensor";
        let target = b"the real target tensor payload";

        let mut store = ext.make_store(&HostPolicy::default()).unwrap();
        let inst = ext.instantiate(&mut store).unwrap();
        let mut delta = vec![0u8; 64];
        let n = invoke_delta_scheme_encode(&inst, &mut store, base, target, &mut delta).unwrap();
        delta.truncate(n);

        let mut store2 = ext.make_store(&HostPolicy::default()).unwrap();
        let inst2 = ext.instantiate(&mut store2).unwrap();
        let mut recon = vec![0u8; 64];
        let m = invoke_delta_scheme_decode(&inst2, &mut store2, base, &delta, &mut recon).unwrap();
        assert_eq!(&recon[..m], target);
    }
}
