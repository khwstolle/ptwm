# Reference extensions

Every `ref_*` directory under this folder is a **passthrough reference
implementation** of one of PTWM's 13 contribution kinds. Their purpose
is to:

1. Demonstrate the ABI surface (Rust → WASM via `wasm32-wasip1` for the
   byte-buffer kinds; Python for the host-only kinds).
2. Anchor the conformance tests under `tests/ptwm/ext_tooling/` and
   `tests/ptwm/host_flavor/`.
3. Serve as the seed templates that `ptwm ext init` generates for
   bundle authors.

**They are not production codecs.** A real plane codec would override
the `encode` / `decode` symbols to do actual entropy coding; a real
classifier would inspect tensor metadata to pick a role; a real
scorer would compute a content-aware quality signal. The reference
implementations here all round-trip data **unchanged** so they can
double as identity tests.

## Inventory

| Kind                  | Path                              | Language | Flavor |
| --------------------- | --------------------------------- | -------- | ------ |
| `plane_codec`         | `ref_plane_codec/rust/`           | Rust     | WASM   |
| `transform`           | `ref_transform/rust/`             | Rust     | WASM   |
| `container_layout`    | `ref_container_layout/rust/`      | Rust     | WASM   |
| `delta_scheme`        | `ref_delta_scheme/rust/`          | Rust     | WASM   |
| `hardware_backend`    | `ref_hardware_backend/rust/`      | Rust     | WASM   |
| `scorer`              | `ref_scorer/rust/`                | Rust     | WASM   |
| `raw_binary`          | `ref_raw_binary/rust/`            | Rust     | WASM   |
| `chain_builder`       | `ref_chain_builder/python/`       | Python   | Host   |
| `chain_explorer`      | `ref_chain_explorer/python/`      | Python   | Host   |
| `classifier`          | `ref_classifier/python/`          | Python   | Host   |
| `integration_adapter` | `ref_integration_adapter/python/` | Python   | Host   |
| `benchmark_metric`    | `ref_benchmark_metric/python/`    | Python   | Host   |
| `training_hook`       | `ref_training_hook/python/`       | Python   | Host   |

## Building the WASM references locally

```sh
rustup target add wasm32-wasip1
cargo build --target wasm32-wasip1 --release \
    -p ref_plane_codec
```

The resulting `target/wasm32-wasip1/release/ref_plane_codec.wasm`
is what `ptwm ext sign` would package alongside the bundle manifest.

## Building a real extension on top of these

Start from one of the language scaffolds:

```sh
ptwm ext init demo-codec --lang rust --kind plane_codec
```

`ptwm ext init` copies from a separate scaffold tree at
[`python/ptwm/ext_tooling/_templates/`](../python/ptwm/ext_tooling/_templates),
**not** from the `ref_*` directories. The init templates are
intentionally smaller than the reference impls; they're the minimal
"hello world" for each kind/language pair, with placeholders for
bundle name / version / canonical id that `init` substitutes. The
`ref_*` implementations here are the conformance anchors; consult
them when you need a working example beyond the init stub. See the
[extension-authoring guide](../docs/site/content/30.guides/03.extension-authoring.md)
for the full lifecycle (`init` → `build` → `sign` → `pack` → `install`).

## See also

- `crates/ptwm-core/src/extension/builtins.rs`: the 27 in-tree
  built-in contributions (6 plane codecs, 16 transforms, 1 chain
  builder, 4 classifiers). These ship with PTWM under `BUILTIN_PUBKEY`
  and are trusted by construction.
- [`docs/architecture/extension-system.md`](../docs/architecture/extension-system.md):
  the full design rationale and per-kind ABI signatures.
