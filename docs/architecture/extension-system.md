# Extension system

PTWM accepts contributions from outside the core crate. The contribution
system was a substantial addition and changed several long-standing
internal interfaces. This document explains the thirteen-kind taxonomy,
the three-flavour delivery model, the canonical-id scheme, and why the
architecture absorbed the cost.

## What an extension is

An extension is a signed bundle containing one or more contributions.
Each contribution implements one of thirteen kinds: `Transform`,
`PlaneCodec`, `ChainBuilder`, `ChainExplorer`, `Classifier`, `Scorer`,
`DeltaScheme`, `IntegrationAdapter`, `ContainerLayout`,
`HardwareBackend`, `BenchmarkMetric`, `TrainingHook`, `RawBinary`.

The kind set is sealed. Adding a fourteenth kind is a wire-format
change because every extension-table entry carries the kind as a
u16. The sealing is intentional: every kind requires the host to
expose an ABI surface for it, and the surface cost grows roughly
linearly with the kind count. The thirteen kinds cover every
extensibility surface the design phase identified as load-bearing.

The kinds are not symmetric in importance. `Transform` and `PlaneCodec`
are the workhorses; almost all third-party contributions will be one
of these. `Classifier`, `ChainBuilder`, and `ChainExplorer` cover the
encode-only side of the PPG. The remaining kinds cover narrower cases
that are worth supporting when the demand arrives, without requiring
another sealed-set change.

## Three flavours

Each contribution can ship in WASM, native cdylib, or Python host
form. The bundle declares which flavours it provides via the
`flavor_hints` bitfield in its extension-table entry; the loader
picks based on host capability and policy.

WASM is the security-first flavour. The Wasmtime runtime executes
the bundle in a sandbox with fuel and memory budgets, no host
imports by default, and no filesystem or network access. Contribution
performance is typically 70% to 80% of native. This is the right
choice when the contribution comes from a publisher the user does
not fully trust.

Native cdylib is the performance-first flavour. The bundle is loaded
via `libloading` after signature verification; once loaded it
executes with full process privileges. There is no sandbox. The use
case is performance-critical paths and contributions that need
hardware access (CUDA kernels, for example).

Python host is for contributions that do not have a hot loop and
that benefit from the existing Python ecosystem. The classifier
contributions ship as Python because they read tensor names and
host metadata; this is awkward in WASM and pointless in native.
Python-host contributions are loaded via the `ptwm.extensions`
entry-point group, with the contribution authoring a `register`
function that calls the relevant registry.

## Canonical ids

Every contribution has a 32-byte `CanonicalId` derived from
`blake3(pubkey || 0x00 || name || 0x00 || version)`. The id is the
identifier the wire format records and the verifier checks. The
zero-byte separators prevent prefix collisions; without them, the
contributions `(pk, "foo", "bar")` and `(pk, "foob", "ar")` would
collide.

Built-in contributions use the sentinel public key
`BUILTIN_PUBKEY = [0xBB; 32]`. The all-0xBB pattern is recognisable
in hex dumps. A real Ed25519 public key cannot accidentally match
the sentinel because the public-key derivation does not produce
that pattern from any valid private key. The verifier short-circuits
on the sentinel because built-ins are trusted by construction.

Third-party canonical ids are derived from the author's real Ed25519
public key. The author's key is the unit of trust: a single key can
sign many contributions, and the user's keyring trusts (or revokes)
the key as a whole. The keyring also supports per-contribution
pinning for cases where the user wants to accept a specific version
without granting transitive trust to the author.

## Extension table per file

The container's extension table is the per-file manifest of every
canonical id the file references. The table is committed by hash in
the header, so a reader can verify the table's integrity without
trusting the table bytes themselves.

The per-file table serves two purposes. It tells the reader which
contributions are needed at decode time, surfacing the dependency
before any tensor decode starts. And it pins each id at the
specific version that was used at encode time; a later upgrade of
the contribution that changes its canonical id will not match the
file's recorded id, so the reader gets a clean `MissingFlavor`
error rather than silently calling a different codec.

The entries in the table are interned by canonical id; if a file
uses the same codec on 100 planes, the table has one entry for that
codec and each plane's `codec_table_idx` points to the same entry.
This keeps the table size bounded by the number of distinct
contributions a file uses, not the number of planes.

## Lifecycle modes

Each contribution declares a lifecycle: `none`, `thread`, or
`process`. The lifecycle controls when the contribution's state is
initialised and reused.

`none` is the simplest: every call to the contribution is
independent. The init happens implicitly inside each call; no state
is reused. Most plane codecs are this kind.

`thread` means one state per rayon worker thread, reused for the
worker's lifetime. This amortises expensive init (loading a model,
opening a file) across many calls within the same thread, without
introducing cross-thread synchronisation. The lifecycle is the right
choice for contributions with substantial init but no need for
sharing across threads.

`process` means one state for the whole process, shared across all
threads. The contribution is responsible for thread safety. This is
appropriate for stateful caches and singletons where the
contribution wants control over its own concurrency strategy.

The lifecycle is declared at the contribution level rather than per
call because the host needs to know whether to keep state around.
Declaring it at the call site would force every caller to know what
the contribution wants, which the contribution author is in a
better position to decide.

## Capabilities

The extension-table entry carries a capability map in canonical CBOR.
The map is sealed-but-extensible: the v1 baseline keys are
`determinism`, `native_deps`, `host_imports`, `hardware_class`,
`sandbox_class`, `mem_factor`, `fuel_factor`. Unknown keys are
allowed at parse time but denied at verify time unless the user's
policy explicitly grants them.

The deny-by-default policy on unknown keys is the security-relevant
choice. A future PTWM upgrade that introduces a new capability key
cannot be exploited by replaying an old extension that declared the
new key without authorisation. The user's keyring entries can scope
trust to a specific allowed-capability list, so a user can trust an
author for pure-decode contributions while declining anything that
declares `native_deps`.

## Trust verification

The verifier in `crates/ptwm-core/src/trust/verifier.rs` runs five
checks in order on every extension-table entry. The first check
that succeeds determines the verdict; later checks are skipped.

1. Built-in short circuit. If the canonical id derives from the
   sentinel public key, the entry is trusted.
2. Contribution hash pin. If the user's keyring has a per-id pin
   that matches, the entry is trusted regardless of the author key.
3. Author key lookup. If the author key is not in the keyring, the
   entry is untrusted.
4. Capability scope. If the entry declares capabilities that the
   user's keyring entry restricts, the entry is untrusted.
5. Signature verification. The entry's Ed25519 signature is checked
   against the author key and the bundle's signed material (the
   blake3 of manifest plus binaries).

The verdict flows back to the reader, which surfaces
`ContributionUntrusted` errors with the specific reason rather than
a generic failure. The intent is that a user who hits an untrusted
contribution can diagnose the cause without reading the verifier
source.

## Discovery

The reader discovers installed bundles by scanning the resolution
path (defaults to `$XDG_DATA_HOME/ptwm/extensions/` plus
`/usr/share/ptwm/extensions/`; overrideable via
`$PTWM_EXTENSION_PATH`). The scan happens at `ContainerReader::open`
time, with an mtime-keyed cache in `$XDG_CACHE_HOME/ptwm/` to
avoid re-parsing manifests on every file open.

The scan-on-open approach has a real cost on slow filesystems. The
alternative is a long-running daemon that watches the extension
path and serves the index over a Unix socket; this would reduce the
per-open cost but adds operational complexity (lifecycle, IPC,
upgrade) that only pays for itself at higher scale than PTWM
currently sees. The mtime cache absorbs most of the cost in
practice; if the daemon model becomes worthwhile later, the reader
side of the protocol is the only piece that needs to change.

## The `.ptwx` variant

PTWM has two on-disk magic strings. The standard `.ptwm` format
requires referenced contributions to be installed locally. The
`.ptwx` format embeds the WASM blobs of its contributions inline in
the extension table; the reader can extract and run them without
contacting the filesystem first.

The `.ptwx` format ships in the Rust core (the magic constant and
fuzz coverage exist) but the Python opt-in gate is deferred to a
later release. Executing WASM from an inline blob without an
out-of-band trust step is a security decision the design did not
want to take by default. The format is in the wire so that
distributors can produce `.ptwx` files today, with the gate
arriving later.
