# Trust model

PTWM accepts third-party code at decode time. The trust model defines
what that means in practice: who can publish a codec, how the user
expresses trust, what happens when something fails verification, and
what attacks the model defends against.

## Threat model

The user is the principal who decides what code runs. The threats
the model defends against are:

A malicious file. An attacker produces a `.ptwm` that references a
codec controlled by the attacker. Without verification the codec
would run on decode and could execute arbitrary code. The model
defends by requiring every non-builtin canonical id in the file's
extension table to chain to a key the user has explicitly trusted.

A malicious published bundle. An attacker publishes a bundle on a
package index, perhaps under a name similar to a legitimate
contribution. The model defends by requiring the user to add the
publisher's key to their keyring before any bundle from that
publisher runs. Adding the key is an explicit step (`ptwm trust add
--key`).

A compromised key. An attacker steals the private key of a legitimate
publisher and publishes a malicious update under the legitimate
canonical-id derivation. The model partly defends through the
contribution-hash pin: a user who has pinned a specific version
(`ptwm trust add --pin <canonical-id>`) does not implicitly trust
later versions even if signed by the same key. The model does not
defend against a compromised key for users who only have an
author-key entry; this is a known, accepted trade-off.

A silent PTWM upgrade. A user upgrades PTWM and the new version
ships a different bundled keyring. The model defends through the
bundled-then-pinned design: the bundled keyring's hash is pinned at
`ptwm trust add --bundled` time. A later upgrade with a different
bundled keyring is detected as a `BundleStatus::Mismatch` and the
new keyring stays inactive until the user explicitly runs `ptwm
trust update --bundled`.

The threats explicitly not in scope are: protection of compressed
weight bytes (the user is expected to use filesystem-level
encryption if needed), denial-of-service via deliberately slow
contributions (the WASM fuel cap bounds this for sandboxed
contributions; native contributions are out of scope by design),
and side-channel attacks against the user's signing key (use a
hardware token).

## Keyring entry types

The keyring supports three entry shapes:

`AuthorKey { pubkey, label }` trusts everything signed by the key.
This is the common case. The user runs `ptwm trust add --key
ed25519:<hex>` to add an entry.

`ContributionHash { canonical_id, label }` pins trust to a specific
canonical id. The verifier checks the pin before checking the
author key, so a pinned id is trusted even if its author key is
not. The use case is "I trust this exact codec at this exact
version, but not future versions from the same author".

`KeyWithCapabilityConstraints { pubkey, allowed_capabilities, label }`
is the same as AuthorKey but restricts the capabilities the
contribution may declare. A contribution that declares a capability
outside the allowed set is rejected as untrusted. The use case is
"I trust this author for pure-decode codecs but not anything that
declares native_deps".

The shapes are CBOR-deserialised from the on-disk keyring at
`$XDG_CONFIG_HOME/ptwm/trust/keys.toml`. The TOML encoding uses
serde tag dispatch (`kind = "author_key"`, etc.), which keeps the
file human-readable and editable.

## Bundled keyring

PTWM ships with a curated bundled keyring of trusted authors. The
keyring's blake3 hash is pinned to
`$XDG_CONFIG_HOME/ptwm/trust/bundled.lock` when the user accepts
it. The three states are:

`FreshInstall` if the lock file does not exist. PTWM treats this as
"the user has not yet decided whether to trust the bundled keyring".
The trust evaluation returns an empty active keyring until the user
runs `ptwm trust add --bundled`.

`Match` if the lock file exists and the hash matches the bundled
keyring shipping with the running PTWM version. The bundled keys
are active.

`Mismatch` if the lock file exists and the hash differs. This
happens when the user upgrades PTWM and the new version ships a
different bundled keyring. The bundled keys are inactive until the
user runs `ptwm trust update --bundled`, which shows the diff
(added and removed entries) and pins the new hash.

The intent is that a user can upgrade PTWM with confidence: a new
version cannot silently grant decode rights to a new publisher
without explicit user consent. The downside is that an upgrade
intended to revoke a compromised publisher's key requires the user
to run `update --bundled` before the revocation takes effect.

## Verification flow

The verifier (`crates/ptwm-core/src/trust/verifier.rs`) is called
from the container open path on every non-builtin extension-table
entry. The flow:

1. Built-in short circuit. Built-in canonical ids derive from the
   sentinel public key; the verifier returns `Trusted` immediately.
2. Contribution hash check. If any keyring entry pins the canonical
   id, the verifier returns `Trusted`.
3. Author key lookup. The verifier finds the author key from the
   discovered installed bundle (matching the canonical id back to a
   manifest in the resolution path). If no installed bundle
   matches, the entry is skipped here; the missing flavour will
   surface separately.
4. Keyring check. If the author key is not in any keyring entry
   (AuthorKey or KeyWithCapabilityConstraints), the verifier returns
   `Untrusted` with the reason "author key not in keyring".
5. Capability scope check. If the keyring entry is a
   KeyWithCapabilityConstraints, the verifier rejects any
   capability declared in the entry that is not in the allowed set.
6. Signature check. The verifier computes the bundle's signed
   material (blake3 of manifest plus binaries on disk, in the
   order documented in `ext_tooling/sign.py`) and verifies the
   Ed25519 signature attached to the extension-table entry against
   the author key.

A failure at any step returns `Untrusted` with a specific reason.
The reader propagates the reason to the user via
`ContributionUntrusted` so the user can diagnose without reading
the verifier source.

## Signed material

The signed material for a bundle is `blake3(manifest_bytes ||
binaries_in_declared_order)`. The manifest is `manifest.toml` as
emitted by `ext_tooling/sign.py`; the binaries are
`<bundle_name>.{wasm, so, dylib}` in that fixed order, with missing
binaries skipped.

The blake3 over manifest plus binaries means tampering with either
the manifest or any binary after signing invalidates the signature.
The signing tool rewrites the manifest to embed the real
author-pubkey and canonical-ids before computing the hash, so the
manifest the verifier rehashes is the same one the signer signed.

A natural alternative is to store the signed material in the
container's extension table rather than recomputing it from disk.
This option costs more on the wire (one hash per contribution per
file) but saves I/O at open time. PTWM chose the recompute-from-disk
shape because the kernel page cache absorbs the I/O cost after the
first read, and keeping the wire format free of per-publisher
overhead makes containers cheaper to redistribute. The trade-off
favours wire economy at the cost of one cold-cache read per
extension per fresh decoder process.

## Capabilities and policy

The capability map on each extension-table entry declares what the
contribution claims to need. The policy file
(`crates/ptwm-core/src/policy/file.rs`) declares what the host
allows. The verifier intersects them: the contribution is rejected
if any declared capability is denied by the policy.

The policy file shape is:

```toml
[allow]
extra = ["blake3:<canonical-id>"]   # additional trust

[ignore]
extensions = ["blake3:<canonical-id>"]   # explicit denial

[per_role.scale]
allow = [...]
deny  = [...]

[capabilities]
allow_host_imports = []
deny_native_deps_unverified = true
```

The `per_role` section lets the host scope policies to specific
classifier roles. A user who wants Order1ScaleAC available only for
scale planes (the role it was designed for) can set
`per_role.scale.allow = [order1_scale_ac_id]` without affecting
other roles.

The resolved policy flows through to the encoder via
`CompressionConfig.from_resolved_policy(rp)`, which restricts the
trial-encode menu to allowed canonical ids. The intent is that
ablation runs can disable a codec without modifying its source: the
policy file is the sole source of truth.

## What the verifier does not check

The verifier does not check that the contribution actually does
what its manifest claims. A contribution declared as a `PlaneCodec`
with `determinism = true` could be non-deterministic in practice;
the verifier has no way to detect this without running the
contribution against a reference distribution. The behavioural
equivalence claim is the contribution author's responsibility, and
the user's recourse if violated is to remove the trust entry.

The verifier does not check the contribution at install time beyond
what `ptwm ext install` validates (the manifest parses, the
signature verifies). A contribution that fails capability check at
verify time is rejected per-file, not per-install. This is
deliberate: the policy may change between install time and decode
time, so the verifier needs to revalidate.
